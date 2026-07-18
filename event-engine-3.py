import cv2
import time
import threading
from collections import deque
from ultralytics import YOLO

class RingBuffer:
    def __init__(self, buffer_seconds=10):
        self.buffer_seconds = buffer_seconds
        self.buffer = deque()

    def add_frame(self, frame):
        _, encoded_img = cv2.imencode('.jpg', frame)
        current_time = time.time()
        self.buffer.append((encoded_img, current_time))
        
        while self.buffer and (current_time - self.buffer[0][1] > self.buffer_seconds):
            self.buffer.popleft()

    def get_encoded_frames(self):
        return list(self.buffer) 

class EventEngine:
    def __init__(self, video_source=0):
        self.cap = cv2.VideoCapture(video_source)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video source: {video_source}")

        print("Camera initialized.")
        self.ring_buffer = RingBuffer(buffer_seconds=10)

        print("Loading YOLOv8 model...")
        self.model = YOLO('yolov8n.pt') 
        self.target_classes = ['person'] 
        self.confidence_threshold = 0.60 

        # STATE MACHINE VARIABLES
        self.STATE_IDLE = 'IDLE'
        self.STATE_POST_RECORDING = 'POST_RECORDING'
        self.STATE_COOLDOWN = 'COOLDOWN'
        
        self.current_state = self.STATE_IDLE
        self.consecutive_detections = 0
        self.required_consecutive_frames = 8 
        
        self.pre_event_frames = []
        self.post_event_frames = []
        self.post_record_duration = 10.0 
        self.cooldown_duration = 20.0    
        self.timer_end = 0
        
        # BRICK 4: Track event IDs
        self.event_counter = 0

    def save_event_data(self, pre_frames, post_frames, event_id):
        """
        This function runs in a BACKGROUND THREAD. 
        It decodes the bytes, saves an MP4 video, and saves a Snapshot.
        """
        # Combine pre and post frames chronologically
        all_frames = pre_frames + post_frames
        if not all_frames:
            return

        print(f"[Thread-{event_id}] Starting video export...")

        # 1. Decode the very first frame just to get the Video Width & Height
        first_frame_bytes = all_frames[0][0]
        first_frame_decoded = cv2.imdecode(first_frame_bytes, cv2.IMREAD_COLOR)
        h, w = first_frame_decoded.shape[:2]

        # 2. Calculate the exact FPS based on the timestamps of the captured frames
        start_time = all_frames[0][1]
        end_time = all_frames[-1][1]
        duration = end_time - start_time
        
        # Prevent division by zero
        fps = len(all_frames) / duration if duration > 0 else 20.0 

        # 3. Initialize OpenCV VideoWriter
        # 'mp4v' is the most compatible codec for OpenCV MP4 creation
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        video_filename = f"event_{event_id}.mp4"
        out = cv2.VideoWriter(video_filename, fourcc, fps, (w, h))

        # 4. Decode and write every frame to the video file
        for encoded_img, ts in all_frames:
            frame = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
            out.write(frame)

        out.release()
        print(f"[Thread-{event_id}] ✅ Video saved: {video_filename} ({duration:.1f}s at {fps:.1f} FPS)")

        # 5. Save the Snapshot (The exact frame that triggered the alarm - last frame of pre_event)
        trigger_frame_bytes = pre_frames[-1][0]
        trigger_frame = cv2.imdecode(trigger_frame_bytes, cv2.IMREAD_COLOR)
        snapshot_filename = f"event_{event_id}_snapshot.jpg"
        cv2.imwrite(snapshot_filename, trigger_frame)
        print(f"[Thread-{event_id}] 📸 Snapshot saved: {snapshot_filename}")

    def run(self):
        print("Starting Event Engine (Brick 4: Threaded Video Export)... Press 'q' to quit.")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            results = self.model(frame, verbose=False)
            detected_target = False
            
            for r in results:
                for box in r.boxes:
                    cls_name = self.model.names[int(box.cls[0])]
                    conf = float(box.conf[0])
                    if cls_name in self.target_classes and conf >= self.confidence_threshold:
                        detected_target = True
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(frame, f"{cls_name} {conf:.1f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # --- STATE MACHINE LOGIC ---
            status_text = ""
            status_color = (0, 255, 0)

            if self.current_state == self.STATE_IDLE:
                self.ring_buffer.add_frame(frame)
                status_text = "STATUS: SCANNING (IDLE)"
                
                if detected_target:
                    self.consecutive_detections += 1
                else:
                    self.consecutive_detections = 0 

                if self.consecutive_detections >= self.required_consecutive_frames:
                    self.event_counter += 1
                    print(f"\n🚨 EVENT {self.event_counter} TRIGGERED! Locking Pre-Event Buffer...")
                    
                    self.pre_event_frames = self.ring_buffer.get_encoded_frames()
                    self.post_event_frames = [] 
                    
                    self.timer_end = time.time() + self.post_record_duration
                    self.current_state = self.STATE_POST_RECORDING
                    self.consecutive_detections = 0

            elif self.current_state == self.STATE_POST_RECORDING:
                _, encoded_img = cv2.imencode('.jpg', frame)
                self.post_event_frames.append((encoded_img, time.time()))
                
                status_text = "STATUS: RECORDING POST-EVENT..."
                status_color = (0, 165, 255) # Orange

                if time.time() >= self.timer_end:
                    print(f"📦 EVENT {self.event_counter} RECORDING COMPLETE! Handing off to background thread...")
                    
                    # BRICK 4: Spawn Background Thread
                    # We pass the lists to the thread. We use `args=` to pass variables.
                    thread = threading.Thread(
                        target=self.save_event_data, 
                        args=(self.pre_event_frames, self.post_event_frames, self.event_counter)
                    )
                    thread.start()
                    
                    # Clear the lists in the main thread to free up RAM immediately.
                    # The background thread has its own reference to the data, so it won't crash.
                    self.pre_event_frames = []
                    self.post_event_frames = []
                    
                    self.timer_end = time.time() + self.cooldown_duration
                    self.current_state = self.STATE_COOLDOWN

            elif self.current_state == self.STATE_COOLDOWN:
                status_text = "STATUS: COOLDOWN"
                status_color = (0, 255, 255) # Yellow
                remaining = int(self.timer_end - time.time())
                cv2.putText(frame, f"Next scan in: {remaining}s", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

                if time.time() >= self.timer_end:
                    print("🔄 Cooldown finished. Resuming IDLE state.")
                    self.current_state = self.STATE_IDLE

            # --- VISUAL DEBUGGING ---
            buffer_status = f"State: {self.current_state} | Events Triggered: {self.event_counter}"
            cv2.putText(frame, buffer_status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, status_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

            display_frame = cv2.resize(frame, (800, 450))
            cv2.imshow("Event Engine - Brick 4 (Threading)", display_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cleanup()

    def cleanup(self):
        print("Cleaning up resources...")
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    CAMERA_SOURCE = 0 
    engine = EventEngine(video_source=CAMERA_SOURCE)
    engine.run()
