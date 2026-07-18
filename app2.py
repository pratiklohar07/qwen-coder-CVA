import cv2
import time
from collections import deque
from ultralytics import YOLO

class RingBuffer:
    def __init__(self, buffer_seconds=10, target_fps=30):
        self.max_frames = buffer_seconds * target_fps
        self.buffer = deque(maxlen=self.max_frames)

    def add_frame(self, frame):
        _, encoded_img = cv2.imencode('.jpg', frame)
        timestamp = time.time()
        self.buffer.append((encoded_img, timestamp))

    def get_all_frames(self):
        decoded_frames = []
        for encoded_img, timestamp in self.buffer:
            frame = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
            decoded_frames.append((frame, timestamp))
        return decoded_frames

class EventEngine:
    def __init__(self, video_source=0):
        self.cap = cv2.VideoCapture(video_source)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video source: {video_source}")

        self.actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.actual_fps <= 1: 
            self.actual_fps = 30 
        
        self.ring_buffer = RingBuffer(buffer_seconds=10, target_fps=int(self.actual_fps))

        print("Loading YOLOv8 model...")
        self.model = YOLO('yolov8n.pt') 
        self.target_classes = ['person'] # Using person for testing
        self.confidence_threshold = 0.60 

        # --- BRICK 3: STATE MACHINE VARIABLES ---
        self.STATE_IDLE = 'IDLE'
        self.STATE_POST_RECORDING = 'POST_RECORDING'
        self.STATE_COOLDOWN = 'COOLDOWN'
        
        self.current_state = self.STATE_IDLE
        
        # Trigger logic
        self.consecutive_detections = 0
        self.required_consecutive_frames = 8 # Must see target for ~8 frames to trigger (prevents glitches)
        
        # Timers & Buffers
        self.pre_event_frames = []
        self.post_event_frames = []
        self.post_record_duration = 10.0 # Record 10s AFTER trigger
        self.cooldown_duration = 20.0    # Wait 20s before allowing another event
        self.timer_end = 0

    def run(self):
        print("Starting Event Engine (State Machine)... Press 'q' to quit.")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Run YOLO on every frame regardless of state (so we can draw boxes)
            results = self.model(frame, verbose=False)
            detected_target = False
            
            for r in results:
                for box in r.boxes:
                    cls_name = self.model.names[int(box.cls[0])]
                    conf = float(box.conf[0])
                    if cls_name in self.target_classes and conf >= self.confidence_threshold:
                        detected_target = True
                        # Draw bounding box
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
                    self.consecutive_detections = 0 # Reset if target is lost

                # TRIGGER EVENT!
                if self.consecutive_detections >= self.required_consecutive_frames:
                    print("🚨 EVENT TRIGGERED! Locking 10s Pre-Event Buffer...")
                    self.pre_event_frames = self.ring_buffer.get_all_frames()
                    self.post_event_frames = [] # Reset post buffer
                    self.timer_end = time.time() + self.post_record_duration
                    self.current_state = self.STATE_POST_RECORDING
                    self.consecutive_detections = 0

            elif self.current_state == self.STATE_POST_RECORDING:
                # Record post-event frames into a temporary list
                _, encoded_img = cv2.imencode('.jpg', frame)
                self.post_event_frames.append((encoded_img, time.time()))
                status_text = "STATUS: RECORDING POST-EVENT..."
                status_color = (0, 165, 255) # Orange

                # Check if 10 seconds have passed
                if time.time() >= self.timer_end:
                    print(f"📦 EVENT COMPLETE! Captured {len(self.pre_event_frames)} pre-frames and {len(self.post_event_frames)} post-frames.")
                    print("   (In Brick 4, we will save these lists as actual MP4 files!)")
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
            buffer_status = f"State: {self.current_state} | Buffer: {len(self.ring_buffer.buffer)}/{self.ring_buffer.max_frames}"
            cv2.putText(frame, buffer_status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, status_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

            display_frame = cv2.resize(frame, (800, 450))
            cv2.imshow("Event Engine - Brick 3 (State Machine)", display_frame)

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
