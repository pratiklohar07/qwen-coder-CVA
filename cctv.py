import cv2
import time
import threading
import os
import json
import queue
import requests
from datetime import datetime
from collections import deque
from ultralytics import YOLO

# ==========================================
# BRICK 6: THE 24/7 CCTV RECORDER
# ==========================================
class CCTVRecorder:
    def __init__(self, output_dir="cctv_archive", segment_minutes=1, target_fps=20.0, resolution=(854, 480)):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Queue to safely pass frames from main thread to DVR thread
        self.queue = queue.Queue(maxsize=300) 
        
        self.segment_seconds = segment_minutes * 60
        self.fps = target_fps
        self.resolution = resolution
        
        # mp4v is standard for OpenCV
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = None
        self.current_file_path = None
        self.start_time = None
        
        self.running = True
        self.thread = threading.Thread(target=self._recording_loop, daemon=True)
        self.thread.start()
        print(f"🔴 24/7 CCTV DVR Started. Saving to: ./{self.output_dir}/ (Segmenting every {segment_minutes} min)")

    def _get_new_filename(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"cctv_{timestamp}.mp4"
        return os.path.join(self.output_dir, filename)

    def _recording_loop(self):
        """Runs continuously in the background."""
        while self.running:
            try:
                frame = self.queue.get(timeout=1.0)
            except queue.Empty:
                continue

            # Check if we need to start a new segment
            if self.writer is None or (time.time() - self.start_time >= self.segment_seconds):
                if self.writer is not None:
                    self.writer.release()
                    print(f"💾 [CCTV] Saved segment: {os.path.basename(self.current_file_path)}")
                
                self.current_file_path = self._get_new_filename()
                self.writer = cv2.VideoWriter(self.current_file_path, self.fourcc, self.fps, self.resolution)
                self.start_time = time.time()

            self.writer.write(frame)
            
        # Cleanup on exit
        if self.writer is not None:
            self.writer.release()
            print(f"💾 [CCTV] Saved final segment: {os.path.basename(self.current_file_path)}")

    def add_frame(self, frame):
        """Called by the main camera loop. Resizes and queues the frame."""
        # We resize CCTV footage to 480p to save massive amounts of space on your 1TB drive.
        # The AI Event Engine will still use the full 1080p original frame!
        frame_resized = cv2.resize(frame, self.resolution)
        
        try:
            # put_nowait ensures the main AI loop NEVER freezes if the hard drive is slow
            self.queue.put_nowait(frame_resized)
        except queue.Full:
            pass # Drop frame if DVR thread is falling behind (better than crashing AI)

    def stop(self):
        self.running = False
        self.thread.join()


# ==========================================
# BRICKS 1-5: THE AI EVENT ENGINE
# ==========================================
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
        
        # Get resolution for AI Engine (Full Quality)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.ring_buffer = RingBuffer(buffer_seconds=10)

        print("Loading YOLOv8 model...")
        self.model = YOLO('yolov8n.pt') 
        self.target_classes = ['person'] 
        self.confidence_threshold = 0.60 

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
        self.event_counter = 0
        
        self.output_dir = "events_data"
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.backend_url = "http://localhost:8000/api/v1/events/upload"
        self.enable_upload = True  

        # --- BRICK 6: INITIALIZE CCTV DVR ---
        # Set to 1 minute for testing. Change to 15 or 60 for production!
        self.cctv_recorder = CCTVRecorder(segment_minutes=1) 

    def upload_to_backend(self, video_path, snapshot_path, metadata_dict):
        try:
            print(f"[Upload] Sending Event {metadata_dict['event_id']} to backend...")
            with open(video_path, 'rb') as video_file, \
                 open(snapshot_path, 'rb') as snapshot_file:
                files = {
                    'video': (os.path.basename(video_path), video_file, 'video/mp4'),
                    'snapshot': (os.path.basename(snapshot_path), snapshot_file, 'image/jpeg')
                }
                data = {'metadata': json.dumps(metadata_dict)}
                response = requests.post(self.backend_url, files=files, data=data, timeout=30)
                
                if response.status_code == 200:
                    print(f"[Upload] ✅ SUCCESS")
                else:
                    print(f"[Upload] ⚠️ Backend Error {response.status_code}")
        except Exception as e:
            print(f"[Upload] ❌ FAILED: {e}")

    def save_event_data(self, pre_frames, post_frames, event_id, detected_class):
        all_frames = pre_frames + post_frames
        if not all_frames: return

        print(f"[Thread-{event_id}] Exporting AI Event...")
        first_frame_decoded = cv2.imdecode(all_frames[0][0], cv2.IMREAD_COLOR)
        h, w = first_frame_decoded.shape[:2]

        start_time = all_frames[0][1]
        duration = all_frames[-1][1] - start_time
        fps = len(all_frames) / duration if duration > 0 else 20.0 

        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        video_filename = f"event_{event_id}_video.mp4"
        video_path = os.path.join(self.output_dir, video_filename)
        out = cv2.VideoWriter(video_path, fourcc, fps, (w, h))

        for encoded_img, ts in all_frames:
            frame = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
            out.write(frame)
        out.release()

        trigger_frame = cv2.imdecode(pre_frames[-1][0], cv2.IMREAD_COLOR)
        snapshot_filename = f"event_{event_id}_snapshot.jpg"
        snapshot_path = os.path.join(self.output_dir, snapshot_filename)
        cv2.imwrite(snapshot_path, trigger_frame)

        metadata = {
            "event_id": event_id, "object_detected": detected_class,
            "timestamp_unix": start_time,
            "timestamp_human": datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S'),
            "duration_seconds": round(duration, 2), "fps": round(fps, 2),
            "resolution": f"{w}x{h}"
        }
        json_path = os.path.join(self.output_dir, f"event_{event_id}_metadata.json")
        with open(json_path, 'w') as f: json.dump(metadata, f, indent=4)

        if self.enable_upload:
            self.upload_to_backend(video_path, snapshot_path, metadata)

    def run(self):
        print("Starting Hybrid Engine (AI Events + 24/7 CCTV)... Press 'q' to quit.")
        
        while True:
            ret, frame = self.cap.read()
            if not ret: break

            # --- TRACK B: Feed the 24/7 CCTV DVR instantly ---
            self.cctv_recorder.add_frame(frame)

            # --- TRACK A: AI Event Engine ---
            results = self.model(frame, verbose=False)
            detected_target = False
            current_detected_class = "Unknown"
            
            for r in results:
                for box in r.boxes:
                    cls_name = self.model.names[int(box.cls[0])]
                    conf = float(box.conf[0])
                    if cls_name in self.target_classes and conf >= self.confidence_threshold:
                        detected_target = True
                        current_detected_class = cls_name
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

            status_text = ""
            status_color = (0, 255, 0)

            if self.current_state == self.STATE_IDLE:
                self.ring_buffer.add_frame(frame)
                status_text = "STATUS: SCANNING (IDLE)"
                if detected_target: self.consecutive_detections += 1
                else: self.consecutive_detections = 0 

                if self.consecutive_detections >= self.required_consecutive_frames:
                    self.event_counter += 1
                    print(f"\n🚨 EVENT {self.event_counter} TRIGGERED!")
                    self.pre_event_frames = self.ring_buffer.get_encoded_frames()
                    self.post_event_frames = [] 
                    self.timer_end = time.time() + self.post_record_duration
                    self.current_state = self.STATE_POST_RECORDING

            elif self.current_state == self.STATE_POST_RECORDING:
                _, encoded_img = cv2.imencode('.jpg', frame)
                self.post_event_frames.append((encoded_img, time.time()))
                status_text = "STATUS: RECORDING POST-EVENT..."
                status_color = (0, 165, 255)
                if time.time() >= self.timer_end:
                    thread = threading.Thread(target=self.save_event_data, args=(self.pre_event_frames, self.post_event_frames, self.event_counter, current_detected_class))
                    thread.start()
                    self.pre_event_frames = []
                    self.post_event_frames = []
                    self.timer_end = time.time() + self.cooldown_duration
                    self.current_state = self.STATE_COOLDOWN

            elif self.current_state == self.STATE_COOLDOWN:
                status_text = "STATUS: COOLDOWN"
                status_color = (0, 255, 255)
                if time.time() >= self.timer_end: self.current_state = self.STATE_IDLE

            # Visuals
            cv2.putText(frame, f"State: {self.current_state} | Events: {self.event_counter}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, status_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

            display_frame = cv2.resize(frame, (800, 450))
            cv2.imshow("Hybrid Engine (AI + CCTV)", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

        self.cleanup()

    def cleanup(self):
        print("Stopping CCTV Recorder...")
        self.cctv_recorder.stop()
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    CAMERA_SOURCE = 0 
    engine = EventEngine(video_source=CAMERA_SOURCE)
    engine.run()
