# recods event after before 10 seconds of detection #saves metadata in form of json #with cool down time and 122 buffer


# event_engine_core.py
# Run this SECOND: python event_engine_core.py

import cv2
import time
import threading
import os
import json
import requests
from datetime import datetime
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
        
        # --- BRICK 5: BACKEND CONFIGURATION ---
        self.backend_url = "http://localhost:8000/api/v1/events/upload"
        self.enable_upload = True  # Set to False if you want local-only mode

    def upload_to_backend(self, video_path, snapshot_path, metadata_dict):
        """
        Sends the event files to the backend server via HTTP POST.
        Uses multipart/form-data (the standard for file uploads).
        """
        try:
            print(f"[Upload] Sending Event {metadata_dict['event_id']} to backend...")
            
            # Open the files in binary mode
            with open(video_path, 'rb') as video_file, \
                 open(snapshot_path, 'rb') as snapshot_file:
                
                # Prepare the multipart payload
                files = {
                    'video': (os.path.basename(video_path), video_file, 'video/mp4'),
                    'snapshot': (os.path.basename(snapshot_path), snapshot_file, 'image/jpeg')
                }
                
                # Metadata is sent as a JSON string in a form field
                data = {
                    'metadata': json.dumps(metadata_dict)
                }
                
                # Send the POST request (timeout after 30 seconds)
                response = requests.post(
                    self.backend_url, 
                    files=files, 
                    data=data, 
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"[Upload] ✅ SUCCESS: {result.get('message')}")
                else:
                    print(f"[Upload] ⚠️ Backend returned status {response.status_code}: {response.text}")

        except requests.exceptions.ConnectionError:
            print(f"[Upload] ❌ FAILED: Cannot connect to backend at {self.backend_url}")
            print(f"[Upload]    Files are saved locally. Will need manual retry later.")
        except requests.exceptions.Timeout:
            print(f"[Upload] ❌ FAILED: Backend took too long to respond (Timeout)")
        except Exception as e:
            print(f"[Upload] ❌ FAILED: Unexpected error: {e}")

    def save_event_data(self, pre_frames, post_frames, event_id, detected_class):
        """
        Runs in a BACKGROUND THREAD.
        Saves files locally, then uploads to backend.
        """
        all_frames = pre_frames + post_frames
        if not all_frames:
            return

        print(f"[Thread-{event_id}] Starting background export...")

        # 1. Get dimensions and calculate FPS
        first_frame_decoded = cv2.imdecode(all_frames[0][0], cv2.IMREAD_COLOR)
        h, w = first_frame_decoded.shape[:2]

        start_time = all_frames[0][1]
        end_time = all_frames[-1][1]
        duration = end_time - start_time
        fps = len(all_frames) / duration if duration > 0 else 20.0 

        # 2. Save MP4 locally
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        video_filename = f"event_{event_id}_video.mp4"
        video_path = os.path.join(self.output_dir, video_filename)
        out = cv2.VideoWriter(video_path, fourcc, fps, (w, h))

        for encoded_img, ts in all_frames:
            frame = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
            out.write(frame)
        out.release()

        # 3. Save Snapshot locally
        trigger_frame = cv2.imdecode(pre_frames[-1][0], cv2.IMREAD_COLOR)
        snapshot_filename = f"event_{event_id}_snapshot.jpg"
        snapshot_path = os.path.join(self.output_dir, snapshot_filename)
        cv2.imwrite(snapshot_path, trigger_frame)

        # 4. Create Metadata
        metadata = {
            "event_id": event_id,
            "object_detected": detected_class,
            "timestamp_unix": start_time,
            "timestamp_human": datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S'),
            "duration_seconds": round(duration, 2),
            "fps": round(fps, 2),
            "resolution": f"{w}x{h}",
            "files": {
                "video": video_filename,
                "snapshot": snapshot_filename
            }
        }
        
        # Save metadata locally
        json_path = os.path.join(self.output_dir, f"event_{event_id}_metadata.json")
        with open(json_path, 'w') as f:
            json.dump(metadata, f, indent=4)

        print(f"[Thread-{event_id}] ✅ Local save complete.")

        # 5. --- BRICK 5: UPLOAD TO BACKEND ---
        if self.enable_upload:
            self.upload_to_backend(video_path, snapshot_path, metadata)

    def run(self):
        print("Starting Event Engine (Brick 5: Backend Upload)... Press 'q' to quit.")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

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
                        cv2.putText(frame, f"{cls_name} {conf:.1f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

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
                    print(f"\n🚨 EVENT {self.event_counter} TRIGGERED!")
                    
                    self.pre_event_frames = self.ring_buffer.get_encoded_frames()
                    self.post_event_frames = [] 
                    
                    self.timer_end = time.time() + self.post_record_duration
                    self.current_state = self.STATE_POST_RECORDING
                    self.consecutive_detections = 0

            elif self.current_state == self.STATE_POST_RECORDING:
                _, encoded_img = cv2.imencode('.jpg', frame)
                self.post_event_frames.append((encoded_img, time.time()))
                
                status_text = "STATUS: RECORDING POST-EVENT..."
                status_color = (0, 165, 255)

                if time.time() >= self.timer_end:
                    print(f"📦 EVENT {self.event_counter} COMPLETE! Starting background thread...")
                    
                    thread = threading.Thread(
                        target=self.save_event_data, 
                        args=(self.pre_event_frames, self.post_event_frames, self.event_counter, current_detected_class)
                    )
                    thread.start()
                    
                    self.pre_event_frames = []
                    self.post_event_frames = []
                    
                    self.timer_end = time.time() + self.cooldown_duration
                    self.current_state = self.STATE_COOLDOWN

            elif self.current_state == self.STATE_COOLDOWN:
                status_text = "STATUS: COOLDOWN"
                status_color = (0, 255, 255)
                remaining = int(self.timer_end - time.time())
                cv2.putText(frame, f"Next scan in: {remaining}s", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

                if time.time() >= self.timer_end:
                    print("🔄 Cooldown finished. Resuming IDLE.")
                    self.current_state = self.STATE_IDLE

            buffer_status = f"State: {self.current_state} | Events: {self.event_counter}"
            cv2.putText(frame, buffer_status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, status_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

            display_frame = cv2.resize(frame, (800, 450))
            cv2.imshow("Event Engine - Brick 5 (Backend Upload)", display_frame)

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
