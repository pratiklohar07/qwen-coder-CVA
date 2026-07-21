import cv2
import time
import threading
import os
import json
import queue
import requests
import asyncio
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from datetime import datetime
from collections import deque
from ultralytics import YOLO

# Graceful fallback for video encoding
try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False
    print("⚠️ Warning: imageio not installed. Videos will be saved as mp4v. Run: pip install imageio imageio-ffmpeg")

# ==========================================
# BRICK 7: LIVE VIDEO STREAMING SERVER
# ==========================================
class LiveStreamServer:
    def __init__(self, port=8080):
        self.port = port
        self.latest_frame_bytes = None
        self.lock = threading.Lock()
        self.app = FastAPI(title="CCTV Live Stream")
        
        @self.app.get("/video_feed")
        async def video_feed():
            return StreamingResponse(self.generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

    def update_frame(self, frame):
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        with self.lock:
            self.latest_frame_bytes = buffer.tobytes()

    async def generate_frames(self):
        while True:
            with self.lock:
                frame = self.latest_frame_bytes
            if frame is None:
                await asyncio.sleep(0.1)
                continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            await asyncio.sleep(0.05) 

    def start(self):
        config = uvicorn.Config(self.app, host="0.0.0.0", port=self.port, log_level="warning")
        server = uvicorn.Server(config)
        
        def run_server():
            # FIX: Properly initialize asyncio event loop for Windows background threads
            asyncio.set_event_loop(asyncio.new_event_loop())
            server.run()
        
        # FIX: Changed target from server.run to run_server
        thread = threading.Thread(target=run_server, daemon=True)
        thread.start()
        print(f"📺 Live Stream Server started at http://localhost:{self.port}/video_feed")

# ==========================================
# BRICK 6: 24/7 CCTV RECORDER
# ==========================================
class CCTVRecorder:
    def __init__(self, output_dir="cctv_archive", segment_minutes=1, target_fps=20.0, resolution=(854, 480)):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.queue = queue.Queue(maxsize=1000) 
        self.segment_seconds = segment_minutes * 60
        self.fps = target_fps
        self.resolution = resolution
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = None
        self.current_file_path = None
        self.start_time = None
        self.running = True
        self.thread = threading.Thread(target=self._recording_loop, daemon=True)
        self.thread.start()
        print(f"🔴 24/7 CCTV DVR Started. Saving to: ./{self.output_dir}/ (Strict {segment_minutes} min chunks)")

    def _get_new_filename(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(self.output_dir, f"cctv_{timestamp}.mp4")

    def _recording_loop(self):
        while self.running:
            try: 
                frame = self.queue.get(timeout=1.0)
            except queue.Empty: 
                continue

            if self.writer is None or (time.time() - self.start_time >= self.segment_seconds):
                if self.writer is not None:
                    self.writer.release()
                    print(f"💾 [CCTV] Saved uniform 60s segment: {os.path.basename(self.current_file_path)}")
                self.current_file_path = self._get_new_filename()
                self.writer = cv2.VideoWriter(self.current_file_path, self.fourcc, self.fps, self.resolution)
                self.start_time = time.time()
            self.writer.write(frame)
        if self.writer is not None: self.writer.release()

    def add_frame(self, frame):
        frame_resized = cv2.resize(frame, self.resolution)
        try: 
            self.queue.put(frame_resized, timeout=0.5)
        except queue.Full: 
            pass 

    def stop(self):
        self.running = False
        self.thread.join()

# ==========================================
# BRICKS 1-5: AI EVENT ENGINE
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

    def get_encoded_frames(self): return list(self.buffer) 

class EventEngine:
    def __init__(self, video_source=0):
        if isinstance(video_source, int):
            self.cap = cv2.VideoCapture(video_source, cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(video_source)

        if not self.cap.isOpened(): 
            raise ValueError(f"Could not open video source: {video_source}")

        self.ring_buffer = RingBuffer(buffer_seconds=10)
        self.model = YOLO('best.pt') 
        self.target_classes = ['Fire-Detection', 'fire'] 
        self.confidence_threshold = 0.50

        self.STATE_IDLE, self.STATE_POST_RECORDING, self.STATE_COOLDOWN = 'IDLE', 'POST_RECORDING', 'COOLDOWN'
        self.current_state = self.STATE_IDLE
        self.consecutive_detections, self.required_consecutive_frames = 0, 12 
        self.pre_event_frames, self.post_event_frames = [], []
        self.post_record_duration, self.cooldown_duration = 10.0, 20.0    
        self.timer_end = 0
        
        self.counter_file = "event_counter.txt"
        if os.path.exists(self.counter_file):
            with open(self.counter_file, 'r') as f:
                self.event_counter = int(f.read().strip())
            print(f"🔄 Resumed Event Counter from local storage: Starting at ID {self.event_counter + 1}")
        else:
            self.event_counter = 0
            
        self.output_dir = "events_data"
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.backend_url = "http://localhost:8000/api/v1/events/upload"
        self.enable_upload = True  

        self.cctv_recorder = CCTVRecorder(segment_minutes=1) 
        
        self.stream_server = LiveStreamServer(port=8080)
        self.stream_server.start()

    def upload_to_backend(self, video_path, snapshot_path, metadata_dict):
        try:
            with open(video_path, 'rb') as vf, open(snapshot_path, 'rb') as sf:
                files = {'video': vf, 'snapshot': sf}
                data = {'metadata': json.dumps(metadata_dict)}
                headers = {'X-API-Key': 'super_secret_edge_key_123'}
                response = requests.post(self.backend_url, files=files, data=data, headers=headers, timeout=30)
                if response.status_code == 200: print("[Upload] ✅ SUCCESS: Sent to Backend")
                else: print(f"[Upload] ⚠️ Backend Error {response.status_code}")
        except Exception as e: 
            print(f"[Upload] ❌ FAILED: {e} (Is the backend running?)")

    def save_event_data(self, pre_frames, post_frames, event_id, detected_class):
        all_frames = pre_frames + post_frames
        if not all_frames: return

        event_folder_name = f"event_{event_id:03d}"
        event_dir = os.path.join(self.output_dir, event_folder_name)
        os.makedirs(event_dir, exist_ok=True)

        print(f"[Thread-{event_id}] Exporting to {event_folder_name}...")

        first_frame_decoded = cv2.imdecode(all_frames[0][0], cv2.IMREAD_COLOR)
        h, w = first_frame_decoded.shape[:2]
        start_time = all_frames[0][1]
        duration = all_frames[-1][1] - start_time
        fps = len(all_frames) / duration if duration > 0 else 20.0 

        video_filename = "video.mp4"
        video_path = os.path.join(event_dir, video_filename)
        
        if HAS_IMAGEIO:
            writer = imageio.get_writer(video_path, fps=fps, codec='libx264', output_params=['-preset', 'fast', '-pix_fmt', 'yuv420p'])
            for encoded_img, ts in all_frames:
                frame = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
                writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            writer.close()
        else:
            out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            for encoded_img, ts in all_frames: out.write(cv2.imdecode(encoded_img, cv2.IMREAD_COLOR))
            out.release()

        trigger_frame = cv2.imdecode(pre_frames[-1][0], cv2.IMREAD_COLOR)
        snapshot_filename = "snapshot.jpg"
        snapshot_path = os.path.join(event_dir, snapshot_filename)
        cv2.imwrite(snapshot_path, trigger_frame)

        metadata = {
            "event_id": event_id, "object_detected": detected_class,
            "timestamp_unix": start_time,
            "timestamp_human": datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S'),
            "duration_seconds": round(duration, 2), "fps": round(fps, 2), "resolution": f"{w}x{h}"
        }
        metadata_path = os.path.join(event_dir, "metadata.json")
        with open(metadata_path, 'w') as f: json.dump(metadata, f, indent=4)

        print(f"[Thread-{event_id}] ✅ Local save complete in {event_folder_name}.")
        if self.enable_upload: self.upload_to_backend(video_path, snapshot_path, metadata)

    def run(self):
        print("Starting Edge Engine... Press 'q' to quit.")
        # FIX: Added try/except/finally to catch silent crashes and keep terminal open
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret: 
                    print("❌ CRITICAL: Camera feed lost (ret=False). The camera disconnected or driver crashed.")
                    break

                self.cctv_recorder.add_frame(frame)
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

                status_text, status_color = "", (0, 255, 0)
                if self.current_state == self.STATE_IDLE:
                    self.ring_buffer.add_frame(frame)
                    status_text = "STATUS: SCANNING (IDLE)"
                    if detected_target: self.consecutive_detections += 1
                    else: self.consecutive_detections = 0 
                    if self.consecutive_detections >= self.required_consecutive_frames:
                        self.event_counter += 1
                        with open(self.counter_file, 'w') as f:
                            f.write(str(self.event_counter))
                        
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
                        threading.Thread(target=self.save_event_data, args=(self.pre_event_frames, self.post_event_frames, self.event_counter, current_detected_class)).start()
                        self.pre_event_frames = self.post_event_frames = []
                        self.timer_end = time.time() + self.cooldown_duration
                        self.current_state = self.STATE_COOLDOWN
                elif self.current_state == self.STATE_COOLDOWN:
                    status_text = "STATUS: COOLDOWN"
                    status_color = (0, 255, 255)
                    if time.time() >= self.timer_end: self.current_state = self.STATE_IDLE

                cv2.putText(frame, f"State: {self.current_state} | Events: {self.event_counter}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, status_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
                self.stream_server.update_frame(frame)

                display_frame = cv2.resize(frame, (800, 450))
                cv2.imshow("Hybrid Engine", display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'): break
                
        except Exception as e:
            print(f"\n❌ FATAL ERROR in main loop: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()
            print("\n⚠️ Engine stopped. Press ENTER to close terminal...")
            input() # Prevents terminal from instantly closing

    def cleanup(self):
        self.cctv_recorder.stop()
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    CAMERA_SOURCE = 0 
    engine = EventEngine(video_source=CAMERA_SOURCE)
    engine.run()
