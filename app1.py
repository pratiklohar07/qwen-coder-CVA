import cv2
import time
from collections import deque
from ultralytics import YOLO

class RingBuffer:
    """
    A memory-efficient circular buffer that stores compressed JPEG frames.
    """
    def __init__(self, buffer_seconds=10, target_fps=30):
        self.max_frames = buffer_seconds * target_fps
        self.buffer = deque(maxlen=self.max_frames)
        self.target_fps = target_fps

    def add_frame(self, frame):
        # Encode frame to JPEG in memory to save RAM
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
        print(f"Camera initialized. Detected FPS: {self.actual_fps}")
        
        self.ring_buffer = RingBuffer(buffer_seconds=10, target_fps=int(self.actual_fps))

        # --- BRICK 2: YOLOv8 INITIALIZATION ---
        print("Loading YOLOv8 model... (this will download 'yolov8n.pt' on first run)")
        # 'n' stands for Nano. It's the fastest model, perfect for live CCTV feeds.
        self.model = YOLO('yolov8n.pt') 
        
        # We will trigger an "event" if we see a person or a dog.
        # Later, you will replace this list with ['fire', 'snake'] and use a custom model.
        self.target_classes = ['person', 'dog', 'cat'] 
        self.confidence_threshold = 0.60 # Only trust detections above 60%

    def run(self):
        print("Starting Event Engine Loop... Press 'q' to quit.")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            # 1. Always add frame to the Ring Buffer first
            self.ring_buffer.add_frame(frame)

            # 2. Run YOLOv8 Inference
            # verbose=False stops YOLO from printing stats to the console every frame
            results = self.model(frame, verbose=False)
            
            detected_target = False
            current_detections = []

            # 3. Process Inference Results
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    cls_id = int(box.cls[0])
                    cls_name = self.model.names[cls_id]
                    conf = float(box.conf[0])
                    
                    # Check if this is one of our target classes
                    if cls_name in self.target_classes and conf >= self.confidence_threshold:
                        detected_target = True
                        current_detections.append(cls_name)
                        
                        # Draw bounding box
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        label = f"{cls_name} {conf:.2f}"
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # --- VISUAL DEBUGGING ---
            buffer_status = f"Buffer: {len(self.ring_buffer.buffer)}/{self.ring_buffer.max_frames} frames"
            cv2.putText(frame, buffer_status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if detected_target:
                status_text = "STATUS: TARGET DETECTED!"
                status_color = (0, 0, 255) # Red
                print(f"⚠️ Detection! Found: {', '.join(set(current_detections))}")
            else:
                status_text = "STATUS: SCANNING..."
                status_color = (0, 255, 0) # Green

            cv2.putText(frame, status_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

            display_frame = cv2.resize(frame, (800, 450))
            cv2.imshow("Event Engine - Brick 2 (YOLOv8 Test)", display_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cleanup()

    def cleanup(self):
        print("Cleaning up resources...")
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    CAMERA_SOURCE = 0 # Change to 1 or 2 if your external webcam is not on index 0
    engine = EventEngine(video_source=CAMERA_SOURCE)
    engine.run()
