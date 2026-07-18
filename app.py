import cv2
import time
from collections import deque
import numpy as np

class RingBuffer:
    """
    A memory-efficient circular buffer that stores compressed JPEG frames.
    """
    def __init__(self, buffer_seconds=10, target_fps=30):
        self.max_frames = buffer_seconds * target_fps
        # Using deque with a maxlen automatically discards the oldest frames
        self.buffer = deque(maxlen=self.max_frames)
        self.target_fps = target_fps
        self.frame_count = 0

    def add_frame(self, frame):
        """
        Compresses the raw OpenCV frame to JPEG and stores it in the buffer.
        """
        # Encode frame to JPEG format in memory (quality 80% to save space)
        # encode_params = [cv2.IMWRITE_JPEG_QUALITY, 80] 
        # (Optional: uncomment above to lower quality and save even more RAM)
        _, encoded_img = cv2.imencode('.jpg', frame)
        
        # Store the byte data and the exact timestamp
        timestamp = time.time()
        self.buffer.append((encoded_img, timestamp))
        self.frame_count += 1

    def get_all_frames(self):
        """
        Returns all frames currently in the buffer, decoded back to OpenCV format.
        """
        decoded_frames = []
        for encoded_img, timestamp in self.buffer:
            # Decode JPEG bytes back into an OpenCV numpy array
            frame = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
            decoded_frames.append((frame, timestamp))
        return decoded_frames

class EventEngine:
    def __init__(self, video_source=0):
        # video_source = 0 (default webcam), 1 (capture card), or 'video.mp4' (test file)
        self.cap = cv2.VideoCapture(video_source)
        
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video source: {video_source}")

        # Get actual FPS of the capture card to calculate buffer size accurately
        self.actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.actual_fps <= 1: 
            self.actual_fps = 30 # Fallback to 30 if OpenCV fails to read FPS
            
        print(f"Camera initialized. Detected FPS: {self.actual_fps}")
        
        # Initialize our 10-second buffer
        self.ring_buffer = RingBuffer(buffer_seconds=10, target_fps=int(self.actual_fps))

    def run(self):
        print("Starting Event Engine Loop... Press 'q' to quit.")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            # 1. Add the current frame to our pre-event memory
            self.ring_buffer.add_frame(frame)

            # --- VISUAL DEBUGGING ---
            # Let's draw the buffer size on the screen to prove it's working
            buffer_status = f"Buffer: {len(self.ring_buffer.buffer)}/{self.ring_buffer.max_frames} frames"
            cv2.putText(frame, buffer_status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Resize for easier viewing on standard monitors
            display_frame = cv2.resize(frame, (800, 450))
            cv2.imshow("Event Engine - Brick 1 (Buffer Test)", display_frame)

            # Break loop on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cleanup()

    def cleanup(self):
        print("Cleaning up resources...")
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # CHANGE THIS to match your capture card! 
    # Usually 0 is built-in webcam, 1 or 2 is the DSLR Capture Card.
    # You can also put a video file path here like "test_fire.mp4" to test without hardware.
    CAMERA_SOURCE = 0 
    
    engine = EventEngine
