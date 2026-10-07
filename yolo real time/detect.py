import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# Force OpenCV/FFmpeg to use TCP for RTSP to prevent packet loss and decoding stutters
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

import argparse
import time
import threading
import cv2
import numpy as np
from ultralytics import YOLO
import torch

class VideoCaptureThread:
    """
    A thread-based VideoCapture wrapper that always returns the most recent frame
    to completely eliminate RTSP stream buffering and delay.
    """
    def __init__(self, source):
        self.source = source
        # Use DSHOW on Windows to prevent MSMF driver failures/hangs
        if isinstance(source, int):
            self.cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.cap.release()
                self.cap = cv2.VideoCapture(source)
        else:
            self.cap = cv2.VideoCapture(source)
            
        self.ret = False
        self.frame = None
        self.running = True
        self.lock = threading.Lock()
        
        # Start background frame grabber
        self.thread = threading.Thread(target=self._grab_frames, daemon=True)
        self.thread.start()

    def _grab_frames(self):
        while self.running:
            try:
                ret, frame = self.cap.read()
                with self.lock:
                    self.ret = ret
                    if ret and frame is not None:
                        self.frame = frame.copy()
                    else:
                        self.frame = None
                if not ret:
                    time.sleep(0.5)
                    self.cap.release()
                    if isinstance(self.source, int):
                        self.cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
                        if not self.cap.isOpened():
                            self.cap.release()
                            self.cap = cv2.VideoCapture(self.source)
                    else:
                        self.cap = cv2.VideoCapture(self.source)
            except Exception as e:
                print(f"[Warning] Frame grab exception: {e}")
                time.sleep(0.5)
            time.sleep(0.01)

    def read(self):
        with self.lock:
            if self.frame is None:
                return False, None
            return self.ret, self.frame.copy()

    def isOpened(self):
        return self.cap.isOpened()

    def set(self, propId, value):
        self.cap.set(propId, value)

    def release(self):
        self.running = False
        self.thread.join(timeout=1.0)
        self.cap.release()

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Real-Time YOLOv26 Object Detection")
    parser.add_argument("--model", type=str, default="yolo26m.pt", help="YOLOv26 model file (e.g., yolo26m.pt, yolo26s.pt)")
    parser.add_argument("--camera", type=str, default="0", help="Camera index (e.g., 0) or RTSP URL stream")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25)")
    parser.add_argument("--imgsz", type=int, default=320, help="YOLO inference resolution (default: 320 for speed)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = args.model

    print(f"Loading model: {model_path} on {device}...")
    try:
        model = YOLO(model_path, task="detect")
        print("Model loaded. Running dry-run warm-up...")
        dummy_frame = np.zeros((args.imgsz, args.imgsz, 3), dtype=np.uint8)
        model(dummy_frame, device=device, verbose=False)
        print("Model warm-up completed!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Handle camera input (convert digits to int, keep strings as RTSP URLs)
    camera_source = args.camera
    if camera_source.isdigit():
        camera_source = int(camera_source)

    print(f"Opening camera source: {camera_source}...")
    cap = VideoCaptureThread(camera_source)

    if not cap.isOpened():
        print(f"Error: Could not open camera {args.camera}.")
        print("Please verify the camera index or check if the camera is used by another application.")
        return

    # Set camera resolution (optional, try 1280x720)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    window_name = "YOLOv26 Real-Time Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    print("\n--------------------------------------------------")
    print(" YOLO Real-Time Detection Started!")
    print(" - Press 'q' to quit.")
    print(" - Press 's' to pause/resume.")
    print(" - Press 'f' to toggle Fullscreen.")
    print("--------------------------------------------------\n")

    prev_time = 0
    paused = False

    while True:
        # Process UI window events on every iteration to keep the window responsive
        # even when camera frames are not ready yet
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            paused = not paused
            if paused:
                print("Paused. Press 's' to resume.")
            else:
                print("Resumed.")
                prev_time = time.time()  # Reset FPS calculation time after pause
        elif key == ord('f'):
            # Toggle fullscreen state
            is_fullscreen = cv2.getWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN) == cv2.WINDOW_FULLSCREEN
            if is_fullscreen:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                print("Fullscreen disabled.")
            else:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                print("Fullscreen enabled.")

        if not paused:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            results = model(frame, imgsz=args.imgsz, conf=args.conf, device=device, verbose=False)

            # Get the annotated frame (YOLO draws bounding boxes automatically)
            annotated_frame = results[0].plot()

            # Calculate FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time else 0
            prev_time = curr_time

            # Overlay FPS on the screen
            fps_text = f"FPS: {fps:.1f}"
            cv2.putText(
                annotated_frame,
                fps_text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

            # Display total counts of detected objects
            counts = {}
            if results[0].boxes:
                for box in results[0].boxes:
                    cls_id = int(box.cls[0])
                    name = model.names[cls_id]
                    counts[name] = counts.get(name, 0) + 1

            count_y = 80
            for name, count in counts.items():
                count_text = f"{name}: {count}"
                cv2.putText(
                    annotated_frame,
                    count_text,
                    (20, count_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2,
                    cv2.LINE_AA
                )
                count_y += 30

            # Show window
            cv2.imshow(window_name, annotated_frame)

    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    print("Camera released. Window closed. Exiting.")

if __name__ == "__main__":
    main()
