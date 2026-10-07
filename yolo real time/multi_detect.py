import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# Force OpenCV/FFmpeg to use TCP for RTSP to prevent packet loss and decoding stutters
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

import argparse
import time
import threading
import math
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
            ret, frame = self.cap.read()
            with self.lock:
                self.ret = ret
                if ret:
                    self.frame = frame.copy()
            if not ret:
                time.sleep(0.5)
                self.cap.release()
                self.cap = cv2.VideoCapture(self.source)
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

# List of all available camera IPs from the SADP tool
CAMERA_IPS = [
    "192.168.1.28",  # Camera 1
    "192.168.1.22",  # Camera 2
    "192.168.1.38",  # Camera 3
    "192.168.1.26",  # Camera 4
    "192.168.1.34",  # Camera 5
    "192.168.1.36",  # Camera 6
    "192.168.1.2",   # Camera 7
    "192.168.1.17",  # Camera 8
    "192.168.1.15",  # Camera 9
    "192.168.1.37",  # Camera 10
    "192.168.1.50"   # Camera 11
]

class CameraStream(threading.Thread):
    def __init__(self, camera_id, ip, password, model_name, conf, imgsz, skip_frames):
        super().__init__()
        self.camera_id = camera_id
        self.ip = ip
        self.password = password
        self.model_name = model_name
        self.conf = conf
        self.imgsz = imgsz
        self.skip_frames = skip_frames
        
        # Stream URL using Channel 102 (Sub-stream for performance)
        self.rtsp_url = f"rtsp://admin:{password}@{ip}:554/Streaming/Channels/102"
        
        self.running = True
        self.frame = None
        self.lock = threading.Lock()
        
        # Determine device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load local YOLO model for this thread
        self.model = YOLO(model_name)
        
        self.daemon = True  # Allows thread to exit when main program exits

    def run(self):
        print(f"[Camera {self.camera_id}] Starting stream for IP {self.ip}...")
        cap = VideoCaptureThread(self.rtsp_url)
        
        frame_count = 0
        last_results = None
        
        while self.running:
            ret, raw_frame = cap.read()
            
            if not ret or raw_frame is None:
                time.sleep(0.01)
                continue
            
            frame_count += 1
            
            # Perform YOLO inference every N frames, and use cached results for intermediate frames
            if frame_count % self.skip_frames == 0 or last_results is None:
                results = self.model(raw_frame, imgsz=self.imgsz, conf=self.conf, device=self.device, verbose=False)
                last_results = results[0]
                annotated_frame = last_results.plot()
            else:
                # Plot the cached boxes on the new frame to keep video motion smooth
                annotated_frame = last_results.plot(img=raw_frame)
            
            # Write annotated frame thread-safely
            with self.lock:
                self.frame = annotated_frame.copy()
            
            # Control frame rate slightly to avoid CPU choking
            time.sleep(0.01)
            
        cap.release()

    def get_frame(self, target_width, target_height):
        with self.lock:
            if self.frame is None:
                # Return placeholder if no frame is available yet
                placeholder = np.zeros((target_height, target_width, 3), dtype=np.uint8)
                cv2.putText(
                    placeholder,
                    f"CAM {self.camera_id} ({self.ip})",
                    (20, target_height // 2 - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )
                cv2.putText(
                    placeholder,
                    "Connecting / Offline...",
                    (20, target_height // 2 + 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 165, 255),
                    1,
                    cv2.LINE_AA
                )
                return placeholder
            
            # Resize the annotated frame to target dimensions
            return cv2.resize(self.frame, (target_width, target_height))

    def stop(self):
        self.running = False


def main():
    parser = argparse.ArgumentParser(description="Multi-Camera YOLOv26 Splitscreen Streamer")
    parser.add_argument("--count", type=int, default=4, help="Number of cameras to display (1 to 11)")
    parser.add_argument("--password", type=str, default="sedra12301", help="Default Hikvision password")
    parser.add_argument("--model", type=str, default="yolo26m.pt", help="YOLOv26 model file")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")
    parser.add_argument("--imgsz", type=int, default=320, help="YOLO inference resolution (default: 320 for speed)")
    parser.add_argument("--skip", type=int, default=1, help="Run inference every N frames (default: 1)")
    args = parser.parse_args()

    # Limit camera count to max 9 for a 3x3 grid layout
    num_cameras = max(1, min(args.count, 9))
    selected_ips = CAMERA_IPS[:num_cameras]
    
    print(f"Initializing {num_cameras} camera streams in splitscreen layout...")
    print(f"Inference Size: {args.imgsz}x{args.imgsz} | Skipping frames: {args.skip}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = args.model

    print("Running dry-run warm-up in main thread...")
    try:
        temp_model = YOLO(model_path)
        dummy_frame = np.zeros((args.imgsz, args.imgsz, 3), dtype=np.uint8)
        temp_model(dummy_frame, device=device, verbose=False)
        print("Dry-run warm-up completed!")
    except Exception as e:
        print(f"Dry-run warm-up warning: {e}")

    # Start threads for each selected camera
    streams = []
    for idx, ip in enumerate(selected_ips, 1):
        stream = CameraStream(idx, ip, args.password, model_path, args.conf, args.imgsz, args.skip)
        stream.start()
        streams.append(stream)

    # Fixed 3x3 Grid Layout (9 panels)
    cols, rows = 3, 3

    # Canvas dimension details
    # We target 1920x1080 grid. Adjust individual cell sizes accordingly.
    cell_width = 1920 // cols
    cell_height = 1080 // rows
    
    window_name = "Multi-Camera YOLOv26 Splitscreen"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    print("\n--------------------------------------------------")
    print(f" Grid layout: {rows} rows x {cols} columns")
    print(" - Press 'q' to quit.")
    print(" - Press 'f' to toggle Fullscreen.")
    print("--------------------------------------------------\n")

    prev_time = 0

    while True:
        # Create empty grid canvas
        canvas = np.zeros((cell_height * rows, cell_width * cols, 3), dtype=np.uint8)
        
        for idx, stream in enumerate(streams):
            r = idx // cols
            c = idx % cols
            
            # Fetch frame from camera thread
            cell_frame = stream.get_frame(cell_width, cell_height)
            
            # Paste into canvas
            canvas[r * cell_height : (r + 1) * cell_height, c * cell_width : (c + 1) * cell_width] = cell_frame
            
        # Draw blank panels for remaining cells in the grid if camera count is not a perfect grid
        for idx in range(num_cameras, rows * cols):
            r = idx // cols
            c = idx % cols
            placeholder = np.zeros((cell_height, cell_width, 3), dtype=np.uint8)
            canvas[r * cell_height : (r + 1) * cell_height, c * cell_width : (c + 1) * cell_width] = placeholder

        # Calculate Grid FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time else 0
        prev_time = curr_time
        
        # Display FPS text in top-left cell
        cv2.putText(
            canvas,
            f"Grid FPS: {fps:.1f}",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 255),
            3,
            cv2.LINE_AA
        )

        # Show final canvas window
        cv2.imshow(window_name, canvas)

        # Key controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('f'):
            is_fullscreen = cv2.getWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN) == cv2.WINDOW_FULLSCREEN
            if is_fullscreen:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                print("Fullscreen disabled.")
            else:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                print("Fullscreen enabled.")

    # Cleanup threads
    print("Shutting down camera threads...")
    for stream in streams:
        stream.stop()
    for stream in streams:
        stream.join(timeout=1.0)
        
    cv2.destroyAllWindows()
    print("Exiting multi-camera stream.")

if __name__ == "__main__":
    main()
