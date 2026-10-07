import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# Force OpenCV/FFmpeg to use TCP for RTSP and minimize buffer sizes for lowest latency
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|max_delay;500000"

import argparse
import time
import threading
import cv2
import numpy as np
from ultralytics import YOLO
from speed_helper import CustomSpeedEstimator as SpeedEstimator
import torch

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

    def get(self, propId):
        return self.cap.get(propId)

    def release(self):
        self.running = False
        self.thread.join(timeout=1.0)
        self.cap.release()

class CameraSpeedStream(threading.Thread):
    def __init__(self, camera_id, ip, password, model_name, conf, imgsz, meter_per_pixel, max_speed, classes):
        super().__init__()
        self.camera_id = camera_id
        self.ip = ip
        self.password = password
        self.model_name = model_name
        self.conf = conf
        self.imgsz = imgsz
        self.meter_per_pixel = meter_per_pixel
        self.max_speed = max_speed
        self.classes = classes
        
        # Stream URL using Channel 102 (Sub-stream for performance)
        self.rtsp_url = f"rtsp://admin:{password}@{ip}:554/Streaming/Channels/102"
        
        self.running = True
        self.frame = None
        self.lock = threading.Lock()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.daemon = True

    def run(self):
        print(f"[Camera {self.camera_id}] Starting speed stream for IP {self.ip}...")
        cap = VideoCaptureThread(self.rtsp_url)
        
        # Wait for the first frame to determine dimensions
        while self.running:
            ret, raw_frame = cap.read()
            if ret and raw_frame is not None:
                break
            time.sleep(0.1)

        if not self.running:
            cap.release()
            return

        # Get FPS from stream
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or fps > 120:
            fps = 30.0

        # Initialize Speed Estimator
        try:
            speed_estimator = SpeedEstimator(
                model=self.model_name,
                device=self.device,
                conf=self.conf,
                imgsz=self.imgsz,
                meter_per_pixel=self.meter_per_pixel,
                max_speed=self.max_speed,
                classes=self.classes,
                fps=fps,
                show=False
            )
        except Exception as e:
            print(f"[Camera {self.camera_id}] Error creating SpeedEstimator: {e}")
            cap.release()
            return

        while self.running:
            ret, raw_frame = cap.read()
            if not ret or raw_frame is None:
                time.sleep(0.01)
                continue
            
            # Process frame for speed estimation
            try:
                results = speed_estimator.process(raw_frame)
                annotated_frame = results.plot_im
                total_tracks = results.total_tracks
            except Exception as e:
                print(f"[Camera {self.camera_id}] Error processing frame: {e}")
                annotated_frame = raw_frame.copy()
                total_tracks = 0

            # Overlay Camera ID and Count
            cv2.putText(
                annotated_frame,
                f"CAM {self.camera_id} ({self.ip}) | Tracking: {total_tracks}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
                cv2.LINE_AA
            )

            # Write annotated frame thread-safely
            with self.lock:
                self.frame = annotated_frame.copy()
            
            time.sleep(0.01)
            
        cap.release()

    def get_frame(self, target_width, target_height):
        with self.lock:
            if self.frame is None:
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
            
            return cv2.resize(self.frame, (target_width, target_height))

    def stop(self):
        self.running = False


def main():
    parser = argparse.ArgumentParser(description="Multi-Camera YOLOv26 Speed Estimator")
    parser.add_argument("--count", type=int, default=4, help="Number of cameras to display (1 to 9)")
    parser.add_argument("--password", type=str, default="sedra12301", help="Default Hikvision password")
    parser.add_argument("--model", type=str, default="yolo26m.pt", help="YOLOv26 model file (e.g. yolo26m.pt)")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")
    parser.add_argument("--imgsz", type=int, default=320, help="YOLO inference resolution")
    parser.add_argument("--meter-per-pixel", type=float, default=0.05, help="Meters represented by one pixel")
    parser.add_argument("--max-speed", type=int, default=120, help="Maximum allowed speed")
    parser.add_argument("--classes", type=str, default=None, help="Comma-separated class IDs to track (e.g., 2,7 for cars and trucks)")
    args = parser.parse_args()

    # Limit camera count to max 9 for 3x3 layout
    num_cameras = max(1, min(args.count, 9))
    selected_ips = CAMERA_IPS[:num_cameras]
    
    print(f"Initializing {num_cameras} camera speed estimators in splitscreen grid...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = args.model

    # Parse classes list
    classes_to_track = None
    if args.classes:
        try:
            classes_to_track = [int(x.strip()) for x in args.classes.split(",") if x.strip()]
        except ValueError:
            print("Invalid classes format. Please use comma-separated integers, e.g. --classes 2,7")
            return

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
        stream = CameraSpeedStream(
            idx, ip, args.password, model_path, args.conf, args.imgsz,
            args.meter_per_pixel, args.max_speed, classes_to_track
        )
        stream.start()
        streams.append(stream)

    # Fixed 3x3 Layout
    cols, rows = 3, 3
    cell_width = 1920 // cols
    cell_height = 1080 // rows
    
    window_name = "Multi-Camera YOLOv26 Speed Estimation Grid"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    print("\n--------------------------------------------------")
    print(" YOLO Multi-Camera Speed Estimation Started!")
    print(" - Press 'q' to quit.")
    print(" - Press 'f' to toggle Fullscreen.")
    print("--------------------------------------------------\n")

    prev_time = 0

    while True:
        canvas = np.zeros((cell_height * rows, cell_width * cols, 3), dtype=np.uint8)
        
        for idx, stream in enumerate(streams):
            r = idx // cols
            c = idx % cols
            cell_frame = stream.get_frame(cell_width, cell_height)
            canvas[r * cell_height : (r + 1) * cell_height, c * cell_width : (c + 1) * cell_width] = cell_frame
            
        for idx in range(num_cameras, rows * cols):
            r = idx // cols
            c = idx % cols
            placeholder = np.zeros((cell_height, cell_width, 3), dtype=np.uint8)
            canvas[r * cell_height : (r + 1) * cell_height, c * cell_width : (c + 1) * cell_width] = placeholder

        # Calculate FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time else 0
        prev_time = curr_time
        
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

        cv2.imshow(window_name, canvas)

        # Keyboard input
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

    # Cleanup
    print("Shutting down camera threads...")
    for stream in streams:
        stream.stop()
    for stream in streams:
        stream.join(timeout=1.0)
        
    cv2.destroyAllWindows()
    print("Exiting multi-camera speed estimation.")

if __name__ == "__main__":
    main()
