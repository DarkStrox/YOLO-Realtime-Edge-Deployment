import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# Force OpenCV/FFmpeg to use TCP for RTSP and minimize buffer sizes for lowest latency
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|max_delay;500000"

import argparse
import time
import threading
import cv2
import numpy as np
from pathlib import Path
import torch
from ultralytics import YOLO
from speed_helper import CustomSpeedEstimator as SpeedEstimator

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

    def get(self, propId):
        return self.cap.get(propId)

    def release(self):
        self.running = False
        self.thread.join(timeout=1.0)
        self.cap.release()

def main():
    parser = argparse.ArgumentParser(description="Real-Time YOLOv26 Object Speed Estimator")
    parser.add_argument("--model", type=str, default="yolo26m.pt", help="YOLOv26 model file (e.g., yolo26m.pt)")
    parser.add_argument("--camera", type=str, default="0", help="Camera index or RTSP URL stream")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--imgsz", type=int, default=320, help="YOLO inference resolution (default: 320 for speed)")
    parser.add_argument("--meter-per-pixel", type=float, default=0.05, help="Meters represented by one pixel (scene scale)")
    parser.add_argument("--max-speed", type=int, default=120, help="Maximum allowed speed to filter outliers")
    parser.add_argument("--classes", type=str, default=None, help="Comma-separated class IDs to track (e.g., 2,7 for cars and trucks)")
    args = parser.parse_args()

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

    # Handle camera input
    camera_source = args.camera
    if camera_source.isdigit():
        camera_source = int(camera_source)

    print(f"Opening camera source: {camera_source}...")
    cap = VideoCaptureThread(camera_source)

    if not cap.isOpened():
        print(f"Error: Could not open camera {args.camera}.")
        return

    # Set camera resolution (optional, try 1280x720)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Get FPS
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 120:
        fps = 30.0
    print(f"Detected video stream FPS: {fps:.1f}")

    print("Initializing Speed Estimator Solution...")
    try:
        speed_estimator = SpeedEstimator(
            model=model_path,
            device=device,
            conf=args.conf,
            imgsz=args.imgsz,
            meter_per_pixel=args.meter_per_pixel,
            max_speed=args.max_speed,
            classes=classes_to_track,
            fps=fps,
            show=False # We handle window drawing ourselves
        )
    except Exception as e:
        print(f"Error creating SpeedEstimator: {e}")
        cap.release()
        return

    window_name = "YOLO Real-Time Speed Estimation"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    print("\n--------------------------------------------------")
    print(" YOLO Real-Time Speed Estimation Started!")
    print(" - Press 'q' to quit.")
    print(" - Press 'f' to toggle Fullscreen.")
    print("--------------------------------------------------\n")

    prev_time = 0

    while True:
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

        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.01)
            continue

        # Run speed estimation process
        results = speed_estimator.process(frame)
        annotated_frame = results.plot_im

        # Calculate FPS
        curr_time = time.time()
        gui_fps = 1 / (curr_time - prev_time) if prev_time else 0
        prev_time = curr_time

        # Overlay FPS and counts on the frame
        cv2.putText(
            annotated_frame,
            f"FPS: {gui_fps:.1f} | Tracking: {results.total_tracks}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA
        )

        cv2.imshow(window_name, annotated_frame)

    cap.release()
    cv2.destroyAllWindows()
    print("Speed estimation completed.")

if __name__ == "__main__":
    main()
