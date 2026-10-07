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

    def set(self, propId, value):
        self.cap.set(propId, value)

    def release(self):
        self.running = False
        self.thread.join(timeout=1.0)
        self.cap.release()


def main():
    parser = argparse.ArgumentParser(description="Real-Time YOLOv26 Person Heatmap Generator")
    parser.add_argument("--model", type=str, default="yolo26m-pose.pt", help="YOLOv26 pose model file (e.g., yolo26m-pose.pt)")
    parser.add_argument("--camera", type=str, default="0", help="Camera index or RTSP URL")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--decay", type=float, default=0.995, help="Heatmap decay rate (lower = faster fade)")
    parser.add_argument("--radius", type=int, default=35, help="Heat spot size radius")
    parser.add_argument("--imgsz", type=int, default=320, help="YOLO inference resolution (default: 320 for speed)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = args.model

    print(f"Loading model: {model_path} on {device}...")
    try:
        model = YOLO(model_path, task="pose")
        print("Model loaded. Running dry-run warm-up...")
        dummy_frame = np.zeros((args.imgsz, args.imgsz, 3), dtype=np.uint8)
        model(dummy_frame, device=device, verbose=False)
        print("Model warm-up completed!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
        return

    # Handle camera input (convert digits to int, keep strings as RTSP URLs)
    camera_source = args.camera
    if camera_source.isdigit():
        camera_source = int(camera_source)

    print(f"Opening low-latency camera stream: {camera_source}...")
    cap = VideoCaptureThread(camera_source)

    # Set camera resolution (optional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Wait for the first frame to determine dimensions
    print("Waiting for camera feed...")
    frame_ready = False
    for _ in range(50):
        ret, frame = cap.read()
        if ret and frame is not None:
            frame_ready = True
            break
        time.sleep(0.1)

    if not frame_ready:
        print("Error: Could not grab frame from camera. Exiting.")
        cap.release()
        return

    h, w, _ = frame.shape
    print(f"Resolution: {w}x{h}")

    # Initialize heatmap accumulator
    heatmap_accum = np.zeros((h, w), dtype=np.float32)

    window_name = "Real-Time Person Heatmap"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    print("\n--------------------------------------------------")
    print(" YOLO Heatmap Generator Started!")
    print(" - Tracking class: 'person' (COCO class 0)")
    print(" - Press 'q' to quit.")
    print(" - Press 'r' to reset heatmap.")
    print(" - Press 'f' to toggle Fullscreen.")
    print("--------------------------------------------------\n")

    prev_time = 0

    while True:
        # Process UI window events on every iteration to keep the window responsive
        # even when camera frames are not ready yet
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            # Reset the heatmap accumulator
            heatmap_accum = np.zeros((h, w), dtype=np.float32)
            print("Heatmap reset.")
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

        results = model(frame, imgsz=args.imgsz, conf=args.conf, device=device, verbose=False)

        # Create temporary canvas for the current frame's heat
        current_heat = np.zeros((h, w), dtype=np.float32)

        # Draw skeletons and generate heatmap from ankle keypoints
        boxes = results[0].boxes
        if results[0].keypoints is not None:
            keypoints_list = results[0].keypoints.xy.cpu().numpy()
            conf_list = results[0].keypoints.conf.cpu().numpy()
            
            connections = [
                (0, 1), (0, 2), (1, 3), (2, 4),
                (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
                (5, 11), (6, 12), (11, 12),
                (11, 13), (13, 15), (12, 14), (14, 16)
            ]
            
            for idx, (pts, confs) in enumerate(zip(keypoints_list, conf_list)):
                # Draw skeleton lines
                for start_idx, end_idx in connections:
                    if confs[start_idx] > 0.5 and confs[end_idx] > 0.5:
                        pt1 = (int(pts[start_idx][0]), int(pts[start_idx][1]))
                        pt2 = (int(pts[end_idx][0]), int(pts[end_idx][1]))
                        cv2.line(frame, pt1, pt2, (0, 255, 0), 2)  # Green bones
                        
                # Draw joint circles
                for (x, y), c in zip(pts, confs):
                    if c > 0.5:
                        cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 255), -1)  # Yellow joints

                # Find feet center point (Ankles are index 15 and 16)
                cx, cy = None, None
                if confs[15] > 0.5 and confs[16] > 0.5:
                    cx = int((pts[15][0] + pts[16][0]) / 2)
                    cy = int((pts[15][1] + pts[16][1]) / 2)
                elif confs[15] > 0.5:
                    cx = int(pts[15][0])
                    cy = int(pts[15][1])
                elif confs[16] > 0.5:
                    cx = int(pts[16][0])
                    cy = int(pts[16][1])
                
                # Fallback to bottom-center of bounding box if ankles not visible
                if (cx is None or cy is None) and boxes is not None and idx < len(boxes):
                    box = boxes[idx]
                    bx1, by1, bx2, by2 = map(int, box.xyxy[0])
                    cx = (bx1 + bx2) // 2
                    cy = by2

                # If we have a valid foot/standing point, draw the heat blob
                if cx is not None and cy is not None:
                    cv2.circle(current_heat, (cx, cy), args.radius, 15.0, -1)

            # Apply Gaussian Blur to smooth the heat spots
            current_heat = cv2.GaussianBlur(current_heat, (51, 51), 0)

        # Accumulate the current heat
        heatmap_accum = cv2.add(heatmap_accum, current_heat)

        # Slowly decay the heatmap over time so old footprints disappear
        heatmap_accum = cv2.multiply(heatmap_accum, args.decay)

        # Normalize the heatmap to 0-255 range for colormap application
        norm_heat = np.clip(heatmap_accum, 0, 255).astype(np.uint8)

        # Apply a ColorMap (JET maps high values to Red, medium to Yellow/Green, low to Blue)
        color_heat = cv2.applyColorMap(norm_heat, cv2.COLORMAP_JET)

        # Filter out cold (blue) areas from overlaying the normal camera feed
        # We only want to draw heat overlay where the intensity is greater than 10
        mask = norm_heat > 10

        # Create the output frame by blending the colorized heatmap on top of the original frame
        blended = frame.copy()
        blended[mask] = cv2.addWeighted(frame, 0.5, color_heat, 0.5, 0)[mask]

        # Calculate FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time else 0
        prev_time = curr_time

        # Overlay FPS and Reset instructions on the frame
        cv2.putText(
            blended,
            f"FPS: {fps:.1f} | People: {len(boxes) if boxes else 0}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )
        cv2.putText(
            blended,
            "Press 'r' to reset map",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        # Display window
        cv2.imshow(window_name, blended)

    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    print("Heatmap stream finished.")

if __name__ == "__main__":
    main()
