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

    def release(self):
        self.running = False
        self.thread.join(timeout=1.0)
        self.cap.release()


class CameraHeatmapStream(threading.Thread):
    def __init__(self, camera_id, ip, password, model_name, conf, imgsz, skip_frames, decay, radius):
        super().__init__()
        self.camera_id = camera_id
        self.ip = ip
        self.password = password
        self.model_name = model_name
        self.conf = conf
        self.imgsz = imgsz
        self.skip_frames = skip_frames
        self.decay = decay
        self.radius = radius
        
        # Stream URL using Channel 102 (Sub-stream for performance)
        self.rtsp_url = f"rtsp://admin:{password}@{ip}:554/Streaming/Channels/102"
        
        self.running = True
        self.frame = None
        self.lock = threading.Lock()
        self.reset_signal = False
        
        # Determine device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Load local YOLO model for this thread
        self.model = YOLO(model_name)
        self.daemon = True

    def reset_heatmap(self):
        self.reset_signal = True

    def run(self):
        print(f"[Camera {self.camera_id}] Starting heatmap stream for IP {self.ip}...")
        cap = VideoCaptureThread(self.rtsp_url)
        
        # Wait for first frame to determine dimensions
        h, w = 0, 0
        while self.running:
            ret, raw_frame = cap.read()
            if ret and raw_frame is not None:
                h, w, _ = raw_frame.shape
                break
            time.sleep(0.1)

        if not self.running:
            cap.release()
            return

        # Initialize heatmap accumulator
        heatmap_accum = np.zeros((h, w), dtype=np.float32)
        frame_count = 0
        last_results = None

        while self.running:
            if self.reset_signal:
                heatmap_accum = np.zeros((h, w), dtype=np.float32)
                self.reset_signal = False
                print(f"[Camera {self.camera_id}] Heatmap reset.")

            ret, raw_frame = cap.read()
            if not ret or raw_frame is None:
                time.sleep(0.01)
                continue
            
            frame_count += 1
            
            # Perform YOLO inference every N frames, and use cached results for intermediate frames
            # Filter detections to 'person' (class 0) only for speed
            if frame_count % self.skip_frames == 0 or last_results is None:
                results = self.model(raw_frame, imgsz=self.imgsz, conf=self.conf, device=self.device, verbose=False)
                last_results = results[0]

            # Generate heatmap overlay on the frame
            current_heat = np.zeros((h, w), dtype=np.float32)
            annotated_frame = raw_frame.copy()
            
            boxes = last_results.boxes if last_results else None
            if last_results and last_results.keypoints is not None:
                keypoints_list = last_results.keypoints.xy.cpu().numpy()
                conf_list = last_results.keypoints.conf.cpu().numpy()
                
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
                            cv2.line(annotated_frame, pt1, pt2, (0, 255, 0), 2)  # Green bones
                            
                    # Draw joint circles
                    for (x, y), c in zip(pts, confs):
                        if c > 0.5:
                            cv2.circle(annotated_frame, (int(x), int(y)), 4, (0, 255, 255), -1)  # Yellow joints

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

                    # If we have a valid standing point, draw the heat blob
                    if cx is not None and cy is not None:
                        cv2.circle(current_heat, (cx, cy), self.radius, 15.0, -1)

                current_heat = cv2.GaussianBlur(current_heat, (51, 51), 0)

            heatmap_accum = cv2.add(heatmap_accum, current_heat)
            heatmap_accum = cv2.multiply(heatmap_accum, self.decay)
            
            norm_heat = np.clip(heatmap_accum, 0, 255).astype(np.uint8)
            color_heat = cv2.applyColorMap(norm_heat, cv2.COLORMAP_JET)
            mask = norm_heat > 10
            
            # Blend heatmap colors
            annotated_frame[mask] = cv2.addWeighted(annotated_frame, 0.5, color_heat, 0.5, 0)[mask]

            # Overlay Camera ID and Count
            cv2.putText(
                annotated_frame,
                f"CAM {self.camera_id} ({self.ip}) | People: {len(boxes) if boxes else 0}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
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
    parser = argparse.ArgumentParser(description="Multi-Camera YOLOv26 Heatmap Streamer")
    parser.add_argument("--count", type=int, default=4, help="Number of cameras to display (1 to 9)")
    parser.add_argument("--password", type=str, default="sedra12301", help="Default Hikvision password")
    parser.add_argument("--model", type=str, default="yolo26m-pose.pt", help="YOLOv26 pose model file (e.g. yolo26m-pose.pt)")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")
    parser.add_argument("--imgsz", type=int, default=320, help="YOLO inference resolution")
    parser.add_argument("--skip", type=int, default=1, help="Run inference every N frames")
    parser.add_argument("--decay", type=float, default=0.995, help="Heatmap decay rate")
    parser.add_argument("--radius", type=int, default=35, help="Heat spot size radius")
    args = parser.parse_args()

    # Limit camera count to max 9 for 3x3 layout
    num_cameras = max(1, min(args.count, 9))
    selected_ips = CAMERA_IPS[:num_cameras]
    
    print(f"Initializing {num_cameras} camera heatmaps in 3x3 splitscreen grid...")

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
        stream = CameraHeatmapStream(idx, ip, args.password, model_path, args.conf, args.imgsz, args.skip, args.decay, args.radius)
        stream.start()
        streams.append(stream)

    # Fixed 3x3 Layout
    cols, rows = 3, 3
    cell_width = 1920 // cols
    cell_height = 1080 // rows
    
    window_name = "Multi-Camera YOLOv26 Heatmap Grid"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    print("\n--------------------------------------------------")
    print(" YOLO Multi-Camera Heatmap Started!")
    print(" - Press 'q' to quit.")
    print(" - Press 'r' to reset all heatmaps.")
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
        elif key == ord('r'):
            # Reset all camera heatmaps
            for stream in streams:
                stream.reset_heatmap()
            print("Reset all heatmaps.")
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
    print("Exiting multi-camera heatmap.")

if __name__ == "__main__":
    main()
