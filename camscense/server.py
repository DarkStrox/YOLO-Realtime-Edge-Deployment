import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import time
import json
import threading
from pathlib import Path

# Make the original YOLO project imports available (openvino_patch, models live there)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
YOLO_DIR = PROJECT_ROOT / "yolo real time"
for p in (PROJECT_ROOT, YOLO_DIR):
    sys.path.append(str(p))

import cv2
import numpy as np
from flask import Flask, Response, send_from_directory, request, jsonify
from ultralytics import YOLO
from speed_helper import CustomSpeedEstimator as SpeedEstimator
import torch

app = Flask(__name__, static_folder=None)

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
CONFIG_PATH = Path(__file__).resolve().parent / "config.json"

DEFAULT_CONFIG = {
    "model": "yolo26m.pt",
    "conf": 0.25,
    "imgsz": 320,
    "camera_password": "sedra12301",
    "camera_ips": [
        "192.168.1.28", "192.168.1.22", "192.168.1.38", "192.168.1.26",
        "192.168.1.34", "192.168.1.36", "192.168.1.2",  "192.168.1.17",
        "192.168.1.15", "192.168.1.37", "192.168.1.50"
    ],
    "video_test": str(YOLO_DIR / "video test" / "crowd.mp4")
}


def load_config():
    if CONFIG_PATH.exists():
        try:
            cfg = json.loads(CONFIG_PATH.read_text())
            for k, v in cfg.items():
                if v is not None:
                    DEFAULT_CONFIG[k] = v
        except Exception as e:
            print(f"[Warn] Could not parse config.json: {e}")
    return DEFAULT_CONFIG


CONFIG = load_config()


def resolve_model_path(model_name, is_pose=False):
    """Resolve a model name to a standard PyTorch PT model file."""
    if is_pose:
        if model_name.endswith(".pt") and "-pose.pt" not in model_name:
            model_name = model_name.replace(".pt", "-pose.pt")
    return str(YOLO_DIR / model_name)


# ----------------------------------------------------------------------------
# Frame grabber for a single source (laptop cam, company cam, or video test)
# ----------------------------------------------------------------------------
class FrameGrabber:
    def __init__(self, source):
        self.source = source
        if isinstance(source, int):
            self.cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.cap.release()
                self.cap = cv2.VideoCapture(source)
        else:
            self.cap = cv2.VideoCapture(source)
        
        # Check if the source is a local video file to pace playback
        self.is_video_file = False
        if isinstance(source, str) and not source.startswith("rtsp://"):
            self.is_video_file = True
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            if self.fps <= 0 or self.fps > 120:
                self.fps = 30.0
            self.frame_delay = 1.0 / self.fps

        self.ret = False
        self.frame = None
        self.running = True
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while self.running:
            start_time = time.time()
            ret, frame = self.cap.read()
            if ret and frame is not None:
                with self.lock:
                    self.ret = True
                    self.frame = frame.copy()
            else:
                # loop video files, retry camera sources
                if not isinstance(self.source, int):
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                else:
                    time.sleep(0.5)
                    self.cap.release()
                    self.cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
                    if not self.cap.isOpened():
                        self.cap.release()
                        self.cap = cv2.VideoCapture(self.source)
            
            if self.is_video_file:
                elapsed = time.time() - start_time
                sleep_time = max(0.001, self.frame_delay - elapsed)
                time.sleep(sleep_time)
            else:
                time.sleep(0.01)

    def read(self):
        with self.lock:
            if self.frame is None:
                return False, None
            return self.ret, self.frame.copy()

    def is_opened(self):
        return self.cap.isOpened()

    def get(self, propId):
        return self.cap.get(propId)

    def release(self):
        self.running = False
        self.thread.join(timeout=1.0)
        self.cap.release()


# ----------------------------------------------------------------------------
# Stream manager: holds the currently running detection session
# ----------------------------------------------------------------------------
class StreamManager:
    def __init__(self):
        self.lock = threading.Lock()
        self.session = None  # dict with grabbers, model, mode, stop flag

    def stop(self):
        with self.lock:
            if self.session:
                self.session["stop"].set()
                for g in self.session["grabbers"]:
                    try:
                        g.release()
                    except Exception:
                        pass
                self.session = None

    def start(self, mode, run_mode="detect", **kwargs):
        with self.lock:
            if self.session:
                self.session["stop"].set()
                for g in self.session["grabbers"]:
                    try:
                        g.release()
                    except Exception:
                        pass
                self.session = None

            is_pose = (run_mode == "heatmap")
            task_type = "pose" if is_pose else "detect"
            model_path = resolve_model_path(CONFIG["model"], is_pose=is_pose)
            model = YOLO(model_path, task=task_type)

            stop = threading.Event()
            grabbers = []

            if mode == "laptop":
                g = FrameGrabber(0)
                grabbers.append(g)
            elif mode == "company":
                indices = kwargs.get("indices")
                if indices is not None:
                    selected_ips = [CONFIG["camera_ips"][i] for i in indices if 0 <= i < len(CONFIG["camera_ips"])]
                else:
                    count = kwargs.get("count", 4)
                    selected_ips = CONFIG["camera_ips"][:count]
                for ip in selected_ips:
                    url = f"rtsp://admin:{CONFIG['camera_password']}@{ip}:554/Streaming/Channels/102"
                    grabbers.append(FrameGrabber(url))
            elif mode == "video":
                video_name = kwargs.get("video")
                if video_name:
                    vp = str(YOLO_DIR / "video test" / video_name)
                else:
                    vp = CONFIG["video_test"]
                count = kwargs.get("count", 4)
                for i in range(count):
                    g = FrameGrabber(vp)
                    grabbers.append(g)
            else:
                raise ValueError(f"Unknown mode: {mode}")

            self.session = {
                "mode": mode,
                "run_mode": run_mode,
                "model": model,
                "model_path": model_path,
                "grabbers": grabbers,
                "stop": stop,
                "kwargs": kwargs,
                "start_time": time.time(),
                "video_name": video_name if mode == "video" else None
            }
            return len(grabbers)

    def get_session(self):
        with self.lock:
            return self.session


manager = StreamManager()


# ----------------------------------------------------------------------------
# Inference generator -> MJPEG frames
# ----------------------------------------------------------------------------
def gen_frames():
    session = manager.get_session()
    if not session:
        return
    model = session["model"]
    grabbers = session["grabbers"]
    stop = session["stop"]
    run_mode = session.get("run_mode", "detect")
    conf = CONFIG["conf"]
    imgsz = CONFIG["imgsz"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # dry-run warm-up
    try:
        model.predict(source=np.zeros((imgsz, imgsz, 3), dtype=np.uint8),
                      imgsz=imgsz, conf=conf, device=device,
                      verbose=False, save=False, show=False)
    except Exception as e:
        print(f"[Warn] dry-run compile failed: {e}")

    prev_time = 0
    # Initialize heatmap accumulators and speed estimators for each grabber
    heatmap_accums = [None] * len(grabbers)
    speed_estimators = [None] * len(grabbers)

    while not stop.is_set():
        sources = []
        for g in grabbers:
            ret, frame = g.read()
            if ret and frame is not None:
                sources.append(frame)

        if not sources:
            time.sleep(0.05)
            continue

        out_frames = []
        for i, raw in enumerate(sources):
            try:
                if run_mode == "heatmap":
                    # Initialize heatmap accumulator for this stream if not set or size changed
                    if heatmap_accums[i] is None or heatmap_accums[i].shape[:2] != raw.shape[:2]:
                        h, w = raw.shape[:2]
                        heatmap_accums[i] = np.zeros((h, w), dtype=np.float32)

                    results = model.predict(source=raw, imgsz=imgsz, conf=conf,
                                             device=device, verbose=False, save=False,
                                             show=False)
                    annotated = raw.copy()
                    h, w = raw.shape[:2]
                    current_heat = np.zeros((h, w), dtype=np.float32)
                    
                    boxes = results[0].boxes
                    if results[0].keypoints is not None and len(results[0].keypoints) > 0:
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
                                if start_idx < len(confs) and end_idx < len(confs):
                                    if confs[start_idx] > 0.5 and confs[end_idx] > 0.5:
                                        pt1 = (int(pts[start_idx][0]), int(pts[start_idx][1]))
                                        pt2 = (int(pts[end_idx][0]), int(pts[end_idx][1]))
                                        cv2.line(annotated, pt1, pt2, (0, 255, 0), 2)  # Green bones
                                    
                            # Draw joint circles
                            for (x, y), c in zip(pts, confs):
                                if c > 0.5:
                                    cv2.circle(annotated, (int(x), int(y)), 4, (0, 255, 255), -1)  # Yellow joints
                                    
                            # Find feet center point (Ankles are index 15 and 16)
                            cx, cy = None, None
                            if len(confs) > 16:
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
                                
                            if cx is not None and cy is not None:
                                radius = 35
                                cv2.circle(current_heat, (cx, cy), radius, 15.0, -1)
                                
                        # Apply Gaussian Blur to smooth the heat spots
                        current_heat = cv2.GaussianBlur(current_heat, (51, 51), 0)
                        
                    # Accumulate and decay
                    heatmap_accums[i] = cv2.add(heatmap_accums[i], current_heat)
                    heatmap_accums[i] = cv2.multiply(heatmap_accums[i], 0.995)
                    
                    # Normalize and colorize
                    norm_heat = np.clip(heatmap_accums[i], 0, 255).astype(np.uint8)
                    color_heat = cv2.applyColorMap(norm_heat, cv2.COLORMAP_JET)
                    
                    # Blend onto frame
                    mask = norm_heat > 10
                    annotated[mask] = cv2.addWeighted(annotated, 0.5, color_heat, 0.5, 0)[mask]
                    
                elif run_mode == "speed":
                    if speed_estimators[i] is None:
                        fps = g.get(cv2.CAP_PROP_FPS)
                        if fps <= 0 or fps > 120:
                            fps = 30.0
                        model_path = session.get("model_path")
                        speed_estimators[i] = SpeedEstimator(
                            model=model_path,
                            device=device,
                            conf=conf,
                            imgsz=imgsz,
                            meter_per_pixel=0.05,
                            max_speed=120,
                            fps=fps,
                            show=False
                        )
                    results = speed_estimators[i].process(raw)
                    annotated = results.plot_im
                    
                else:
                    results = model.predict(source=raw, imgsz=imgsz, conf=conf,
                                             device=device, verbose=False, save=False,
                                             show=False)
                    annotated = results[0].plot()
            except Exception as e:
                print(f"[Error] inference failed: {e}")
                annotated = raw.copy() if raw is not None else np.zeros((360, 640, 3), np.uint8)
                cv2.putText(annotated, "inference error", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            out_frames.append(annotated)

        # Build a grid if more than one source
        if len(out_frames) == 1:
            canvas = out_frames[0]
        else:
            n = len(out_frames)
            cols = 3 if n > 4 else (2 if n > 1 else 1)
            rows = (n + cols - 1) // cols
            cw, ch = 640, 360
            canvas = np.zeros((ch * rows, cw * cols, 3), dtype=np.uint8)
            for i, f in enumerate(out_frames):
                r, c = divmod(i, cols)
                canvas[r * ch:(r + 1) * ch, c * cw:(c + 1) * cw] = cv2.resize(f, (cw, ch))

        curr = time.time()
        fps = 1 / (curr - prev_time) if prev_time else 0
        prev_time = curr
        cv2.putText(canvas, f"FPS: {fps:.1f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

        ok, buf = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not ok:
            continue
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
        time.sleep(0.01)


# ----------------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------------
@app.route("/")
def index():
    return send_from_directory(Path(__file__).resolve().parent, "index.html")


@app.route("/api/videos", methods=["GET"])
def api_videos():
    video_dir = YOLO_DIR / "video test"
    videos = []
    if video_dir.exists():
        for f in video_dir.iterdir():
            if f.is_file() and f.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv"):
                videos.append(f.name)
    return jsonify({"videos": sorted(videos)})


@app.route("/api/config", methods=["GET"])
def api_config():
    s = manager.get_session()
    return jsonify({
        "model": CONFIG["model"],
        "conf": CONFIG["conf"],
        "imgsz": CONFIG["imgsz"],
        "modes": ["laptop", "company", "video"],
        "camera_count": len(CONFIG["camera_ips"]),
        "active_mode": s["mode"] if s else None,
        "active_run_mode": s.get("run_mode") if s else None,
        "active_video": s.get("video_name") if s else None,
    })


@app.route("/api/start", methods=["POST"])
def api_start():
    data = request.get_json(silent=True) or {}
    mode = data.get("mode")
    if mode not in ("laptop", "company", "video"):
        return jsonify({"error": "mode must be laptop, company or video"}), 400
    run_mode = data.get("run_mode", "detect")
    if run_mode not in ("detect", "heatmap", "speed"):
        return jsonify({"error": "run_mode must be detect, heatmap or speed"}), 400
    count = int(data.get("count", 4))
    indices = data.get("indices")
    if indices is not None:
        try:
            indices = [int(x) for x in indices]
        except Exception:
            return jsonify({"error": "indices must be a list of integers"}), 400
    try:
        n = manager.start(mode, run_mode=run_mode, count=count, indices=indices, video=data.get("video"))
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    return jsonify({"status": "started", "mode": mode, "run_mode": run_mode, "streams": n, "video": data.get("video")})


@app.route("/api/stop", methods=["POST"])
def api_stop():
    manager.stop()
    return jsonify({"status": "stopped"})


@app.route("/api/status")
def api_status():
    s = manager.get_session()
    if not s:
        return jsonify({"active": False})
    return jsonify({
        "active": True,
        "mode": s["mode"],
        "run_mode": s.get("run_mode", "detect"),
        "streams": len(s["grabbers"]),
        "uptime": round(time.time() - s["start_time"], 1),
        "active_video": s.get("video_name")
    })


@app.route("/video_feed")
def video_feed():
    if not manager.get_session():
        return jsonify({"error": "no active stream"}), 400
    return Response(gen_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    print("CamScense running at http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, threaded=True)
