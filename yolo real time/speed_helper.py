import time
import math
from collections import deque
from ultralytics.solutions import SpeedEstimator
from ultralytics.utils.plotting import Annotator, colors

class CustomSpeedEstimator(SpeedEstimator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.custom_track_history = {}  # track_id -> deque of (center, timestamp)
        self.max_history_len = 15       # number of frames for rolling average
        
    def process(self, im0):
        self.frame_count += 1
        self.extract_tracks(im0)
        
        annotator = Annotator(im0, line_width=self.line_width)
        
        for box, track_id, cls, conf in zip(self.boxes, self.track_ids, self.clss, self.confs):
            x1, y1, x2, y2 = map(float, box)
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            center = (cx, cy)
            
            # Initialize history for new track IDs
            if track_id not in self.custom_track_history:
                self.custom_track_history[track_id] = deque(maxlen=self.max_history_len)
            self.custom_track_history[track_id].append((center, time.time()))
            
            history = self.custom_track_history[track_id]
            speed_text = "Estimating..."
            
            # Calculate speed based on the oldest and newest points in the rolling history
            if len(history) >= 2:
                p0, t0 = history[0]
                p1, t1 = history[-1]
                dt = t1 - t0
                if dt > 0.1:  # Update speed if at least 0.1s has elapsed
                    dx = p1[0] - p0[0]
                    dy = p1[1] - p0[1]
                    pixel_dist = math.sqrt(dx*dx + dy*dy)
                    meters = pixel_dist * self.meter_per_pixel
                    speed_kmh = int(min((meters / dt) * 3.6, self.max_speed))
                    speed_text = f"{speed_kmh} km/h"
            
            class_name = self.names[int(cls)]
            label = f"ID:{track_id} {class_name} | {speed_text}"
            annotator.box_label(box, label=label, color=colors(track_id, True))
            
        plot_im = annotator.result()
        
        # Clean up old tracks from custom_track_history to avoid memory leaks
        current_track_ids = set(self.track_ids)
        now = time.time()
        for tid in list(self.custom_track_history.keys()):
            if tid not in current_track_ids:
                # Expire tracks if not seen in 2.0 seconds
                if now - self.custom_track_history[tid][-1][1] > 2.0:
                    self.custom_track_history.pop(tid, None)
                    
        return type('SolutionResults', (object,), {'plot_im': plot_im, 'total_tracks': len(self.track_ids)})()
