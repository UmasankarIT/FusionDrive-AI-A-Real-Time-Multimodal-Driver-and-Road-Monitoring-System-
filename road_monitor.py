
import cv2
import numpy as np
import logging
from collections import deque
import time

from road_rnn import RoadRNN

logger = logging.getLogger("RoadMonitor")

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
LANE_CANNY_LOW            = 50
LANE_CANNY_HIGH           = 150
LANE_HOUGH_THRESHOLD      = 20
LANE_MIN_LENGTH           = 30
LANE_MAX_GAP              = 25

VEHICLE_DISTANCE_THRESHOLD    = 0.3
PEDESTRIAN_DISTANCE_THRESHOLD = 0.4
HAZARD_CONFIDENCE_THRESHOLD   = 0.65
YOLO_ROI_TOP_FRACTION = 0.45

# Adaptive brightness thresholds
BRIGHTNESS_NIGHT   = 60     # below this = night
BRIGHTNESS_DUSK    = 120    # below this = dusk/dawn
CONF_DAY           = 0.75
CONF_DUSK          = 0.78
CONF_NIGHT         = 0.82

# CV vehicle detection — tune if too many / too few detections
CV_MIN_CONTOUR_AREA    = 2000   # pixels² — smaller blobs ignored
CV_ASPECT_MIN          = 1.0    # width/height ratio lower bound
CV_ASPECT_MAX          = 5.0    # width/height ratio upper bound
CV_ROI_TOP_FRACTION    = 0.35   # look at bottom (1 - this) fraction of road region
CV_MAX_VEHICLES        = 6      # keep only the N closest detections

HAZARD_LEVEL_CLEAR    = 0
HAZARD_LEVEL_CAUTION  = 1
HAZARD_LEVEL_WARNING  = 2
HAZARD_LEVEL_CRITICAL = 3

COLOR_GREEN  = (0, 255, 0)
COLOR_YELLOW = (0, 255, 255)
COLOR_ORANGE = (0, 165, 255)
COLOR_RED    = (0, 0, 255)
COLOR_WHITE  = (255, 255, 255)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def estimate_object_distance(bbox, frame_shape, prev_bbox=None, obj_type=None):
    h, w = frame_shape[:2]

    x1, y1, x2, y2 = bbox
    bbox_h = y2 - y1

    # --- Normalized features ---
    norm_h = bbox_h / h
    center_y = (y1 + y2) / 2
    norm_y = center_y / h

    # --- Base distance ---
    dist = 0.55 * (norm_h ** 0.7) + 0.45 * norm_y

    # --- Growth rate (approach detection) ---
    if prev_bbox is not None:
        prev_h = prev_bbox[3] - prev_bbox[1]
        growth = max(0, (bbox_h - prev_h) / h)
        dist += 0.25 * growth

    # --- Center priority ---
    center_x = (x1 + x2) / 2
    offset = abs(center_x - w/2) / (w/2)
    dist += 0.15 * (1 - offset)

    # --- Object scaling ---
    if obj_type == "truck":
        dist *= 1.1
    elif obj_type == "bike":
        dist *= 0.9
    elif obj_type == "pedestrian":
        dist *= 1.2

    return float(np.clip(dist, 0.0, 1.0))


def hazard_color(level: int) -> tuple:
    return (COLOR_RED    if level == HAZARD_LEVEL_CRITICAL else
            COLOR_ORANGE if level == HAZARD_LEVEL_WARNING  else
            COLOR_YELLOW if level == HAZARD_LEVEL_CAUTION  else
            COLOR_GREEN)


def detect_lanes(frame: np.ndarray, road_x_start: int = 0) -> dict:
    """
    Detect lane markings using Canny + HoughLinesP.

    Parameters
    ----------
    frame        : full display frame (BGR)
    road_x_start : x-pixel where the road region begins.
                   Pass 0 for full-frame, or w//2 for right-half split-screen.

    Returns dict with left_lane, right_lane, confidence.
    """
    # Crop to road region only
    road = frame[:, road_x_start:] if road_x_start > 0 else frame
    h, w = road.shape[:2]

    gray    = cv2.cvtColor(road, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges   = cv2.Canny(blurred, LANE_CANNY_LOW, LANE_CANNY_HIGH)

    # ROI — bottom 55% of road region
    roi_mask = np.zeros_like(edges)
    roi_mask[int(h * 0.45):, :] = 255
    roi_edges = cv2.bitwise_and(edges, roi_mask)

    lines = cv2.HoughLinesP(
        roi_edges,
        rho=1, theta=np.pi / 180,
        threshold=LANE_HOUGH_THRESHOLD,
        minLineLength=LANE_MIN_LENGTH,
        maxLineGap=LANE_MAX_GAP,
    )

    left_lane = right_lane = None
    lane_confidence = 0.0

    if lines is not None:
        left_lines, right_lines = [], []
        for line in lines:
            line_data = line[0] if len(line.shape) > 1 else line
            x1, y1, x2, y2 = line_data
            if x2 == x1:
                continue
            slope = (y2 - y1) / (x2 - x1)
            if slope < -0.3 and x1 < w // 2:
                left_lines.append(line_data)
            elif slope > 0.3 and x1 > w // 2:
                right_lines.append(line_data)

        if left_lines:
            left_lane = np.mean(left_lines, axis=0)
        if right_lines:
            right_lane = np.mean(right_lines, axis=0)

        # Confidence = proportion of long, clear lines (max 5 needed for full confidence)
        strong = sum(
            1 for l in lines
            if abs(l[0][2] - l[0][0]) + abs(l[0][3] - l[0][1]) > LANE_MIN_LENGTH * 2
        )
        lane_confidence = min(1.0, strong / 5.0)

    return {
        "left_lane"  : left_lane,
        "right_lane" : right_lane,
        "confidence" : lane_confidence,
    }


def detect_road_surface(frame: np.ndarray, road_x_start: int = 0) -> dict:
    """
    Estimate road surface clarity using colour-based segmentation.
    Returns confidence (0–1) and road mask.
    """
    road = frame[:, road_x_start:] if road_x_start > 0 else frame
    h, w = road.shape[:2]

    hsv        = cv2.cvtColor(road, cv2.COLOR_BGR2HSV)
    road_mask  = cv2.inRange(hsv,
                             np.array([0,   0,   0  ]),
                             np.array([180, 50, 150 ]))
    kernel     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    road_mask  = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(road_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    edge_confidence = 0.0
    if contours:
        largest         = max(contours, key=cv2.contourArea)
        edge_confidence = min(1.0, cv2.contourArea(largest) / (h * w))

    return {"confidence": edge_confidence}


# ─────────────────────────────────────────────
# MAIN CLASS
# ─────────────────────────────────────────────

class RoadMonitor:
    """
    Real-time road monitoring. Call process(frame) each frame.

    Split-screen usage
    ------------------
    If your video/camera shows a split-screen (e.g. driver left, road right),
    set road_x_start so detection only runs on the road portion:

        monitor = RoadMonitor()
        monitor.road_x_start = 640   # road starts at x=640 in a 1280-wide frame
    """

    def __init__(self, enable_yolo: bool = False):
        logger.info("Initializing RoadMonitor…")

        # ── YOLO (optional) ───────────────────────────
        self.yolo_model = None
        self.use_yolo   = False

        if enable_yolo:
            try:
                from ultralytics import YOLO
                logger.info("Loading YOLOv11n-seg model…")
                self.yolo_model = YOLO("yolo11n-seg_openvino_model")   # segmentation model
                self.use_yolo   = True
                logger.info("YOLOv11n-seg loaded successfully.")
            except ImportError:
                logger.warning("ultralytics not installed — falling back to CV detection.")
            except Exception as e:
                logger.warning(f"YOLOv11n-seg load failed ({e}) — falling back to CV detection.")

        # ── Split-screen crop ─────────────────────────
        # Set this to the x-pixel where the road region starts.
        # 0 = full frame (default / single-camera mode)
        self.road_x_start: int = 0

        # ── Smoothing buffer ──────────────────────────
        self.hazard_buffer = deque(maxlen=20)

        # ── FPS tracking ──────────────────────────────
        self.fps_estimate     = 30.0
        self._fps_last_time   = time.time()
        self._fps_frame_count = 0

        # ── State ─────────────────────────────────────
        self.lane_status  = "UNKNOWN"
        self.road_status  = "UNKNOWN"
        self.hazard_level = HAZARD_LEVEL_CLEAR

        self.scene_mode     = "DAY"
        self.conf_threshold = CONF_DAY
        self.lane_departure = "NONE"

        # ── RNN layer ─────────────────────────────────
        self.rnn = RoadRNN()

        logger.info(
            f"RoadMonitor ready — "
            f"mode: {'YOLOv11n-seg' if self.use_yolo else 'CV (no YOLO)'}"
        )

    # ── FPS ───────────────────────────────────────────

    def _update_fps(self) -> None:
        self._fps_frame_count += 1
        if self._fps_frame_count < 30:
            return
        elapsed = time.time() - self._fps_last_time
        if elapsed > 0:
            new_fps = self._fps_frame_count / elapsed
            if abs(new_fps - self.fps_estimate) > 2:
                self.fps_estimate = new_fps
        self._fps_last_time   = time.time()
        self._fps_frame_count = 0

    # ── YOLO segmentation helpers ─────────────────────

    def _mask_distance(self, mask_xy: np.ndarray, frame_h: int) -> float:
        """
        Compute distance from a segmentation mask polygon.
        Uses the lowest point of the mask (closest to camera on road plane)
        and the mask's vertical span — more accurate than bbox-centre method.
        Returns normalised distance 0 (far) → 1 (very close).
        """
        if mask_xy is None or len(mask_xy) == 0:
            return 0.0
        ys       = mask_xy[:, 1]
        lowest_y = float(np.max(ys))
        span_y   = float(np.max(ys) - np.min(ys))
        norm_y   = lowest_y / frame_h
        norm_span = span_y  / frame_h
        return float(np.clip(norm_y * 0.6 + norm_span * 0.4, 0.0, 1.0))

    # ── YOLO detection ────────────────────────────────

    def _detect_objects_yolo(self, frame: np.ndarray) -> dict:
        """
        Run YOLOv11n-seg on the road region.
        Uses segmentation masks for more accurate distance estimation
        and draws filled mask overlays on the frame.
        Falls back to bbox distance if no mask is available for a detection.
        """
        empty = {"vehicles": [], "pedestrians": [], "traffic_signs": [], "all_detections": []}
        if not self.use_yolo or self.yolo_model is None:
            return empty

        road_frame = frame[:, self.road_x_start:] if self.road_x_start > 0 else frame
        fh         = frame.shape[0]

        y_roi      = int(fh * YOLO_ROI_TOP_FRACTION)
        road_frame = road_frame[y_roi:, :]

        try:
            results   = self.yolo_model(road_frame, verbose=False)
            vehicles, pedestrians, traffic_signs, all_detections = [], [], [], []

            if not results or len(results) == 0:
                return empty

            r         = results[0]
            has_masks = r.masks is not None and len(r.masks) > 0

            for i, box in enumerate(r.boxes):
                cls_id = int(box.cls[0])
                conf   = float(box.conf[0])
                if conf < self.conf_threshold:
                    continue

                # Bounding box — shift to full-frame coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1 += self.road_x_start
                x2 += self.road_x_start
                y1 += y_roi
                y2 += y_roi
                bbox = (x1, y1, x2, y2)

                # ── Extract mask polygon ──────────────────────────────────
                mask_poly = None
                if has_masks and i < len(r.masks):
                    try:
                        pts = r.masks[i].xy   # list of Nx2 arrays
                        if pts is not None and len(pts) > 0:
                            poly = np.array(pts[0], dtype=np.int32)
                            poly[:, 0] += self.road_x_start  # shift to full-frame x
                            poly[:, 1] += y_roi              # shift to full-frame y
                            mask_poly   = poly
                    except Exception:
                        mask_poly = None

                # ── Distance: mask-based if available, bbox fallback ──────
                dist = (self._mask_distance(mask_poly, fh)
                        if mask_poly is not None
                        else estimate_object_distance(bbox, frame.shape))

                det = {
                    "bbox"      : bbox,
                    "confidence": conf,
                    "distance"  : dist,
                    "class_id"  : cls_id,
                    "source"    : "yolo_seg",
                    "mask_poly" : mask_poly,   # used by _draw_detections
                }

                if cls_id in [2, 3, 5, 7]:   # car, motorcycle, bus, truck
                    vehicles.append(det)
                elif cls_id == 0:             # person
                    pedestrians.append(det)
                elif cls_id in [9, 11]:       # traffic light, stop sign
                    traffic_signs.append({"bbox": bbox, "confidence": conf,
                                          "class_id": cls_id})
                all_detections.append(det)

            return {"vehicles": vehicles, "pedestrians": pedestrians,
                    "traffic_signs": traffic_signs, "all_detections": all_detections}

        except Exception as e:
            logger.error(f"YOLO-seg detection error: {e}")
            return empty

    # ── CV fallback detection ─────────────────────────

    def _detect_objects_cv(self, frame: np.ndarray) -> dict:
        """
        Contour-based vehicle detection — works without YOLO.

        Strategy:
          1. Crop to the road region (respects road_x_start)
          2. Focus on the lower portion where vehicles appear
          3. Find large, roughly rectangular blobs = candidate vehicles
          4. Filter by aspect ratio and minimum area
          5. Shift coordinates back to full-frame space
        """
        h = frame.shape[0]
        road = frame[:, self.road_x_start:] if self.road_x_start > 0 else frame
        rh, rw = road.shape[:2]

        # Work only in the lower part of the road region
        y_offset = int(rh * CV_ROI_TOP_FRACTION)
        roi      = road[y_offset:, :]

        gray     = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred  = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        edges    = cv2.Canny(thresh, 30, 90)
        dilated  = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)

        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        vehicles = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < CV_MIN_CONTOUR_AREA:
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)
            aspect = bw / (bh + 1e-6)
            if not (CV_ASPECT_MIN < aspect < CV_ASPECT_MAX and bh > 30):
                continue

            # Convert to full-frame coordinates
            x1 = x  + self.road_x_start
            y1 = y  + y_offset
            x2 = x1 + bw
            y2 = y1 + bh

            dist_val = estimate_object_distance((x1, y1, x2, y2), frame.shape)
            conf_val = float(np.clip(area / 20000.0, 0.3, 0.95))

            vehicles.append({
                "bbox"      : (x1, y1, x2, y2),
                "confidence": conf_val,
                "distance"  : dist_val,
                "class_id"  : -1,       # -1 = CV detected, no YOLO class
                "source"    : "cv",
            })

        # Keep only the closest N to reduce noise
        vehicles.sort(key=lambda v: v["distance"], reverse=True)
        vehicles = vehicles[:CV_MAX_VEHICLES]

        return {
            "vehicles"      : vehicles,
            "pedestrians"   : [],
            "traffic_signs" : [],
            "all_detections": vehicles,
        }

    # ── Hazard assessment ─────────────────────────────

    def _assess_hazard_level(self, detections: dict) -> int:
        hazard = HAZARD_LEVEL_CLEAR

        for v in detections.get("vehicles", []):
            d = v["distance"]
            if d > 0.7:   hazard = max(hazard, HAZARD_LEVEL_CRITICAL)
            elif d > 0.5: hazard = max(hazard, HAZARD_LEVEL_WARNING)
            elif d > VEHICLE_DISTANCE_THRESHOLD:
                          hazard = max(hazard, HAZARD_LEVEL_CAUTION)

        for p in detections.get("pedestrians", []):
            d = p["distance"]
            if d > 0.6:   hazard = max(hazard, HAZARD_LEVEL_CRITICAL)
            elif d > 0.4: hazard = max(hazard, HAZARD_LEVEL_WARNING)
            elif d > PEDESTRIAN_DISTANCE_THRESHOLD:
                          hazard = max(hazard, HAZARD_LEVEL_CAUTION)

        if detections.get("traffic_signs"):
            hazard = max(hazard, HAZARD_LEVEL_CAUTION)

        return hazard

    # ── Draw detections ───────────────────────────────

    def _draw_detections(self, frame: np.ndarray,
                         detections: dict, hazard_level: int) -> None:
        """
        Draw detections on frame.
        - YOLO-seg: filled semi-transparent mask + outline + label
        - CV / bbox-only: plain rectangle + label
        """
        col = hazard_color(hazard_level)

        def _draw_one(det: dict, label: str, color: tuple):
            mask_poly = det.get("mask_poly")
            x1, y1, x2, y2 = det["bbox"]

            if mask_poly is not None and len(mask_poly) >= 3:
                # Semi-transparent filled mask
                overlay = frame.copy()
                cv2.fillPoly(overlay, [mask_poly], color)
                cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)
                # Mask outline
                cv2.polylines(frame, [mask_poly], isClosed=True, color=color, thickness=2)
            else:
                # Plain bounding box fallback
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            cv2.putText(frame, label, (x1, max(y1 - 4, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1)

        for v in detections.get("vehicles", []):
            src   = v.get("source", "")
            tag   = "V" if "yolo" in src else "v"
            _draw_one(v, f"{tag} {v['distance']:.2f}", col)

        for p in detections.get("pedestrians", []):
            _draw_one(p, f"P {p['distance']:.2f}", COLOR_RED)

    # ── Draw lane lines ───────────────────────────────


    def _draw_lanes(self, frame: np.ndarray, lane_info: dict) -> None:
        """Draw extended lane lines and filled lane polygon on the frame."""
        h = frame.shape[0]
        y_top    = int(h * 0.50)   # extend lines up to 50% of frame height
        y_bottom = h

        def extend_line(pts):
            if pts is None:
                return None
            x1, y1, x2, y2 = map(float, pts)
            x1 += self.road_x_start
            x2 += self.road_x_start
            if abs(x2 - x1) < 1e-3:
                return None
            slope     = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1
            if abs(slope) < 1e-3:
                return None
            # Extrapolate to y_bottom and y_top
            xb = int((y_bottom - intercept) / slope)
            xt = int((y_top    - intercept) / slope)
            return (xb, y_bottom, xt, y_top)

        left  = extend_line(lane_info.get("left_lane"))
        right = extend_line(lane_info.get("right_lane"))

    # Draw filled lane polygon between the two lines
        if left and right:
            pts = np.array([
                [left[0],  left[1]],
                [left[2],  left[3]],
                [right[2], right[3]],
                [right[0], right[1]],
            ], dtype=np.int32)
            overlay = frame.copy()
            cv2.fillPoly(overlay, [pts], (0, 200, 0))
            cv2.addWeighted(overlay, 0.20, frame, 0.80, 0, frame)

    # Draw the lane lines on top
        if left:
            cv2.line(frame, (left[0],  left[1]),  (left[2],  left[3]),  COLOR_GREEN, 3)
        if right:
            cv2.line(frame, (right[0], right[1]), (right[2], right[3]), COLOR_GREEN, 3)

    # ── Main process ──────────────────────────────────

    def process(self, frame: np.ndarray) -> dict:
        """
        Process one frame. Returns road telemetry dict.
        main.py handles all HUD text — this module draws
        bounding boxes and lane lines only.
        """
        self._update_fps()

        # Auto day/dusk/night detection
        avg_brightness = float(np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)))
        if avg_brightness < BRIGHTNESS_NIGHT:
            self.scene_mode     = "NIGHT"
            self.conf_threshold = CONF_NIGHT
        elif avg_brightness < BRIGHTNESS_DUSK:
            self.scene_mode     = "DUSK"
            self.conf_threshold = CONF_DUSK
        else:
            self.scene_mode     = "DAY"
            self.conf_threshold = CONF_DAY

        # Lane detection (road region only)
        lane_info        = detect_lanes(frame, self.road_x_start)
        self.lane_status = "DETECTED" if lane_info["confidence"] > 0.3 else "NOT_DETECTED"

        # Road surface
        road_info        = detect_road_surface(frame, self.road_x_start)
        self.road_status = "CLEAR" if road_info["confidence"] > 0.3 else "UNCLEAR"

        # Object detection — YOLO if available, CV otherwise
        if self.use_yolo:
            detections = self._detect_objects_yolo(frame)
        else:
            detections = self._detect_objects_cv(frame)

        # Hazard level with temporal smoothing
        raw_hazard = self._assess_hazard_level(detections)
        self.hazard_buffer.append(raw_hazard)
        smoothed = (int(round(float(np.mean(self.hazard_buffer))))
                    if self.hazard_buffer else HAZARD_LEVEL_CLEAR)
        smoothed = int(np.clip(smoothed, 0, 3))
        self.hazard_level = smoothed

        # Draw bboxes + lane lines
        self._draw_detections(frame, detections, smoothed)
        self._draw_lanes(frame, lane_info)

        # ── RNN temporal refinement ───────────────────
        raw_result = {
            "hazard_level"   : smoothed,
            "lane_status"    : self.lane_status,
            "lane_departure" : self.lane_departure,
            "scene_mode"     : self.scene_mode,
            "road_status"    : self.road_status,
            "vehicles"       : detections.get("vehicles",      []),
            "pedestrians"    : detections.get("pedestrians",   []),
            "traffic_signs"  : detections.get("traffic_signs", []),
            "all_detections" : detections.get("all_detections",[]),
            "frame"          : frame,
        }
        rnn_out = self.rnn.update(raw_result)

        # RNN can raise OR lower the hazard level based on temporal context.
        # If the RNN is confident (>0.65) let it override; otherwise blend up only.
        if rnn_out["rnn_confidence"] > 0.65:
            final_hazard = rnn_out["hazard_level"]
        else:
            # Conservative: take the max so we never suppress a real hazard
            final_hazard = max(smoothed, rnn_out["hazard_level"])

        self.hazard_level = final_hazard

        return {
            "hazard_level"      : final_hazard,
            "lane_status"       : self.lane_status,
            "lane_departure"    : self.lane_departure,
            "scene_mode"        : self.scene_mode,
            "road_status"       : self.road_status,
            "vehicles"          : detections.get("vehicles",      []),
            "pedestrians"       : detections.get("pedestrians",   []),
            "traffic_signs"     : detections.get("traffic_signs", []),
            "all_detections"    : detections.get("all_detections",[]),
            # RNN extras — available to fusion_engine
            "collision_prob"    : rnn_out["collision_prob"],
            "approach_rate"     : rnn_out["approach_rate"],
            "rnn_confidence"    : rnn_out["rnn_confidence"],
            "collision_warning" : rnn_out["collision_warning"],
            "rnn_override"      : rnn_out["override"],
            "frame"             : frame,
        }