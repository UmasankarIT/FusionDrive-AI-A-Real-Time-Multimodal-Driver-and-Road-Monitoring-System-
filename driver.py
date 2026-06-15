"""
Driver Monitoring Module  (MediaPipe-Only)
==========================================
Detects drowsiness and distraction using MediaPipe FaceMesh.

Detects:
  - EAR  (Eye Aspect Ratio)      -> eyes closing / closed
  - MAR  (Mouth Aspect Ratio)    -> yawning
  - Head pose (yaw / pitch)      -> head turned L/R/up/down
  - Gaze ratio (iris)            -> eyes looking away
  - Fatigue Score (0-10)         -> replaces PERCLOS; webcam-grade fatigue tracking

Fatigue Score breakdown (0-10, webcam-friendly):
  Component 1 — Microsleep count : each closure > alert threshold = +2 pts (cap 6)
  Component 2 — Yawn frequency   : yawns in last 5 min, 3+ = up to +3 pts
  Component 3 — Max closure len  : longest single closure >= 2s = up to +1 pt
  Slow decay (0.5 pts/min) when driver is fully alert.

Why better than PERCLOS here:
  - Counts EVENTS not rates  -> no 60s window dependency, resets instantly
  - Works at 10-21fps        -> PERCLOS needs 60+ fps for statistical validity
  - Microsleeps are real     -> each one is a discrete dangerous event
  - Yawn count is universal  -> reliable fatigue signal at any framerate

Status strings (used by fusion_engine):
  "Alert"          -> fine
  "Eyes Closing"   -> EAR drooping
  "DROWSY"         -> eyes closed too long OR high fatigue score
  "Yawning"        -> MAR sustained high
  "Head LEFT/RIGHT/UP/DOWN"
  "Gaze LEFT/RIGHT"
  Multiple joined with " | "
"""

import cv2
import numpy as np
import mediapipe as mp
from scipy.spatial import distance as dist
from collections import deque
import time
import logging

logger = logging.getLogger("DriverMonitor")

# ─────────────────────────────────────────────
# THRESHOLDS
# ─────────────────────────────────────────────
EAR_CLOSED_WARN_SEC  = 0.35
EAR_CLOSED_ALERT_SEC = 0.70
MAR_OPEN_SEC         = 0.60
HEAD_WARN_SEC        = 0.5
HEAD_ALERT_SEC       = 1.2

EAR_THRESHOLD        = 0.28   # fallback before adaptive calibration
BLINK_MAX_FRAMES     = 5      # closures <= this are natural blinks (~250ms@20fps)
MAR_THRESHOLD        = 0.55

YAW_WARN_DEG         = 12
YAW_ALERT_DEG        = 20
PITCH_WARN_DEG       = 8
PITCH_ALERT_DEG      = 15

GAZE_LEFT_THRESHOLD  = 0.40
GAZE_RIGHT_THRESHOLD = 0.60
GAZE_WARN_SEC        = 0.50

# ─────────────────────────────────────────────
# FATIGUE SCORE CONSTANTS
# ─────────────────────────────────────────────
FATIGUE_MICROSLEEP_PTS   = 2      # pts per microsleep event
FATIGUE_MICROSLEEP_MAX   = 6      # cap microsleep contribution at 6 pts
FATIGUE_YAWN_WINDOW_SEC  = 300    # 5-min rolling window for yawn count
FATIGUE_YAWN_THRESH      = 3      # yawns before points start
FATIGUE_YAWN_MAX_PTS     = 3      # cap yawn contribution at 3 pts
FATIGUE_CLOSURE_SEC      = 2.0    # closure must be this long to add closure pts
FATIGUE_CLOSURE_MAX_PTS  = 1      # cap closure length contribution at 1 pt
FATIGUE_DECAY_PER_MIN    = 0.5    # pts/min natural decay when fully alert

# MediaPipe landmark indices
LEFT_EYE_IDX   = [33,  160, 158, 133, 153, 144]
RIGHT_EYE_IDX  = [362, 385, 387, 263, 373, 380]
MOUTH_IDX      = [61,  39,  0,   269, 291, 405, 17, 181]
LEFT_IRIS_IDX  = [468, 469, 470, 471]
RIGHT_IRIS_IDX = [473, 474, 475, 476]

HEAD_POSE_IDX = {
    "nose_tip"    : 1,
    "chin"        : 152,
    "left_eye"    : 33,
    "right_eye"   : 263,
    "mouth_left"  : 61,
    "mouth_right" : 291,
}
HEAD_MODEL_POINTS = np.array([
    [  0.0,    0.0,    0.0  ],
    [  0.0, -330.0,  -65.0 ],
    [-225.0,  170.0, -135.0],
    [ 225.0,  170.0, -135.0],
    [-150.0, -150.0, -125.0],
    [ 150.0, -150.0, -125.0],
], dtype=np.float64)

COLOR_GREEN  = (0,   255,   0)
COLOR_ORANGE = (0,   165, 255)
COLOR_RED    = (0,     0, 255)
FONT         = cv2.FONT_HERSHEY_SIMPLEX


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def eye_aspect_ratio(eye: np.ndarray) -> float:
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C) if C > 0 else 0.0


def mouth_aspect_ratio(mouth: np.ndarray) -> float:
    A = dist.euclidean(mouth[1], mouth[7])
    B = dist.euclidean(mouth[2], mouth[6])
    C = dist.euclidean(mouth[3], mouth[5])
    D = dist.euclidean(mouth[0], mouth[4])
    return (A + B + C) / (2.0 * D) if D > 0 else 0.0


def extract_landmarks(lm_list, indices: list, w: int, h: int) -> np.ndarray:
    return np.array(
        [[lm_list[i].x * w, lm_list[i].y * h] for i in indices],
        dtype=np.float64,
    )


def get_gaze_ratio(eye_pts: np.ndarray, iris_pts: np.ndarray) -> float:
    eye_width = eye_pts[3][0] - eye_pts[0][0]
    if abs(eye_width) < 1e-3:
        return 0.5
    iris_x = float(np.mean(iris_pts[:, 0]))
    return float(np.clip((iris_x - eye_pts[0][0]) / eye_width, 0.0, 1.0))


def score_color(score: int) -> tuple:
    return COLOR_RED if score == 2 else COLOR_ORANGE if score == 1 else COLOR_GREEN


# ─────────────────────────────────────────────
# MAIN CLASS
# ─────────────────────────────────────────────

class DriverMonitor:

    def __init__(self):
        logger.info("Loading MediaPipe FaceMesh...")
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh    = self.mp_face_mesh.FaceMesh(
            static_image_mode        = False,
            max_num_faces            = 1,
            refine_landmarks         = True,
            min_detection_confidence = 0.2,
            min_tracking_confidence  = 0.2,
        )

        # FPS tracking
        self.fps_estimate     = 30.0
        self._fps_last_time   = time.time()
        self._fps_frame_count = 0
        self._update_thresholds()

        # Sustained-condition counters
        self.ear_counter      = 0
        self.mar_counter      = 0
        self.head_counter     = 0
        self.gaze_counter     = 0
        self._blink_counter   = 0   # consecutive closed frames for blink filter
        self._ear_open_streak = 0

        # Gaze smoothing
        self.gaze_buffer = deque(maxlen=6)

        # Head pose calibration
        self.ref_yaw      = 0.0
        self.ref_pitch    = 0.0
        self.calibrated   = False
        self.calib_frames: list = []
        self.calib_needed = 60

        # Adaptive EAR threshold
        self.adaptive_ear_samples: list = []
        self.adaptive_ear_ready   = False
        self.adaptive_ear_thresh  = EAR_THRESHOLD

        # ── Fatigue Score state ────────────────────────
        self.fatigue_score         = 0.0
        self._microsleep_pts       = 0.0   # contribution from microsleep events
        self._microsleep_count     = 0     # total microsleeps this session
        self._yawn_times           = deque()  # wall-clock times of yawn onsets
        self._max_closure_sec      = 0.0   # longest single eye closure (seconds)
        self._closure_start_time   = 0.0   # when current closure began
        self._in_closure           = False
        self._last_fatigue_ts      = time.monotonic()

        # Misc
        self._last_head_dir  = ""
        self._last_gaze_dir  = "CENTER"
        self.no_face_counter = 0
        self.NO_FACE_ALERT_FRAMES = 30

        logger.info("DriverMonitor ready.")

    # ──────────────────────────────────────────────
    # FPS / THRESHOLD MANAGEMENT
    # ──────────────────────────────────────────────

    def _update_fps(self) -> None:
        self._fps_frame_count += 1
        if self._fps_frame_count < 30:
            return
        elapsed = time.time() - self._fps_last_time
        if elapsed > 0:
            new_fps = self._fps_frame_count / elapsed
            if abs(new_fps - self.fps_estimate) > 2:
                self.fps_estimate = new_fps
                self._update_thresholds()
        self._fps_last_time   = time.time()
        self._fps_frame_count = 0

    def _update_thresholds(self) -> None:
        fps = max(1.0, self.fps_estimate)
        self.ear_warn_frames   = max(5,  int(EAR_CLOSED_WARN_SEC  * fps))
        self.ear_alert_frames  = max(8,  int(EAR_CLOSED_ALERT_SEC * fps))
        self.mar_frames        = max(4,  int(MAR_OPEN_SEC          * fps))
        self.head_warn_frames  = max(4,  int(HEAD_WARN_SEC         * fps))
        self.head_alert_frames = max(6,  int(HEAD_ALERT_SEC        * fps))
        self.gaze_warn_frames  = max(4,  int(GAZE_WARN_SEC         * fps))

    # ──────────────────────────────────────────────
    # ADAPTIVE EAR CALIBRATION
    # ──────────────────────────────────────────────

    def calibrate_ear(self, ear_value: float) -> None:
        if self.adaptive_ear_ready or ear_value <= 0.15:
            return
        self.adaptive_ear_samples.append(ear_value)
        if len(self.adaptive_ear_samples) >= 120:
            arr       = np.array(self.adaptive_ear_samples)
            mean_open = float(np.median(arr))
            # 0.72 multiplier: 30% wider gap vs 0.78 — safer for low-EAR drivers
            self.adaptive_ear_thresh = float(np.clip(mean_open * 0.72, 0.16, 0.28))
            self.adaptive_ear_ready  = True
            self.ear_counter = self._blink_counter = self._ear_open_streak = 0
            self.mar_counter = self.head_counter = self.gaze_counter = 0
            logger.info(
                f"Adaptive EAR ready: thresh={self.adaptive_ear_thresh:.3f} "
                f"(median={mean_open:.3f})"
            )

    # ──────────────────────────────────────────────
    # HEAD POSE (solvePnP)
    # ──────────────────────────────────────────────

    def get_head_pose(self, lm, w: int, h: int):
        image_points = np.array([
            [lm[HEAD_POSE_IDX["nose_tip"]   ].x * w, lm[HEAD_POSE_IDX["nose_tip"]   ].y * h],
            [lm[HEAD_POSE_IDX["chin"]       ].x * w, lm[HEAD_POSE_IDX["chin"]       ].y * h],
            [lm[HEAD_POSE_IDX["left_eye"]   ].x * w, lm[HEAD_POSE_IDX["left_eye"]   ].y * h],
            [lm[HEAD_POSE_IDX["right_eye"]  ].x * w, lm[HEAD_POSE_IDX["right_eye"]  ].y * h],
            [lm[HEAD_POSE_IDX["mouth_left"] ].x * w, lm[HEAD_POSE_IDX["mouth_left"] ].y * h],
            [lm[HEAD_POSE_IDX["mouth_right"]].x * w, lm[HEAD_POSE_IDX["mouth_right"]].y * h],
        ], dtype=np.float64)

        focal_length = float(w)
        cam_matrix   = np.array([
            [focal_length, 0.0,          w / 2.0],
            [0.0,          focal_length, h / 2.0],
            [0.0,          0.0,          1.0    ],
        ], dtype=np.float64)

        success, rvec, _ = cv2.solvePnP(
            HEAD_MODEL_POINTS, image_points, cam_matrix,
            np.zeros((4, 1)), flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            return None, None

        rmat, _ = cv2.Rodrigues(rvec)
        # Correct ZYX decomposition: pitch=X (nod), yaw=Y (turn)
        sy = np.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
        if sy >= 1e-6:
            pitch = float(np.degrees(np.arctan2( rmat[2, 1], rmat[2, 2])))
            yaw   = float(np.degrees(np.arctan2(-rmat[2, 0], sy)))
        else:
            pitch = float(np.degrees(np.arctan2(-rmat[1, 2], rmat[1, 1])))
            yaw   = float(np.degrees(np.arctan2(-rmat[2, 0], sy)))

        if not self.calibrated:
            self.calib_frames.append((yaw, pitch))
            if len(self.calib_frames) >= self.calib_needed:
                arr = np.array(self.calib_frames)
                for col_idx, attr in [(0, "ref_yaw"), (1, "ref_pitch")]:
                    col    = arr[:, col_idx]
                    q1, q3 = np.percentile(col, [25, 75])
                    iqr    = q3 - q1
                    clean  = col[(col >= q1 - 1.5 * iqr) & (col <= q3 + 1.5 * iqr)]
                    setattr(self, attr, float(np.median(clean)))
                self.calibrated   = True
                self.head_counter = 0
                self.gaze_counter = 0
                logger.info(
                    f"Head calibrated: ref_yaw={self.ref_yaw:.1f} "
                    f"ref_pitch={self.ref_pitch:.1f}"
                )

        return yaw - self.ref_yaw, pitch - self.ref_pitch

    # ──────────────────────────────────────────────
    # FATIGUE SCORE  (replaces PERCLOS)
    # ──────────────────────────────────────────────

    def _update_fatigue(self, eyes_closed: bool, just_woke: bool,
                        closed_frames: int, yawn_just_started: bool) -> None:
        """
        Three-component fatigue score (0-10). Resets instantly when eyes open.

        Component 1: Microsleep count (0-6 pts)
          - Triggered when a real drowsy closure ends (> ear_alert_frames)
          - +2 pts per event, capped at 6

        Component 2: Yawn frequency (0-3 pts)
          - Counts yawn ONSETS in a rolling 5-minute window
          - 3+ yawns starts adding pts; capped at 3

        Component 3: Max closure length (0-1 pt)
          - Tracks the longest single eye closure this session
          - Closures >= 2s add up to 1 pt

        Decay: 0.5 pts/min when driver is fully alert (no closed eyes, no yawning)
        """
        now = time.monotonic()

        # ── Component 1: microsleeps ───────────────────────────────────────
        if just_woke and closed_frames > self.ear_alert_frames:
            self._microsleep_count += 1
            self._microsleep_pts    = min(
                float(FATIGUE_MICROSLEEP_MAX),
                self._microsleep_pts + FATIGUE_MICROSLEEP_PTS
            )
            logger.info(
                f"Microsleep #{self._microsleep_count} "
                f"({closed_frames}fr). microsleep_pts={self._microsleep_pts:.1f}"
            )

        # ── Component 2: track closure duration ────────────────────────────
        if eyes_closed:
            if not self._in_closure:
                self._in_closure        = True
                self._closure_start_time = now
        else:
            if self._in_closure:
                dur = now - self._closure_start_time
                if dur > self._max_closure_sec:
                    self._max_closure_sec = dur
            self._in_closure = False

        # ── Component 3: yawn frequency ────────────────────────────────────
        if yawn_just_started:
            self._yawn_times.append(now)
        # Evict old yawns outside rolling window
        cutoff = now - FATIGUE_YAWN_WINDOW_SEC
        while self._yawn_times and self._yawn_times[0] < cutoff:
            self._yawn_times.popleft()

        # ── Compute raw score ───────────────────────────────────────────────
        micro_pts = self._microsleep_pts   # 0-6

        recent_yawns = len(self._yawn_times)
        yawn_pts = 0.0
        if recent_yawns >= FATIGUE_YAWN_THRESH:
            yawn_pts = float(np.clip(
                (recent_yawns - FATIGUE_YAWN_THRESH + 1) * 1.0,
                0.0, FATIGUE_YAWN_MAX_PTS
            ))

        closure_pts = 0.0
        if self._max_closure_sec >= FATIGUE_CLOSURE_SEC:
            closure_pts = float(np.clip(
                (self._max_closure_sec - FATIGUE_CLOSURE_SEC) * 0.5 + 0.5,
                0.0, FATIGUE_CLOSURE_MAX_PTS
            ))

        raw = micro_pts + yawn_pts + closure_pts   # 0-10

        # ── Slow decay when alert ───────────────────────────────────────────
        dt = now - self._last_fatigue_ts
        self._last_fatigue_ts = now
        if (not eyes_closed) and (not yawn_just_started) and raw > 0:
            decay = (FATIGUE_DECAY_PER_MIN / 60.0) * dt
            self._microsleep_pts  = max(0.0, self._microsleep_pts  - decay * 0.5)
            self._max_closure_sec = max(0.0, self._max_closure_sec - decay * 0.3)

        self.fatigue_score = float(np.clip(raw, 0.0, 10.0))

    def get_fatigue_level(self) -> str:
        s = self.fatigue_score
        if s >= 9: return "CRITICAL"
        if s >= 6: return "WARNING"
        if s >= 3: return "CAUTION"
        return "ALERT"

    # ──────────────────────────────────────────────
    # RECALIBRATE
    # ──────────────────────────────────────────────

    def recalibrate(self) -> None:
        self.calibrated           = False
        self.calib_frames         = []
        self.adaptive_ear_ready   = False
        self.adaptive_ear_samples = []
        self.adaptive_ear_thresh  = EAR_THRESHOLD
        self.ear_counter = self._blink_counter = self._ear_open_streak = 0
        self.mar_counter = self.head_counter   = self.gaze_counter     = 0
        self.gaze_buffer.clear()
        # Reset fatigue
        self.fatigue_score       = 0.0
        self._microsleep_pts     = 0.0
        self._microsleep_count   = 0
        self._yawn_times.clear()
        self._max_closure_sec    = 0.0
        self._in_closure         = False
        self._last_fatigue_ts    = time.monotonic()
        self._update_thresholds()
        logger.info("Recalibrating - look straight ahead.")

    # ──────────────────────────────────────────────
    # DRAW LANDMARKS
    # ──────────────────────────────────────────────

    def _draw_landmarks(self, frame, left_eye_pts, right_eye_pts,
                        mouth_pts, driver_score) -> None:
        col = score_color(driver_score)
        for pts in (left_eye_pts, right_eye_pts, mouth_pts):
            for x, y in pts:
                cv2.circle(frame, (int(x), int(y)), 2, col, -1)
        all_pts = np.vstack([left_eye_pts, right_eye_pts, mouth_pts])
        x1 = max(0, int(all_pts[:, 0].min()) - 10)
        y1 = max(0, int(all_pts[:, 1].min()) - 20)
        x2 = int(all_pts[:, 0].max()) + 10
        y2 = int(all_pts[:, 1].max()) + 20
        cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)

    # ──────────────────────────────────────────────
    # MAIN PROCESS FRAME
    # ──────────────────────────────────────────────

    def process(self, frame: np.ndarray) -> dict:
        self._update_fps()

        h, w    = frame.shape[:2]
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        driver_score   = 0
        status_parts   = []
        gaze_direction = "CENTER"
        ear_val = ear_left = ear_right = ear_min = 0.0
        mar_val = yaw_val = pitch_val = 0.0

        if results.multi_face_landmarks:
            self.no_face_counter = 0   # reset only when a face is found
            lm = results.multi_face_landmarks[0].landmark

            left_eye_pts   = extract_landmarks(lm, LEFT_EYE_IDX,   w, h)
            right_eye_pts  = extract_landmarks(lm, RIGHT_EYE_IDX,  w, h)
            mouth_pts      = extract_landmarks(lm, MOUTH_IDX,      w, h)
            left_iris_pts  = extract_landmarks(lm, LEFT_IRIS_IDX,  w, h)
            right_iris_pts = extract_landmarks(lm, RIGHT_IRIS_IDX, w, h)

            ear_left  = eye_aspect_ratio(left_eye_pts)
            ear_right = eye_aspect_ratio(right_eye_pts)
            ear_val   = (ear_left + ear_right) / 2.0
            ear_min   = min(ear_left, ear_right)
            mar_val   = mouth_aspect_ratio(mouth_pts)

            gaze_l = get_gaze_ratio(left_eye_pts,  left_iris_pts)
            gaze_r = get_gaze_ratio(right_eye_pts, right_iris_pts)
            self.gaze_buffer.append((gaze_l + gaze_r) / 2.0)
            gaze_ratio = float(np.mean(self.gaze_buffer))
            if gaze_ratio < GAZE_LEFT_THRESHOLD:
                gaze_direction = "RIGHT"
            elif gaze_ratio > GAZE_RIGHT_THRESHOLD:
                gaze_direction = "LEFT"
            self._last_gaze_dir = gaze_direction

            yaw_raw, pitch_raw = self.get_head_pose(lm, w, h)
            if yaw_raw is not None:
                yaw_val, pitch_val = yaw_raw, pitch_raw

            self.calibrate_ear(ear_val)
            thresh = self.adaptive_ear_thresh if self.adaptive_ear_ready else EAR_THRESHOLD

            if not self.calibrated or not self.adaptive_ear_ready:
                return {
                    "driver_score"  : 0,
                    "ear"           : float(ear_val),
                    "ear_left"      : float(ear_left),
                    "ear_right"     : float(ear_right),
                    "ear_min"       : float(ear_min),
                    "mar"           : float(mar_val),
                    "yaw"           : float(yaw_val),
                    "pitch"         : float(pitch_val),
                    "fatigue_score" : 0.0,
                    "fatigue_level" : "ALERT",
                    "microsleeps"   : 0,
                    "gaze_direction": "CENTER",
                    "status"        : "Calibrating",
                    "frame"         : frame,
                }

            # ── EAR scoring with blink filter ──────────────
            eyes_closed = (ear_val < thresh) or (ear_min < thresh * 0.88)
            just_woke   = False

            if eyes_closed:
                self._blink_counter   += 1
                self._ear_open_streak  = 0
                if self._blink_counter > BLINK_MAX_FRAMES:
                    self.ear_counter += 1
            else:
                just_woke           = (self._blink_counter > BLINK_MAX_FRAMES)
                self.ear_counter    = 0   # reset immediately on eye open
                self._blink_counter = 0
                self._ear_open_streak += 1

            # ── MAR scoring — detect yawn ONSET for fatigue tracking ───────
            yawn_just_started = False
            if mar_val > MAR_THRESHOLD:
                self.mar_counter += 1
                if self.mar_counter == self.mar_frames:
                    yawn_just_started = True   # first frame crossing threshold
            else:
                self.mar_counter = max(0, self.mar_counter - 1)

            yawn_active = (self.mar_counter >= self.mar_frames)

            # ── Update fatigue score ────────────────────────
            self._update_fatigue(
                eyes_closed      = eyes_closed,
                just_woke        = just_woke,
                closed_frames    = self._blink_counter if eyes_closed else 0,
                yawn_just_started = yawn_just_started,
            )

            # ── Drowsy / Eyes Closing alerts ───────────────
            if self.ear_counter >= self.ear_alert_frames:
                driver_score = max(driver_score, 2)
                status_parts.append("DROWSY")
            elif self.ear_counter >= self.ear_warn_frames:
                driver_score = max(driver_score, 1)
                status_parts.append("Eyes Closing")

            # ── Yawn alert ─────────────────────────────────
            if yawn_active:
                driver_score = max(driver_score, 1)
                status_parts.append("Yawning")

            # ── Fatigue score escalation ────────────────────
            # Fatigue escalates driver_score between individual drowsy events.
            # A score >= 6 (WARNING) bumps to score 1 even if eyes are open.
            # A score >= 9 (CRITICAL) bumps to score 2 — session-level danger.
            fl = self.get_fatigue_level()
            if fl == "CRITICAL" and driver_score < 2:
                driver_score = 2
                if "DROWSY" not in status_parts:
                    status_parts.append("DROWSY")
            elif fl == "WARNING" and driver_score < 1:
                driver_score = 1
                if "Eyes Closing" not in status_parts:
                    status_parts.append("Eyes Closing")

            self._draw_landmarks(frame, left_eye_pts, right_eye_pts,
                                 mouth_pts, driver_score)

        else:
            self.no_face_counter += 1
            score = 2 if self.no_face_counter >= self.NO_FACE_ALERT_FRAMES else 1
            return {
                "driver_score"  : score,
                "ear"           : 0.0,
                "ear_left"      : 0.0,
                "ear_right"     : 0.0,
                "ear_min"       : 0.0,
                "mar"           : 0.0,
                "yaw"           : 0.0,
                "pitch"         : 0.0,
                "fatigue_score" : self.fatigue_score,
                "fatigue_level" : self.get_fatigue_level(),
                "microsleeps"   : self._microsleep_count,
                "gaze_direction": "CENTER",
                "status"        : "No Face Detected",
                "frame"         : frame,
            }

        # ── Head direction scoring ─────────────────────────────────────────
        if abs(yaw_val) >= YAW_WARN_DEG or abs(pitch_val) >= PITCH_WARN_DEG:
            if abs(yaw_val) >= abs(pitch_val):
                self._last_head_dir = "LEFT" if yaw_val < 0 else "RIGHT"
            else:
                self._last_head_dir = "DOWN" if pitch_val > 0 else "UP"
            self.head_counter += 1
        else:
            self._last_head_dir = ""   # clear stale label
            self.head_counter   = max(0, self.head_counter - 1)

        if self.head_counter >= self.head_alert_frames:
            driver_score = max(driver_score, 2)
            status_parts.append(f"Head {self._last_head_dir}")
        elif self.head_counter >= self.head_warn_frames:
            driver_score = max(driver_score, 1)
            status_parts.append(f"Head {self._last_head_dir}")

        # ── Gaze scoring ───────────────────────────────────────────────────
        if gaze_direction != "CENTER":
            self.gaze_counter += 1
        else:
            self.gaze_counter = max(0, self.gaze_counter - 1)

        if self.gaze_counter >= self.gaze_warn_frames:
            driver_score = max(driver_score, 1)
            status_parts.append(f"Gaze {gaze_direction}")

        if not self.calibrated:
            cv2.putText(frame, "Calibrating - look straight ahead",
                        (8, 18), FONT, 0.42, (0, 255, 255), 1, cv2.LINE_AA)

        status_text = " | ".join(status_parts) if status_parts else "Alert"

        return {
            "driver_score"  : driver_score,
            "ear"           : float(ear_val),
            "ear_left"      : float(ear_left),
            "ear_right"     : float(ear_right),
            "ear_min"       : float(ear_min),
            "mar"           : float(mar_val),
            "yaw"           : float(yaw_val),
            "pitch"         : float(pitch_val),
            "fatigue_score" : self.fatigue_score,
            "fatigue_level" : self.get_fatigue_level(),
            "microsleeps"   : self._microsleep_count,
            "gaze_direction": gaze_direction,
            "status"        : status_text,
            "frame"         : frame,
        }