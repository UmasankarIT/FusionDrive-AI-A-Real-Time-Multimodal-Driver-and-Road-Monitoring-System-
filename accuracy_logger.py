import logging
import time
from collections import deque
from typing import Dict

logger = logging.getLogger("AccuracyLogger")

# ─────────────────────────────────────────────
# TUNABLE WEIGHTS
# ─────────────────────────────────────────────

# Road accuracy
W_RNN_CONF      = 0.35   # rnn_confidence
W_HAZARD_CONS   = 0.25   # hazard-level temporal consistency
W_COLL_STAB     = 0.20   # collision-probability stability
W_SURF          = 0.10   # road-surface clarity (road_status)
W_LANE_ROAD     = 0.10   # lane detection quality (road side)

# Driver accuracy
W_EAR           = 0.30   # EAR validity
W_MAR           = 0.15   # MAR validity
W_HEADPOSE      = 0.25   # head-pose confidence
W_FATIGUE_SM    = 0.15   # fatigue-score smoothness
W_FACE_CONT     = 0.15   # face continuity

# Lane accuracy
W_LANE_CONF     = 0.40   # lane_confidence from road_result
W_BOTH_LANES    = 0.25   # both left + right detected
W_LANE_CONS     = 0.25   # temporal consistency of lane_status
W_SCENE         = 0.10   # scene-mode bonus/penalty

WINDOW          = 10     # frames for rolling consistency windows
EMA_ALPHA       = 0.15   # smoothing factor for displayed values

# ─────────────────────────────────────────────
# VALID RANGES  (used for sanity-check scoring)
# ─────────────────────────────────────────────
EAR_VALID_LOW   = 0.10   # below this = fully closed / unreliable
EAR_VALID_HIGH  = 0.45   # above this = mediapipe noise
MAR_VALID_LOW   = 0.00
MAR_VALID_HIGH  = 0.80
YAW_MAX         = 45.0   # degrees — beyond = face turned away
PITCH_MAX       = 35.0


class AccuracyLogger:
    """
    Drop-in accuracy tracker.  No external dependencies beyond the standard
    library and numpy (already present in the project).
    """

    def __init__(self, window: int = WINDOW):
        self._win = window

        # Rolling buffers
        self._hazard_buf: deque   = deque(maxlen=window)
        self._coll_buf:   deque   = deque(maxlen=window)
        self._lane_buf:   deque   = deque(maxlen=window)   # "DETECTED" / "NOT_DETECTED"
        self._fatigue_buf: deque  = deque(maxlen=window)

        # EMA-smoothed display values
        self._road_acc:   float   = 0.5
        self._driver_acc: float   = 0.5
        self._lane_acc:   float   = 0.5

        # Session totals (for report())
        self._n_frames:   int     = 0
        self._road_sum:   float   = 0.0
        self._driver_sum: float   = 0.0
        self._lane_sum:   float   = 0.0
        self._start_time: float   = time.time()

        # Optional overlay toggle (key 'A')
        self.show_overlay: bool   = False

        logger.info("AccuracyLogger initialised  (window=%d)", window)

    # ─────────────────────────────────────────────
    # PUBLIC API
    # ─────────────────────────────────────────────

    def update(self, road_result: dict, driver_result: dict) -> None:
        """Call once per processed frame."""
        raw_road   = self._compute_road_accuracy(road_result)
        raw_driver = self._compute_driver_accuracy(driver_result)
        raw_lane   = self._compute_lane_accuracy(road_result)

        # EMA smoothing for display
        a = EMA_ALPHA
        self._road_acc   = a * raw_road   + (1 - a) * self._road_acc
        self._driver_acc = a * raw_driver + (1 - a) * self._driver_acc
        self._lane_acc   = a * raw_lane   + (1 - a) * self._lane_acc

        # Clip to [0, 1]
        self._road_acc   = max(0.0, min(1.0, self._road_acc))
        self._driver_acc = max(0.0, min(1.0, self._driver_acc))
        self._lane_acc   = max(0.0, min(1.0, self._lane_acc))

        # Session totals
        self._n_frames   += 1
        self._road_sum   += self._road_acc
        self._driver_sum += self._driver_acc
        self._lane_sum   += self._lane_acc

    def get_accuracy(self) -> Dict[str, float]:
        """
        Returns current smoothed accuracy values in [0, 1].

        Keys: "road", "driver", "lane"
        """
        return {
            "road":   round(self._road_acc,   4),
            "driver": round(self._driver_acc, 4),
            "lane":   round(self._lane_acc,   4),
        }

    def handle_key(self, key: int) -> None:
        """Pass raw cv2.waitKey() result here for hotkey handling."""
        if key in (ord('a'), ord('A')):
            self.show_overlay = not self.show_overlay
            logger.info("Accuracy overlay: %s", "ON" if self.show_overlay else "OFF")

    def report(self) -> None:
        """Print a session-end summary to the log."""
        elapsed = time.time() - self._start_time
        n       = max(self._n_frames, 1)
        logger.info("=" * 52)
        logger.info("  Accuracy Report  (%.0f s, %d frames)", elapsed, n)
        logger.info("  Road   accuracy  : %.1f %%", 100 * self._road_sum   / n)
        logger.info("  Driver accuracy  : %.1f %%", 100 * self._driver_sum / n)
        logger.info("  Lane   accuracy  : %.1f %%", 100 * self._lane_sum   / n)
        logger.info("=" * 52)

    # ─────────────────────────────────────────────
    # ROAD ACCURACY
    # ─────────────────────────────────────────────

    def _compute_road_accuracy(self, r: dict) -> float:
        """
        Compute instantaneous road-accuracy score [0, 1].

        Components
        ----------
        rnn_confidence    – direct signal from the RoadRNN
        hazard_consistency – how stable the hazard level has been
        collision_stability – variance of collision_prob over window
        road_surface       – "CLEAR" → 1.0, "UNCLEAR" → 0.4
        lane_quality       – "DETECTED" → 1.0, else 0.3
        """
        rnn_conf = float(r.get("rnn_confidence", 0.5))

        # Hazard level temporal consistency
        h_level = int(r.get("hazard_level", 0))
        self._hazard_buf.append(h_level)
        h_cons = _temporal_consistency(self._hazard_buf)

        # Collision probability stability
        c_prob = float(r.get("collision_prob", 0.0))
        self._coll_buf.append(c_prob)
        c_stab = _value_stability(self._coll_buf)

        # Road surface quality
        surf_score = 1.0 if r.get("road_status", "CLEAR") == "CLEAR" else 0.4

        # Lane presence on road side
        lane_score = 1.0 if r.get("lane_status", "NOT_DETECTED") == "DETECTED" else 0.3

        score = (
            W_RNN_CONF    * rnn_conf   +
            W_HAZARD_CONS * h_cons     +
            W_COLL_STAB   * c_stab     +
            W_SURF        * surf_score +
            W_LANE_ROAD   * lane_score
        )
        return float(max(0.0, min(1.0, score)))

    # ─────────────────────────────────────────────
    # DRIVER ACCURACY
    # ─────────────────────────────────────────────

    def _compute_driver_accuracy(self, d: dict) -> float:
        """
        Compute instantaneous driver-accuracy score [0, 1].

        Higher = MediaPipe landmarks are reliable and metrics are stable.
        Lower  = face absent, metrics out of valid range, or erratic readings.

        Components
        ----------
        ear_score       – EAR within physiologically valid range
        mar_score       – MAR within valid range
        headpose_score  – yaw/pitch not at extremes
        fatigue_smooth  – fatigue score changing slowly (stable signal)
        face_continuity – face has been consistently present
        """
        ear = float(d.get("ear", 0.0))
        mar = float(d.get("mar", 0.0))
        yaw = abs(float(d.get("yaw", 0.0)))
        pitch = abs(float(d.get("pitch", 0.0)))
        fatigue = float(d.get("fatigue_score", 0.0))
        status  = str(d.get("status", ""))

        # Face presence: no face or calibrating → low score
        face_present = ("No Face" not in status) and ("Calibrat" not in status)
        face_cont    = 1.0 if face_present else 0.1

        # EAR validity
        if not face_present or ear == 0.0:
            ear_score = 0.2
        elif EAR_VALID_LOW <= ear <= EAR_VALID_HIGH:
            ear_score = 1.0
        else:
            # Penalise proportionally to how far outside the range
            dist = max(EAR_VALID_LOW - ear, ear - EAR_VALID_HIGH, 0.0)
            ear_score = max(0.0, 1.0 - dist / 0.15)

        # MAR validity
        if not face_present or mar == 0.0:
            mar_score = 0.3
        elif MAR_VALID_LOW <= mar <= MAR_VALID_HIGH:
            mar_score = 1.0
        else:
            dist = max(0.0, mar - MAR_VALID_HIGH)
            mar_score = max(0.0, 1.0 - dist / 0.4)

        # Head-pose confidence: degrades as yaw/pitch approach limits
        yaw_ok   = max(0.0, 1.0 - yaw   / YAW_MAX)
        pitch_ok = max(0.0, 1.0 - pitch / PITCH_MAX)
        headpose_score = (yaw_ok + pitch_ok) / 2.0

        # Fatigue-score smoothness (temporal stability)
        self._fatigue_buf.append(fatigue)
        fatigue_smooth = _value_stability(self._fatigue_buf)

        score = (
            W_EAR        * ear_score      +
            W_MAR        * mar_score      +
            W_HEADPOSE   * headpose_score +
            W_FATIGUE_SM * fatigue_smooth +
            W_FACE_CONT  * face_cont
        )
        return float(max(0.0, min(1.0, score)))

    # ─────────────────────────────────────────────
    # LANE ACCURACY
    # ─────────────────────────────────────────────

    def _compute_lane_accuracy(self, r: dict) -> float:
        """
        Compute instantaneous lane-accuracy score [0, 1].

        Components
        ----------
        lane_conf      – raw Hough-line confidence from detect_lanes()
        both_lanes     – both left and right lanes detected
        lane_cons      – temporal consistency of lane_status string
        scene_penalty  – night/dusk reduces expected confidence
        """
        # lane_confidence is embedded in road_result if we propagate it,
        # otherwise we reconstruct from lane_status.
        lane_conf  = float(r.get("lane_confidence", 0.0))   # may be absent
        lane_status = str(r.get("lane_status", "NOT_DETECTED"))
        scene_mode  = str(r.get("scene_mode", "DAY"))

        # If lane_confidence not in result, approximate from status
        if lane_conf == 0.0:
            lane_conf = 0.8 if lane_status == "DETECTED" else 0.15

        # Both lanes — road_result carries left_lane / right_lane dicts when available
        left_ok  = r.get("left_lane")  is not None
        right_ok = r.get("right_lane") is not None
        if not left_ok and not right_ok:
            # Fallback: if lane is DETECTED but we don't have individual flags,
            # give partial credit
            both_score = 0.6 if lane_status == "DETECTED" else 0.0
        else:
            both_score = (0.5 * float(left_ok) + 0.5 * float(right_ok))

        # Temporal consistency of lane_status
        self._lane_buf.append(lane_status)
        lane_cons = _temporal_consistency_str(self._lane_buf)

        # Scene-mode factor (night degrades lane visibility)
        scene_factor = {"DAY": 1.0, "DUSK": 0.75, "NIGHT": 0.55}.get(scene_mode, 1.0)

        score = (
            W_LANE_CONF * lane_conf   +
            W_BOTH_LANES * both_score +
            W_LANE_CONS  * lane_cons  +
            W_SCENE      * scene_factor
        )
        return float(max(0.0, min(1.0, score)))


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _temporal_consistency(buf: deque) -> float:
    """
    Score 0-1: how consistent the last N values have been.
    Uses mode-frequency: if all values identical → 1.0.
    """
    if not buf:
        return 0.5
    from collections import Counter
    cnt = Counter(buf)
    mode_freq = cnt.most_common(1)[0][1]
    return mode_freq / len(buf)


def _temporal_consistency_str(buf: deque) -> float:
    """Same as above but for string buffers."""
    return _temporal_consistency(buf)


def _value_stability(buf: deque) -> float:
    """
    Score 0-1: stability of a continuous value.
    High variance → low score.  Stable → 1.0.
    """
    if len(buf) < 2:
        return 0.5
    import numpy as _np
    arr = _np.array(list(buf), dtype=float)
    std = float(_np.std(arr))
    # Normalise: std of 0 → 1.0; std of 0.3+ → ~0.0
    return float(max(0.0, 1.0 - std / 0.3))