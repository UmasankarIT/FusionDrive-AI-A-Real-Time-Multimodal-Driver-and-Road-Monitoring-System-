import time
import logging
import numpy as np
from enum import Enum
from typing import Dict, Tuple

logger = logging.getLogger("FusionEngine")


# ─────────────────────────────────────────────
# ENUMS
# ─────────────────────────────────────────────

class AlertType(Enum):
    SAFE               = 0
    EYES_CLOSING       = 1
    DROWSY             = 2
    YAWNING            = 3
    HEAD_TURN          = 4
    GAZE_AWAY          = 5
    HEAD_AND_GAZE      = 6
    PEDESTRIAN         = 7
    VEHICLE_CLOSE      = 8
    LANE_DRIFT         = 9
    COMBINED_RISK      = 10
    MICROSLEEP_RISK    = 11


class Sev(Enum):
    SAFE     = "SAFE"
    CAUTION  = "CAUTION"
    WARNING  = "WARNING"
    CRITICAL = "CRITICAL"


# ─────────────────────────────────────────────
# THRESHOLDS
# ─────────────────────────────────────────────

# How many consecutive seconds each condition must persist
# before escalating severity.  Shorter = more sensitive.
T_EYES_CLOSING_CAUTION  = 0.4    # "Eyes Closing" → CAUTION
T_EYES_CLOSING_WARN     = 0.9    # "Eyes Closing" → WARNING
T_DROWSY_WARN           = 0.3    # "DROWSY"       → WARNING  (fast — already critical in driver.py)
T_DROWSY_CRIT           = 1.5    # "DROWSY"       → CRITICAL

T_YAWN_CAUTION          = 0.6    # "Yawning"      → CAUTION
T_YAWN_WARN             = 2.0    # "Yawning"      → WARNING (sustained = fatigue)

T_HEAD_GRACE            = 1.2    # head turn grace — mirrors / U-turns
T_HEAD_WARN             = 2.0    # head turn → WARNING
T_HEAD_CRIT             = 3.5    # head turn → CRITICAL

T_GAZE_GRACE            = 0.8    # brief glance away
T_GAZE_WARN             = 1.8    # gaze away → WARNING
T_GAZE_CRIT             = 3.0    # gaze away → CRITICAL

# Road distance thresholds (normalised 0–1)
VEH_CAUTION             = 0.45
VEH_WARN                = 0.60
VEH_CRIT                = 0.75
PED_CAUTION             = 0.40
PED_WARN                = 0.55
PED_CRIT                = 0.70

# Alert cooldowns (seconds) — same alert won't repeat within this window
COOLDOWN = {
    Sev.CAUTION : 4.0,
    Sev.WARNING : 3.0,
    Sev.CRITICAL: 1.5,
}


# ─────────────────────────────────────────────
# MESSAGES  — one per (AlertType, severity)
# ─────────────────────────────────────────────
# {dir}     = head / gaze direction string
# {dur:.0f} = sustained seconds
# {pct}     = perclos percentage string

MESSAGES: Dict[AlertType, Dict[str, str]] = {

    AlertType.EYES_CLOSING: {
        "CAUTION" : "Eyes are getting heavy — stay alert.",
        "WARNING" : "Eyes closing! Open your eyes and focus on the road.",
        "CRITICAL": "Eyes closing too long — pull over if feeling drowsy.",
    },

    AlertType.DROWSY: {
        "CAUTION" : "Drowsiness detected — take a rest break soon.",
        "WARNING" : "Driver drowsy! Eyes heavy — focus on the road now.",
        "CRITICAL": "DROWSY DRIVER — Pull over immediately. Do not continue.",
    },

    AlertType.YAWNING: {
        "CAUTION" : "Yawning detected — fatigue building. Consider a break.",
        "WARNING" : "Repeated yawning — you are fatigued. Take a rest stop.",
        "CRITICAL": "Severe fatigue — stop driving at the next safe location.",
    },

    AlertType.HEAD_TURN: {
        "CAUTION" : "Head turned {dir} — keep eyes on the road ahead.",
        "WARNING" : "Head turned {dir} for {dur:.0f}s! Focus on the road.",
        "CRITICAL": "Head off road {dur:.0f}s — CRITICAL. Watch the road!",
    },

    AlertType.GAZE_AWAY: {
        "CAUTION" : "Eyes looking {dir} — keep your gaze on the road.",
        "WARNING" : "Eyes off road {dur:.0f}s — look ahead immediately!",
        "CRITICAL": "Eyes off road {dur:.0f}s — DANGER. Focus ahead now!",
    },

    AlertType.HEAD_AND_GAZE: {
        "CAUTION" : "Head and eyes both off road — please focus ahead.",
        "WARNING" : "Head turned AND eyes off road — HIGH distraction risk!",
        "CRITICAL": "SEVERE DISTRACTION — Head and eyes off road. Look ahead!",
    },

    AlertType.PEDESTRIAN: {
        "CAUTION" : "Pedestrian nearby — reduce speed and stay alert.",
        "WARNING" : "Pedestrian ahead! Slow down, prepare to stop.",
        "CRITICAL": "PEDESTRIAN IN PATH — Brake now!",
    },

    AlertType.VEHICLE_CLOSE: {
        "CAUTION" : "Vehicle ahead — maintain a safe following distance.",
        "WARNING" : "Too close to vehicle! Increase following distance.",
        "CRITICAL": "COLLISION RISK — Brake immediately!",
    },

    AlertType.LANE_DRIFT: {
        "CAUTION" : "Lane markings unclear — stay centred in your lane.",
        "WARNING" : "Lane not detected — possible drift. Check position.",
        "CRITICAL": "Lane lost — dangerous drift possible. Steer carefully.",
    },

    AlertType.COMBINED_RISK: {
        "CAUTION" : "Fatigue + road hazard ahead — stay sharp.",
        "WARNING" : "Drowsy driver + road hazard — HIGH RISK. Focus now!",
        "CRITICAL": "CRITICAL: Impaired driver + road danger — PULL OVER.",
    },

    AlertType.MICROSLEEP_RISK: {
        "CAUTION" : "High eye-closure rate — microsleep risk building.",
        "WARNING" : "PERCLOS high — microsleep risk! Pull over soon.",
        "CRITICAL": "MICROSLEEP IMMINENT — Stop the vehicle immediately.",
    },

    AlertType.SAFE: {
        "SAFE": "System nominal. Drive safely.",
    },
}


def _msg(atype: AlertType, sev: Sev, **kw) -> str:
    sev_map  = MESSAGES.get(atype, {})
    template = sev_map.get(sev.value, sev_map.get("WARNING", "Stay alert."))
    try:
        return template.format(**kw)
    except (KeyError, ValueError):
        return template


# ─────────────────────────────────────────────
# RISK SCORES  (0.0 – 1.0)
# ─────────────────────────────────────────────

def _driver_risk(driver: Dict, t: Dict) -> float:
    score  = 0.0
    ds     = driver.get("driver_score", 0)
    fl     = driver.get("fatigue_level", "ALERT")   # "ALERT"/"CAUTION"/"WARNING"/"CRITICAL"

    score += {0: 0.0, 1: 0.25, 2: 0.65}.get(ds, 0.0)

    # Fatigue contribution — uses fatigue_level from driver.py (replaces PERCLOS)
    if   fl == "CRITICAL": score += 0.35
    elif fl == "WARNING":  score += 0.20
    elif fl == "CAUTION":  score += 0.08

    off_t  = max(t.get("head", 0.0), t.get("gaze", 0.0))
    score += min(0.25, off_t / 10.0)

    return float(np.clip(score, 0.0, 1.0))


def _road_risk(road: Dict) -> float:
    score        = 0.0
    hazard_level = road.get("hazard_level", 0)
    vehicles     = road.get("vehicles",    [])
    pedestrians  = road.get("pedestrians", [])

    score += {0: 0.0, 1: 0.15, 2: 0.40, 3: 0.70}.get(hazard_level, 0.0)

    close_veh = max((v.get("distance", 0) for v in vehicles),    default=0.0)
    close_ped = max((p.get("distance", 0) for p in pedestrians), default=0.0)

    if close_veh >= VEH_CRIT:      score += 0.40
    elif close_veh >= VEH_WARN:    score += 0.25
    elif close_veh >= VEH_CAUTION: score += 0.10

    if close_ped >= PED_CRIT:      score += 0.50
    elif close_ped >= PED_WARN:    score += 0.30
    elif close_ped >= PED_CAUTION: score += 0.15

    if road.get("lane_status") == "NOT_DETECTED":
        score += 0.10

    return float(np.clip(score, 0.0, 1.0))


# ─────────────────────────────────────────────
# MAIN CLASS
# ─────────────────────────────────────────────

class FusionEngine:
    """
    Fuses DriverMonitor + RoadMonitor outputs into precise,
    non-overlapping, human-realistic safety alerts.
    """

    def __init__(self):
        # Sustained-condition timers (wall-clock seconds)
        self._t: Dict[str, float] = {
            "eyes_closing": 0.0,
            "drowsy"      : 0.0,
            "yawning"     : 0.0,
            "head"        : 0.0,   # head off road
            "gaze"        : 0.0,   # gaze off road
        }

        # Track the current head/gaze direction for message filling
        self._head_dir = ""
        self._gaze_dir = ""

        # Cooldown: AlertType → wall-clock time when it last fired
        self._last_fired: Dict[AlertType, float] = {}

        self.total_alerts  = 0
        self.alert_history = []

        self._last_ts = time.monotonic()
        logger.info("FusionEngine ready.")

    # ─────────────────────────────────────────
    # dt
    # ─────────────────────────────────────────

    def _dt(self) -> float:
        now = time.monotonic()
        dt  = float(np.clip(now - self._last_ts, 0.001, 0.5))
        self._last_ts = now
        return dt

    # ─────────────────────────────────────────
    # UPDATE TIMERS
    # ─────────────────────────────────────────

    def _update_timers(self, driver: Dict, dt: float) -> None:
        """
        Increment or decay each condition timer based on driver status.
        Uses EXACT status strings set by driver.py.
        """
        status  = driver.get("status", "Alert")
        parts   = [p.strip() for p in status.split("|")]

        # ── Eyes closing / drowsy ────────────────────────
        is_drowsy        = any("DROWSY"        in p for p in parts)
        is_eyes_closing  = any("Eyes Closing"  in p for p in parts)

        if is_drowsy:
            self._t["drowsy"]       += dt
            self._t["eyes_closing"]  = 0.0
        elif is_eyes_closing:
            self._t["eyes_closing"] += dt
            self._t["drowsy"]        = max(0.0, self._t["drowsy"] - dt * 2)
        else:
            self._t["drowsy"]       = max(0.0, self._t["drowsy"]       - dt * 3)
            self._t["eyes_closing"] = max(0.0, self._t["eyes_closing"] - dt * 2)

        # ── Yawning ──────────────────────────────────────
        is_yawning = any("Yawning" in p for p in parts)
        if is_yawning:
            self._t["yawning"] += dt
        else:
            self._t["yawning"] = max(0.0, self._t["yawning"] - dt * 1.5)

        # ── Head direction ───────────────────────────────
        # Matches "Head LEFT", "Head RIGHT", "Head UP", "Head DOWN"
        head_parts = [p for p in parts if p.startswith("Head ")]
        if head_parts:
            self._t["head"] += dt
            # Extract direction from e.g. "Head LEFT"
            self._head_dir = head_parts[0].replace("Head ", "").strip()
        else:
            # Decay faster than increment — brief turns clear quickly
            self._t["head"] = max(0.0, self._t["head"] - dt * 2.5)

        # ── Gaze direction ───────────────────────────────
        # Matches "Gaze LEFT", "Gaze RIGHT"
        gaze_parts = [p for p in parts if p.startswith("Gaze ")]
        if gaze_parts:
            self._t["gaze"] += dt
            self._gaze_dir = gaze_parts[0].replace("Gaze ", "").strip()
        else:
            self._t["gaze"] = max(0.0, self._t["gaze"] - dt * 2.0)

    # ─────────────────────────────────────────
    # PICK ALERT
    # ─────────────────────────────────────────

    def _pick(self, driver: Dict, road: Dict,
              d_risk: float, r_risk: float) -> Tuple[AlertType, Sev]:
        """Return the single highest-priority alert for this frame."""

        t           = self._t
        fl          = driver.get("fatigue_level", "ALERT")   # from driver.py
        hazard      = road.get("hazard_level", 0)
        vehicles = road.get("vehicles",    [])
        peds     = road.get("pedestrians", [])

        close_ped = max((p.get("distance", 0) for p in peds),     default=0.0)
        close_veh = max((v.get("distance", 0) for v in vehicles), default=0.0)

        # ── 1. Pedestrian ────────────────────────────────
        if close_ped >= PED_CRIT:
            return AlertType.PEDESTRIAN, Sev.CRITICAL
        if close_ped >= PED_WARN:
            return AlertType.PEDESTRIAN, (Sev.CRITICAL if d_risk > 0.3 else Sev.WARNING)
        if close_ped >= PED_CAUTION:
            return AlertType.PEDESTRIAN, Sev.CAUTION

        # ── 1b. RNN collision warning (temporal — catches gradual approach) ──
        coll_prob = road.get("collision_prob", 0.0)
        if road.get("collision_warning", False):
            sev = Sev.CRITICAL if coll_prob > 0.80 else Sev.WARNING
            return AlertType.VEHICLE_CLOSE, sev

        # ── 2. Vehicle collision ─────────────────────────
        if close_veh >= VEH_CRIT:
            return AlertType.VEHICLE_CLOSE, (Sev.CRITICAL if d_risk > 0.3 else Sev.WARNING)
        if close_veh >= VEH_WARN:
            return AlertType.VEHICLE_CLOSE, Sev.WARNING
        if close_veh >= VEH_CAUTION:
            return AlertType.VEHICLE_CLOSE, Sev.CAUTION

        # ── 3. Combined risk ─────────────────────────────
        if d_risk >= 0.55 and r_risk >= 0.45:
            return AlertType.COMBINED_RISK, Sev.CRITICAL
        if d_risk >= 0.35 and r_risk >= 0.28:
            return AlertType.COMBINED_RISK, Sev.WARNING

        # ── 4. Microsleep / Fatigue ──────────────────────
        # Uses fatigue_level from driver.py — works correctly at low framerates
        # unlike PERCLOS which needs 60+ fps to be statistically valid.
        if   fl == "CRITICAL": return AlertType.MICROSLEEP_RISK, Sev.CRITICAL
        elif fl == "WARNING":  return AlertType.MICROSLEEP_RISK, Sev.WARNING
        elif fl == "CAUTION":  return AlertType.MICROSLEEP_RISK, Sev.CAUTION

        # ── 5. Drowsy (sustained) ────────────────────────
        if t["drowsy"] >= T_DROWSY_CRIT:
            return AlertType.DROWSY, Sev.CRITICAL
        if t["drowsy"] >= T_DROWSY_WARN:
            return AlertType.DROWSY, (Sev.CRITICAL if hazard >= 2 else Sev.WARNING)

        # ── 6. Head AND gaze both off road ───────────────
        if t["head"] > T_HEAD_GRACE and t["gaze"] > T_GAZE_GRACE:
            if t["head"] >= T_HEAD_CRIT or t["gaze"] >= T_GAZE_CRIT:
                return AlertType.HEAD_AND_GAZE, Sev.CRITICAL
            elif t["head"] >= T_HEAD_WARN or t["gaze"] >= T_GAZE_WARN:
                return AlertType.HEAD_AND_GAZE, Sev.WARNING
            else:
                return AlertType.HEAD_AND_GAZE, Sev.CAUTION

        # ── 7. Head turned (sustained, past grace) ───────
        if t["head"] >= T_HEAD_CRIT:
            return AlertType.HEAD_TURN, Sev.CRITICAL
        if t["head"] >= T_HEAD_WARN:
            return AlertType.HEAD_TURN, (Sev.CRITICAL if hazard >= 2 else Sev.WARNING)
        if t["head"] > T_HEAD_GRACE:
            return AlertType.HEAD_TURN, Sev.CAUTION

        # ── 8. Gaze away (sustained, past grace) ─────────
        if t["gaze"] >= T_GAZE_CRIT:
            return AlertType.GAZE_AWAY, Sev.CRITICAL
        if t["gaze"] >= T_GAZE_WARN:
            return AlertType.GAZE_AWAY, (Sev.CRITICAL if hazard >= 2 else Sev.WARNING)
        if t["gaze"] > T_GAZE_GRACE:
            return AlertType.GAZE_AWAY, Sev.CAUTION

        # ── 9. Eyes closing (mild) ───────────────────────
        if t["eyes_closing"] >= T_EYES_CLOSING_WARN:
            return AlertType.EYES_CLOSING, (Sev.WARNING if hazard >= 1 else Sev.CAUTION)
        if t["eyes_closing"] >= T_EYES_CLOSING_CAUTION:
            return AlertType.EYES_CLOSING, Sev.CAUTION

        # ── 10. Yawning ──────────────────────────────────
        if t["yawning"] >= T_YAWN_WARN:
            return AlertType.YAWNING, Sev.WARNING
        if t["yawning"] >= T_YAWN_CAUTION:
            return AlertType.YAWNING, Sev.CAUTION

        # ── 11. Road hazard only ─────────────────────────
        if hazard == 3: return AlertType.VEHICLE_CLOSE, Sev.CRITICAL
        if hazard == 2: return AlertType.VEHICLE_CLOSE, Sev.WARNING
        if hazard == 1: return AlertType.VEHICLE_CLOSE, Sev.CAUTION

        # ── 12. Lane drift ───────────────────────────────
        if road.get("lane_status") == "NOT_DETECTED":
            return AlertType.LANE_DRIFT, Sev.CAUTION

        return AlertType.SAFE, Sev.SAFE

    # ─────────────────────────────────────────
    # COOLDOWN CHECK
    # ─────────────────────────────────────────

    def _ok(self, atype: AlertType, sev: Sev) -> bool:
        if atype == AlertType.SAFE:
            return False
        cd   = COOLDOWN.get(sev, 3.0)
        last = self._last_fired.get(atype, 0.0)
        return (time.monotonic() - last) >= cd

    # ─────────────────────────────────────────
    # PUBLIC: process()
    # ─────────────────────────────────────────

    def process(self, driver_data: Dict, road_data: Dict) -> Dict:
        dt = self._dt()
        self._update_timers(driver_data, dt)

        sustained = {"head": self._t["head"], "gaze": self._t["gaze"]}
        d_risk    = _driver_risk(driver_data, sustained)
        r_risk    = _road_risk(road_data)
        combined  = float(np.clip(d_risk * 0.55 + r_risk * 0.45, 0.0, 1.0))
        
        # 🚨 skip alerts if driver not ready
        if driver_data.get("status") == "No Face Detected":
            return {
                "alert_type": "SAFE",
                "alert_severity": "SAFE",
                "alert_message": "Waiting for driver...",
                "driver_risk": 0.0,
                "road_risk": 0.0,
                "combined_risk": 0.0,
                "should_alert": False,
                "suppressed": False,
                "total_alerts": self.total_alerts,
            }

        atype, sev = self._pick(driver_data, road_data, d_risk, r_risk)

        # Build message with live context
        dur = max(self._t["head"], self._t["gaze"], self._t["drowsy"])
        msg = _msg(
            atype, sev,
            dir = self._head_dir or self._gaze_dir,
            dur = dur,
            pct = f"{driver_data.get('fatigue_score', 0.0) / 10.0:.0%}",
        )

        should_alert = self._ok(atype, sev)
        suppressed   = (atype != AlertType.SAFE) and (not should_alert)

        if should_alert:
            self._last_fired[atype] = time.monotonic()
            self.total_alerts += 1
            self.alert_history.append({
                "type"    : atype.name,
                "severity": sev.value,
                "message" : msg,
                "time"    : time.strftime("%H:%M:%S"),
            })
            if len(self.alert_history) > 300:
                self.alert_history = self.alert_history[-150:]

        return {
            "alert_type"    : atype.name,
            "alert_severity": sev.value,
            "alert_message" : msg,
            "driver_risk"   : d_risk,
            "road_risk"     : r_risk,
            "combined_risk" : combined,
            "should_alert"  : should_alert,
            "suppressed"    : suppressed,
            "total_alerts"  : self.total_alerts,
        }

    def get_statistics(self) -> Dict:
        return {
            "total_alerts" : self.total_alerts,
            "alert_history": self.alert_history[-10:],
        }

    def reset(self) -> None:
        for k in self._t:
            self._t[k] = 0.0
        self._last_fired.clear()
        self.total_alerts  = 0
        self.alert_history = []
        logger.info("FusionEngine reset.")