"""
╔══════════════════════════════════════════════════════════════════╗
║       Integrated Driver & Road Monitoring System                 ║
║       Layout : 70 % road feed  |  30 % driver feed  (1280×720)   ║
╚══════════════════════════════════════════════════════════════════╝

Road video + live driver camera
────────────────────────────────
    python main.py --video road.mp4

Extra options
─────────────
    --enable-yolo          Enable YOLOv8 on road feed
    --yolo-skip  N         Run YOLO every N frames (default 2)
    --output     out.mp4   Record combined output
    --headless             No GUI / server mode

"""

import cv2
import argparse
import logging
import sys
import time
import numpy as np

from driver        import DriverMonitor
from road_monitor  import RoadMonitor
from fusion_engine import FusionEngine
from alert_system  import AlertSystem
from accuracy_logger import AccuracyLogger

# ═══════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════

logging.basicConfig(
    level   = logging.INFO,
    format  = "[%(asctime)s] %(levelname)-8s  %(name)s — %(message)s",
    datefmt = "%H:%M:%S",
)
logger = logging.getLogger("MainSystem")

# ═══════════════════════════════════════════════════════════════════
# LAYOUT
# ═══════════════════════════════════════════════════════════════════
DISPLAY_W    = 1280
DISPLAY_H    = 720
ROAD_W       = int(DISPLAY_W * 0.70)   # 896 px
DRIVER_W     = DISPLAY_W - ROAD_W      # 384 px
FPS_INTERVAL = 30
WINDOW_TITLE = "Driver & Road Monitoring System"
LINE_H       = 22    # fixed HUD line height — prevents text overlap

# ═══════════════════════════════════════════════════════════════════
# COLOUR PALETTE  (BGR)
# ═══════════════════════════════════════════════════════════════════
WHITE   = (255, 255, 255)
GREEN   = (0,   220,   0)
LIME    = (0,   255, 128)
ORANGE  = (0,   160, 255)
RED     = (0,    40, 220)
CRIMSON = (0,     0, 200)
CYAN    = (255, 220,   0)
YELLOW  = (0,   220, 220)
GRAY    = (150, 150, 150)
DARK    = (20,   20,  20)
BLACK   = (0,     0,   0)
FONT    = cv2.FONT_HERSHEY_SIMPLEX

SEVERITY: dict = {
    "SAFE"    : ("  SAFE  ",    GREEN  ),
    "CAUTION" : (" CAUTION ",   ORANGE ),
    "WARNING" : (" WARNING ",   RED    ),
    "CRITICAL": (" CRITICAL ",  CRIMSON),
}

HAZARD_LABEL = {
    0: ("CLEAR",    GREEN  ),
    1: ("CAUTION",  ORANGE ),
    2: ("WARNING",  RED    ),
    3: ("CRITICAL", CRIMSON),
}


# ═══════════════════════════════════════════════════════════════════
# CAMERA HELPERS
# ═══════════════════════════════════════════════════════════════════

def list_cameras(limit: int = 6) -> list:
    found = []
    for i in range(limit):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            found.append(i)
            cap.release()
    return found


def open_source(source, tag: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        if isinstance(source, int):
            cams = list_cameras()
            logger.error(f"[{tag}] Cannot open camera {source}. Available: {cams or 'none'}.")
        else:
            logger.error(f"[{tag}] Cannot open file: '{source}'.")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    logger.info(f"[{tag}] Opened ← {source}")
    return cap


def crop_driver_region(frame: np.ndarray) -> np.ndarray:
    """Centre-crop + mirror for single-cam driver view."""
    h, w  = frame.shape[:2]
    x1, x2 = w // 3, 2 * w // 3
    return cv2.flip(frame[:, x1:x2].copy(), 1)


def placeholder(w: int, h: int, text: str) -> np.ndarray:
    img = np.full((h, w, 3), DARK, dtype=np.uint8)
    (tw, th), _ = cv2.getTextSize(text, FONT, 0.55, 1)
    cv2.putText(img, text, ((w - tw) // 2, (h + th) // 2), FONT, 0.55, GRAY, 1)
    return img

def enhance_driver_frame(frame: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


# ═══════════════════════════════════════════════════════════════════
# HUD HELPERS
# ═══════════════════════════════════════════════════════════════════

def _alpha_rect(frame, pt1, pt2, colour, alpha=0.60):
    sub  = frame[pt1[1]:pt2[1], pt1[0]:pt2[0]]
    rect = np.full_like(sub, colour)
    cv2.addWeighted(rect, alpha, sub, 1 - alpha, 0, sub)
    frame[pt1[1]:pt2[1], pt1[0]:pt2[0]] = sub


def _badge(frame, text: str, x: int, y: int, colour: tuple):
    """Solid filled badge. Returns x-end so caller can continue text after."""
    (tw, th), _ = cv2.getTextSize(text, FONT, 0.40, 1)
    pad = 5
    x2  = x + tw + pad * 2
    y1  = y - th - pad
    cv2.rectangle(frame, (x, y1), (x2, y + pad), colour, -1)
    cv2.putText(frame, text, (x + pad, y), FONT, 0.40, BLACK, 1, cv2.LINE_AA)
    return x2 + 6


def _txt(frame, text: str, x: int, y: int,
         colour=WHITE, scale=0.38, bold=False):
    cv2.putText(frame, text, (x, y), FONT, scale, colour,
                2 if bold else 1, cv2.LINE_AA)


def _clean(msg: str) -> str:
    s = msg.encode("ascii", errors="replace").decode("ascii").replace("?", "")
    for tag in ("[CRITICAL]", "[WARNING]", "[CAUTION]", "[SAFE]"):
        s = s.replace(tag, "")
    return s.strip()


def _draw_accuracy_bar(frame, label: str, value: float, x: int, y: int,
                       bar_w: int = 100, bar_h: int = 10):
    """
    Draw a compact labelled accuracy bar.

        [Road Acc  ████████░░  82%]

    Parameters
    ----------
    frame   : canvas to draw on
    label   : short string, e.g. "Road Acc"
    value   : float 0-1
    x, y    : top-left anchor of the whole widget
    bar_w   : pixel width of the filled bar
    bar_h   : pixel height of the bar
    """
    pct = max(0.0, min(1.0, value))

    # Colour gradient: green → orange → red
    if pct >= 0.75:
        bar_col = (0, 210, 0)          # green
    elif pct >= 0.50:
        bar_col = (0, 165, 255)        # orange
    else:
        bar_col = (0, 60, 220)         # red

    # Label
    lbl_w, lbl_h = cv2.getTextSize(label, FONT, 0.32, 1)[0]
    cv2.putText(frame, label, (x, y), FONT, 0.32, (180, 180, 180), 1, cv2.LINE_AA)
    bx = x + lbl_w + 4

    # Background track
    cv2.rectangle(frame, (bx, y - bar_h), (bx + bar_w, y), (50, 50, 50), -1)

    # Filled portion
    filled = int(bar_w * pct)
    if filled > 0:
        cv2.rectangle(frame, (bx, y - bar_h), (bx + filled, y), bar_col, -1)

    # Percentage text
    pct_txt = f"{int(pct * 100):3d}%"
    cv2.putText(frame, pct_txt, (bx + bar_w + 4, y), FONT, 0.32,
                bar_col, 1, cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════════════
# HUD — MAIN RENDERER
# ═══════════════════════════════════════════════════════════════════

def draw_hud(frame, driver_result, road_result,
             fusion_result, fps, calibrated, paused, single_cam,
             accuracy: dict = None):
    H, W = frame.shape[:2]
    DX   = ROAD_W

    # ── Backgrounds ─────────────────────────────────────────────────
    _alpha_rect(frame, (0,  0),      (295, 168), DARK,  0.65)
    _alpha_rect(frame, (DX, 0),      (W,   238), DARK,  0.65)
    _alpha_rect(frame, (0,  H - 98), (W,   H  ), BLACK, 0.82)
    cv2.line(frame, (DX, 0), (DX, H - 98), (60, 60, 60), 1)

    # ─────────────────────────────────────────────────────────────────
    # LEFT — Road
    # ─────────────────────────────────────────────────────────────────
    haz_idx        = min(road_result.get("hazard_level", 0), 3)
    haz_lbl, h_col = HAZARD_LABEL[haz_idx]
    n_veh          = len(road_result.get("vehicles",    []))
    n_ped          = len(road_result.get("pedestrians", []))
    lane_st        = road_result.get("lane_status", "—")
    road_st        = road_result.get("road_status", "—")

    lx = 10
    ly = 18
    _txt(frame, "ROAD MONITOR",           lx, ly,            CYAN,  0.44, bold=True);  ly += LINE_H + 3
    _txt(frame, f"Hazard:      {haz_lbl}",lx, ly,            h_col, 0.40);             ly += LINE_H
    _txt(frame, f"Vehicles:    {n_veh}",  lx, ly,            WHITE, 0.38);             ly += LINE_H
    _txt(frame, f"Pedestrians: {n_ped}",  lx, ly,            WHITE, 0.38);             ly += LINE_H
    _txt(frame, f"Lane:        {lane_st}",lx, ly,            WHITE, 0.38);             ly += LINE_H
    _txt(frame, f"Condition:   {road_st}",lx, ly,            WHITE, 0.38);             ly += LINE_H
    _txt(frame, f"FPS:         {fps:.1f}",lx, ly,            LIME,  0.38)

    # ─────────────────────────────────────────────────────────────────
    # RIGHT — Driver
    # ─────────────────────────────────────────────────────────────────

    d_score       = driver_result.get("driver_score",   0)
    ear_l         = driver_result.get("ear_left",       0.0)
    ear_r         = driver_result.get("ear_right",      0.0)
    ear_avg       = driver_result.get("ear",            0.0)
    mar           = driver_result.get("mar",            0.0)
    yaw           = driver_result.get("yaw",            0.0)
    pitch         = driver_result.get("pitch",          0.0)
    fatigue_score = driver_result.get("fatigue_score",  0.0)
    fatigue_level = driver_result.get("fatigue_level",  "ALERT")
    microsleeps   = driver_result.get("microsleeps",    0)
    gaze          = driver_result.get("gaze_direction", "CENTER")
    status        = driver_result.get("status",         "Alert")

    s_col = GREEN if d_score == 0 else ORANGE if d_score == 1 else RED
    f_col = GREEN if fatigue_level == "ALERT" else YELLOW if fatigue_level == "CAUTION" else ORANGE if fatigue_level == "WARNING" else RED

    dx = DX + 10
    dy = 18
    _txt(frame, "DRIVER MONITOR",                                    dx, dy, CYAN,  0.44, bold=True); dy += LINE_H + 4
    _txt(frame, f"Status:  {status}",                                dx, dy, s_col, 0.42, bold=True); dy += LINE_H + 3
    _txt(frame, f"EAR  L:{ear_l:.2f}  R:{ear_r:.2f}  avg:{ear_avg:.2f}",
                                                                     dx, dy, WHITE, 0.36);             dy += LINE_H
    _txt(frame, f"MAR:     {mar:.2f}",                               dx, dy, WHITE, 0.36);             dy += LINE_H
    _txt(frame, f"Yaw:     {yaw:+.1f}    Pitch: {pitch:+.1f}",      dx, dy, WHITE, 0.36);             dy += LINE_H
    _txt(frame, f"Fatigue: {fatigue_score:.1f}/10  [{fatigue_level}]",
                                                                     dx, dy, f_col, 0.36);             dy += LINE_H
    _txt(frame, f"Microsleeps: {microsleeps}",                       dx, dy, f_col, 0.36);             dy += LINE_H
    _txt(frame, f"Gaze:    {gaze}",                                  dx, dy, WHITE, 0.36);             dy += LINE_H + 2

    if not calibrated:
        _txt(frame, "Calibrating — look straight ahead", dx, dy, YELLOW, 0.33);          dy += LINE_H
    if paused:
        _txt(frame, "[ PAUSED — Press P to resume ]",    dx, dy, ORANGE, 0.36, bold=True)
    if single_cam:
        _txt(frame, "Single-cam mode", DX + 6, H - 90, GRAY, 0.30)

    # ─────────────────────────────────────────────────────────────────
    # BOTTOM BAR — Severity badge + alert message + risk scores
    # ─────────────────────────────────────────────────────────────────
    alert_msg     = fusion_result.get("alert_message",  "System nominal.")
    alert_sev     = fusion_result.get("alert_severity", "SAFE")
    driver_risk   = fusion_result.get("driver_risk",    0.0)
    road_risk     = fusion_result.get("road_risk",      0.0)
    combined_risk = fusion_result.get("combined_risk",  0.0)

    badge_txt, bar_col = SEVERITY.get(alert_sev, (" INFO ", WHITE))
    clean_msg          = _clean(alert_msg) or "System nominal."

    # Row 1 — badge + message  (fixed y so nothing overlaps)
    row1_y = H - 54
    msg_x  = _badge(frame, badge_txt, 10, row1_y, bar_col)
    _txt(frame, clean_msg, msg_x, row1_y, bar_col, 0.46)

    # Row 2 — risk scores
    risk_txt = (f"Driver risk: {driver_risk:.2f}   "
                f"Road risk: {road_risk:.2f}   "
                f"Combined: {combined_risk:.2f}")
    _txt(frame, risk_txt, 10, H - 40, GRAY, 0.34)

    # Row 3 — accuracy meters (Road / Driver / Lane)
    if accuracy:
        _draw_accuracy_bar(frame, "Road Acc",   accuracy.get("road",   0.0),  10,  H - 20)
        _draw_accuracy_bar(frame, "Driver Acc", accuracy.get("driver", 0.0),  240, H - 20)
        _draw_accuracy_bar(frame, "Lane Acc",   accuracy.get("lane",   0.0),  470, H - 20)

    # Border flash
    if alert_sev in ("WARNING", "CRITICAL"):
        cv2.rectangle(frame, (2, 2), (W - 2, H - 2), bar_col, 2)


# ═══════════════════════════════════════════════════════════════════
# ARGUMENT PARSER
# ═══════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description     = "Integrated Driver & Road Monitoring System",
        formatter_class = argparse.ArgumentDefaultsHelpFormatter,
        epilog          = "Run with no arguments to use a single webcam (camera 0).",
    )
    p.add_argument("--video",         type=str, default=None,
                   help="Road video file. Overrides --road-camera.")
    p.add_argument("--road-camera",   type=int, default=0,
                   help="Camera index for road feed.")
    p.add_argument("--driver-camera", type=int, default=None,
                   help="Camera index for driver feed. Omit = single-cam mode.")
    p.add_argument("--enable-yolo",   action="store_true",
                   help="Enable YOLOv8 on road feed.")
    p.add_argument("--yolo-skip",     type=int, default=2,
                   help="Run YOLO every N frames.")
    p.add_argument("--output",        type=str, default=None,
                   help="Save combined output to .mp4 file.")
    p.add_argument("--headless",      action="store_true",
                   help="No GUI window.")
    p.add_argument("--face-top",      type=float, default=0.0,
                   help="Top of face crop as fraction of driver panel height (0.0 = full frame).")
    p.add_argument("--face-bottom",   type=float, default=1.0,
                   help="Bottom of face crop as fraction of driver panel height (1.0 = full frame).")
    return p


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main() -> None:
    args     = build_parser().parse_args()
    road_src = args.video if args.video else args.road_camera

    # Resolve driver source and mode
    if args.driver_camera is not None:
        driver_src = args.driver_camera
        single_cam = False
        if args.video is None and args.driver_camera == args.road_camera:
            logger.warning("Road and driver cameras share the same index — single-cam mode.")
            single_cam = True
            driver_src = None
    elif args.video is not None:
        driver_src = 0
        single_cam = False
        logger.info("Road feed from video — driver feed defaulting to camera 0.")
    else:
        driver_src = None
        single_cam = True

    logger.info("=" * 62)
    logger.info("  Integrated Driver & Road Monitoring System")
    logger.info(f"  Mode   : {'Single-camera' if single_cam else 'Dual-camera'}")
    logger.info(f"  Road   : {road_src}")
    logger.info(f"  Driver : {'crop of road frame' if single_cam else driver_src}")
    logger.info("=" * 62)

    road_cap   = open_source(road_src, "Road  ")
    driver_cap = None if single_cam else open_source(driver_src, "Driver")

    src_fps = road_cap.get(cv2.CAP_PROP_FPS) or 30.0
    logger.info(f"Road source FPS : {src_fps:.1f}")

    out_writer = None
    if args.output:
        fourcc     = cv2.VideoWriter.fourcc(*"mp4v")
        out_writer = cv2.VideoWriter(args.output, fourcc, src_fps, (DISPLAY_W, DISPLAY_H))
        if not out_writer.isOpened():
            logger.error(f"Cannot write to: {args.output}")
            sys.exit(1)
        logger.info(f"Recording → {args.output}")

    driver_monitor = DriverMonitor()
    road_monitor   = RoadMonitor(enable_yolo=args.enable_yolo)
    fusion_engine  = FusionEngine()
    alert_system   = AlertSystem()
    accuracy_logger = AccuracyLogger()
    logger.info("All modules initialised.")

    if not args.headless:
        logger.info("Controls:  Q=quit  R=recalibrate  S=screenshot  P=pause")

    frame_count: int   = 0
    fps_counter: int   = 0
    fps_timer:   float = time.time()
    display_fps: float = 0.0
    paused:      bool  = False
    last_road:   dict  = {}
    last_driver: dict  = {}   # cache last driver result for skipped frames
    drv_ph             = placeholder(DRIVER_W, DISPLAY_H, "No Driver Camera")

    try:
        while True:

            # Pause loop
            if paused and not args.headless:
                key = cv2.waitKey(80) & 0xFF
                if   key in (ord('p'), ord('P')): paused = False; logger.info("Resumed.")
                elif key in (ord('q'), ord('Q')): logger.info("Quit."); break
                continue

            # Read road frame
            ret_r, road_frame = road_cap.read()
            if not ret_r or road_frame is None:
                if args.video:
                    road_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                logger.warning("Road feed ended.")
                break

            # Read / derive driver frame
            if single_cam:
                driver_frame = crop_driver_region(road_frame)
            else:
                ret_d, driver_frame = driver_cap.read()
                if not ret_d or driver_frame is None:
                    driver_frame = drv_ph.copy()

            frame_count += 1
            fps_counter += 1

            road_frame = cv2.resize(road_frame, (ROAD_W, DISPLAY_H))

            # Process road monitor on half-res frame — faster CV/YOLO, then draw on full res
            road_small  = cv2.resize(road_frame, (ROAD_W // 2, DISPLAY_H // 2))
            run_yolo    = (frame_count % max(args.yolo_skip, 1) == 0 or not last_road)
            if run_yolo:
                road_result_small = road_monitor.process(road_small)
                # Scale bounding boxes back up to full display resolution
                scale_x = ROAD_W      / (ROAD_W // 2)
                scale_y = DISPLAY_H   / (DISPLAY_H // 2)
                for det_list in ("vehicles", "pedestrians", "all_detections"):
                    for obj in road_result_small.get(det_list, []):
                        x1, y1, x2, y2 = obj["bbox"]
                        obj["bbox"] = (int(x1*scale_x), int(y1*scale_y),
                                       int(x2*scale_x), int(y2*scale_y))
                road_result_small["frame"] = road_frame   # swap in full-res frame for HUD
                last_road = road_result_small
            road_result = last_road
            # Resizing to the narrow display panel (384×720) destroys face detection.
            # Crop the face region at full native res, pad to square, resize to 480×480.
            raw_dh, raw_dw = driver_frame.shape[:2]
            y1_raw = int(raw_dh * args.face_top)
            y2_raw = int(raw_dh * args.face_bottom)
            face_crop = driver_frame[y1_raw:y2_raw, :]

            # Letterbox to square (no aspect-ratio distortion)
            fh, fw   = face_crop.shape[:2]
            side     = max(fh, fw)
            pt       = (side - fh) // 2
            pl       = (side - fw) // 2
            face_sq  = cv2.copyMakeBorder(
                face_crop, pt, side - fh - pt, pl, side - fw - pl,
                cv2.BORDER_CONSTANT, value=(0, 0, 0),
            )
            face_sq = cv2.resize(face_sq, (320, 320))  # 320×320 is sufficient for MediaPipe, ~2× faster than 480

            # Process MediaPipe every other frame — halves CPU cost, imperceptible at >10fps
            if frame_count % 2 == 0 or not last_driver:
                driver_result = driver_monitor.process(enhance_driver_frame(face_sq))
                last_driver   = driver_result
            else:
                driver_result = last_driver

            # Update accuracy logger AFTER both results are ready
            accuracy_logger.update(road_result, driver_result)

            # Resize driver frame for display ONLY after processing
            driver_frame = cv2.resize(driver_frame, (DRIVER_W, DISPLAY_H))

            fusion_result = fusion_engine.process(driver_result, road_result)

            # Compose frame
            canvas = np.hstack([road_frame, driver_frame])
            draw_hud(canvas, driver_result, road_result, fusion_result,
                     display_fps, driver_monitor.calibrated, paused, single_cam,
                     accuracy=accuracy_logger.get_accuracy())
            alert_system.update(canvas, fusion_result)

            # FPS
            if fps_counter >= FPS_INTERVAL:
                elapsed     = time.time() - fps_timer
                display_fps = fps_counter / elapsed if elapsed > 0 else 0.0
                fps_counter = 0
                fps_timer   = time.time()

            if out_writer:
                out_writer.write(canvas)

            if not args.headless:
                cv2.imshow(WINDOW_TITLE, canvas)
                key = cv2.waitKey(1) & 0xFF
                if   key in (ord('q'), ord('Q')): logger.info("Quit."); break
                elif key in (ord('r'), ord('R')): driver_monitor.recalibrate(); logger.info("Recalibrated.")
                elif key in (ord('s'), ord('S')):
                    path = f"screenshot_{time.strftime('%Y%m%d_%H%M%S')}.png"
                    cv2.imwrite(path, canvas); logger.info(f"Screenshot → {path}")
                elif key in (ord('p'), ord('P')): paused = True; logger.info("Paused.")
                else: accuracy_logger.handle_key(key)  # pass to accuracy logger for GT input

            sev = fusion_result.get("alert_severity", "SAFE")
            if fusion_result.get("should_alert") and sev != "SAFE":
                logger.warning(f"ALERT [{sev}]  {fusion_result.get('alert_message', '')}")

    except KeyboardInterrupt:
        logger.info("Stopped by Ctrl+C.")

    finally:
        logger.info("Releasing resources…")
        road_cap.release()
        if driver_cap:
            driver_cap.release()
        if out_writer:
            out_writer.release()
        alert_system.stop()
        accuracy_logger.report()
        if not args.headless:
            cv2.destroyAllWindows()

        recent = fusion_engine.get_statistics().get("alert_history", [])[-5:]
        logger.info("-" * 50)
        logger.info(f"Frames processed : {frame_count}")
        logger.info(f"Total alerts     : {fusion_engine.total_alerts}")
        if recent:
            logger.info("Last 5 alerts:")
            for a in recent:
                logger.info(f"  [{a['severity']:<8}]  {a['message']}")
        logger.info("Shutdown complete.")


if __name__ == "__main__":
    main()