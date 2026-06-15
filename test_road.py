import cv2
import argparse
import time
from road_monitor import RoadMonitor

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",       type=str, default=None)
    parser.add_argument("--camera",      type=int, default=0)
    parser.add_argument("--enable-yolo", action="store_true")
    args = parser.parse_args()

    source = args.video if args.video else args.camera
    cap    = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {source}")
        return

    monitor     = RoadMonitor(enable_yolo=args.enable_yolo)
    fps_timer   = time.time()
    fps_counter = 0
    display_fps = 0.0

    FONT   = cv2.FONT_HERSHEY_SIMPLEX
    GREEN  = (0, 255, 0)
    ORANGE = (0, 165, 255)
    RED    = (0, 0, 255)
    WHITE  = (255, 255, 255)
    CYAN   = (0, 255, 255)
    DARK   = (30, 30, 30)

    def h_color(lvl):
        return RED if lvl >= 3 else ORANGE if lvl == 2 else (0,255,255) if lvl == 1 else GREEN

    print("Controls: Q = quit | S = screenshot | SPACE = pause")

    paused = False

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("End of video — looping back.")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            frame = cv2.resize(frame, (1280, 720))
            result = monitor.process(frame)

            fps_counter += 1
            if fps_counter >= 30:
                display_fps = fps_counter / (time.time() - fps_timer)
                fps_timer   = time.time()
                fps_counter = 0

            # ── Draw HUD ──────────────────────────────────
            h, w = frame.shape[:2]

            # Background panel
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (280, 130), DARK, -1)
            cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

            hazard  = result["hazard_level"]
            h_text  = ["CLEAR", "CAUTION", "WARNING", "CRITICAL"][hazard]
            n_v     = len(result["vehicles"])
            n_p     = len(result["pedestrians"])
            lane    = result["lane_status"]
            road    = result["road_status"]

            cv2.putText(frame, "ROAD MONITOR TEST",     (8,  18), FONT, 0.50, CYAN,          1)
            cv2.putText(frame, f"Hazard: {h_text}",     (8,  38), FONT, 0.50, h_color(hazard), 2)
            cv2.putText(frame, f"Vehicles:    {n_v}",   (8,  58), FONT, 0.45, WHITE,          1)
            cv2.putText(frame, f"Pedestrians: {n_p}",   (8,  76), FONT, 0.45, WHITE,          1)
            cv2.putText(frame, f"Lane:  {lane}",        (8,  94), FONT, 0.45, WHITE,          1)
            cv2.putText(frame, f"Road:  {road}",        (8, 112), FONT, 0.45, WHITE,          1)
            cv2.putText(frame, f"FPS: {display_fps:.1f}", (w-110, 18), FONT, 0.45, GREEN,     1)

            # YOLO status badge
            yolo_label = "YOLO: ON" if monitor.use_yolo else "YOLO: OFF (CV only)"
            cv2.putText(frame, yolo_label, (8, h - 15), FONT, 0.40, CYAN, 1)

        cv2.imshow("Road Monitor Test", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q')):
            break
        elif key in (ord('s'), ord('S')):
            path = f"road_test_{int(time.time())}.png"
            cv2.imwrite(path, frame)
            print(f"Screenshot saved: {path}")
        elif key == ord(' '):
            paused = not paused
            print("PAUSED" if paused else "RESUMED")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()