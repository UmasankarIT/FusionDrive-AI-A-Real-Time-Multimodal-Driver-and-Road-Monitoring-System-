# 🚗 Integrated Driver & Road Monitoring System

A real-time dual-channel safety monitoring system that uses computer vision, deep learning, and temporal reasoning to detect driver fatigue and road hazards simultaneously.

---

## Demo

> Single-camera or dual-camera mode. 70% road feed | 30% driver feed.

```
┌─────────────────────────────┬──────────────┐
│                             │  DRIVER CAM  │
│        ROAD FEED            │  EAR / MAR   │
│   [lane overlay + boxes]    │  Head Pose   │
│                             │  Fatigue     │
├─────────────────────────────┴──────────────┤
│  ⚠ WARNING  Vehicle detected 8m ahead     │
│  Driver risk: 0.42   Road risk: 0.61       │
└────────────────────────────────────────────┘
```

---

## Features

- **Driver Monitoring** — Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR), head pose (yaw/pitch), gaze tracking, fatigue score (0–10), microsleep detection
- **Road Monitoring** — Lane detection (Canny + HoughLinesP), vehicle/pedestrian detection (YOLOv11 or contour-based CV fallback), adaptive day/dusk/night modes
- **Temporal RNN** — Pure-NumPy dual LSTM (HazardLSTM + CollisionLSTM), no PyTorch/TensorFlow required, online self-training during session
- **Fusion Engine** — 12 alert types, 4 severity levels (SAFE / CAUTION / WARNING / CRITICAL), per-condition grace timers and cooldowns
- **Alert System** — Threaded audio beeps (escalating pattern), TTS speech on new alerts, visual border flash + red overlay on CRITICAL
- **Accuracy Logger** — Live EMA-smoothed Road / Driver / Lane accuracy bars with session report on exit

---

## Architecture

```
main.py
├── driver.py          ← MediaPipe FaceMesh — EAR, MAR, head pose, gaze, fatigue
├── road_monitor.py    ← Lane detection + vehicle detection (YOLO or CV)
│   └── road_rnn.py   ← HazardLSTM + CollisionLSTM (pure NumPy)
├── fusion_engine.py   ← Alert arbitration — 12 alert types, 4 severity levels
├── alert_system.py   ← Audio beeps + TTS + visual overlays
└── accuracy_logger.py ← Live accuracy tracking + session report
```

---

## Requirements

```
Python >= 3.9
opencv-python
mediapipe
numpy
pyttsx3
```

Optional (for YOLO mode):
```
ultralytics   (YOLOv11)
```

Install everything:
```bash
pip install -r requirements.txt
```

---

## Quick Start

**Single webcam (road + driver from same camera):**
```bash
python main.py
```

**Road video file + webcam for driver:**
```bash
python main.py --video road.mp4
```

**Dual camera setup:**
```bash
python main.py --road-camera 0 --driver-camera 1
```

**Enable YOLOv11 object detection:**
```bash
python main.py --video road.mp4 --enable-yolo
```

**Save output to file:**
```bash
python main.py --video road.mp4 --output result.mp4
```

**Headless / server mode (no GUI):**
```bash
python main.py --headless
```

---

## Keyboard Controls

| Key | Action |
|-----|--------|
| `Q` | Quit |
| `R` | Recalibrate driver monitor |
| `S` | Screenshot (saved to current directory) |
| `P` | Pause / Resume |
| `A` | Toggle accuracy overlay |

---

## Module Details

### `driver.py` — DriverMonitor
Uses MediaPipe FaceMesh to track 6 signals per frame:
- **EAR** (Eye Aspect Ratio) — eye closure detection
- **MAR** (Mouth Aspect Ratio) — yawn detection
- **Head pose** — yaw / pitch via `solvePnP`
- **Gaze ratio** — iris position tracking
- **Fatigue Score (0–10)** — event-based, works at 10–21 fps (unlike PERCLOS which needs 60+)
- **Microsleep detection** — discrete dangerous eye closure events

### `road_monitor.py` — RoadMonitor
- **Lane detection** — Canny edges → HoughLinesP → left/right line averaging → filled polygon overlay
- **Object detection** — YOLOv11n-seg (if enabled) or contour-based CV fallback
- **Distance estimation** — bbox size + vertical position, normalised 0→1
- **Adaptive thresholds** — auto day/dusk/night mode based on frame brightness

### `road_rnn.py` — RoadRNN
- Two pure-NumPy LSTMs, no ML framework required
- **HazardLSTM** — 7-dim input → hazard class 0–3
- **CollisionLSTM** — 4-dim input → collision probability
- Seeded with rule-derived weights → sensible output from frame 1
- Online teacher-forcing → improves during the session

### `fusion_engine.py` — FusionEngine
- 12 alert types: `EYES_CLOSING`, `DROWSY`, `YAWNING`, `HEAD_TURN`, `GAZE_AWAY`, `HEAD_AND_GAZE`, `PEDESTRIAN`, `VEHICLE_CLOSE`, `LANE_DRIFT`, `COMBINED_RISK`, `MICROSLEEP_RISK`
- Per-condition grace timers (e.g. 1.2s before HEAD_TURN fires)
- Per-severity cooldowns prevent alert flooding
- Outputs combined driver + road risk score (0–1)

### `alert_system.py` — AlertSystem
- Background daemon thread for audio (non-blocking)
- Escalating beep patterns: 4s / 2.5s / 1s intervals for CAUTION / WARNING / CRITICAL
- TTS speaks only on new alert events (no repetition)
- Visual: coloured border flash + red screen overlay on CRITICAL

### `accuracy_logger.py` — AccuracyLogger
- **Road accuracy** — RNN confidence + hazard consistency + collision stability
- **Driver accuracy** — EAR/MAR validity + head pose confidence + fatigue smoothness
- **Lane accuracy** — Hough confidence + both-lanes detection + temporal consistency
- EMA smoothing (α = 0.15) for stable display values
- Full session report on shutdown

---

## Notes

- `alert_system.py` uses `winsound` and SAPI5 TTS — **Windows only** by default. Linux/macOS users can swap in `aplay`/`espeak` or `afplay`/`say`.
- YOLOv11 is optional. The system runs fully without it using contour-based CV detection.
- The RNN trains itself online — no pre-training or dataset needed.

---