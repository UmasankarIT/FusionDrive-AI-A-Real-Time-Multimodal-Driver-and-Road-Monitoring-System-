import os
import sys
import time
import logging
import threading
import numpy as np
import cv2

logger = logging.getLogger("AlertSystem")

# ─────────────────────────────────────────────
# BEEP INTERVALS  — how often to beep while severity is active
# ─────────────────────────────────────────────
BEEP_INTERVAL = {
    "CAUTION" : 4.0,
    "WARNING" : 2.5,
    "CRITICAL": 1.0,
}

# ─────────────────────────────────────────────
# BEEP PATTERNS  (freq_hz, duration_ms, pause_ms)
# Used by both winsound and pygame backends.
# ─────────────────────────────────────────────
BEEP_PATTERNS = {
    "CAUTION" : [(700,  150,    0)],
    "WARNING" : [(900,  180,  100),  (900,  180,   0)],
    "CRITICAL": [(1200, 120,   80),  (1200, 120,  80),
                 (1200, 120,   80),  (1200, 120,   0)],
}
BEEP_VOLUME = {"CAUTION": 0.45, "WARNING": 0.70, "CRITICAL": 1.00}

# ─────────────────────────────────────────────
# TTS MESSAGES
# ─────────────────────────────────────────────
TTS_MESSAGES = {
    ("EYES_CLOSING",    "CAUTION") : "Eyes getting heavy.",
    ("EYES_CLOSING",    "WARNING") : "Eyes closing. Focus on the road.",
    ("EYES_CLOSING",    "CRITICAL"): "Eyes closing. Pull over now.",

    ("DROWSY",          "CAUTION") : "Drowsiness detected. Take a break soon.",
    ("DROWSY",          "WARNING") : "Driver drowsy. Stay focused.",
    ("DROWSY",          "CRITICAL"): "Drowsy driver. Pull over immediately.",

    ("YAWNING",         "CAUTION") : "Yawning detected. Fatigue building.",
    ("YAWNING",         "WARNING") : "Repeated yawning. Take a rest stop.",
    ("YAWNING",         "CRITICAL"): "Severe fatigue. Stop driving soon.",

    ("HEAD_TURN",       "CAUTION") : "Head turned. Eyes on the road.",
    ("HEAD_TURN",       "WARNING") : "Head off road. Look ahead now.",
    ("HEAD_TURN",       "CRITICAL"): "Head off road. Critical. Watch the road.",

    ("GAZE_AWAY",       "CAUTION") : "Eyes off road. Look ahead.",
    ("GAZE_AWAY",       "WARNING") : "Eyes off road. Look ahead now.",
    ("GAZE_AWAY",       "CRITICAL"): "Eyes off road. Danger. Focus ahead.",

    ("HEAD_AND_GAZE",   "CAUTION") : "Head and eyes off road. Focus ahead.",
    ("HEAD_AND_GAZE",   "WARNING") : "Head and eyes off road. High risk.",
    ("HEAD_AND_GAZE",   "CRITICAL"): "Severe distraction. Look ahead now.",

    ("PEDESTRIAN",      "CAUTION") : "Pedestrian nearby. Reduce speed.",
    ("PEDESTRIAN",      "WARNING") : "Pedestrian ahead. Slow down.",
    ("PEDESTRIAN",      "CRITICAL"): "Pedestrian in path. Brake now.",

    ("VEHICLE_CLOSE",   "CAUTION") : "Vehicle ahead. Keep distance.",
    ("VEHICLE_CLOSE",   "WARNING") : "Too close to vehicle. Increase distance.",
    ("VEHICLE_CLOSE",   "CRITICAL"): "Collision risk. Brake immediately.",

    ("LANE_DRIFT",      "CAUTION") : "Lane unclear. Stay centred.",
    ("LANE_DRIFT",      "WARNING") : "Lane not detected. Check position.",
    ("LANE_DRIFT",      "CRITICAL"): "Lane lost. Steer carefully.",

    ("COMBINED_RISK",   "CAUTION") : "Fatigue and road hazard. Stay sharp.",
    ("COMBINED_RISK",   "WARNING") : "Drowsy driver and road hazard. High risk.",
    ("COMBINED_RISK",   "CRITICAL"): "Critical. Impaired driver and road danger. Pull over.",

    ("MICROSLEEP_RISK", "CAUTION") : "Fatigue building. Consider a break.",
    ("MICROSLEEP_RISK", "WARNING") : "Microsleep risk. Pull over soon.",
    ("MICROSLEEP_RISK", "CRITICAL"): "Microsleep imminent. Stop the vehicle.",
}

def _tts_text(alert_type: str, severity: str) -> str:
    return TTS_MESSAGES.get((alert_type, severity), f"Alert. {severity.lower()}.")


# ─────────────────────────────────────────────
# BACKEND INIT
# ─────────────────────────────────────────────

def _init_beep_backend():
    """
    Returns ('pygame', mixer) or ('winsound', None) or ('silent', None).
    On Windows, winsound is always available as a guaranteed fallback.
    """
    # ── Try pygame first (better tone quality) ────────────────────
    try:
        import pygame
        pygame.mixer.pre_init(frequency=44100, size=-16, channels=2, buffer=512)
        pygame.mixer.init()
        logger.info(f"Beep: pygame mixer ready {pygame.mixer.get_init()}.")
        return "pygame", pygame.mixer
    except ImportError:
        logger.info("Beep: pygame not installed — using winsound. "
                    "(Optional upgrade: pip install pygame)")
    except Exception as e:
        logger.warning(f"Beep: pygame failed ({e}) — falling back to winsound.")

    # ── winsound — always present on Windows ─────────────────────
    try:
        import winsound
        # Quick test: Beep(frequency, duration_ms)
        winsound.Beep(440, 1)   # silent 1ms test — just checks the API works
        logger.info("Beep: winsound ready (Windows built-in).")
        return "winsound", None
    except ImportError:
        logger.error(
            "Beep: winsound not available — are you on Windows?\n"
            "  If yes, this is unexpected. Try: pip install pygame instead."
        )
    except Exception as e:
        logger.error(f"Beep: winsound failed ({e}).")

    logger.warning("Beep: no audio backend — visual alerts only.")
    return "silent", None


def _init_tts():
    """
    Returns pyttsx3 engine or None.
    On Windows, pyttsx3 uses the built-in SAPI5 engine — no espeak needed.
    """
    try:
        import pyttsx3
        engine = pyttsx3.init()          # uses SAPI5 on Windows automatically
        engine.setProperty("rate",   160)
        engine.setProperty("volume", 1.0)
        # Prefer a female voice (more attention-grabbing while driving)
        voices = engine.getProperty("voices")
        for v in voices:
            name = v.name.lower()
            if "zira" in name or "female" in name or "hazel" in name:
                engine.setProperty("voice", v.id)
                break
        engine.stop()
        logger.info("TTS: pyttsx3/SAPI5 ready.")
        return engine
    except ImportError:
        logger.warning(
            "TTS: pyttsx3 not installed.\n"
            "  Fix: pip install pyttsx3"
        )
    except Exception as e:
        logger.warning(f"TTS: unavailable ({e})")
    return None


def _make_pygame_beep(mixer, freq: int, duration_ms: int, volume: float):
    """Synthesise a pure sine-tone pygame Sound object."""
    import pygame
    sr   = 44100
    n    = int(sr * duration_ms / 1000)
    t    = np.linspace(0, duration_ms / 1000, n, endpoint=False)
    mono = (np.sin(2 * np.pi * freq * t) * 32767 * volume).astype(np.int16)
    wave = np.column_stack((mono, mono))
    return pygame.sndarray.make_sound(wave)


def _sev_rank(sev: str) -> int:
    return {"SAFE": 0, "CAUTION": 1, "WARNING": 2, "CRITICAL": 3}.get(sev, 0)


# ─────────────────────────────────────────────
# VISUAL CONFIG
# ─────────────────────────────────────────────
_YELLOW = (0,   220, 220)
_ORANGE = (0,   140, 255)
_RED    = (0,     0, 220)

BORDER_COLOR     = {"CAUTION": _YELLOW, "WARNING": _ORANGE, "CRITICAL": _RED}
BORDER_THICKNESS = {"CAUTION": 3,       "WARNING": 5,       "CRITICAL": 7}
FLASH_DURATION   = {"CAUTION": 0.6,     "WARNING": 0.9,     "CRITICAL": 1.3}
CRITICAL_ALPHA   = 0.18


# ─────────────────────────────────────────────
# MAIN CLASS
# ─────────────────────────────────────────────

class AlertSystem:
    """
    Drop-in replacement for the original AlertSystem.
    Call update(canvas, fusion_result) every frame after draw_hud().

    Audio architecture
    ──────────────────
    _beep_loop() runs in a daemon thread and fires beeps on a timer
    keyed to the CURRENT severity — independent of fusion_engine cooldowns.

      CAUTION  → beep every 4.0 s   (1 soft beep)
      WARNING  → beep every 2.5 s   (2 medium beeps)
      CRITICAL → beep every 1.0 s   (4 rapid high beeps)
      SAFE     → silent

    TTS speaks only when fusion_engine fires a new alert (should_alert=True),
    so voice messages are not spammed every second.
    """

    def __init__(self):
        self._backend, self._mixer = _init_beep_backend()
        self._tts_engine           = _init_tts()
        self._sounds               = {}

        if self._backend == "pygame":
            self._preload_sounds()

        # Severity shared with beep thread (protected by lock)
        self._current_severity: str = "SAFE"
        self._severity_lock         = threading.Lock()

        # Visual flash state
        self._flash_until: float = 0.0
        self._flash_sev:   str   = "SAFE"

        # Single-fire guards
        self._beep_lock = threading.Lock()
        self._beep_busy = False
        self._tts_lock  = threading.Lock()
        self._tts_busy  = False

        # Beep loop stop signal
        self._stop_evt  = threading.Event()

        # Start continuous beep thread
        self._beep_thread = threading.Thread(
            target=self._beep_loop, daemon=True, name="BeepLoop"
        )
        self._beep_thread.start()

        self._log_status()

    # ──────────────────────────────────────────
    # STATUS
    # ──────────────────────────────────────────

    def _log_status(self):
        beep_ok = self._backend != "silent"
        tts_ok  = self._tts_engine is not None
        logger.info(
            f"AlertSystem ready — "
            f"Beeps: {'ON (' + self._backend + ')' if beep_ok else 'OFF'} | "
            f"TTS: {'ON (pyttsx3/SAPI5)' if tts_ok else 'OFF (pip install pyttsx3)'}"
        )
        if not beep_ok:
            logger.error(
                "NO BEEP BACKEND on Windows — this is unexpected.\n"
                "  Try: pip install pygame"
            )

    # ──────────────────────────────────────────
    # SOUND PRELOAD  (pygame only)
    # ──────────────────────────────────────────

    def _preload_sounds(self):
        """Pre-synthesise all beep tones so playback has zero latency."""
        for sev, pattern in BEEP_PATTERNS.items():
            vol    = BEEP_VOLUME[sev]
            sounds = []
            for freq, dur_ms, _ in pattern:
                try:
                    s = _make_pygame_beep(self._mixer, freq, dur_ms, vol)
                    sounds.append((s, dur_ms))
                except Exception as e:
                    logger.warning(f"Sound preload failed [{sev}]: {e}")
            self._sounds[sev] = sounds
        logger.info("Beep sounds preloaded (pygame).")

    # ──────────────────────────────────────────
    # CONTINUOUS BEEP LOOP
    # ──────────────────────────────────────────

    def _beep_loop(self):
        """
        Background daemon thread.
        Fires the correct beep pattern at the interval for the current severity.
        Escalates immediately (no wait) when severity rank increases.
        """
        last_beep_time: float = 0.0
        last_rank:      int   = 0

        while not self._stop_evt.is_set():
            with self._severity_lock:
                sev = self._current_severity

            now      = time.monotonic()
            rank     = _sev_rank(sev)
            interval = BEEP_INTERVAL.get(sev, 9999.0)

            # Escalation → fire immediately, don't wait for the next interval
            if rank > last_rank:
                last_beep_time = 0.0
            last_rank = rank

            if sev != "SAFE" and (now - last_beep_time) >= interval:
                self._fire_beep(sev)
                last_beep_time = now

            time.sleep(0.05)   # 50 ms poll — reacts quickly to severity changes

    def _fire_beep(self, severity: str):
        """Dispatch the beep pattern in a short-lived thread."""
        if self._backend == "silent":
            return
        with self._beep_lock:
            if self._beep_busy:
                return
            self._beep_busy = True

        def _run():
            try:
                if self._backend == "pygame":
                    sounds  = self._sounds.get(severity, [])
                    pattern = BEEP_PATTERNS.get(severity, [])
                    for (sound, dur_ms), (_, _, pause_ms) in zip(sounds, pattern):
                        sound.play()
                        time.sleep((dur_ms + pause_ms) / 1000.0)

                elif self._backend == "winsound":
                    import winsound
                    for freq, dur_ms, pause_ms in BEEP_PATTERNS.get(severity, []):
                        # winsound.Beep blocks for dur_ms then returns —
                        # this is fine because we're already in a thread.
                        winsound.Beep(freq, dur_ms)
                        if pause_ms:
                            time.sleep(pause_ms / 1000.0)

            except Exception as e:
                logger.warning(f"Beep playback error: {e}")
            finally:
                with self._beep_lock:
                    self._beep_busy = False

        threading.Thread(target=_run, daemon=True).start()

    # ──────────────────────────────────────────
    # TTS
    # ──────────────────────────────────────────

    def _speak(self, text: str):
        """
        Speak text in a background thread.
        Skips if TTS is already speaking (prevents queue buildup).
        Creates a fresh pyttsx3 engine per call — avoids SAPI5 COM
        threading issues on Windows when called from multiple threads.
        """
        if self._tts_engine is None:
            return
        with self._tts_lock:
            if self._tts_busy:
                return
            self._tts_busy = True

        def _run():
            try:
                import pyttsx3
                eng = pyttsx3.init()
                eng.setProperty("rate",   160)
                eng.setProperty("volume", 1.0)
                # Keep the same voice preference
                voices = eng.getProperty("voices")
                for v in voices:
                    name = v.name.lower()
                    if "zira" in name or "female" in name or "hazel" in name:
                        eng.setProperty("voice", v.id)
                        break
                eng.say(text)
                eng.runAndWait()
                eng.stop()
            except Exception as e:
                logger.warning(f"TTS error: {e}")
            finally:
                with self._tts_lock:
                    self._tts_busy = False

        threading.Thread(target=_run, daemon=True).start()

    # ──────────────────────────────────────────
    # VISUAL OVERLAY
    # ──────────────────────────────────────────

    def _draw_visual(self, frame: np.ndarray, severity: str):
        H, W  = frame.shape[:2]
        col   = BORDER_COLOR.get(severity, _YELLOW)
        thick = BORDER_THICKNESS.get(severity, 3)
        cv2.rectangle(frame, (2, 2), (W - 2, H - 2), col, thick)
        if severity == "CRITICAL":
            overlay    = np.zeros_like(frame, dtype=np.uint8)
            overlay[:] = _RED
            cv2.addWeighted(overlay, CRITICAL_ALPHA, frame, 1 - CRITICAL_ALPHA, 0, frame)

    # ──────────────────────────────────────────
    # PUBLIC: update()
    # ──────────────────────────────────────────

    def update(self, frame: np.ndarray, fusion_result: dict) -> None:
        """
        Call once per frame after draw_hud().

        Every call:
          - Updates _current_severity → beep loop adjusts automatically
          - Draws persistent border while any hazard is active
          - Draws brighter flash border immediately after a new alert fires
          - Fires TTS only on new alert events (should_alert=True)
        """
        severity     = fusion_result.get("alert_severity", "SAFE")
        alert_type   = fusion_result.get("alert_type",     "SAFE")
        should_alert = fusion_result.get("should_alert",   False)
        now          = time.monotonic()

        # Always push severity to beep thread
        with self._severity_lock:
            self._current_severity = severity if severity != "SAFE" else "SAFE"

        # TTS + flash on new alert events only
        if should_alert and severity != "SAFE":
            self._flash_until = now + FLASH_DURATION.get(severity, 0.6)
            self._flash_sev   = severity
            self._speak(_tts_text(alert_type, severity))

        # Clear flash state when safe
        if severity == "SAFE":
            self._flash_until = 0.0
            self._flash_sev   = "SAFE"

        # Draw visual:
        #   - Bright flash during flash window (new alert just fired)
        #   - Persistent dim border while any hazard remains
        if now < self._flash_until and self._flash_sev != "SAFE":
            self._draw_visual(frame, self._flash_sev)
        elif severity != "SAFE":
            self._draw_visual(frame, severity)

    # ──────────────────────────────────────────
    # SHUTDOWN
    # ──────────────────────────────────────────

    def stop(self):
        """Clean shutdown — call on program exit."""
        self._stop_evt.set()
        if self._backend == "pygame":
            try:
                import pygame
                pygame.mixer.quit()
            except Exception:
                pass
        if self._tts_engine is not None:
            try:
                self._tts_engine.stop()
            except Exception:
                pass
        logger.info("AlertSystem stopped.")


# ─────────────────────────────────────────────
# QUICK SELF-TEST
# Run:  python alert_system.py --test
# ─────────────────────────────────────────────

def _self_test():
    logging.basicConfig(level=logging.INFO,
                        format="[%(levelname)s] %(name)s — %(message)s")
    print("\n=== AlertSystem self-test (Windows) ===\n")

    backend, mixer = _init_beep_backend()
    tts            = _init_tts()

    print(f"Beep backend : {backend}")
    print(f"TTS          : {'pyttsx3/SAPI5' if tts else 'NOT available — pip install pyttsx3'}")
    print()

    if backend == "silent":
        print("ERROR: No beep backend found on Windows.")
        print("Fix:   pip install pygame")
        return

    # Preload pygame sounds if needed
    sounds = {}
    if backend == "pygame":
        for sev, pattern in BEEP_PATTERNS.items():
            vol = BEEP_VOLUME[sev]
            sounds[sev] = []
            for freq, dur_ms, _ in pattern:
                sounds[sev].append((_make_pygame_beep(mixer, freq, dur_ms, vol), dur_ms))

    def play(sev):
        print(f"  Playing {sev} beep...")
        if backend == "pygame":
            pattern = BEEP_PATTERNS[sev]
            for (sound, dur_ms), (_, _, pause_ms) in zip(sounds[sev], pattern):
                sound.play()
                time.sleep((dur_ms + pause_ms) / 1000.0)
        elif backend == "winsound":
            import winsound
            for freq, dur_ms, pause_ms in BEEP_PATTERNS[sev]:
                winsound.Beep(freq, dur_ms)
                if pause_ms:
                    time.sleep(pause_ms / 1000.0)
        time.sleep(0.3)

    play("CAUTION")
    play("WARNING")
    play("CRITICAL")

    if tts:
        print("  Speaking TTS test message...")
        import pyttsx3
        eng = pyttsx3.init()
        eng.setProperty("rate", 160)
        eng.say("Alert system test. Audio is working correctly.")
        eng.runAndWait()
        eng.stop()

    if backend == "pygame":
        import pygame
        pygame.mixer.quit()

    print("\n=== Test complete ===")
    print("If you heard 3 beep patterns (1, 2, 4 beeps) and a voice, everything is working.")
    print("If not, run:  pip install pygame pyttsx3\n")


if __name__ == "__main__":
    if "--test" in sys.argv:
        _self_test()
    else:
        print("Run with --test to verify your audio setup:")
        print("  python alert_system.py --test")