"""
road_rnn.py — Temporal RNN Layer for Road Monitor
==================================================
Two lightweight LSTMs that run on top of road_monitor.py detections:

  HazardLSTM      — classifies hazard level (0-3) from a sequence of
                    per-frame road features. Replaces the simple threshold
                    logic with temporal context ("is danger building?").

  CollisionLSTM   — tracks the closest vehicle's distance over time,
                    estimates approach rate, and outputs a collision
                    probability for the next ~1 second.

No dataset required
-------------------
Both models are seeded with rule-derived weights so they produce sensible
outputs from frame 1.  They also do lightweight online updates (gradient
descent on the rule-based "teacher" signal) so accuracy improves over the
session without any pre-training.

Integration
-----------
    from road_rnn import RoadRNN

    rnn = RoadRNN()
    rnn_out = rnn.update(road_result)   # call after road_monitor.process()

    # rnn_out keys:
    #   "hazard_level"        int   0-3  (RNN-refined, use instead of raw)
    #   "collision_prob"      float 0-1  (probability of collision in ~1 s)
    #   "approach_rate"       float      (positive = getting closer, px/frame)
    #   "rnn_confidence"      float 0-1  (how certain the LSTM is)
    #   "override"            bool       (True = RNN disagrees with CV/YOLO)
    #   "collision_warning"   bool       (True = imminent, alert fusion engine)
"""

import numpy as np
import logging
from collections import deque
from typing import Dict, Tuple

logger = logging.getLogger("RoadRNN")

# ─────────────────────────────────────────────────────────
# HYPER-PARAMETERS
# ─────────────────────────────────────────────────────────
SEQ_LEN          = 20      # frames of history each LSTM sees (~1 s at 20 fps)
HAZARD_INPUT_DIM = 7       # features fed to HazardLSTM per frame
COLLISION_INPUT_DIM = 4    # features fed to CollisionLSTM per frame
HIDDEN_DIM       = 32      # LSTM hidden units (small = fast, no GPU needed)
LEARNING_RATE    = 0.004   # online learning rate
ONLINE_UPDATE_EVERY = 5    # update weights every N frames (reduces CPU cost)

COLLISION_THRESH = 0.55    # probability above which collision_warning fires
APPROACH_ALPHA   = 0.25    # EMA smoothing for approach rate


# ─────────────────────────────────────────────────────────
# TINY NUMPY LSTM  (no torch/tf dependency)
# ─────────────────────────────────────────────────────────

class NumpyLSTM:
    """
    Single-layer LSTM implemented in pure NumPy.
    Supports forward pass + BPTT for one step of online learning.
    """

    def __init__(self, input_dim: int, hidden_dim: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        d, h = input_dim, hidden_dim

        # Xavier initialisation — keeps gradients stable
        scale = np.sqrt(2.0 / (d + h))

        # Gates: input(i), forget(f), cell(g), output(o)
        # Weight matrices: W_x [4h×d], W_h [4h×h], bias [4h]
        self.Wx = rng.normal(0, scale, (4 * h, d)).astype(np.float32)
        self.Wh = rng.normal(0, scale, (4 * h, h)).astype(np.float32)
        self.b  = np.zeros(4 * h, dtype=np.float32)

        # Forget gate bias initialised to +1 — helps remember long sequences
        self.b[h:2*h] = 1.0

        self.h_dim = h
        self.reset()

    def reset(self):
        self.h = np.zeros(self.h_dim, dtype=np.float32)   # hidden state
        self.c = np.zeros(self.h_dim, dtype=np.float32)   # cell state

        # Cache for BPTT
        self._cache = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """One step forward. x: [input_dim]. Returns hidden state [hidden_dim]."""
        h, c = self.h, self.c
        gates = self.Wx @ x + self.Wh @ h + self.b   # [4h]
        hd    = self.h_dim

        i = _sigmoid(gates[0*hd : 1*hd])   # input gate
        f = _sigmoid(gates[1*hd : 2*hd])   # forget gate
        g = np.tanh  (gates[2*hd : 3*hd])  # cell gate
        o = _sigmoid (gates[3*hd : 4*hd])  # output gate

        c_new = f * c + i * g
        h_new = o * np.tanh(c_new)

        self._cache = (x, h, c, i, f, g, o, c_new, h_new)
        self.h = h_new
        self.c = c_new
        return h_new

    def backward_and_update(self, dh: np.ndarray, lr: float) -> None:
        """
        One-step BPTT gradient update given upstream gradient dh [hidden_dim].
        Clips gradients to prevent exploding gradients.
        """
        if self._cache is None:
            return
        x, h_prev, c_prev, i, f, g, o, c_new, h_new = self._cache
        hd = self.h_dim

        tanh_c = np.tanh(c_new)
        do = dh * tanh_c
        dc = dh * o * (1 - tanh_c ** 2)

        di = dc * g
        df = dc * c_prev
        dg = dc * i

        # Gate pre-activation gradients
        di_pre = di * i * (1 - i)
        df_pre = df * f * (1 - f)
        dg_pre = dg * (1 - g ** 2)
        do_pre = do * o * (1 - o)

        d_gates = np.concatenate([di_pre, df_pre, dg_pre, do_pre])
        np.clip(d_gates, -1.0, 1.0, out=d_gates)

        # Update Wx [4h × input_dim] and Wh [4h × hidden_dim] separately
        dWx = np.outer(d_gates, x)
        dWh = np.outer(d_gates, h_prev)

        self.Wx -= lr * dWx
        self.Wh -= lr * dWh
        self.b  -= lr * d_gates


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -15, 15)))


# ─────────────────────────────────────────────────────────
# LINEAR HEAD  (LSTM hidden → output logits)
# ─────────────────────────────────────────────────────────

class LinearHead:
    def __init__(self, in_dim: int, out_dim: int, seed: int = 1):
        rng  = np.random.default_rng(seed)
        self.W = rng.normal(0, 0.1, (out_dim, in_dim)).astype(np.float32)
        self.b = np.zeros(out_dim, dtype=np.float32)

    def forward(self, h: np.ndarray) -> np.ndarray:
        return self.W @ h + self.b

    def update(self, h: np.ndarray, grad_out: np.ndarray, lr: float) -> np.ndarray:
        """Returns gradient w.r.t. h for BPTT."""
        self.W -= lr * np.outer(grad_out, h)
        self.b -= lr * grad_out
        return self.W.T @ grad_out


# ─────────────────────────────────────────────────────────
# FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────

def extract_hazard_features(road: Dict) -> np.ndarray:
    """
    7-dim feature vector for HazardLSTM:
      [0] closest vehicle distance (0–1)
      [1] n_vehicles (clipped, normalised)
      [2] closest pedestrian distance (0–1)
      [3] n_pedestrians (clipped, normalised)
      [4] lane detected (0/1)
      [5] raw hazard level / 3 (normalised)
      [6] road unclear (0/1)
    """
    vehs  = road.get("vehicles",    [])
    peds  = road.get("pedestrians", [])
    close_v = max((v.get("distance", 0.0) for v in vehs), default=0.0)
    close_p = max((p.get("distance", 0.0) for p in peds), default=0.0)
    n_v  = float(np.clip(len(vehs),  0, 6) / 6.0)
    n_p  = float(np.clip(len(peds),  0, 4) / 4.0)
    lane = 1.0 if road.get("lane_status") == "DETECTED" else 0.0
    hl   = float(road.get("hazard_level", 0)) / 3.0
    road_unc = 0.0 if road.get("road_status") == "CLEAR" else 1.0
    return np.array([close_v, n_v, close_p, n_p, lane, hl, road_unc],
                    dtype=np.float32)


def extract_collision_features(road: Dict, prev_dist: float,
                                approach_rate: float) -> np.ndarray:
    """
    4-dim feature vector for CollisionLSTM:
      [0] closest vehicle distance now
      [1] smoothed approach rate (positive = closing)
      [2] distance delta (now - prev, positive = moving away)
      [3] n_vehicles normalised
    """
    vehs    = road.get("vehicles", [])
    close_v = max((v.get("distance", 0.0) for v in vehs), default=0.0)
    delta   = float(close_v - prev_dist)
    n_v     = float(np.clip(len(vehs), 0, 6) / 6.0)
    return np.array([close_v, approach_rate, delta, n_v], dtype=np.float32)


# ─────────────────────────────────────────────────────────
# RULE-BASED TEACHER  (generates soft labels for online learning)
# ─────────────────────────────────────────────────────────

def rule_hazard_label(feat: np.ndarray) -> np.ndarray:
    """
    Produce a soft 4-class label from the rule engine.
    Returns probability vector [p0, p1, p2, p3].
    """
    close_v, n_v, close_p, n_p, lane, hl_norm, road_unc = feat
    score = (close_v * 0.40 + close_p * 0.30 +
             n_v    * 0.10 + n_p    * 0.10 +
             (1 - lane) * 0.05 + road_unc * 0.05)
    if   score > 0.70: label = 3
    elif score > 0.45: label = 2
    elif score > 0.20: label = 1
    else:              label = 0
    # Soft label: 0.85 on target, spread 0.05 across others
    soft = np.full(4, 0.05, dtype=np.float32)
    soft[label] = 0.80
    return soft


def rule_collision_label(feat: np.ndarray) -> float:
    """
    Produce a collision probability in [0,1] from simple physics intuition.
    """
    close_v, approach_rate, delta, n_v = feat
    # Closing quickly at close range = high risk
    prob = float(np.clip(
        close_v * 0.5 + max(0.0, -delta) * 1.5 + max(0.0, approach_rate) * 0.8,
        0.0, 1.0
    ))
    return prob


# ─────────────────────────────────────────────────────────
# SOFTMAX / CROSS-ENTROPY
# ─────────────────────────────────────────────────────────

def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def ce_grad(probs: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Gradient of cross-entropy loss w.r.t. logits."""
    return probs - target


# ─────────────────────────────────────────────────────────
# MAIN CLASS
# ─────────────────────────────────────────────────────────

class RoadRNN:
    """
    Plug-in temporal RNN layer for RoadMonitor.
    Call rnn.update(road_result) every frame after road_monitor.process().
    """

    def __init__(self):
        logger.info("Initialising RoadRNN (HazardLSTM + CollisionLSTM)…")

        # ── HazardLSTM ─────────────────────────────────────
        self.h_lstm  = NumpyLSTM(HAZARD_INPUT_DIM,    HIDDEN_DIM, seed=42)
        self.h_head  = LinearHead(HIDDEN_DIM, 4,                  seed=43)

        # ── CollisionLSTM ───────────────────────────────────
        self.c_lstm  = NumpyLSTM(COLLISION_INPUT_DIM, HIDDEN_DIM, seed=44)
        self.c_head  = LinearHead(HIDDEN_DIM, 1,                  seed=45)

        # Seed weights with rule-compatible biases (not random zero)
        self._seed_hazard_head()
        self._seed_collision_head()

        # ── History buffers ─────────────────────────────────
        self.hazard_feat_buf    = deque(maxlen=SEQ_LEN)
        self.collision_feat_buf = deque(maxlen=SEQ_LEN)

        # ── Online update bookkeeping ───────────────────────
        self._frame_count    = 0
        self._pending_h_feat = []   # (feat, target) pairs waiting for update
        self._pending_c_feat = []

        # ── State for collision tracking ────────────────────
        self._prev_close_v   = 0.0
        self._approach_rate  = 0.0  # EMA of closing speed

        # ── Output smoothing ────────────────────────────────
        self._smooth_hazard  = 0.0
        self._smooth_coll    = 0.0

        logger.info("RoadRNN ready — rule-seeded, no pre-training required.")

    # ──────────────────────────────────────────────────────
    # WEIGHT SEEDING
    # ──────────────────────────────────────────────────────

    def _seed_hazard_head(self):
        """
        Bias the hazard head so that without any learning it roughly
        maps hidden activations to the rule-based hazard levels.
        """
        # Class 0 (clear) → slight positive bias
        # Class 3 (critical) → slight negative bias (harder to trigger)
        self.h_head.b = np.array([0.5, 0.1, -0.1, -0.5], dtype=np.float32)

    def _seed_collision_head(self):
        """Collision head: start conservative (low collision probability)."""
        self.c_head.b = np.array([-1.0], dtype=np.float32)

    # ──────────────────────────────────────────────────────
    # FORWARD
    # ──────────────────────────────────────────────────────

    def _run_hazard(self, feat: np.ndarray) -> Tuple[int, float, np.ndarray]:
        """
        Run HazardLSTM on one new feature vector.
        Returns (predicted_class, confidence, probabilities).
        """
        hidden = self.h_lstm.forward(feat)
        logits = self.h_head.forward(hidden)
        probs  = softmax(logits)
        cls    = int(np.argmax(probs))
        conf   = float(probs[cls])
        return cls, conf, probs

    def _run_collision(self, feat: np.ndarray) -> float:
        """
        Run CollisionLSTM on one feature vector.
        Returns collision probability [0, 1].
        """
        hidden = self.c_lstm.forward(feat)
        logit  = self.c_head.forward(hidden)[0]
        prob   = float(_sigmoid(np.array([logit]))[0])
        return prob

    # ──────────────────────────────────────────────────────
    # ONLINE UPDATE  (teacher forcing with rule labels)
    # ──────────────────────────────────────────────────────

    def _do_online_update(self):
        """Update weights on accumulated (feat, label) pairs."""
        lr = LEARNING_RATE

        # Hazard LSTM update
        for feat, target_soft in self._pending_h_feat:
            hidden = self.h_lstm.forward(feat)
            logits = self.h_head.forward(hidden)
            probs  = softmax(logits)
            grad   = ce_grad(probs, target_soft)
            dh     = self.h_head.update(hidden, grad, lr)
            self.h_lstm.backward_and_update(dh, lr)

        # Collision LSTM update
        for feat, target_prob in self._pending_c_feat:
            hidden  = self.c_lstm.forward(feat)
            logit   = self.c_head.forward(hidden)[0]
            pred_p  = float(_sigmoid(np.array([logit]))[0])
            grad_c  = np.array([pred_p - target_prob], dtype=np.float32)
            dh      = self.c_head.update(hidden, grad_c, lr)
            self.c_lstm.backward_and_update(dh, lr)

        self._pending_h_feat.clear()
        self._pending_c_feat.clear()

    # ──────────────────────────────────────────────────────
    # PUBLIC API
    # ──────────────────────────────────────────────────────

    def update(self, road_result: Dict) -> Dict:
        """
        Call this every frame after road_monitor.process().

        Parameters
        ----------
        road_result : dict returned by RoadMonitor.process()

        Returns
        -------
        dict with keys:
            hazard_level       int   0–3 (RNN-refined)
            collision_prob     float 0–1
            approach_rate      float (positive = vehicle closing in)
            rnn_confidence     float 0–1
            override           bool  (RNN changed the raw hazard level)
            collision_warning  bool
        """
        self._frame_count += 1

        # ── Approach rate (EMA of distance delta) ───────────
        vehs    = road_result.get("vehicles", [])
        close_v = max((v.get("distance", 0.0) for v in vehs), default=0.0)
        prev_close_v = self._prev_close_v          # save BEFORE updating
        delta        = close_v - prev_close_v
        # Positive delta = vehicle getting closer (distance normalised 0→1)
        self._approach_rate = (
            APPROACH_ALPHA * delta +
            (1 - APPROACH_ALPHA) * self._approach_rate
        )
        self._prev_close_v = close_v               # update AFTER delta calc

        # ── Feature extraction ───────────────────────────────
        h_feat = extract_hazard_features(road_result)
        c_feat = extract_collision_features(
            road_result, prev_close_v, self._approach_rate  # use saved prev
        )

        # ── Forward pass ─────────────────────────────────────
        raw_hazard = road_result.get("hazard_level", 0)
        rnn_hazard, confidence, probs = self._run_hazard(h_feat)
        coll_prob  = self._run_collision(c_feat)

        # ── EMA smoothing of outputs ─────────────────────────
        self._smooth_hazard = 0.3 * rnn_hazard + 0.7 * self._smooth_hazard
        self._smooth_coll   = 0.3 * coll_prob  + 0.7 * self._smooth_coll

        final_hazard = int(round(float(np.clip(self._smooth_hazard, 0, 3))))
        final_coll   = float(self._smooth_coll)

        # ── Accumulate for online update ─────────────────────
        teacher_h = rule_hazard_label(h_feat)
        teacher_c = rule_collision_label(c_feat)
        self._pending_h_feat.append((h_feat.copy(), teacher_h))
        self._pending_c_feat.append((c_feat.copy(), teacher_c))

        if self._frame_count % ONLINE_UPDATE_EVERY == 0:
            self._do_online_update()

        # ── Build output ─────────────────────────────────────
        override          = (final_hazard != raw_hazard)
        collision_warning = (final_coll >= COLLISION_THRESH and close_v > 0.3)

        if self._frame_count % 90 == 0:   # log every ~3 s at 30 fps
            logger.debug(
                f"RNN | hazard raw={raw_hazard} rnn={final_hazard} "
                f"conf={confidence:.2f} | coll={final_coll:.2f} "
                f"approach={self._approach_rate:+.3f}"
            )

        return {
            "hazard_level"      : final_hazard,
            "collision_prob"    : final_coll,
            "approach_rate"     : float(self._approach_rate),
            "rnn_confidence"    : confidence,
            "override"          : override,
            "collision_warning" : collision_warning,
        }

    def reset(self):
        """Call when starting a new driving session."""
        self.h_lstm.reset()
        self.c_lstm.reset()
        self._prev_close_v  = 0.0
        self._approach_rate = 0.0
        self._smooth_hazard = 0.0
        self._smooth_coll   = 0.0
        self._pending_h_feat.clear()
        self._pending_c_feat.clear()
        logger.info("RoadRNN reset.")

    def get_stats(self) -> Dict:
        return {
            "frames_processed" : self._frame_count,
            "approach_rate"    : self._approach_rate,
            "smooth_hazard"    : self._smooth_hazard,
            "smooth_collision" : self._smooth_coll,
        }