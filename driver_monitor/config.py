"""Central settings. Everything tunable lives here."""

# ---- timing -------------------------------------------------------------
WINDOW_SECONDS = 5.0        # temporal window
WINDOW_FRAMES = 75          # GRU input length (5 s at 15 Hz); window_array() resamples to this
CALIB_SECONDS = 30.0        # per-driver calibration length (proposal: 30 s; test 10/20 s too)
CALIB_MIN_FRAMES = 30       # need at least this many face frames to finish calibration

# ---- reliability / degraded --------------------------------------------
R_MIN = 0.35                # below this, a cue counts as unreliable
DEGRADED_SECONDS = 3.0      # all cues below R_MIN for this long -> "monitoring degraded"

# ---- alert policy -------------------------------------------------------
STATES = ("alert", "fatigue", "distraction")
ACTIONS = ("none", "soft", "visual_audio", "urgent")
EFFECTIVENESS = (0.0, 0.5, 0.8, 1.0)     # how much each action reduces the miss cost
INTENSITY_COST = (0.0, 1.0, 3.0, 8.0)    # false-alarm cost grows with alert intensity
FA_COST = 1.0
MISS_RATIO = 10.0                         # miss : false-alarm (sweep 3..20 in the report)
MISS_RATIO_MIN, MISS_RATIO_MAX = 3.0, 20.0
ANNOYANCE_K = 0.25                        # extra FA cost per alert event in the last 10 min
ANNOYANCE_WINDOW_S = 600.0
PROB_EMA = 0.3                            # smoothing of P(state) before the policy

# ---- stub-model thresholds (only used by the placeholder temporal model) --
STUB_WARMUP_S = 2.0
EAR_CLOSED_REL = 0.85       # ear/ear_baseline below this starts to count as closed
MAR_YAWN_DELTA = 0.4        # mar - mar_baseline above this counts as mouth wide open
YAW_AWAY_DEG = 25.0
PITCH_AWAY_DEG = 20.0

# ---- head-pose sign conventions (flip if yaw/pitch move the wrong way) ----
YAW_SIGN = 1.0
PITCH_SIGN = 1.0

# ---- audio --------------------------------------------------------------
AUDIO_COOLDOWN_S = {2: 6.0, 3: 2.5}       # seconds between repeats, per alert level

# ---- MediaPipe landmark indices (468-point mesh) -------------------------
LM_R_EYE = [33, 160, 158, 133, 153, 144]
LM_L_EYE = [362, 385, 387, 263, 373, 380]
LM_MOUTH_V = [(82, 87), (13, 14), (312, 317)]
LM_MOUTH_H = (78, 308)
LM_MOUTH_ALL = [82, 87, 13, 14, 312, 317, 78, 308]
