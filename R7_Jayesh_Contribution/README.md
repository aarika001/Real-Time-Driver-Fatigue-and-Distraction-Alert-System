# R7 — Reliability Estimator & Expected-Utility Alert Policy

**Owner:** Jayesh Singh Mehta (R7)  
**Team:** NeuroNova  
**Project:** Real-Time Driver Fatigue and Distraction Alert System  
**Track:** Applications of Deep Learning  

---

## 1. Overview and Role Responsibilities

As defined in the Project Proposal (Sections 1, 2, 3.2, 3.3, and 5), Role R7 designs, implements, and evaluates:
1. **Reliability-Aware Multi-Cue Fusion:** Logistic regression estimators predicting cue trustworthiness from optical and geometric quality signals (brightness, contrast, glare fraction, sharpness, head deviation, and landmark jitter), down-weighting degraded sensors and raising a "Monitoring Degraded" banner when reliability stays low for $T$ seconds.
2. **Expected-Utility Alert Policy:** A rational decision agent maximizing expected utility:
   $$a^* = \arg\max_{a \in \mathcal{A}} \sum_{s \in \mathcal{S}} P(s) \, U(s, a)$$
   incorporating asymmetric safety costs (miss cost $\gg$ false-alarm cost) and a dynamic driver annoyance penalty over a 10-minute sliding window (mitigating nuisance alarms).
3. **Parameter Sweep & Ablation Framework:** Sweeping miss-to-false-alarm cost ratios (3:1 to 20:1) to identify optimal operating points and running systematic ablation studies (without reliability, without calibration, and fixed 0.5 threshold baseline).

---

## 2. Mathematical Formulations

### A. Cue Reliability Estimation
For each cue $k \in \{\text{eye}, \text{mouth}, \text{head\_pose}\}$, a logistic regression model predicts reliability $r_k \in [0, 1]$:
$$r_k = \sigma(\mathbf{w}_k^T \mathbf{x}_k + b_k) = \frac{1}{1 + e^{-(\mathbf{w}_k^T \mathbf{x}_k + b_k)}}$$

Quality features $\mathbf{x}_k$:
* **Eye:** Brightness deviation from nominal, RMS contrast, glare pixel fraction, Laplacian sharpness, head deviation z-score, landmark jitter (px), face detection flag.
* **Mouth:** Head deviation z-score, landmark jitter (px), sharpness proxy, face detection flag.
* **Head Pose:** Extreme angle magnitude, landmark jitter, detector confidence, face detection flag.

### B. Expected-Utility Decision Model
* **States $\mathcal{S}$:** $\{\text{ALERT}, \text{FATIGUE}, \text{DISTRACTION}\}$
* **Actions $\mathcal{A}$:** $\{\text{NONE}, \text{SOFT\_VISUAL}, \text{VISUAL\_AUDIO}, \text{URGENT}\}$
* **Utility Matrix $U(s, a)$:**
  $$U(s, a) = - \text{Cost}(s, a)$$
  where:
  $$\text{Cost}(\text{FATIGUE}, \text{NONE}) = C_{\text{miss}} = \rho \cdot C_{\text{fa}}$$
  $$\text{Cost}(\text{ALERT}, a) = C_{\text{fa}}(a) + \Delta_{\text{annoyance}}$$

### C. Driver Annoyance Escalation (10-Minute Window)
To prevent driver annoyance shutdowns (Subaru survey evidence):
$$\Delta_{\text{annoyance}} = \alpha \cdot \left(\sum_{i \in \text{recent alerts}} w(a_i) \cdot \left(0.5 + 0.5 \frac{t - t_i}{T_{\text{window}}}\right)\right)^{\gamma}$$
where $T_{\text{window}} = 600\text{ s}$ (10 minutes).

---

## 3. Directory Layout

```
R7_Jayesh_Contribution/
├── README.md                      # This documentation file
├── __init__.py                    # Public API exports
├── reliability/                   # Milestone: Reliability Estimator
│   ├── __init__.py
│   ├── quality_features.py        # Quality signal extractor & jitter tracker
│   ├── estimator.py               # CueReliabilityEstimator (Logistic Regression)
│   ├── fusion.py                  # ReliabilityAwareFusion (dynamic cue down-weighting)
│   └── degraded_state.py          # MonitoringDegradedTracker (T-second persistence)
├── policy/                        # Milestone: Expected-Utility Policy
│   ├── __init__.py
│   ├── utility_matrix.py          # DriverState, AlertAction, UtilityMatrix U(s, a)
│   ├── annoyance.py               # AnnoyanceTracker (10-minute sliding window)
│   ├── expected_utility.py        # ExpectedUtilityAlertPolicy (argmax EU(a))
│   └── baseline_policy.py         # FixedThresholdBaselinePolicy (0.5 ablation baseline)
├── evaluation/                    # Milestone: Evaluation & Ablations
│   ├── __init__.py
│   ├── cost_sweep.py              # CostRatioSweep (sweeps rho from 3:1 to 20:1)
│   └── ablations.py               # AblationRunner (compares full vs 3 ablations)
└── tests/                         # Pytest test suite
    ├── __init__.py
    ├── test_quality_and_reliability.py
    ├── test_policy_and_decision.py
    └── test_evaluation_and_ablations.py
```

---

## 4. Integration Contracts

* **R4 (Dhwaj - Eye & Mouth):** Consumes `eye_brightness`, `eye_contrast`, `eye_glare_frac`, `eye_sharpness` from `cues.schemas.R4FrameOutput` / `cues.eye_crop.EyeQuality`.
* **R5 (Aaditya - Head Pose):** Consumes `head_angle_from_neutral(delta_pitch, delta_yaw)` from `head-pose.head_pose_core`.
* **R6 (Sejal - Temporal GRU & Calibration):** Consumes calibrated state probability beliefs $P(s) = [P(\text{ALERT}), P(\text{FATIGUE}), P(\text{DISTRACTION})]$ as input to `ExpectedUtilityAlertPolicy.decide()`.
* **R8 (Dev - Real-time UI):** Hands `DecisionOutput` (`selected_action`, `action_name`, `is_monitoring_degraded`, `status_message`) to the live overlay and alert actuators.

---

## 5. Running Tests

Run the complete R7 test suite using Python 3.12:

```bash
cd Real-Time-Driver-Fatigue-and-Distraction-Alert-System
py -3 -m pytest R7_Jayesh_Contribution/tests -v
```
