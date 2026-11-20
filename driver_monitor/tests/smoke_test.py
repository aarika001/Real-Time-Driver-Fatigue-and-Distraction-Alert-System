"""Runs without a camera or MediaPipe: synthetic percepts through calibration -> policy -> HUD."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import degrade, reliability
from audio import AudioAlert
from calibration import Calibrator
from interfaces import Percept
from overlay import Hud, draw_hud
from policy import AlertPolicy
from temporal import TemporalModel
import config

Q = dict(brightness=0.5, contrast=0.5, glare_frac=0.0, sharpness=0.6, jitter=0.1, det_conf=1.0)


def run(scenario, fps=15, seconds=25, calib_s=5):
    cal, tm, pol = Calibrator(calib_s), TemporalModel(), AlertPolicy()
    acts, degraded = [], []
    for i in range(int(fps * seconds)):
        t = i / fps
        p = scenario(t)
        rel = reliability.estimate(p)
        if not cal.done:
            cal.add(p, t); continue
        d = pol.decide(tm.update(cal.normalise(p), rel, t), rel, t)
        acts.append(d.action_idx); degraded.append(d.degraded)
    return acts, degraded, tm


def normal(t):    return Percept(t, True, 0.0, 0.30, 0.02, 0.0, 0.0, dict(Q))
def drowsy(t):    # after calib, eyes closed most of the time
    closed = t > 8
    return Percept(t, True, 1.0 if closed else 0.0, 0.10 if closed else 0.30, 0.02, 0, 0, dict(Q))
def mirror(t):    # brief 1 s head turns
    turn = (t % 8) < 1 and t > 8
    return Percept(t, True, 0, 0.30, 0.02, 40.0 if turn else 0.0, 0, dict(Q))
def looking_away(t): return Percept(t, True, 0, 0.30, 0.02, 40.0 if t > 8 else 0.0, 0, dict(Q))
def dark(t):      return Percept(t, True, 0, 0.30, 0.02, 0, 0, dict(Q, brightness=0.03)) if t > 8 else normal(t)

a, _, tm = run(normal);        assert max(a) <= 1, "audio/urgent false alarm on normal driving"
a, _, _ = run(drowsy);         assert max(a) >= 2, "missed sustained eye closure"
a, _, _ = run(mirror);         assert max(a) <= 1, "mirror check triggered an audio/urgent alert"
a, _, _ = run(looking_away);   assert max(a) >= 2, "missed sustained look-away"
a, dg, _ = run(dark);          assert any(dg), "no degraded banner in the dark"
assert tm.window_array().shape == (config.WINDOW_FRAMES, 13)

# policy: ratio changes behaviour (higher miss cost -> alerts earlier)
from interfaces import TemporalOut, Reliability
lo, hi = AlertPolicy(3), AlertPolicy(20)
out = TemporalOut(np.array([0.8, 0.2, 0.0])); r = Reliability(1, 1, 1)
assert lo.decide(out, r, 0).action_idx <= hi.decide(out, r, 0).action_idx

# degrade + overlay + audio(disabled) execute
frame = np.full((480, 640, 3), 120, np.uint8)
st = degrade.DegradeState(active={"glare", "lowlight", "occlusion", "offaxis"})
assert degrade.apply(frame, st, {"eyes": (320, 200), "mouth": (320, 320)}).shape == frame.shape
st.active = {"facelost"}; assert degrade.apply(frame, st).max() == 0
d = AlertPolicy().decide(TemporalOut(np.array([0.1, 0.9, 0.0])), r, 1.0)
h = Hud(percept=normal(0), rel=r, tout=TemporalOut(np.array([0.1, 0.9, 0.0])), decision=d)
draw_hud(frame.copy(), h)
AudioAlert(enabled=False).handle(d, 0.0)
print("smoke test passed")
