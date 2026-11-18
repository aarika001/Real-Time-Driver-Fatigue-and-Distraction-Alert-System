"""Expected-utility alert policy: a* = argmax_a sum_s P(s) U(s, a).

Working first version (R7 refines: cost sweep, validation-based operating point).
"""
from collections import deque
import numpy as np
import config
from interfaces import Decision, TemporalOut, Reliability


class AlertPolicy:
    def __init__(self, miss_ratio: float = config.MISS_RATIO):
        self.miss_ratio = miss_ratio
        self.recent = deque()           # timestamps of alert *events* (none -> alert)
        self.prev = 0
        self.p = None                   # EMA-smoothed state probabilities
        self.low_since = None

    def set_ratio(self, r: float):
        self.miss_ratio = float(np.clip(r, config.MISS_RATIO_MIN, config.MISS_RATIO_MAX))

    def utility(self, a: int, s: int, n_recent: int) -> float:
        if s == 0:      # driver is fine: any alert is a false alarm, costlier when loud / repeated
            return -config.FA_COST * config.INTENSITY_COST[a] * (1 + config.ANNOYANCE_K * n_recent)
        return -config.FA_COST * self.miss_ratio * (1 - config.EFFECTIVENESS[a])   # missed or weak warning

    def decide(self, tout: TemporalOut, rel: Reliability, t: float) -> Decision:
        a = config.PROB_EMA
        self.p = tout.probs if self.p is None else (1 - a) * self.p + a * tout.probs
        while self.recent and t - self.recent[0] > config.ANNOYANCE_WINDOW_S:
            self.recent.popleft()
        n = len(self.recent)

        eu = [sum(self.p[s] * self.utility(act, s, n) for s in range(3)) for act in range(4)]
        best = int(np.argmax(eu))

        # degraded: every cue unreliable for DEGRADED_SECONDS
        if rel.best() < config.R_MIN:
            self.low_since = t if self.low_since is None else self.low_since
        else:
            self.low_since = None
        degraded = self.low_since is not None and t - self.low_since >= config.DEGRADED_SECONDS
        if degraded:
            best = 0                    # don't guess; the banner says so

        if best > 0 and self.prev == 0:
            self.recent.append(t)
        self.prev = best
        return Decision(
            action_idx=best, action=config.ACTIONS[best],
            dominant="fatigue" if self.p[1] >= self.p[2] else "distraction",
            degraded=degraded, eu=dict(zip(config.ACTIONS, eu)), n_recent_alerts=n,
        )
