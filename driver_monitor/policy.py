"""Expected-utility alert policy."""
from R7_Jayesh_Contribution.policy.expected_utility import ExpectedUtilityAlertPolicy
from R7_Jayesh_Contribution.reliability.degraded_state import MonitoringDegradedTracker, DegradedStatus
import config
from interfaces import Decision, TemporalOut, Reliability

class AlertPolicy:
    def __init__(self, miss_ratio: float = config.MISS_RATIO):
        self.miss_ratio = miss_ratio
        self.r7_policy = ExpectedUtilityAlertPolicy(cost_ratio=miss_ratio)
        self.r7_degraded = MonitoringDegradedTracker(
            threshold=config.R_MIN, 
            required_duration_s=config.DEGRADED_SECONDS
        )

    def set_ratio(self, r: float):
        self.miss_ratio = float(r)
        self.r7_policy.set_cost_ratio(r)

    def decide(self, tout: TemporalOut, rel: Reliability, t: float) -> Decision:
        # Check overall reliability via R7's tracker
        overall_rel = rel.best()
        deg_status = self.r7_degraded.update(overall_rel, t)
        
        # Get decision from R7's expected utility agent
        r7_out = self.r7_policy.decide(tout.probs, t, degraded_status=deg_status)
        
        # Map to your expected interface output
        dominant_state = "fatigue" if tout.probs[1] >= tout.probs[2] else "distraction"
        return Decision(
            action_idx=r7_out.selected_action.value,
            action=r7_out.action_name.lower(),
            dominant=dominant_state,
            degraded=deg_status.is_degraded,
            eu=r7_out.expected_utilities,
            n_recent_alerts=self.r7_policy.annoyance.count_recent_alerts(t)
        )