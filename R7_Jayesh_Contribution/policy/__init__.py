"""R7 Expected-Utility Alert Policy Subpackage.

Implements the decision-theoretic framework for driver alerts:
  - DriverState & AlertAction enums
  - UtilityMatrix with asymmetric miss vs false-alarm costs and annoyance penalty
  - AnnoyanceTracker for 10-minute sliding window alert history
  - ExpectedUtilityAlertPolicy (argmax EU(a))
  - FixedThresholdBaselinePolicy (0.5 threshold ablation baseline)
"""

from .utility_matrix import DriverState, AlertAction, UtilityMatrix
from .annoyance import AnnoyanceTracker
from .expected_utility import ExpectedUtilityAlertPolicy, DecisionOutput
from .baseline_policy import FixedThresholdBaselinePolicy

__all__ = [
    "DriverState",
    "AlertAction",
    "UtilityMatrix",
    "AnnoyanceTracker",
    "ExpectedUtilityAlertPolicy",
    "DecisionOutput",
    "FixedThresholdBaselinePolicy",
]
