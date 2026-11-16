"""R7 Reliability Subpackage.

Implements quality signal feature extraction, cue reliability estimation via
logistic regression, multi-cue reliability-aware fusion, and the
'monitoring degraded' state machine.
"""

from .quality_features import QualityFeatureExtractor, QualitySignals
from .estimator import CueReliabilityEstimator
from .fusion import ReliabilityAwareFusion
from .degraded_state import MonitoringDegradedTracker

__all__ = [
    "QualityFeatureExtractor",
    "QualitySignals",
    "CueReliabilityEstimator",
    "ReliabilityAwareFusion",
    "MonitoringDegradedTracker",
]
