"""R7 Evaluation and Ablations Subpackage.

Implements the cost-ratio sweep (3:1 to 20:1) and system ablation comparisons
specified in proposal Sections 3.2, 3.3, and 5.
"""

from .cost_sweep import CostRatioSweep, SweepResult
from .ablations import AblationRunner, AblationResult

__all__ = [
    "CostRatioSweep",
    "SweepResult",
    "AblationRunner",
    "AblationResult",
]
