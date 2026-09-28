"""Full-cycle power accounting for a TEF case.

A pumping cycle has three distinct powers that must never be collapsed:

- the configured generator limit (the QSM mechanical ``max_power`` cap, the
  swept axis of the generator study);
- the full-cycle peak mechanical power, ``max|P_mech(t)|`` over the whole
  cycle including the transition phases, which is what the drivetrain
  physically handles and therefore the correct generator cost basis;
- the rated cycle electrical power, the maximum cycle-average electrical
  power, which drives the power curve, the AEP and the capacity factor.

The peak is read from the QSM time histories; the phase-average powers in
the power-curve YAML understate it.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class CasePowers:
    """The three distinct powers of a pumping cycle [W]."""

    generator_power_limit_mechanical_w: float
    peak_mechanical_power_w: Optional[float]
    rated_cycle_electrical_power_w: float

    @property
    def crest_peak_over_rated(self) -> Optional[float]:
        """Crest factor from the full-cycle peak, or None when unavailable."""
        if not self.peak_mechanical_power_w or not self.rated_cycle_electrical_power_w:
            return None
        return self.peak_mechanical_power_w / self.rated_cycle_electrical_power_w

    @property
    def crest_limit_over_rated(self) -> Optional[float]:
        """Crest factor from the configured limit."""
        if not self.rated_cycle_electrical_power_w:
            return None
        return (self.generator_power_limit_mechanical_w /
                self.rated_cycle_electrical_power_w)


def full_cycle_peak_mechanical_power(power_curves_path) -> Optional[float]:
    """Max ``|P_mech(t)|`` over all wind-speed profiles [W].

    Read from the QSM time-history ``.npz`` written alongside the power
    curves. Returns None when the time-history file is absent, so the
    caller can fall back to the power-curve reel-out power.

    Args:
        power_curves_path: Path to ``power_curves.yml``; its ``.npz``
            sibling holds the time histories.

    Returns:
        float: The full-cycle peak mechanical power [W], or None.
    """
    npzPath = Path(power_curves_path).with_suffix('.npz')
    if not npzPath.exists():
        return None
    histories = np.load(npzPath, allow_pickle=True)
    peak = 0.0
    for key in histories.files:
        if key.endswith('_power') and not key.endswith('electrical_power'):
            peak = max(peak, float(np.max(np.abs(histories[key]))))
    return peak or None
