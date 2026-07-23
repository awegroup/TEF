"""TEF storage sizing block.

Runs after AWESPA and before EcoMo: EcoMo reads the rated storage
capacity from the system file, so this block updates the storage
entries in ``system.yml`` based on the computed power curves.

Primary method (``cycle_energy_imbalance``): the storage must smooth
the pumping cycle to the cycle-average electrical power. With constant
power per phase the cumulative imbalance E_j - P_avg * t_j is
piecewise linear in time, so its extrema occur at phase boundaries;
the required capacity at one operating point is the range of the
cumulative imbalance, and the rated capacity is the maximum over all
wind speeds and profiles times a safety factor. Fallback when phase
durations are unavailable: the retraction (reel-in plus negative
transition) electrical energy only — conservative, no full smoothing.

Public interface:
    update_storage_capacity_after_power_curves(...)
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tef.io import load_yaml, write_yaml
from tef.system_scaling import StorageSizingSettings

J_PER_WH = 3600.0

# Pumping cycle phase order used for the cumulative imbalance.
_PHASES = ('reel_out', 'transition_rori', 'reel_in', 'transition_riro')


def _phase_values(section: Dict[str, Any], suffix: str,
                  ) -> Optional[List[float]]:
    """Per-phase values (energy or time) in cycle order, or None when
    any phase entry is missing."""
    values = []
    for phase in _PHASES:
        value = section.get(f'{phase}_{suffix}')
        if value is None:
            return None
        values.append(float(value))
    return values


def required_capacity_cycle_imbalance(performance: Dict[str, Any],
                                      ) -> Optional[float]:
    """Required storage energy [J] for one operating point using the
    cycle-average smoothing definition. Returns None when the needed
    phase energies or durations are missing."""
    phaseEnergies = _phase_values(
        performance.get('electrical_energy', {}), 'energy')
    phaseTimes = _phase_values(performance.get('timing', {}), 'time')
    if phaseEnergies is None or phaseTimes is None:
        return None
    cycleTime = sum(phaseTimes)
    if cycleTime <= 0.0:
        return None
    averagePower = sum(phaseEnergies) / cycleTime

    cumulative = 0.0
    boundaries = [0.0]
    for phaseEnergy, phaseTime in zip(phaseEnergies, phaseTimes):
        cumulative += phaseEnergy - averagePower * phaseTime
        boundaries.append(cumulative)
    return max(boundaries) - min(boundaries)


def required_capacity_retraction_fallback(performance: Dict[str, Any],
                                          ) -> Optional[float]:
    """Fallback required storage energy [J]: retraction energy only
    (reel-in plus negative transition electrical energy)."""
    energy = performance.get('electrical_energy', {})
    reelIn = energy.get('reel_in_energy')
    if reelIn is None:
        return None
    total = abs(min(float(reelIn), 0.0))
    for phase in ('transition_rori', 'transition_riro'):
        value = energy.get(f'{phase}_energy')
        if value is not None:
            total += abs(min(float(value), 0.0))
    return total


def storage_capacity_from_power_curves(power_curves: Dict[str, Any],
                                       ) -> Tuple[Optional[float], str]:
    """Maximum required storage energy [Wh] over all profiles and wind
    speeds, and the method actually used ('unavailable' when no
    operating point provides usable energy data)."""
    maxRequired = None
    method = 'cycle_energy_imbalance'
    for curve in power_curves.get('power_curves', []):
        for point in curve.get('wind_speed_data', []):
            if not point.get('successful', False):
                continue
            performance = point.get('performance', {})
            required = required_capacity_cycle_imbalance(performance)
            if required is None:
                required = required_capacity_retraction_fallback(performance)
                if required is None:
                    continue
                method = 'retraction_energy_fallback'
            if maxRequired is None or required > maxRequired:
                maxRequired = required
    if maxRequired is None:
        return None, 'unavailable'
    return maxRequired / J_PER_WH, method


def update_storage_capacity_after_power_curves(
        system_path: Path,
        power_curves_path: Path,
        settings: StorageSizingSettings,
        output_system_path: Optional[Path] = None) -> dict:
    """Size the storage capacity and update the system file.

    Args:
        system_path: System file to update (already scaled).
        power_curves_path: AWESPA power curves for this case.
        settings: Storage sizing settings.
        output_system_path: Destination for the updated system file;
            defaults to updating ``system_path`` in place.

    Returns:
        dict: Sizing record (method, capacity_wh, safety_factor,
        updated_storage_entries) for the case summary.
    """
    system = load_yaml(system_path)
    capacityWh = None
    method = 'unavailable'

    if settings.mode == 'from_power_curves_if_available':
        powerCurves = load_yaml(power_curves_path)
        capacityWh, method = storage_capacity_from_power_curves(powerCurves)
        if capacityWh is not None:
            capacityWh *= settings.safety_factor

    if capacityWh is None:
        if settings.fallback_mode != 'scale_with_generator_power':
            raise ValueError(
                f"Unknown storage sizing fallback mode: "
                f"{settings.fallback_mode!r}")
        generatorPower = (system['components']['ground_station']
                          ['generators'][0]['max_power'])
        capacityWh = (settings.reference_capacity_wh * generatorPower
                      / settings.reference_generator_max_power_w)
        method = 'scale_with_generator_power'

    updatedEntries = []
    for storage in system['components']['ground_station'].get(
            'storages', []):
        if (settings.update_all_storage_entries
                or storage.get('type') == settings.selected_storage_type):
            storage['capacity'] = capacityWh
            updatedEntries.append(storage.get('name', storage.get('type')))

    destination = (Path(output_system_path) if output_system_path is not None
                   else Path(system_path))
    write_yaml(system, destination)

    return {
        'method': method,
        'capacity_wh': capacityWh,
        'safety_factor': settings.safety_factor,
        'updated_storage_entries': updatedEntries,
    }
