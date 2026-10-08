"""Run a hardware test case over the polar grid (polar efficiency x fixed
CL) at one or more fixed wing areas, then plot it.

Every input lives in a case file (``config/hardware_cases/*.yml``): the
hardware limits (ground station power, maximum tether force, maximum
reeling speed, tether length, wing areas, ...), the operational
constraints (minimum tether length, minimum height, elevation bounds,
...), the wind speed range and the polar grid. Entries can be overridden
for one run with ``--set key=value`` (hardware/constraints keys directly,
anything else as a dotted path; lists as ``key=a,b``; ``key=none``
restores a tether-length-relative default).

For each wing area the script writes a polar grid study, runs it (cases
already computed for the same case are reused) and writes, under
``results/<case>/S<area>/plots``:

    polar_grid_heatmaps_S<area>.png      LCoE and AEP maps (cell values)
    polar_grid_contours_S<area>.png      LCoE and AEP contours
    polar_grid_power_curves_S<area>.png  power curves per CL x L/D_max
    polar_grid_operation_S<area>.png     cycle and reel-out power, reel-out
                                         tether force, reel-out and reel-in
                                         speed against wind speed

Examples:
    python scripts/run_hardware_case.py
    python scripts/run_hardware_case.py --case config/hardware_cases/other.yml
    python scripts/run_hardware_case.py --set areas_m2=22 \\
        --set min_tether_length_m=50 --set wind_speeds.cut_in=5
"""

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import yaml

from plot_polar_grid import plot_study
from tef.aero_polars import run_polar_grid_study
from tef.io import load_yaml, project_root, write_yaml

DEFAULT_CASE = Path('config/hardware_cases/13kw_150m_1t.yml')
# Integer-valued entries kept as integers; every other number is a float
# so the case hash does not depend on how a number is written.
INTEGER_KEYS = ('n_points', 'workers')


def _floats(value, key=None):
    if isinstance(value, dict):
        return {k: _floats(v, k) for k, v in value.items()}
    if isinstance(value, list):
        return [_floats(v, key) for v in value]
    if (isinstance(value, int) and not isinstance(value, bool)
            and key not in INTEGER_KEYS):
        return float(value)
    return value


def resolve_lengths(hw: dict, cons: dict) -> dict:
    """Tether length bounds in metres, defaults from the base fractions;
    raises when they are inconsistent."""
    length = hw['tether_length_m']
    reelOut = cons['reel_out_end_length_m'] or [0.8 * length, 0.95 * length]
    lengths = {
        'reel_in_end': [cons['min_tether_length_m'],
                        cons['max_reel_in_end_length_m'] or 0.8 * length],
        'reel_out_end': [float(v) for v in reelOut],
        'min_stroke': cons['min_stroke_m'] or 0.1 * length,
    }
    inLow, inHigh = lengths['reel_in_end']
    outLow, outHigh = lengths['reel_out_end']
    if not (0 < inLow <= inHigh and outLow <= outHigh <= length
            and inLow + lengths['min_stroke'] <= outHigh):
        raise SystemExit(f"Inconsistent tether lengths for L = {length:g} m: "
                         f"{lengths}")
    return lengths


def qsm_override(case: dict, base_dir: Path) -> dict:
    """QSM settings for the constraints: lengths as fractions of the
    tether length, optimiser start point clipped into the bounds."""
    hw, cons = case['hardware'], case['constraints']
    length = hw['tether_length_m']
    lengths = resolve_lengths(hw, cons)
    fracIn = [v / length for v in lengths['reel_in_end']]
    fracOut = [v / length for v in lengths['reel_out_end']]
    elevOut = [float(v) for v in cons['elevation_traction_deg']]
    elevRori = [float(v) for v in cons['elevation_end_rori_deg']]
    settings = load_yaml(base_dir / 'inertiafree-qsm_settings.yml')
    x0 = list(settings['optimization']['optimizer']['x0'])
    for i, (low, high) in zip((2, 3, 4, 5),
                              (fracIn, fracOut, elevOut, elevRori)):
        x0[i] = min(max(x0[i], low), high)
    return {
        'optimization': {
            'wind_speeds': case['wind_speeds'],
            'optimizer': {'x0': x0},
            'bounds': {
                'fraction_tether_length_retraction_end_min': fracIn[0],
                'fraction_tether_length_retraction_end_max': fracIn[1],
                'fraction_tether_length_traction_end_min': fracOut[0],
                'fraction_tether_length_traction_end_max': fracOut[1],
                'elevation_angle_traction_min': elevOut[0],
                'elevation_angle_traction_max': elevOut[1],
                'elevation_angle_end_trans_rori_min': elevRori[0],
                'elevation_angle_end_trans_rori_max': elevRori[1],
            },
            'constraints': {
                'min_tether_length_fraction_difference':
                    lengths['min_stroke'] / length,
                'max_difference_elevation_angle_steps':
                    float(cons['max_elevation_step_deg']),
            },
        },
        'cycle': {'minimum_height': float(cons['minimum_height_m']),
                  'minimum_tether_force':
                      float(cons['minimum_tether_force_n'])},
    }


def _wing_mass_override(area: float, base_dir: Path) -> dict:
    """Extend the wing mass table below its smallest anchor along the
    first segment's log-log slope (the table refuses smaller areas)."""
    model = load_yaml(base_dir / 'scaling_settings.yml')['wing_mass_model']
    areas = [float(a) for a in model['anchor_flat_areas_m2']]
    masses = [float(m) for m in model['anchor_masses_kg']]
    if area >= areas[0]:
        return {}
    slope = math.log(masses[1] / masses[0]) / math.log(areas[1] / areas[0])
    mass = masses[0] * (area / areas[0]) ** slope
    return {'wing_mass_model': {
        'anchor_flat_areas_m2': [area] + areas,
        'anchor_masses_kg': [round(mass, 2)] + masses}}


def build_study(case: dict, area: float, results_dir: Path) -> dict:
    """Polar grid study configuration for one wing area."""
    hw = case['hardware']
    baseDir = project_root() / case['base_config_dir']
    ratio = load_yaml(baseDir / 'scaling_settings.yml')['geometry'][
        'projected_to_flat_area_ratio']
    study = {
        'metadata': {'name': f"{case['metadata']['name']}, "
                             f"S = {area:g} m2"},
        'base_config_dir': case['base_config_dir'],
        'results_dir': results_dir.relative_to(project_root()).as_posix(),
        'design': {
            'flat_area_m2': area,
            'allowable_tether_stress_pa': hw['allowable_tether_stress_pa'],
            'generator_max_power_w': hw['generator_kw'] * 1e3,
            'max_tether_speed_m_s': hw['max_tether_speed_m_s'],
            'tether_length_m': hw['tether_length_m'],
            # Maximum tether force = wing loading x projected area.
            'max_wing_loading_n_m2_projected':
                hw['max_tether_force_n'] / (ratio * area),
        },
        'polar': case['polar'],
        'reel_in': {'cl': case['reel_in_cl']},
        'reference_kites': case.get('reference_kites') or {},
        'qsm_override': qsm_override(case, baseDir),
        'case_options': {'zero_power_beyond_cut_out': True,
                         'peak_power_cap_factor': hw['peak_power_cap_factor']},
        'execution': {'workers': case['execution']['workers'],
                      'reuse_cases': True},
    }
    massOverride = _wing_mass_override(area, baseDir)
    if massOverride:
        study['scaling_override'] = massOverride
    return study


def _case_record(case: dict) -> dict:
    """Everything that changes a case's result (the areas only add
    folders)."""
    hw = case['hardware']
    return {'hardware': {k: v for k, v in hw.items() if k != 'areas_m2'},
            'constraints': case['constraints'],
            'wind_speeds': case['wind_speeds'], 'polar': case['polar'],
            'reel_in_cl': case['reel_in_cl']}


def _case_name(case: dict) -> str:
    """Readable hardware name plus a short hash of the whole case."""
    hw = case['hardware']
    digest = hashlib.sha1(json.dumps(_case_record(case), sort_keys=True)
                          .encode()).hexdigest()[:6]
    return (f"hw_{hw['generator_kw']:g}kW_"
            f"{hw['max_tether_force_n'] / 1e3:.3g}kN_"
            f"{hw['max_tether_speed_m_s']:g}ms_"
            f"{hw['tether_length_m']:g}m").replace('.', 'p') + f"_{digest}"


def apply_overrides(pairs, case: dict) -> None:
    """``key=value`` overrides of the case: a hardware or constraints key,
    or a dotted path (``wind_speeds.cut_in``)."""
    for pair in pairs or []:
        key, _, text = pair.partition('=')
        if '.' in key:
            *parents, leaf = key.split('.')
            target = case
            for part in parents:
                target = target.get(part) if isinstance(target, dict) else None
        else:
            leaf = key
            target = next((case[g] for g in ('hardware', 'constraints')
                           if key in case[g]), None)
        if not isinstance(target, dict) or leaf not in target:
            raise SystemExit(
                f"Unknown parameter {key!r}; hardware: "
                f"{sorted(case['hardware'])}; constraints: "
                f"{sorted(case['constraints'])}; or a dotted path")
        value = yaml.safe_load(f"[{text}]" if ',' in text else text)
        if isinstance(value, str) and value.lower() == 'none':
            value = None
        if isinstance(target[leaf], list) and not isinstance(value, list):
            value = [value]
        target[leaf] = value


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--case', type=Path, default=DEFAULT_CASE,
                        help='Hardware case file (all inputs and constraints)')
    parser.add_argument('--set', action='append', metavar='KEY=VALUE',
                        help='Override a case entry for this run')
    parser.add_argument('--name', help='Results folder name under results/ '
                        '(default: hardware limits + case hash)')
    parser.add_argument('--plot-only', action='store_true',
                        help='Replot existing results without running')
    args = parser.parse_args()

    root = project_root()
    casePath = args.case if args.case.is_absolute() else root / args.case
    case = load_yaml(casePath)
    apply_overrides(args.set, case)
    case = _floats(case)
    hw = case['hardware']
    resolve_lengths(hw, case['constraints'])  # fail before writing anything

    caseDir = root / 'results' / (args.name or _case_name(case))
    caseDir.mkdir(parents=True, exist_ok=True)
    # Reused cases are only valid for the same case.
    record = _case_record(case)
    recordPath = caseDir / 'hardware.yml'
    areas = list(hw['areas_m2'])
    if recordPath.exists():
        previous = load_yaml(recordPath)
        previousAreas = previous.pop('areas_m2', [])
        previous.pop('case_file', None)
        if json.dumps(_floats(previous), sort_keys=True) != json.dumps(
                record, sort_keys=True):
            raise SystemExit(f"{recordPath} holds a different case; pass "
                             f"another --name.")
        areas = sorted(set(previousAreas) | set(areas))
    write_yaml({**record, 'areas_m2': areas,
                'case_file': casePath.relative_to(root).as_posix()},
               recordPath)
    # The case as run (overrides applied), next to its results.
    write_yaml(case, caseDir / 'case.yml')
    print(f"Case folder: {caseDir}")

    start = time.perf_counter()
    for area in hw['areas_m2']:
        areaDir = caseDir / f"S{area:05.1f}".replace('.', 'p')
        studyPath = areaDir / 'study.yml'
        if not args.plot_only:
            areaDir.mkdir(parents=True, exist_ok=True)
            write_yaml(build_study(case, area, areaDir), studyPath)
            print(f"\n=== S = {area:g} m2 ===")
            frame = run_polar_grid_study(studyPath)
            frame = frame[~frame['error']]
            print("LCoE [EUR/MWh] (rows L/D_max, columns CL):")
            print(frame.pivot(index='ld_max', columns='cl',
                              values='lcoe_eur_per_mwh').round(0).to_string())
            print("AEP [MWh]:")
            print(frame.pivot(index='ld_max', columns='cl',
                              values='aep_mwh').round(0).to_string())
        print(f"Plots written to {plot_study(studyPath, fixed_area=area)}")
    print(f"\nTotal wall time: {(time.perf_counter() - start) / 60:.1f} min")


if __name__ == '__main__':
    main()
