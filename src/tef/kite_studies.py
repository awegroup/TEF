"""Kite comparison study: grid-optimise several kites (each its own base
configuration, e.g. different reel-out aerodynamics) at one fixed
generator power limit, evaluating all cases in parallel.

Every (kite, wing area, tether stress) node is a complete, independent
TEF case, so the grid runs in a process pool rather than sequentially
as in :mod:`tef.design_studies`. A shared QSM settings override (e.g. the
wind-speed range) is applied to every kite's inputs so all kites are
evaluated with identical numerical settings; a per-kite scaling override
(e.g. a different wing mass model) is applied on top of that kite's
base configuration. ``case_options`` are passed to every
:class:`TefCaseRunner` (e.g. the cut-out zero point and peak cap).

Public interface:
    load_kite_study_config, run_kite_study
"""

import contextlib
import itertools
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from tef.io import load_yaml, project_root, write_yaml
from tef.pipeline import TefCaseRunner, _deep_update, prepare_study_inputs
from tef.system_scaling import DesignVariables


@dataclass
class KiteStudyConfig:
    """Parsed kite comparison study configuration."""

    name: str
    results_dir: Path
    kites: Dict[str, Path]
    kite_scaling_overrides: Dict[str, Dict[str, Any]]
    qsm_override: Dict[str, Any]
    case_options: Dict[str, Any]
    design_defaults: Dict[str, float]
    flat_areas_m2: List[float]
    tether_stresses_pa: List[float]
    workers: int
    reuse_cases: bool
    raw: Dict[str, Any] = field(default_factory=dict)


def load_kite_study_config(path: Path) -> KiteStudyConfig:
    """Load a kite study YAML. Relative directories resolve against the
    TEF project root."""
    data = load_yaml(path)
    root = project_root()

    def resolve(path_str: str) -> Path:
        candidate = Path(path_str)
        return candidate if candidate.is_absolute() else root / candidate

    grid = data['grid']
    execution = data.get('execution') or {}
    return KiteStudyConfig(
        name=data.get('metadata', {}).get('name', Path(path).stem),
        results_dir=resolve(data['results_dir']),
        kites={name: resolve(item['base_config_dir'])
               for name, item in data['kites'].items()},
        kite_scaling_overrides={
            name: item.get('scaling_override') or {}
            for name, item in data['kites'].items()},
        qsm_override=data.get('qsm_override') or {},
        case_options=data.get('case_options') or {},
        design_defaults={k: float(v)
                         for k, v in data['design_defaults'].items()},
        flat_areas_m2=[float(v) for v in grid['flat_area_m2']],
        tether_stresses_pa=[float(v)
                            for v in grid['allowable_tether_stress_pa']],
        workers=int(execution.get('workers', max(1, os.cpu_count() - 2))),
        reuse_cases=bool(execution.get('reuse_cases', True)),
        raw=data,
    )


def _case_name(flat_area_m2: float, tether_stress_pa: float) -> str:
    return (f"S{flat_area_m2:05.1f}_sig{tether_stress_pa / 1e9:.3f}"
            .replace('.', 'p'))


def _run_case(job: Dict[str, Any]) -> Dict[str, Any]:
    """Run one TEF case (process-pool worker); model output goes to
    ``run.log`` in the case folder. Failures are returned, not raised."""
    caseDir = Path(job['case_dir'])
    caseDir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    try:
        with open(caseDir / 'run.log', 'w', encoding='utf-8') as log, \
                contextlib.redirect_stdout(log):
            TefCaseRunner(
                case_dir=caseDir,
                design=DesignVariables(case_name=caseDir.name,
                                       **job['design']),
                inputs_dir=Path(job['inputs_dir']),
                validate=False,
                run_storage_sizing=True,
                overwrite=True,
                verbose=False,
                **job['case_options'],
            ).run()
        error = None
    except Exception:  # recorded per case so one failure keeps the grid
        error = traceback.format_exc()
        (caseDir / 'error.txt').write_text(error, encoding='utf-8')
    runtime = time.perf_counter() - start
    write_yaml({'runtime_s': runtime, 'error': error},
               caseDir / 'case_status.yml')
    return {'case_dir': str(caseDir), 'runtime_s': runtime, 'error': error}


def _case_record(kite: str, case_dir: Path) -> Dict[str, Any]:
    """One grid-surface row read back from a finished case folder."""
    design = load_yaml(case_dir / 'design.yml')['design_variables']
    status = load_yaml(case_dir / 'case_status.yml')
    record = {
        'kite': kite,
        'case': case_dir.name,
        'flat_area_m2': design['flat_area_m2'],
        'allowable_tether_stress_gpa':
            design['allowable_tether_stress_pa'] / 1e9,
        'runtime_min': status['runtime_s'] / 60,
        'error': bool(status['error']),
    }
    if status['error']:
        return record
    summary = load_yaml(case_dir / 'tef_summary.yml')
    scaling = summary['scaling']
    powers = summary['power_accounting']
    record.update({
        'airborne_mass_kg': scaling.get('total_airborne_mass_kg'),
        'max_tether_force_kn': scaling.get('max_tether_force_n', 0) / 1e3,
        'tether_diameter_mm': scaling.get('tether_diameter_m', 0) * 1e3,
        'aep_mwh': summary['performance']['aep_mwh'],
        'capacity_factor': summary['performance']['capacity_factor'],
        'rated_power_kw': powers['rated_cycle_electrical_power_w'] / 1e3,
        'peak_mech_power_kw': (powers['peak_mechanical_power_w'] or 0) / 1e3,
        'sizing_peak_power_kw': (powers.get('peak_mechanical_power_sizing_w')
                                 or powers['peak_mechanical_power_w']
                                 or 0) / 1e3,
        'wing_mass_kg': scaling.get('wing_and_bridle_mass_kg'),
        'icc_keur': summary['economics']['icc_eur'] / 1e3,
        'omc_keur_per_year': summary['economics']['omc_eur_per_year'] / 1e3,
        'lcoe_eur_per_mwh': summary['economics']['lcoe_eur_per_mwh'],
    })
    return record


def run_kite_study(study_config_path: Path) -> pd.DataFrame:
    """Evaluate every kite over the (wing area x tether stress) grid.

    Writes, under the study's ``results_dir``: per kite an ``inputs/``
    folder (with the shared QSM override applied) and one folder per
    grid case, plus ``grid_surface.csv`` (all cases) and
    ``optimum_by_kite.csv`` / ``.yml`` (lowest LCoE per kite, flagged
    when it lies on a grid bound).

    Returns:
        The optimum table (one row per kite).
    """
    config = load_kite_study_config(study_config_path)
    resultsDir = config.results_dir
    studyStart = time.perf_counter()

    kiteInputs: Dict[str, Path] = {}
    for kite, baseDir in config.kites.items():
        inputs = prepare_study_inputs(baseDir,
                                      resultsDir / kite / 'inputs')
        for path, override in (
                (inputs.qsm_settings, config.qsm_override),
                (inputs.scaling_settings,
                 config.kite_scaling_overrides[kite])):
            if override:
                settings = load_yaml(path)
                _deep_update(settings, override)
                write_yaml(settings, path)
        kiteInputs[kite] = inputs.inputs_dir

    # Interleave kites so partial results cover all of them evenly.
    jobs, caseDirs = [], []
    for area, stress, kite in itertools.product(
            config.flat_areas_m2, config.tether_stresses_pa, config.kites):
        caseDir = resultsDir / kite / _case_name(area, stress)
        caseDirs.append((kite, caseDir))
        status = caseDir / 'case_status.yml'
        if (config.reuse_cases and status.exists()
                and not load_yaml(status)['error']):
            continue
        jobs.append({
            'case_dir': str(caseDir),
            'inputs_dir': str(kiteInputs[kite]),
            'case_options': config.case_options,
            'design': {**config.design_defaults,
                       'flat_area_m2': area,
                       'allowable_tether_stress_pa': stress},
        })

    print(f"{len(config.kites)} kites x {len(config.flat_areas_m2)} areas "
          f"x {len(config.tether_stresses_pa)} stresses = "
          f"{len(caseDirs)} cases ({len(jobs)} to run), "
          f"{config.workers} workers")
    if jobs:
        for var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                    'MKL_NUM_THREADS'):
            os.environ.setdefault(var, '1')
        done = 0
        with ProcessPoolExecutor(max_workers=config.workers) as pool:
            futures = [pool.submit(_run_case, job) for job in jobs]
            for future in as_completed(futures):
                outcome = future.result()
                done += 1
                caseDir = Path(outcome['case_dir'])
                state = 'FAILED' if outcome['error'] else 'ok'
                print(f"  [{done}/{len(jobs)}] {caseDir.parent.name}/"
                      f"{caseDir.name}: {state} "
                      f"({outcome['runtime_s'] / 60:.1f} min)", flush=True)

    frame = pd.DataFrame([_case_record(kite, caseDir)
                          for kite, caseDir in caseDirs])
    frame.to_csv(resultsDir / 'grid_surface.csv', index=False)

    optima = []
    feasible = frame[~frame['error']]
    for kite in config.kites:
        rows = feasible[feasible['kite'] == kite]
        if rows.empty:
            continue
        best = rows.loc[rows['lcoe_eur_per_mwh'].idxmin()].to_dict()
        best['area_on_bound'] = best['flat_area_m2'] in (
            min(config.flat_areas_m2), max(config.flat_areas_m2))
        best['stress_on_bound'] = best['allowable_tether_stress_gpa'] * 1e9 \
            in (min(config.tether_stresses_pa), max(config.tether_stresses_pa))
        optima.append(best)
    optimum = pd.DataFrame(optima)
    optimum.to_csv(resultsDir / 'optimum_by_kite.csv', index=False)
    write_yaml({
        'metadata': {'name': config.name,
                     'generated_by': 'TEF kite comparison study'},
        'timing': {
            'total_wall_min': (time.perf_counter() - studyStart) / 60,
            'case_cpu_min': float(frame['runtime_min'].sum()),
        },
        'n_failed_cases': int(frame['error'].sum()),
        'optima': optima,
    }, resultsDir / 'optimum_by_kite.yml')
    return optimum
