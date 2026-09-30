"""Aerodynamic polar study (Stage A): LCoE sensitivity of a fixed design
to the kite's maximum lift-to-drag ratio and maximum lift coefficient.

The QSM takes one fixed (CL, CD) pair for the reel-out phase, so it
cannot choose the angle of attack. This module emulates that choice:

1. Each kite is described by a parabolic polar ``CD = CD0 + k CL^2``,
   capped at ``CL_max``. ``k`` is anchored on a reference operating
   point (the TU Delft V3 reel-out pair) taken as the reference polar's
   L/D_max point, so ``k = (CD_ref / 2) / CL_ref^2`` is held fixed and
   ``CD0 = 1 / (4 k (L/D)_max^2)`` follows from each (L/D)_max.
2. Every polar is sampled at a few operating points between its best-L/D
   lift coefficient and ``CL_max``. Each operating point is a full QSM
   power curve; all of them run in parallel on one shared scaled system,
   since the aerodynamics do not change the scaling block's output.
3. Per wind speed, the operating point with the highest cycle-average
   electrical power is selected. The resulting envelope power curve
   (YAML and time histories) replaces the single QSM run of the case, so
   storage sizing, the peak-power generator sizing, AEP and EcoMo all see
   the envelope.

The reel-in (depowered) coefficients are left at their base values.

Public interface:
    ParabolicPolar, envelope_power_curves,
    load_aero_polar_study_config, run_aero_polar_study
"""

import contextlib
import copy
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from tef.io import load_yaml, project_root, write_yaml
from tef.pipeline import TefCaseRunner, _deep_update, prepare_study_inputs
from tef.system_scaling import DesignVariables, build_scaled_system
from tef.wrappers import AwespaPowerRunner


# ---------------------------------------------------------------------------
# Polar model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ParabolicPolar:
    """Parabolic kite polar ``CD = CD0 + k CL^2`` capped at ``cl_max``."""

    ld_max: float
    cl_max: float
    k: float

    @classmethod
    def anchored(cls, ld_max: float, cl_max: float,
                 reference_cl: float, reference_cd: float,
                 ) -> 'ParabolicPolar':
        """Polar whose ``k`` follows from a reference point taken as the
        L/D_max point of its own polar (there CD0 = k CL^2 = CD / 2)."""
        k = 0.5 * reference_cd / reference_cl ** 2
        return cls(ld_max=ld_max, cl_max=cl_max, k=k)

    @property
    def cd0(self) -> float:
        return 1.0 / (4.0 * self.k * self.ld_max ** 2)

    @property
    def cl_at_ld_max(self) -> float:
        return math.sqrt(self.cd0 / self.k)

    def cd(self, cl: float) -> float:
        return self.cd0 + self.k * cl ** 2

    def operating_points(self, max_points: int, min_spacing: float,
                         ) -> List[Tuple[float, float]]:
        """(CL, CD) pairs from the best-L/D lift coefficient up to
        ``cl_max``, at most ``max_points`` and at least ``min_spacing``
        apart. A polar whose ``cl_max`` lies below its best-L/D point
        is flown at ``cl_max`` only."""
        clLow = min(self.cl_at_ld_max, self.cl_max)
        span = self.cl_max - clLow
        count = max(1, min(max_points,
                           int(math.ceil(span / min_spacing - 1e-9)) + 1))
        lifts = np.linspace(clLow, self.cl_max, count) if count > 1 \
            else [self.cl_max]
        return [(round(float(cl), 3), round(self.cd(float(cl)), 4))
                for cl in lifts]

    @property
    def name(self) -> str:
        return (f"ld{self.ld_max:g}_clmax{self.cl_max:g}"
                .replace('.', 'p'))


def _operating_point_name(cl: float, cd: float) -> str:
    return f"cl{cl:.3f}_cd{cd:.4f}".replace('.', 'p')


# ---------------------------------------------------------------------------
# Envelope over operating points
# ---------------------------------------------------------------------------

def _cycle_power(entry: Dict[str, Any]) -> float:
    return float(entry['performance']['electrical_power']
                 ['average_cycle_power'])


def envelope_power_curves(sources: Sequence[Path], output_path: Path,
                          ) -> List[Dict[str, Any]]:
    """Write the per-wind-speed best of several power curves.

    All sources must share the wind resource and wind-speed grid. At each
    (profile, wind speed) the successful entry with the highest
    cycle-average electrical power is taken, together with its time
    histories. The metadata's nominal powers and cut-in wind speed are
    recomputed as the QSM defines them.

    Args:
        sources: ``power_curves.yml`` paths (``.npz`` siblings required).
        output_path: Envelope ``power_curves.yml`` to write.

    Returns:
        list: One record per (profile, wind speed) naming the selected
        source index and its cycle power.
    """
    curves = [load_yaml(source) for source in sources]
    windSpeeds = curves[0]['reference_wind_speeds']
    for source, curve in zip(sources, curves):
        if not np.allclose(curve['reference_wind_speeds'], windSpeeds):
            raise ValueError(f"Wind-speed grid of {source} differs")

    envelope = copy.deepcopy(curves[0])
    histories = [np.load(Path(s).with_suffix('.npz'), allow_pickle=True)
                 for s in sources]
    arrays: Dict[str, np.ndarray] = {}
    selection = []
    for p, profile in enumerate(envelope['power_curves']):
        profileId = profile['profile_id']
        for i in range(len(profile['wind_speed_data'])):
            candidates = [(j, curve['power_curves'][p]['wind_speed_data'][i])
                          for j, curve in enumerate(curves)]
            successful = [c for c in candidates if c[1].get('successful')]
            best, entry = max(successful or candidates,
                              key=lambda c: _cycle_power(c[1]))
            profile['wind_speed_data'][i] = copy.deepcopy(entry)
            prefix = f"p{profileId}_ws{i}_"
            for key in histories[best].files:
                if key.startswith(prefix):
                    arrays[key] = histories[best][key]
            selection.append({
                'profile_id': profileId,
                'wind_speed': float(entry['wind_speed']),
                'source': best,
                'successful': bool(entry.get('successful')),
                'cycle_power_w': _cycle_power(entry),
            })

    allEntries = [e for prof in envelope['power_curves']
                  for e in prof['wind_speed_data']]
    modelConfig = envelope['metadata']['model_config']
    modelConfig['nominal_mechanical_power'] = max(
        e['performance']['power']['average_cycle_power'] for e in allEntries)
    modelConfig['nominal_electrical_power'] = max(
        _cycle_power(e) for e in allEntries)
    firstProfile = envelope['power_curves'][0]['wind_speed_data']
    modelConfig['cut_in_wind_speed'] = next(
        (e['wind_speed'] for e in firstProfile
         if e['performance']['power']['average_cycle_power'] > 0),
        windSpeeds[0])
    envelope['metadata']['note'] = (
        'Envelope over QSM aerodynamic operating points (TEF aero polar '
        'study). ' + envelope['metadata'].get('note', ''))

    outputPath = Path(output_path)
    write_yaml(envelope, outputPath)
    np.savez_compressed(outputPath.with_suffix('.npz'), **arrays)
    return selection


# ---------------------------------------------------------------------------
# One QSM power curve per operating point (process-pool worker)
# ---------------------------------------------------------------------------

def _run_operating_point(job: Dict[str, Any]) -> Tuple[str, float]:
    """Run one QSM power curve with the reel-out aerodynamics overridden.
    Model output goes to ``qsm.log`` in the operating-point folder."""
    opDir = Path(job['op_dir'])
    opDir.mkdir(parents=True, exist_ok=True)
    settings = load_yaml(job['qsm_settings'])
    _deep_update(settings, job['qsm_override'])
    settingsPath = opDir / 'inertiafree-qsm_settings.yml'
    write_yaml(settings, settingsPath)

    start = time.perf_counter()
    with open(opDir / 'qsm.log', 'w', encoding='utf-8') as log, \
            contextlib.redirect_stdout(log):
        AwespaPowerRunner(
            system_path=Path(job['system']),
            qsm_settings_path=settingsPath,
            wind_resource_path=Path(job['wind_resource']),
            output_power_curves_path=opDir / 'power_curves.yml',
            validate=False,
            verbose=False,
        ).run()
    return job['name'], time.perf_counter() - start


# ---------------------------------------------------------------------------
# Study configuration and driver
# ---------------------------------------------------------------------------

@dataclass
class AeroPolarStudyConfig:
    """Parsed aero polar study configuration."""

    name: str
    base_config_dir: Path
    results_dir: Path
    design: Dict[str, Any]
    reference_cl: float
    reference_cd: float
    ld_max: List[float]
    cl_max: List[float]
    max_operating_points: int
    min_cl_spacing: float
    workers: int
    reuse_operating_points: bool
    raw: Dict[str, Any] = field(default_factory=dict)

    def polars(self) -> List[ParabolicPolar]:
        return [ParabolicPolar.anchored(ld, cl, self.reference_cl,
                                        self.reference_cd)
                for ld in self.ld_max for cl in self.cl_max]


def load_aero_polar_study_config(path: Path) -> AeroPolarStudyConfig:
    """Load an aero polar study YAML. Relative directories resolve
    against the TEF project root."""
    data = load_yaml(path)
    root = project_root()

    def resolve(path_str: str) -> Path:
        candidate = Path(path_str)
        return candidate if candidate.is_absolute() else root / candidate

    polar = data['polar']
    execution = data.get('execution') or {}
    return AeroPolarStudyConfig(
        name=data.get('metadata', {}).get('name', Path(path).stem),
        base_config_dir=resolve(data['base_config_dir']),
        results_dir=resolve(data['results_dir']),
        design={k: float(v) for k, v in data['design'].items()},
        reference_cl=float(polar['reference']['cl']),
        reference_cd=float(polar['reference']['cd']),
        ld_max=[float(v) for v in polar['ld_max']],
        cl_max=[float(v) for v in polar['cl_max']],
        max_operating_points=int(polar.get('max_operating_points', 4)),
        min_cl_spacing=float(polar.get('min_cl_spacing', 0.1)),
        workers=int(execution.get('workers', max(1, os.cpu_count() - 2))),
        reuse_operating_points=bool(
            execution.get('reuse_operating_points', True)),
        raw=data,
    )


def _without_metadata(system: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in system.items() if k != 'metadata'}


def run_aero_polar_study(study_config_path: Path) -> pd.DataFrame:
    """Run the aero polar study for one fixed design.

    Writes, under the study's ``results_dir``: the shared scaled system
    (``_system/``), one folder per QSM operating point
    (``operating_points/``), one complete case per polar (``polars/``,
    with ``aero_polar.yml`` recording the polar and the per-wind-speed
    selection), and ``aero_polar_summary.csv``.

    Returns:
        The summary table (one row per polar).
    """
    config = load_aero_polar_study_config(study_config_path)
    resultsDir = config.results_dir
    inputs = prepare_study_inputs(config.base_config_dir,
                                  resultsDir / 'inputs')
    design = DesignVariables(case_name='design', **config.design)
    studyStart = time.perf_counter()

    # Aerodynamics do not enter the scaling block, so one scaled system
    # serves every operating point (checked again per case below).
    sharedSystem = resultsDir / '_system' / 'system.yml'
    sharedSystem.parent.mkdir(parents=True, exist_ok=True)
    build_scaled_system(
        base_system_path=inputs.base_system,
        scaling_settings_path=inputs.scaling_settings,
        design=design,
        output_system_path=sharedSystem,
        validate=False,
    )

    polars = config.polars()
    points: Dict[str, Tuple[float, float]] = {}
    polarPoints: Dict[str, List[str]] = {}
    for polar in polars:
        names = []
        for cl, cd in polar.operating_points(config.max_operating_points,
                                             config.min_cl_spacing):
            name = _operating_point_name(cl, cd)
            points[name] = (cl, cd)
            names.append(name)
        polarPoints[polar.name] = names

    opRoot = resultsDir / 'operating_points'
    cycleOverride: Dict[str, Any] = {}
    if design.minimum_tether_force_n is not None:
        cycleOverride = {'cycle': {
            'minimum_tether_force': design.minimum_tether_force_n}}
    jobs = []
    for name, (cl, cd) in points.items():
        if (config.reuse_operating_points
                and (opRoot / name / 'power_curves.npz').exists()):
            continue
        override = {'aerodynamics': {
            'kite_lift_coefficient_reel_out': cl,
            'kite_drag_coefficient_reel_out': cd}}
        _deep_update(override, cycleOverride)
        jobs.append({
            'name': name,
            'op_dir': str(opRoot / name),
            'system': str(sharedSystem),
            'qsm_settings': str(inputs.qsm_settings),
            'wind_resource': str(inputs.wind_resource),
            'qsm_override': override,
        })

    print(f"{len(polars)} polars, {len(points)} operating points "
          f"({len(jobs)} to run, {len(points) - len(jobs)} reused), "
          f"{config.workers} workers")
    runtimes: Dict[str, float] = {}
    qsmStart = time.perf_counter()
    if jobs:
        # One thread per QSM process; the pool supplies the parallelism.
        for var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                    'MKL_NUM_THREADS'):
            os.environ.setdefault(var, '1')
        with ProcessPoolExecutor(max_workers=config.workers) as pool:
            for name, seconds in pool.map(_run_operating_point, jobs):
                runtimes[name] = seconds
                print(f"  {name}: {seconds / 60:.1f} min")
    qsmWall = time.perf_counter() - qsmStart
    write_yaml({name: {'cl': points[name][0], 'cd': points[name][1],
                       'runtime_s': runtimes.get(name)}
                for name in points},
               opRoot / 'operating_points.yml')

    records = []
    for polar in polars:
        names = polarPoints[polar.name]
        sources = [opRoot / n / 'power_curves.yml' for n in names]
        caseDir = resultsDir / 'polars' / polar.name
        selectionHolder: Dict[str, Any] = {}

        def builder(system_path, _qsm_settings_path, output_path,
                    sources=sources, holder=selectionHolder):
            if (_without_metadata(load_yaml(system_path))
                    != _without_metadata(load_yaml(sharedSystem))):
                raise RuntimeError(
                    f"Scaled system of {system_path} differs from the "
                    f"shared system the operating points were run on")
            holder['selection'] = envelope_power_curves(sources,
                                                        output_path)

        caseStart = time.perf_counter()
        result = TefCaseRunner(
            case_dir=caseDir,
            design=DesignVariables(case_name=polar.name, **config.design),
            inputs_dir=inputs.inputs_dir,
            validate=False,
            run_storage_sizing=True,
            overwrite=True,
            verbose=False,
            power_curve_builder=builder,
        ).run()
        chainSeconds = time.perf_counter() - caseStart

        selection = selectionHolder['selection']
        for record in selection:
            cl, cd = points[names[record.pop('source')]]
            record.update({'cl': cl, 'cd': cd})
        write_yaml({
            'polar': {'ld_max': polar.ld_max, 'cl_max': polar.cl_max,
                      'k': polar.k, 'cd0': polar.cd0,
                      'cl_at_ld_max': polar.cl_at_ld_max},
            'operating_points': [
                {'cl': points[n][0], 'cd': points[n][1],
                 'path': os.path.relpath(opRoot / n, caseDir)
                 .replace(os.sep, '/')} for n in names],
            'selection': selection,
        }, caseDir / 'aero_polar.yml')

        selectedCl = [r['cl'] for r in selection if r['cycle_power_w'] > 0]
        records.append({
            'polar': polar.name,
            'ld_max': polar.ld_max,
            'cl_max': polar.cl_max,
            'cd0': polar.cd0,
            'cl_at_ld_max': polar.cl_at_ld_max,
            'n_operating_points': len(names),
            'selected_cl_min': min(selectedCl) if selectedCl else None,
            'selected_cl_max': max(selectedCl) if selectedCl else None,
            'aep_mwh': result.aep_mwh,
            'capacity_factor': result.capacity_factor,
            'rated_power_kw': result.emergent_rated_power_w / 1e3,
            'peak_mech_power_kw': (result.peak_mechanical_power_w or 0) / 1e3,
            'storage_kwh': (result.storage_capacity_wh or 0) / 1e3,
            'icc_keur': result.icc_eur / 1e3,
            'omc_keur_per_year': result.omc_eur_per_year / 1e3,
            'lcoe_eur_per_mwh': result.lcoe_eur_per_mwh,
            'qsm_runtime_min': sum(runtimes.get(n, 0.0) for n in names) / 60,
            'chain_runtime_s': chainSeconds,
        })

    frame = pd.DataFrame(records)
    frame.to_csv(resultsDir / 'aero_polar_summary.csv', index=False)
    write_yaml({
        'metadata': {'name': config.name,
                     'generated_by': 'TEF aero polar study'},
        'design': config.design,
        'timing': {'qsm_wall_min': qsmWall / 60,
                   'qsm_cpu_min': sum(runtimes.values()) / 60,
                   'total_wall_min':
                       (time.perf_counter() - studyStart) / 60},
        'polars': records,
    }, resultsDir / 'aero_polar_summary.yml')
    return frame


# ---------------------------------------------------------------------------
# Fixed-CL polar grid (polar efficiency x operating lift coefficient)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScaledPolar:
    """Kite polar scaled by one efficiency factor around a design lift
    coefficient: ``CD = CL_d / (2 E) * (1 + (CL / CL_d)^2)``.

    ``E`` is the maximum lift-to-drag ratio, reached at ``CL_d``. Raising
    ``E`` lowers the parasitic (``CD0 = CL_d / (2E)``) and induced
    (``k = 1 / (2 E CL_d)``) drag together, as a cleaner, higher aspect
    ratio kite would, so a better polar also helps at high lift.
    """

    ld_max: float
    cl_design: float

    @property
    def cd0(self) -> float:
        return self.cl_design / (2.0 * self.ld_max)

    @property
    def k(self) -> float:
        return 1.0 / (2.0 * self.ld_max * self.cl_design)

    def cd(self, cl: float) -> float:
        return self.cd0 + self.k * cl ** 2

    @staticmethod
    def implied_ld_max(cl: float, cd: float, cl_design: float) -> float:
        """L/D_max of the family member passing through (CL, CD)."""
        return cl_design / (2.0 * cd) * (1.0 + (cl / cl_design) ** 2)


def run_polar_grid_study(study_config_path: Path) -> pd.DataFrame:
    """Evaluate a fixed design over a grid of polar efficiency (L/D_max)
    and fixed reel-out lift coefficient, one complete case per node.

    Each case flies one (CL, CD) pair at every wind speed, as the QSM
    does; CD follows from :class:`ScaledPolar`. Cases run in parallel.
    Writes ``polar_grid.csv`` and ``polar_grid_summary.yml`` (with the
    reference kites' implied L/D_max) under the study's ``results_dir``.

    With a ``grid.flat_area_m2`` list, every cell is evaluated at each
    wing area and ``polar_grid_optima.csv`` holds the lowest-LCoE area
    per cell. With ``reel_in.cl``, the reel-in coefficients follow the
    same polar at that lift coefficient.

    Returns:
        The grid table (one row per evaluated case).
    """
    from tef.kite_studies import _case_record, _run_case

    data = load_yaml(study_config_path)
    root = project_root()
    resultsDir = root / data['results_dir']
    polarCfg = data['polar']
    clDesign = float(polarCfg['cl_design'])
    ldValues = [float(v) for v in polarCfg['ld_max']]
    clValues = [float(v) for v in polarCfg['cl']]
    execution = data.get('execution') or {}
    workers = int(execution.get('workers', max(1, os.cpu_count() - 2)))
    reuse = bool(execution.get('reuse_cases', True))
    studyStart = time.perf_counter()

    inputs = prepare_study_inputs(root / data['base_config_dir'],
                                  resultsDir / 'inputs')
    if data.get('qsm_override'):
        settings = load_yaml(inputs.qsm_settings)
        _deep_update(settings, data['qsm_override'])
        write_yaml(settings, inputs.qsm_settings)

    # Optional wing-area grid per cell (the design's area otherwise); the
    # reel-in coefficients optionally follow each wing's own polar.
    areas = [float(a) for a in (data.get('grid') or {}).get(
        'flat_area_m2', [data['design']['flat_area_m2']])]
    reelInCl = (data.get('reel_in') or {}).get('cl')

    nodes, jobs = [], []
    for ldMax in ldValues:
        polar = ScaledPolar(ldMax, clDesign)
        for cl in clValues:
            cd = round(polar.cd(cl), 5)
            aero = {'kite_lift_coefficient_reel_out': cl,
                    'kite_drag_coefficient_reel_out': cd}
            if reelInCl is not None:
                aero['kite_lift_coefficient_reel_in'] = float(reelInCl)
                aero['kite_drag_coefficient_reel_in'] = round(
                    polar.cd(float(reelInCl)), 5)
            cellDir = resultsDir / (f"E{ldMax:04.1f}_CL{cl:.2f}"
                                    .replace('.', 'p'))
            for area in areas:
                caseDir = (cellDir / f"S{area:05.1f}".replace('.', 'p')
                           if len(areas) > 1 else cellDir)
                nodes.append((ldMax, cl, cd, caseDir))
                status = caseDir / 'case_status.yml'
                if (reuse and status.exists()
                        and not load_yaml(status)['error']):
                    continue
                options = dict(data.get('case_options') or {})
                options['qsm_override'] = {'aerodynamics': dict(aero)}
                design = {k: float(v) for k, v in data['design'].items()}
                design['flat_area_m2'] = area
                jobs.append({
                    'case_dir': str(caseDir),
                    'inputs_dir': str(inputs.inputs_dir),
                    'case_options': options,
                    'design': design,
                })

    print(f"{len(ldValues)} L/D_max x {len(clValues)} CL x {len(areas)} "
          f"areas = {len(nodes)} cases ({len(jobs)} to run), "
          f"{workers} workers")
    if jobs:
        for var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                    'MKL_NUM_THREADS'):
            os.environ.setdefault(var, '1')
        done = 0
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for outcome in pool.map(_run_case, jobs):
                done += 1
                state = 'FAILED' if outcome['error'] else 'ok'
                caseDir = Path(outcome['case_dir'])
                print(f"  [{done}/{len(jobs)}] "
                      f"{caseDir.parent.name}/{caseDir.name}: {state} "
                      f"({outcome['runtime_s'] / 60:.1f} min)", flush=True)

    records = []
    for ldMax, cl, cd, caseDir in nodes:
        record = _case_record('polar', caseDir)
        record.update({'ld_max': ldMax, 'cl': cl, 'cd': cd,
                       'kite_ld': cl / cd})
        records.append(record)
    frame = pd.DataFrame(records)
    frame.to_csv(resultsDir / 'polar_grid.csv', index=False)
    # Lowest-LCoE wing area per (L/D_max, CL) cell.
    feasible = frame[~frame['error']]
    optima = feasible.loc[feasible.groupby(['ld_max', 'cl'])
                          ['lcoe_eur_per_mwh'].idxmin()].copy()
    optima['area_on_bound'] = optima['flat_area_m2'].isin(
        [min(areas), max(areas)]) if len(areas) > 1 else False
    optima.to_csv(resultsDir / 'polar_grid_optima.csv', index=False)

    kites = {name: {'cl': float(k['cl']), 'cd': float(k['cd']),
                    'implied_ld_max': ScaledPolar.implied_ld_max(
                        float(k['cl']), float(k['cd']), clDesign)}
             for name, k in (data.get('reference_kites') or {}).items()}
    write_yaml({
        'metadata': {'name': data.get('metadata', {}).get('name', ''),
                     'generated_by': 'TEF polar grid study'},
        'polar': {'family': 'CD = CL_d/(2E) * (1 + (CL/CL_d)^2)',
                  'cl_design': clDesign, 'reel_in_cl': reelInCl},
        'flat_areas_m2': areas,
        'reference_kites': kites,
        'timing': {'total_wall_min': (time.perf_counter() - studyStart) / 60,
                   'case_cpu_min': float(frame['runtime_min'].sum())},
        'n_failed_cases': int(frame['error'].sum()),
    }, resultsDir / 'polar_grid_summary.yml')
    return frame
