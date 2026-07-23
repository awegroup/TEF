"""TEF studies: study configuration loading, sequential case
execution, combined summaries, and the two built-in sweeps (wing area,
generator specific power).

Public interface:
    run_area_sweep(study_config_path)
    run_generator_power_sweep(study_config_path)
    load_study_config, make_design, run_cases (for custom studies)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from tef.io import load_yaml, project_root, write_yaml
from tef.pipeline import (
    TefCaseResult,
    TefCaseRunner,
    prepare_study_inputs,
)
from tef.system_scaling import DesignVariables

# DesignVariables fields that may be set through design_defaults.
_DEFAULTABLE_FIELDS = (
    'max_wing_loading_n_m2_projected',
    'allowable_tether_stress_pa',
    'generator_max_power_w',
    'generator_specific_power_w_m2_projected',
    'rated_power_w',
    'crest_factor',
    'max_tether_speed_m_s',
    'tether_length_m',
    'minimum_tether_force_n',
)

CSV_COLUMNS = [
    'case_name',
    'flat_area_m2',
    'projected_area_m2',
    'wing_and_bridle_mass_kg',
    'kcu_mass_kg',
    'sensor_mass_kg',
    'total_airborne_mass_kg',
    'max_tether_force_n',
    'tether_diameter_m',
    'generator_max_power_w',
    'storage_capacity_wh',
    'aep_mwh',
    'capacity_factor',
    'lcoe_eur_per_mwh',
    'icc_eur',
    'omc_eur_per_year',
    'status',
]


# ---------------------------------------------------------------------------
# Study configuration
# ---------------------------------------------------------------------------

@dataclass
class StudyConfig:
    """Parsed study configuration."""

    name: str
    base_config_dir: Path
    results_dir: Path
    design_defaults: Dict[str, Any] = field(default_factory=dict)
    validate: bool = True
    run_storage_sizing: bool = True
    continue_on_failure: bool = True
    overwrite: bool = False
    raw: Dict[str, Any] = field(default_factory=dict)


def load_study_config(study_config_path: Path) -> StudyConfig:
    """Load a study YAML file. Relative directories resolve against the
    TEF project root."""
    data = load_yaml(study_config_path)
    root = project_root()

    def resolve(path_str: str) -> Path:
        path = Path(path_str)
        return path if path.is_absolute() else root / path

    defaults = {k: v for k, v in (data.get('design_defaults') or {}).items()
                if v is not None}
    unknown = set(defaults) - set(_DEFAULTABLE_FIELDS)
    if unknown:
        raise ValueError(
            f"Unknown design_defaults keys in {study_config_path}: "
            f"{sorted(unknown)}")

    execution = data.get('execution') or {}
    return StudyConfig(
        name=data.get('metadata', {}).get('name',
                                          Path(study_config_path).stem),
        base_config_dir=resolve(data['base_config_dir']),
        results_dir=resolve(data['results_dir']),
        design_defaults=defaults,
        validate=execution.get('validate', True),
        run_storage_sizing=execution.get('run_storage_sizing', True),
        continue_on_failure=execution.get('continue_on_failure', True),
        overwrite=execution.get('overwrite', False),
        raw=data,
    )


def make_design(config: StudyConfig, case_name: str,
                **overrides: Any) -> DesignVariables:
    """Build the DesignVariables for one case from the study defaults
    plus per-case overrides."""
    kwargs = dict(config.design_defaults)
    kwargs.update(overrides)
    return DesignVariables(case_name=case_name, **kwargs)


# ---------------------------------------------------------------------------
# Case execution and summaries
# ---------------------------------------------------------------------------

def case_record(result, status: str = 'ok',
                error: str = None) -> Dict[str, Any]:
    """Flatten one case outcome into a summary record.

    ``result`` is a TefCaseResult, or None for failed cases.
    """
    record = {column: None for column in CSV_COLUMNS}
    record['status'] = status
    if error is not None:
        record['error'] = error
    if result is not None:
        for column in CSV_COLUMNS:
            if column != 'status' and hasattr(result, column):
                record[column] = getattr(result, column)
    return record


def write_study_summary(records: List[Dict[str, Any]], study_name: str,
                        results_dir: Path) -> pd.DataFrame:
    """Write study_summary.yml and study_summary.csv for a study.

    Returns:
        pd.DataFrame: The summary table (one row per case).
    """
    resultsDir = Path(results_dir)
    write_yaml({
        'metadata': {'name': study_name,
                     'generated_by': 'TEF study runner'},
        'cases': records,
    }, resultsDir / 'study_summary.yml')

    frame = pd.DataFrame(records)
    known = [c for c in CSV_COLUMNS if c in frame.columns]
    frame = frame.reindex(columns=known + [c for c in frame.columns
                                           if c not in known])
    frame.to_csv(resultsDir / 'study_summary.csv', index=False)
    return frame


def run_cases(config: StudyConfig,
              cases: List[Tuple[str, DesignVariables]],
              ) -> Tuple[List[Optional[TefCaseResult]],
                         List[Dict[str, Any]]]:
    """Run a list of (case_name, design) pairs sequentially against
    the study's shared inputs directory.

    Failed cases are recorded with status 'failed' and, when
    ``continue_on_failure`` is set, do not stop the study.

    Returns:
        Results per case (None for failures) and the flat summary
        records written to study_summary.{yml,csv}.
    """
    inputs = prepare_study_inputs(config.base_config_dir,
                                  config.results_dir / 'inputs')
    results: List[Optional[TefCaseResult]] = []
    records: List[Dict[str, Any]] = []
    for caseName, design in cases:
        print(f"\n=== TEF case: {caseName} ===")
        runner = TefCaseRunner(
            case_dir=config.results_dir / caseName,
            design=design,
            inputs_dir=inputs.inputs_dir,
            validate=config.validate,
            run_storage_sizing=config.run_storage_sizing,
            overwrite=config.overwrite,
        )
        try:
            result = runner.run()
        except Exception as exc:  # noqa: BLE001 - study must record failures
            if not config.continue_on_failure:
                raise
            print(f"Case {caseName} FAILED: {exc}")
            results.append(None)
            record = case_record(None, status='failed', error=str(exc))
            record['case_name'] = caseName
            record['flat_area_m2'] = design.flat_area_m2
            records.append(record)
            continue
        results.append(result)
        records.append(case_record(result))

    write_study_summary(records, config.name, config.results_dir)
    return results, records


# ---------------------------------------------------------------------------
# Built-in studies
# ---------------------------------------------------------------------------

def area_case_name(flat_area_m2: float) -> str:
    return f"area_{int(round(flat_area_m2)):03d}"


def run_area_sweep(study_config_path: Path,
                   ) -> List[Optional[TefCaseResult]]:
    """Run one full TEF case per wing area in the study config."""
    config = load_study_config(study_config_path)
    areas = config.raw['sweep']['flat_area_m2']
    cases = [
        (area_case_name(area),
         make_design(config, area_case_name(area), flat_area_m2=float(area)))
        for area in areas
    ]
    results, _ = run_cases(config, cases)
    return results


def generator_case_name(flat_area_m2: float, specific_power: float) -> str:
    return (f"area_{int(round(flat_area_m2)):03d}"
            f"_sp_{int(round(specific_power)):04d}")


def run_generator_power_sweep(study_config_path: Path,
                              ) -> List[Optional[TefCaseResult]]:
    """Run the area x specific-generator-power grid and write the
    LCoE-optimal generator size per area to optimum_by_area.csv."""
    config = load_study_config(study_config_path)
    areas = config.raw['areas_m2']
    specificPowers = (
        config.raw['generator_specific_power_w_m2_projected']['values'])

    cases = []
    for area in areas:
        for specificPower in specificPowers:
            name = generator_case_name(area, specificPower)
            cases.append((name, make_design(
                config, name,
                flat_area_m2=float(area),
                generator_specific_power_w_m2_projected=float(
                    specificPower))))

    results, records = run_cases(config, cases)

    frame = pd.DataFrame([r for r in records if r['status'] == 'ok'])
    if not frame.empty:
        optimum = frame.loc[
            frame.groupby('flat_area_m2')['lcoe_eur_per_mwh'].idxmin()]
        optimum.to_csv(config.results_dir / 'optimum_by_area.csv',
                       index=False)
    return results
