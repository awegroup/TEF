"""Design-optimisation study: minimise LCoE over a design space at each
of several fixed generator power limits.

This is the one module where the optimiser meets the models. For each
generator ``max_power`` on the outer axis it builds the LCoE objective
from :func:`tef.pipeline.evaluate_design`, runs the optimiser from
:mod:`tef.optimisation` over the inner :class:`DesignSpace` (wing area and
tether stress), persists the optimum as a full case, and writes the grid
surface and a combined optimum-per-generator table. The rated cycle power
is emergent, not an input.

The optimiser never sees the models: it is handed a numeric objective and
a design space. Swapping a model behind ``evaluate_design`` (a different
mass, cost or power model, selected in the base config) does not touch
this module's optimisation logic.

Public interface:
    DesignEvaluator
    load_design_study_config, run_design_optimisation
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from tef.design_space import DesignSpace, DesignVariableSpec
from tef.io import load_yaml, project_root, write_yaml
from tef.optimisation import optimise
from tef.pipeline import (
    TefCaseResult,
    TefCaseRunner,
    evaluate_design,
    prepare_study_inputs,
)
from tef.system_scaling import DesignVariables


# ---------------------------------------------------------------------------
# Objective builder (design vector -> LCoE at a fixed generator limit)
# ---------------------------------------------------------------------------

@dataclass
class DesignEvaluator:
    """Builds the LCoE objective for one fixed generator power limit.

    The generator ``max_power`` is held at the study's outer-axis value;
    the inner design variables (wing area, tether stress) are optimised
    and the rated cycle power emerges from the QSM.

    Attributes mirror the study configuration; see
    :func:`run_design_optimisation`.
    """

    design_space: DesignSpace
    generator_max_power_w: float
    scratch_dir: Path
    base_config_dir: Path
    run_storage_sizing: bool = True

    def _evaluate(self, design: DesignVariables) -> TefCaseResult:
        """Run the full chain for one design in the shared scratch dir."""
        return evaluate_design(
            design, work_dir=self.scratch_dir,
            base_config_dir=self.base_config_dir,
            validate=False, verbose=False,
            run_storage_sizing=self.run_storage_sizing)

    def solved_design(self, vector, case_name: str = 'scratch',
                      ) -> DesignVariables:
        """Return the design for a vector at the fixed generator limit."""
        return self.design_space.to_design(
            vector, case_name=case_name,
            generator_max_power_w=self.generator_max_power_w)

    def lcoe(self, vector) -> float:
        """LCoE objective for one design vector."""
        return self._evaluate(self.solved_design(vector)).lcoe_eur_per_mwh


# ---------------------------------------------------------------------------
# Study configuration
# ---------------------------------------------------------------------------

@dataclass
class DesignStudyConfig:
    """Parsed design-optimisation study configuration."""

    name: str
    base_config_dir: Path
    results_dir: Path
    generator_max_power_w: List[float]
    design_space: DesignSpace
    grid_points: Any
    refine_method: Optional[str]
    run_storage_sizing: bool
    overwrite: bool
    raw: Dict[str, Any] = field(default_factory=dict)


def _coerce_number(value):
    # YAML 1.1 parses an unsigned exponent (e.g. 4.0e8) as a string, so
    # coerce numeric-looking design defaults to float.
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return value
    return value


def load_design_study_config(study_config_path: Path) -> DesignStudyConfig:
    """Load a design-optimisation study YAML. Relative directories
    resolve against the TEF project root."""
    data = load_yaml(study_config_path)
    root = project_root()

    def resolve(path_str: str) -> Path:
        path = Path(path_str)
        return path if path.is_absolute() else root / path

    variables = tuple(
        DesignVariableSpec(name=item['name'],
                           lower=float(item['lower']),
                           upper=float(item['upper']))
        for item in data['variables'])
    fixed = {k: _coerce_number(v)
             for k, v in (data.get('design_defaults') or {}).items()
             if v is not None}
    designSpace = DesignSpace(variables=variables, fixed=fixed)

    refine = (data.get('refine') or {}).get('method')
    if refine in (None, 'none'):
        refine = None
    execution = data.get('execution') or {}

    return DesignStudyConfig(
        name=data.get('metadata', {}).get('name',
                                          Path(study_config_path).stem),
        base_config_dir=resolve(data['base_config_dir']),
        results_dir=resolve(data['results_dir']),
        generator_max_power_w=[float(p)
                               for p in data['generator_max_power_w']],
        design_space=designSpace,
        grid_points=data.get('grid_points', 5),
        refine_method=refine,
        run_storage_sizing=execution.get('run_storage_sizing', True),
        overwrite=execution.get('overwrite', True),
        raw=data,
    )


# ---------------------------------------------------------------------------
# Result records
# ---------------------------------------------------------------------------

def _generator_dir_name(generator_max_power_w: float) -> str:
    return f"gen_{int(round(generator_max_power_w / 1000)):04d}kW"


def optimum_record(generator_max_power_w: float, design: DesignVariables,
                   result: TefCaseResult) -> Dict[str, Any]:
    """Flatten one optimum into a summary row: the generator limit, the
    optimum design, the emergent rated power, both crest factors, the
    cost-driver quantities and LCoE."""
    emergentRated = result.emergent_rated_power_w
    peak = result.peak_mechanical_power_w
    return {
        'generator_max_power_kw': generator_max_power_w / 1000.0,
        'emergent_rated_power_kw': emergentRated / 1000.0,
        'peak_mechanical_power_kw': peak / 1000.0 if peak else None,
        'crest_peak_over_rated': peak / emergentRated
        if peak and emergentRated else None,
        'crest_limit_over_rated': generator_max_power_w / emergentRated
        if emergentRated else None,
        'flat_area_m2': result.flat_area_m2,
        'allowable_tether_stress_gpa':
            (design.allowable_tether_stress_pa / 1e9
             if design.allowable_tether_stress_pa is not None else None),
        'tether_diameter_mm': result.tether_diameter_m * 1000.0,
        # Cost-driver 1: mass.
        'total_airborne_mass_kg': result.total_airborne_mass_kg,
        'wing_and_bridle_mass_kg': result.wing_and_bridle_mass_kg,
        'kcu_mass_kg': result.kcu_mass_kg,
        # Cost-driver 2: storage.
        'storage_capacity_wh': result.storage_capacity_wh,
        # Cost-driver 3: tether replacement enters via LCoE/OMC below.
        'aep_mwh': result.aep_mwh,
        'capacity_factor': result.capacity_factor,
        'lcoe_eur_per_mwh': result.lcoe_eur_per_mwh,
        'icc_eur': result.icc_eur,
        'omc_eur_per_year': result.omc_eur_per_year,
    }


def write_grid_surface(evaluations, design_space: DesignSpace,
                       path: Path) -> None:
    """Write the full grid surface (one row per evaluated vector) so the
    LCoE surface can be plotted."""
    rows = []
    for evaluation in evaluations:
        row = dict(zip(design_space.names, evaluation.vector))
        row['lcoe_eur_per_mwh'] = evaluation.value
        row['status'] = 'ok' if evaluation.ok else 'failed'
        row['error'] = evaluation.error
        rows.append(row)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Study runner
# ---------------------------------------------------------------------------

def run_design_optimisation(study_config_path: Path) -> pd.DataFrame:
    """Optimise the design at each generator limit and write the results.

    For each generator ``max_power`` the LCoE is minimised over the design
    space (grid, optionally refined), the optimum is persisted as a full
    case under ``results/<study>/<gen>/optimum/``, and the grid surface is
    written next to it. A combined ``optimum_by_generator_power.csv`` and
    ``.yml`` gather the optimum of every generator limit.

    Args:
        study_config_path: Path to the design-optimisation study YAML.

    Returns:
        The combined optimum table (one row per generator limit).
    """
    config = load_design_study_config(study_config_path)
    scratchDir = config.results_dir / '_scratch'
    inputs = prepare_study_inputs(config.base_config_dir,
                                  config.results_dir / 'inputs')

    records: List[Dict[str, Any]] = []
    for generatorPower in config.generator_max_power_w:
        generatorName = _generator_dir_name(generatorPower)
        print(f"\n=== Design optimisation at "
              f"{generatorPower / 1000:g} kW generator ===")

        evaluator = DesignEvaluator(
            design_space=config.design_space,
            generator_max_power_w=generatorPower,
            scratch_dir=scratchDir,
            base_config_dir=config.base_config_dir,
            run_storage_sizing=config.run_storage_sizing,
        )

        outcome = optimise(evaluator.lcoe, config.design_space,
                           method='grid', points=config.grid_points)
        if config.refine_method is not None and outcome.succeeded:
            outcome = optimise(evaluator.lcoe, config.design_space,
                               method=config.refine_method,
                               x0=outcome.best_vector)

        generatorDir = config.results_dir / generatorName
        write_grid_surface(outcome.evaluations, config.design_space,
                           generatorDir / 'grid_surface.csv')

        if not outcome.succeeded:
            print(f"  no feasible design found at "
                  f"{generatorPower / 1000:g} kW")
            for reason in sorted({e.error for e in outcome.evaluations
                                  if e.error}):
                print(f"    reason: {reason}")
            continue

        # Persist the optimum as a full, inspectable case.
        bestDesign = evaluator.solved_design(outcome.best_vector,
                                             case_name='optimum')
        bestResult = TefCaseRunner(
            case_dir=generatorDir / 'optimum',
            design=bestDesign,
            inputs_dir=inputs.inputs_dir,
            validate=False,
            run_storage_sizing=config.run_storage_sizing,
            overwrite=config.overwrite,
            verbose=False,
        ).run()

        records.append(
            optimum_record(generatorPower, bestDesign, bestResult))
        print(f"  optimum: S={bestResult.flat_area_m2:g} m2, "
              f"LCoE={bestResult.lcoe_eur_per_mwh:.1f} EUR/MWh")

    frame = pd.DataFrame(records)
    frame.to_csv(config.results_dir / 'optimum_by_generator_power.csv',
                 index=False)
    write_yaml({'metadata': {'name': config.name,
                             'generated_by': 'TEF design optimisation'},
                'optima': records},
               config.results_dir / 'optimum_by_generator_power.yml')
    return frame
