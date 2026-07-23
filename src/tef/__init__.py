"""TEF: techno-economic framework for soft-wing AWE systems."""

from tef.pipeline import (
    CasePaths,
    StudyInputs,
    TefCaseResult,
    TefCaseRunner,
    evaluate_design,
    prepare_study_inputs,
)
from tef.storage_sizing import update_storage_capacity_after_power_curves
from tef.system_scaling import (
    DesignVariables,
    ScaledSystemQuantities,
    ScalingSettings,
    build_scaled_system,
    compute_scaled_quantities,
    scaling_summary_section,
)
from tef.design_space import DesignSpace, DesignVariableSpec
from tef.optimisation import (
    OptimisationResult,
    grid_optimise,
    optimise,
    solve_monotonic_input,
)
from tef.design_studies import (
    DesignEvaluator,
    load_design_study_config,
    run_design_optimisation,
)
from tef.studies import run_area_sweep, run_generator_power_sweep
from tef.wrappers import AepRunner, AwespaPowerRunner, EcomoRunner

__all__ = [
    'CasePaths', 'StudyInputs', 'TefCaseResult', 'TefCaseRunner',
    'evaluate_design', 'prepare_study_inputs',
    'update_storage_capacity_after_power_curves',
    'DesignVariables', 'ScaledSystemQuantities', 'ScalingSettings',
    'build_scaled_system', 'compute_scaled_quantities',
    'scaling_summary_section',
    'DesignSpace', 'DesignVariableSpec',
    'OptimisationResult', 'grid_optimise', 'optimise', 'solve_monotonic_input',
    'DesignEvaluator', 'load_design_study_config', 'run_design_optimisation',
    'run_area_sweep', 'run_generator_power_sweep',
    'AepRunner', 'AwespaPowerRunner', 'EcomoRunner',
]
