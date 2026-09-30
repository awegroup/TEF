"""TEF case pipeline: executes the full model chain for one design
point.

Sequence: system scaling -> AWESPA power curves -> AEP -> storage
sizing -> EcoMo -> TEF summary.

File layout: the inputs shared by all cases of a study (base system,
scaling settings, QSM settings, wind resource, cost inputs, economic
settings template) live once in an ``inputs/`` directory prepared with
:func:`prepare_study_inputs`. A case directory holds only the
case-specific files: design.yml, system.yml, power_curves.yml,
aep_results.yml, economic_settings.yml, ecomo_results.yml and one
consolidated tef_summary.yml. The generated economic settings
reference the shared inputs relatively (EcoMo resolves all paths
relative to the settings file).

Public interface:
    prepare_study_inputs(base_config_dir, inputs_dir)
    CasePaths, StudyInputs, TefCaseResult, TefCaseRunner
    evaluate_design(design, work_dir)   # optimisation entry point
"""

import copy
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from tef.io import base_config_dir as default_base_config_dir
from tef.io import load_yaml, write_yaml
from tef.power_accounting import CasePowers, full_cycle_peak_mechanical_power
from tef.storage_sizing import update_storage_capacity_after_power_curves
from tef.system_scaling import (
    DesignVariables,
    ScalingSettings,
    build_scaled_system,
    scaling_summary_section,
)
from tef.wrappers import AepRunner, AwespaPowerRunner, EcomoRunner

# Base-config filenames copied into a study's inputs directory
# (source name -> shared input name).
INPUT_FILES = {
    'base_system.yml': 'base_system.yml',
    'scaling_settings.yml': 'scaling_settings.yml',
    'wind_resource.yml': 'wind_resource.yml',
    'inertiafree-qsm_settings.yml': 'inertiafree-qsm_settings.yml',
    'economic_cost_inputs_V3.yml': 'economic_cost_inputs.yml',
    'economic_settings_template.yml': 'economic_settings_template.yml',
}


@dataclass(frozen=True)
class StudyInputs:
    """Shared input files of a study, prepared once per study."""

    inputs_dir: Path

    @property
    def base_system(self) -> Path:
        return self.inputs_dir / 'base_system.yml'

    @property
    def scaling_settings(self) -> Path:
        return self.inputs_dir / 'scaling_settings.yml'

    @property
    def wind_resource(self) -> Path:
        return self.inputs_dir / 'wind_resource.yml'

    @property
    def qsm_settings(self) -> Path:
        return self.inputs_dir / 'inertiafree-qsm_settings.yml'

    @property
    def economic_cost_inputs(self) -> Path:
        return self.inputs_dir / 'economic_cost_inputs.yml'

    @property
    def economic_settings_template(self) -> Path:
        return self.inputs_dir / 'economic_settings_template.yml'


def prepare_study_inputs(base_config_dir: Path,
                         inputs_dir: Path) -> StudyInputs:
    """Sync the shared input files from the base configuration into a
    study's ``inputs/`` directory.

    A file is (re)copied when it is missing OR the base source is newer
    than the copy, so editing a base config always propagates to the
    next run. ``copy2`` preserves the source mtime, so an unchanged base
    leaves equal mtimes and is never re-copied - a plain re-run stays
    reproducible and does not touch its recorded inputs. This replaces
    the old skip-if-present behaviour, under which a stale copy in
    ``inputs/`` (or ``_scratch/inputs/``) silently shadowed base edits.
    """
    inputsDir = Path(inputs_dir)
    inputsDir.mkdir(parents=True, exist_ok=True)
    for sourceName, inputName in INPUT_FILES.items():
        source = Path(base_config_dir) / sourceName
        destination = inputsDir / inputName
        if (not destination.exists()
                or source.stat().st_mtime > destination.stat().st_mtime):
            shutil.copy2(source, destination)
    return StudyInputs(inputsDir)


@dataclass(frozen=True)
class CasePaths:
    """Case-specific file locations inside one case directory."""

    case_dir: Path

    @property
    def design(self) -> Path:
        return self.case_dir / 'design.yml'

    @property
    def system(self) -> Path:
        return self.case_dir / 'system.yml'

    @property
    def qsm_settings_override(self) -> Path:
        # Only written when the design overrides a QSM setting.
        return self.case_dir / 'inertiafree-qsm_settings.yml'

    @property
    def power_curves(self) -> Path:
        return self.case_dir / 'power_curves.yml'

    @property
    def aep_results(self) -> Path:
        return self.case_dir / 'aep_results.yml'

    @property
    def economic_settings(self) -> Path:
        return self.case_dir / 'economic_settings.yml'

    @property
    def ecomo_results(self) -> Path:
        return self.case_dir / 'ecomo_results.yml'

    @property
    def tef_summary(self) -> Path:
        return self.case_dir / 'tef_summary.yml'


@dataclass(frozen=True)
class TefCaseResult:
    """Key metrics and file locations for one evaluated design point."""

    case_name: str
    case_dir: Path

    flat_area_m2: float
    projected_area_m2: float
    generator_max_power_w: float
    max_tether_force_n: float
    tether_diameter_m: float
    wing_and_bridle_mass_kg: float
    kcu_mass_kg: float
    sensor_mass_kg: float
    total_airborne_mass_kg: float
    # None when storage sizing is disabled for the run.
    storage_capacity_wh: Optional[float]

    aep_mwh: float
    capacity_factor: float
    # Emergent rated (plateau) power from the power curve; the generator
    # max_power is an input, this is what the curve actually rates at.
    emergent_rated_power_w: float
    # Full-cycle peak mechanical power (sizes the generator); None when the
    # QSM time histories are unavailable.
    peak_mechanical_power_w: Optional[float]
    lcoe_eur_per_mwh: float
    icc_eur: float
    omc_eur_per_year: float

    power_curves_path: Path
    aep_results_path: Path
    ecomo_results_path: Path
    tef_summary_path: Path


def _relative_to_case(path: Path, case_dir: Path) -> str:
    """Path string relative to the case directory (EcoMo resolves all
    input files relative to the settings file's directory)."""
    return os.path.relpath(Path(path), Path(case_dir)).replace(os.sep, '/')


def write_case_economic_settings(template_path: Path,
                                 case_dir: Path,
                                 system_path: Path,
                                 cost_inputs_path: Path,
                                 wind_resource_path: Path,
                                 power_curves_path: Path,
                                 aep_results_path: Path,
                                 output_path: Path,
                                 peak_mechanical_power_override_w=None) -> Path:
    """Generate the case-local economic_settings.yml from the template.

    The file references are rewritten and, when supplied, the full-cycle
    peak mechanical power is injected as
    ``system_extras.peak_mechanical_power_override`` so EcoMo sizes the
    generator on it. Every other economic assumption (business, operations,
    replacements, the rest of system_extras) comes from the template.
    """
    settings = load_yaml(template_path)
    caseDir = Path(case_dir)

    settings['input_files'] = {
        'system': _relative_to_case(system_path, caseDir),
        'cost_inputs': _relative_to_case(cost_inputs_path, caseDir),
        'performance': None,
        'aep_results': _relative_to_case(aep_results_path, caseDir),
        'power_curves': _relative_to_case(power_curves_path, caseDir),
    }
    settings.setdefault('wind_resource', {})['resource_file'] = (
        _relative_to_case(wind_resource_path, caseDir))
    if peak_mechanical_power_override_w is not None:
        settings.setdefault('system_extras', {})[
            'peak_mechanical_power_override'] = peak_mechanical_power_override_w

    write_yaml(settings, output_path)
    return Path(output_path)


def _deep_update(target: dict, override: dict) -> None:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value


def append_zero_power_point(power_curves_path: Path,
                            offset_m_s: float = 0.01) -> None:
    """Append an unsuccessful zero-power entry just past the last wind
    speed of every profile.

    AWESPA's AEP interpolates the power curve onto the wind-resource bins
    with ``np.interp``, which holds the end values constant outside the
    curve; without this point every wind above the cut-out would be
    credited with the cut-out power. EcoMo and the storage sizing skip
    unsuccessful entries, so only the AEP sees the zero.
    """
    data = load_yaml(power_curves_path)
    windSpeed = float(data['reference_wind_speeds'][-1]) + offset_m_s
    data['reference_wind_speeds'].append(windSpeed)
    zero = {'average_cycle_power': 0.0}
    for profile in data['power_curves']:
        profile['wind_speed_data'].append({
            'wind_speed': windSpeed,
            'successful': False,
            'performance': {'power': dict(zero),
                            'electrical_power': dict(zero),
                            'timing': {}},
        })
    write_yaml(data, power_curves_path)


class TefCaseRunner:
    """Runs one complete TEF case against a prepared inputs directory."""

    def __init__(self, case_dir: Path, design: DesignVariables,
                 inputs_dir: Path, validate: bool = True,
                 run_storage_sizing: bool = True, overwrite: bool = False,
                 verbose: bool = True,
                 power_curve_builder: Optional[
                     Callable[[Path, Path, Path], None]] = None,
                 zero_power_beyond_cut_out: bool = False,
                 peak_power_cap_factor: Optional[float] = None,
                 qsm_override: Optional[dict] = None):
        """``power_curve_builder(system_path, qsm_settings_path,
        output_power_curves_path)`` replaces the single AWESPA power run
        when given (e.g. an envelope over several aerodynamic operating
        points); it must write ``power_curves.yml`` and its ``.npz``.

        ``zero_power_beyond_cut_out`` appends a zero-power point just past
        the last computed wind speed, so the AEP (which holds the end
        values of the curve outside its range) produces nothing above the
        cut-out. ``peak_power_cap_factor`` caps the peak mechanical power
        that sizes the drivetrain at that multiple of the generator limit;
        the uncapped peak is still reported. ``qsm_override`` is merged
        into the study's QSM settings for this case only (e.g. its reel-out
        aerodynamic coefficients)."""
        self.paths = CasePaths(Path(case_dir))
        self.design = design
        self.inputs = StudyInputs(Path(inputs_dir))
        self.validate = validate
        self.runStorageSizing = run_storage_sizing
        self.overwrite = overwrite
        self.verbose = verbose
        self.powerCurveBuilder = power_curve_builder
        self.zeroPowerBeyondCutOut = zero_power_beyond_cut_out
        self.peakPowerCapFactor = peak_power_cap_factor
        self.qsmOverride = qsm_override or {}

    @property
    def case_name(self) -> str:
        return self.design.case_name or self.paths.case_dir.name

    def _prepare_case_dir(self) -> Path:
        """Create the case directory, record the design, and return the
        QSM settings path (case-local only when the design overrides a
        QSM setting)."""
        caseDir = self.paths.case_dir
        if caseDir.exists() and any(caseDir.iterdir()):
            if not self.overwrite:
                raise FileExistsError(
                    f"Case directory {caseDir} already exists and is not "
                    f"empty; pass overwrite=True to replace it")
        caseDir.mkdir(parents=True, exist_ok=True)

        write_yaml({'design_variables': self.design.to_dict()},
                   self.paths.design)

        override = copy.deepcopy(self.qsmOverride)
        if self.design.minimum_tether_force_n is not None:
            _deep_update(override, {'cycle': {
                'minimum_tether_force': self.design.minimum_tether_force_n}})
        if not override:
            return self.inputs.qsm_settings
        qsmSettings = load_yaml(self.inputs.qsm_settings)
        _deep_update(qsmSettings, override)
        write_yaml(qsmSettings, self.paths.qsm_settings_override)
        return self.paths.qsm_settings_override

    def _write_tef_summary(self, quantities, storage_summary,
                           aep_mwh: float, capacity_factor: float,
                           metrics: dict, case_powers,
                           sizing_peak_power_w=None) -> None:
        write_yaml({
            'design': {
                'flat_area_m2': quantities.flat_area_m2,
                'projected_area_m2': quantities.projected_area_m2,
                'generator_max_power_w': quantities.generator_max_power_w,
            },
            'scaling': scaling_summary_section(quantities),
            'storage_sizing': storage_summary,
            'power_accounting': {
                'generator_power_limit_mechanical_w':
                    case_powers.generator_power_limit_mechanical_w,
                'peak_mechanical_power_w':
                    case_powers.peak_mechanical_power_w,
                'peak_mechanical_power_sizing_w': sizing_peak_power_w,
                'rated_cycle_electrical_power_w':
                    case_powers.rated_cycle_electrical_power_w,
                'crest_peak_over_rated': case_powers.crest_peak_over_rated,
                'crest_limit_over_rated': case_powers.crest_limit_over_rated,
            },
            'performance': {
                'aep_mwh': aep_mwh,
                'capacity_factor': capacity_factor,
            },
            'economics': {
                'lcoe_eur_per_mwh': metrics['LCoE'],
                'icc_eur': metrics['ICC'],
                'omc_eur_per_year': metrics['OMC'],
            },
            'paths': {
                'system': self.paths.system.name,
                'power_curves': self.paths.power_curves.name,
                'aep_results': self.paths.aep_results.name,
                'ecomo_results': self.paths.ecomo_results.name,
                'inputs_dir': _relative_to_case(self.inputs.inputs_dir,
                                                self.paths.case_dir),
            },
        }, self.paths.tef_summary)

    def run(self) -> TefCaseResult:
        """Execute the complete model chain and return the case result."""
        paths = self.paths
        qsmSettingsPath = self._prepare_case_dir()

        quantities = build_scaled_system(
            base_system_path=self.inputs.base_system,
            scaling_settings_path=self.inputs.scaling_settings,
            design=self.design,
            output_system_path=paths.system,
            validate=self.validate,
        )

        if self.powerCurveBuilder is not None:
            self.powerCurveBuilder(paths.system, qsmSettingsPath,
                                   paths.power_curves)
        else:
            AwespaPowerRunner(
                system_path=paths.system,
                qsm_settings_path=qsmSettingsPath,
                wind_resource_path=self.inputs.wind_resource,
                output_power_curves_path=paths.power_curves,
                validate=self.validate,
                verbose=self.verbose,
            ).run()
        if self.zeroPowerBeyondCutOut:
            append_zero_power_point(paths.power_curves)

        AepRunner(
            power_curve_path=paths.power_curves,
            wind_resource_path=self.inputs.wind_resource,
            output_path=paths.aep_results,
        ).run()

        # The three distinct powers: the configured limit, the full-cycle
        # peak (sizes the generator), and the rated cycle electrical power.
        casePowers = CasePowers(
            generator_power_limit_mechanical_w=quantities.generator_max_power_w,
            peak_mechanical_power_w=full_cycle_peak_mechanical_power(
                paths.power_curves),
            rated_cycle_electrical_power_w=load_yaml(
                paths.aep_results)['power_summary']['max_rated_power_w'],
        )

        # Drivetrain sizing peak: optionally capped at a multiple of the
        # generator limit (a controller that holds transitions within the
        # generator's overload rating).
        sizingPeakW = casePowers.peak_mechanical_power_w
        if self.peakPowerCapFactor is not None and sizingPeakW is not None:
            sizingPeakW = min(sizingPeakW, self.peakPowerCapFactor
                              * quantities.generator_max_power_w)

        storageSummary = None
        if self.runStorageSizing:
            scalingSettings = ScalingSettings.load(
                self.inputs.scaling_settings)
            storageSummary = update_storage_capacity_after_power_curves(
                system_path=paths.system,
                power_curves_path=paths.power_curves,
                settings=scalingSettings.storage_sizing,
            )

        write_case_economic_settings(
            template_path=self.inputs.economic_settings_template,
            case_dir=paths.case_dir,
            system_path=paths.system,
            cost_inputs_path=self.inputs.economic_cost_inputs,
            wind_resource_path=self.inputs.wind_resource,
            power_curves_path=paths.power_curves,
            aep_results_path=paths.aep_results,
            output_path=paths.economic_settings,
            peak_mechanical_power_override_w=sizingPeakW,
        )

        eco = EcomoRunner(
            economic_settings_path=paths.economic_settings,
            output_path=paths.ecomo_results,
            validate=self.validate,
            verbose=self.verbose,
        ).run()
        metrics = eco['metrics']

        aepData = load_yaml(paths.aep_results)
        aepMwh = aepData['annual_energy_production']['total']['aep_mwh']
        capacityFactor = aepData['power_summary']['capacity_factor']
        emergentRatedPowerW = aepData['power_summary']['max_rated_power_w']

        self._write_tef_summary(quantities, storageSummary, aepMwh,
                                capacityFactor, metrics, casePowers,
                                sizingPeakW)

        return TefCaseResult(
            case_name=self.case_name,
            case_dir=paths.case_dir,
            flat_area_m2=quantities.flat_area_m2,
            projected_area_m2=quantities.projected_area_m2,
            generator_max_power_w=quantities.generator_max_power_w,
            max_tether_force_n=quantities.max_tether_force_n,
            tether_diameter_m=quantities.tether_diameter_m,
            wing_and_bridle_mass_kg=quantities.wing_and_bridle_mass_kg,
            kcu_mass_kg=quantities.kcu_mass_kg,
            sensor_mass_kg=quantities.sensor_mass_kg,
            total_airborne_mass_kg=quantities.total_airborne_mass_kg,
            storage_capacity_wh=(storageSummary['capacity_wh']
                                 if storageSummary is not None else None),
            aep_mwh=aepMwh,
            capacity_factor=capacityFactor,
            emergent_rated_power_w=emergentRatedPowerW,
            peak_mechanical_power_w=casePowers.peak_mechanical_power_w,
            lcoe_eur_per_mwh=metrics['LCoE'],
            icc_eur=metrics['ICC'],
            omc_eur_per_year=metrics['OMC'],
            power_curves_path=paths.power_curves,
            aep_results_path=paths.aep_results,
            ecomo_results_path=paths.ecomo_results,
            tef_summary_path=paths.tef_summary,
        )


def evaluate_design(design: DesignVariables,
                    work_dir: Path,
                    base_config_dir: Optional[Path] = None,
                    validate: bool = False,
                    verbose: bool = False,
                    run_storage_sizing: bool = True) -> TefCaseResult:
    """Optimisation entry point: evaluate one design in a reusable
    working directory.

    Prepares ``work_dir/inputs`` once (reused across calls), then runs
    the full chain in ``work_dir/<case_name>`` with overwrite enabled.
    Defaults are tuned for an optimisation loop: no awesIO validation,
    no verbose model output.

    Args:
        design: The design point to evaluate.
        work_dir: Working directory reused across evaluations.
        base_config_dir: Source of the shared inputs; defaults to the
            repository's ``config/base``.
        validate: Enable awesIO validation of generated files.
        verbose: Enable verbose AWESPA/EcoMo output.
        run_storage_sizing: Run the storage sizing block.

    Returns:
        TefCaseResult: Metrics of the evaluated design (LCoE etc.).
    """
    workDir = Path(work_dir)
    inputs = prepare_study_inputs(
        base_config_dir if base_config_dir is not None
        else default_base_config_dir(),
        workDir / 'inputs')
    caseName = design.case_name or 'case'
    return TefCaseRunner(
        case_dir=workDir / caseName,
        design=design,
        inputs_dir=inputs.inputs_dir,
        validate=validate,
        run_storage_sizing=run_storage_sizing,
        overwrite=True,
        verbose=verbose,
    ).run()
