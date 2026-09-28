"""TEF system scaling block.

Turns top-level design variables (wing area, wing loading, tether
stress, generator sizing) into a physically consistent awesIO system
file plus a scaling summary. All relations follow the soft-wing
scaling models from the thesis: log-space interpolation (or power law)
for the LEI wing mass, the component-decomposition KCU model
calibrated at 60 m2, constant sensor mass, constant maximum wing
loading for the tether force, and constant allowable stress for the
tether diameter.

Public interface:
    DesignVariables, ScalingSettings, ScaledSystemQuantities
    compute_scaled_quantities(design, settings)
    build_scaled_system(...)
    scaling_summary_section(quantities)
"""

import copy
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from tef.io import load_yaml, write_yaml


# ---------------------------------------------------------------------------
# Design variables and settings
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DesignVariables:
    """Top-level design variables for one TEF case.

    Only ``flat_area_m2`` is required; every other variable falls back
    to the reference value or default mode in the scaling settings.
    """

    flat_area_m2: float

    max_wing_loading_n_m2_projected: Optional[float] = None
    allowable_tether_stress_pa: Optional[float] = None

    generator_max_power_w: Optional[float] = None
    generator_specific_power_w_m2_projected: Optional[float] = None
    rated_power_w: Optional[float] = None
    crest_factor: Optional[float] = None

    max_tether_speed_m_s: Optional[float] = None
    tether_length_m: Optional[float] = None
    minimum_tether_force_n: Optional[float] = None

    case_name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Return the design variables as a plain dictionary."""
        return {
            'flat_area_m2': self.flat_area_m2,
            'max_wing_loading_n_m2_projected':
                self.max_wing_loading_n_m2_projected,
            'allowable_tether_stress_pa': self.allowable_tether_stress_pa,
            'generator_max_power_w': self.generator_max_power_w,
            'generator_specific_power_w_m2_projected':
                self.generator_specific_power_w_m2_projected,
            'rated_power_w': self.rated_power_w,
            'crest_factor': self.crest_factor,
            'max_tether_speed_m_s': self.max_tether_speed_m_s,
            'tether_length_m': self.tether_length_m,
            'minimum_tether_force_n': self.minimum_tether_force_n,
            'case_name': self.case_name,
        }


@dataclass(frozen=True)
class ReferenceSettings:
    flat_area_m2: float = 25.0
    projected_area_m2: float = 19.75
    wing_and_bridle_mass_kg: float = 10.6
    kcu_mass_kg: float = 8.4
    tether_diameter_m: float = 0.006
    max_tether_force_n: float = 8200.0
    tether_length_m: float = 500.0
    tether_density_kg_m3: float = 970.0
    generator_max_power_w: float = 40000.0
    max_tether_speed_m_s: float = 10.0
    generator_efficiency: float = 1.0


@dataclass(frozen=True)
class GeometrySettings:
    projected_to_flat_area_ratio: float = 0.79


@dataclass(frozen=True)
class WingMassModelSettings:
    type: str = 'tabulated_log_interpolation'
    anchor_flat_areas_m2: List[float] = field(
        default_factory=lambda: [25.0, 50.0, 100.0, 200.0, 400.0])
    anchor_masses_kg: List[float] = field(
        default_factory=lambda: [10.60, 23.97, 55.79, 133.75, 330.21])
    power_law_reference_area_m2: float = 25.0
    power_law_reference_mass_kg: float = 10.60
    power_law_exponent: float = 1.22
    allow_extrapolation: bool = False


@dataclass(frozen=True)
class KcuMassModelSettings:
    type: str = 'component_scaling'
    reference_area_m2: float = 60.0
    reference_mass_kg: float = 26.2
    fixed_fraction: float = 0.01
    linear_fraction: float = 0.53
    superlinear_fraction: float = 0.45
    superlinear_exponent: float = 1.5
    calibration_mode: str = 'raw_thesis'


@dataclass(frozen=True)
class SensorMassModelSettings:
    type: str = 'constant'
    mass_kg: float = 3.4
    include_in_control_system_mass: bool = True


@dataclass(frozen=True)
class TetherSizingSettings:
    fibre_area_fraction: float = 0.85
    default_allowable_stress_pa: float = 341194910.6806645
    # Rope-level ultimate (fibre) strength of the tether material [Pa].
    # Distinct from the allowable working stress that sizes the diameter:
    # EcoMo uses breaking_strength to cap the fibre stress in the life
    # models and to size the winch drum wall, so it must be the material
    # ultimate, not the sizing allowable.
    material_breaking_strength_pa: float = 1.5e9
    write_allowable_stress_to_material: bool = True
    write_breaking_strength_compatibility_field: bool = True


@dataclass(frozen=True)
class GeneratorSizingSettings:
    default_mode: str = 'specific_power'
    default_specific_power_w_m2_projected: float = 2025.3164556962024
    allow_explicit_max_power: bool = True
    allow_rated_power_and_crest_factor: bool = True


@dataclass(frozen=True)
class DrumSettings:
    max_tether_speed_m_s: float = 10.0
    keep_constant_with_scale: bool = True


@dataclass(frozen=True)
class StorageSizingSettings:
    mode: str = 'from_power_curves_if_available'
    method: str = 'cycle_energy_imbalance'
    safety_factor: float = 1.0
    fallback_mode: str = 'scale_with_generator_power'
    reference_capacity_wh: float = 1137.7
    reference_generator_max_power_w: float = 40000.0
    selected_storage_type: str = 'capacitor_bank'
    update_all_storage_entries: bool = True
    # Rolling-mean window (in wind-speed points) applied to the per-cycle
    # required-capacity curve before taking its maximum, so a single
    # coincidental high-wind cycle cannot solo-set the capacity. 1 = no
    # smoothing (raw max); 3 = average each point with its two neighbours.
    smoothing_window: int = 3


@dataclass(frozen=True)
class ValidationSettings:
    validate_awesio_system: bool = True


_SECTION_TYPES = {
    'reference': ReferenceSettings,
    'geometry': GeometrySettings,
    'wing_mass_model': WingMassModelSettings,
    'kcu_mass_model': KcuMassModelSettings,
    'sensor_mass_model': SensorMassModelSettings,
    'tether_sizing': TetherSizingSettings,
    'generator_sizing': GeneratorSizingSettings,
    'drum': DrumSettings,
    'storage_sizing': StorageSizingSettings,
    'validation': ValidationSettings,
}


@dataclass(frozen=True)
class ScalingSettings:
    """Parsed contents of scaling_settings.yml."""

    reference: ReferenceSettings = field(default_factory=ReferenceSettings)
    geometry: GeometrySettings = field(default_factory=GeometrySettings)
    wing_mass_model: WingMassModelSettings = field(
        default_factory=WingMassModelSettings)
    kcu_mass_model: KcuMassModelSettings = field(
        default_factory=KcuMassModelSettings)
    sensor_mass_model: SensorMassModelSettings = field(
        default_factory=SensorMassModelSettings)
    tether_sizing: TetherSizingSettings = field(
        default_factory=TetherSizingSettings)
    generator_sizing: GeneratorSizingSettings = field(
        default_factory=GeneratorSizingSettings)
    drum: DrumSettings = field(default_factory=DrumSettings)
    storage_sizing: StorageSizingSettings = field(
        default_factory=StorageSizingSettings)
    validation: ValidationSettings = field(default_factory=ValidationSettings)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ScalingSettings':
        """Build settings from a parsed YAML dictionary.

        Unknown keys inside a known section raise, so typos in the
        settings file fail loudly instead of silently using defaults.
        """
        sections = {}
        for name, sectionCls in _SECTION_TYPES.items():
            sectionData = data.get(name)
            if sectionData is None:
                sections[name] = sectionCls()
            else:
                unknown = set(sectionData) - set(
                    sectionCls.__dataclass_fields__)
                if unknown:
                    raise ValueError(
                        f"Unknown keys in scaling settings section "
                        f"'{name}': {sorted(unknown)}")
                sections[name] = sectionCls(**sectionData)
        return cls(**sections)

    @classmethod
    def load(cls, path: Path) -> 'ScalingSettings':
        """Load and parse a scaling settings YAML file."""
        return cls.from_dict(load_yaml(path))


@dataclass(frozen=True)
class ScaledSystemQuantities:
    """Derived quantities for one scaled system design."""

    flat_area_m2: float
    projected_area_m2: float

    wing_and_bridle_mass_kg: float
    kcu_mass_kg: float
    sensor_mass_kg: float
    control_system_mass_written_kg: float
    total_airborne_mass_kg: float

    max_wing_loading_n_m2_projected: float
    max_tether_force_n: float
    allowable_tether_stress_pa: float
    tether_diameter_m: float
    tether_length_m: float
    tether_density_kg_m3: float

    generator_max_power_w: float
    generator_specific_power_w_m2_projected: float
    max_tether_speed_m_s: float
    generator_efficiency: float

    mass_model_type: str
    kcu_model_type: str
    kcu_calibration_mode: str
    sensor_included_in_control_system_mass: bool


# ---------------------------------------------------------------------------
# Scaling equations
# ---------------------------------------------------------------------------

def projected_area(flat_area_m2: float, ratio: float) -> float:
    """Projected area from flat area: A_proj = r * S_flat."""
    return ratio * flat_area_m2


def max_tether_force(projected_area_m2: float,
                     wing_loading_n_m2: float) -> float:
    """Maximum tether force from maximum wing loading:
    F_t,max = W_l,max * A_proj."""
    return wing_loading_n_m2 * projected_area_m2


def tether_diameter(force_n: float, fibre_area_fraction: float,
                    allowable_stress_pa: float) -> float:
    """Tether diameter: d_t = sqrt(4 F / (pi f_A sigma_allow))."""
    return math.sqrt(
        4.0 * force_n / (math.pi * fibre_area_fraction * allowable_stress_pa))


def wing_mass(flat_area_m2: float, settings: ScalingSettings) -> float:
    """Wing-and-bridle mass from the configured wing mass model."""
    model = settings.wing_mass_model
    if model.type == 'tabulated_log_interpolation':
        anchorAreas = np.asarray(model.anchor_flat_areas_m2, dtype=float)
        anchorMasses = np.asarray(model.anchor_masses_kg, dtype=float)
        if not model.allow_extrapolation:
            if flat_area_m2 < anchorAreas[0] or flat_area_m2 > anchorAreas[-1]:
                raise ValueError(
                    f"flat_area_m2={flat_area_m2} outside anchor range "
                    f"[{anchorAreas[0]}, {anchorAreas[-1]}] and "
                    f"extrapolation is disabled")
        logMass = np.interp(np.log(flat_area_m2), np.log(anchorAreas),
                            np.log(anchorMasses))
        return float(np.exp(logMass))
    if model.type == 'power_law':
        return model.power_law_reference_mass_kg * (
            flat_area_m2 / model.power_law_reference_area_m2
        ) ** model.power_law_exponent
    raise ValueError(f"Unknown wing mass model type: {model.type!r}")


def kcu_mass_raw(flat_area_m2: float, settings: ScalingSettings) -> float:
    """Raw thesis KCU mass model (fixed + linear + superlinear parts)."""
    model = settings.kcu_mass_model
    if model.type != 'component_scaling':
        raise ValueError(f"Unknown KCU mass model type: {model.type!r}")
    sizeRatio = flat_area_m2 / model.reference_area_m2
    return model.reference_mass_kg * (
        model.fixed_fraction
        + model.linear_fraction * sizeRatio
        + model.superlinear_fraction
        * sizeRatio ** model.superlinear_exponent)


def kcu_mass(flat_area_m2: float, settings: ScalingSettings) -> float:
    """KCU mass including the configured calibration mode."""
    model = settings.kcu_mass_model
    rawMass = kcu_mass_raw(flat_area_m2, settings)
    if model.calibration_mode == 'raw_thesis':
        return rawMass
    if model.calibration_mode == 'match_v3_at_25':
        referenceRaw = kcu_mass_raw(settings.reference.flat_area_m2,
                                    settings)
        return rawMass * settings.reference.kcu_mass_kg / referenceRaw
    raise ValueError(
        f"Unknown KCU calibration mode: {model.calibration_mode!r}")


def sensor_mass(settings: ScalingSettings) -> float:
    """Sensor package mass (constant with size)."""
    model = settings.sensor_mass_model
    if model.type != 'constant':
        raise ValueError(f"Unknown sensor mass model type: {model.type!r}")
    return model.mass_kg


def resolve_wing_loading(design: DesignVariables,
                         settings: ScalingSettings) -> float:
    """Maximum wing loading: design value or the V3 reference value."""
    if design.max_wing_loading_n_m2_projected is not None:
        return design.max_wing_loading_n_m2_projected
    ref = settings.reference
    return ref.max_tether_force_n / ref.projected_area_m2


def resolve_allowable_stress(design: DesignVariables,
                             settings: ScalingSettings) -> float:
    """Allowable tether stress: design value or the V3-equivalent
    default that reproduces the 6 mm reference tether."""
    if design.allowable_tether_stress_pa is not None:
        return design.allowable_tether_stress_pa
    return settings.tether_sizing.default_allowable_stress_pa


def generator_max_power(projected_area_m2: float, design: DesignVariables,
                        settings: ScalingSettings) -> float:
    """Generator power limit, resolved by priority.

    1. Explicit ``generator_max_power_w``.
    2. ``rated_power_w * crest_factor`` when both are given.
    3. ``generator_specific_power_w_m2_projected * A_proj``.
    4. Default specific power from the settings.
    """
    sizing = settings.generator_sizing
    if design.generator_max_power_w is not None:
        if not sizing.allow_explicit_max_power:
            raise ValueError(
                "Explicit generator_max_power_w given but "
                "generator_sizing.allow_explicit_max_power is false")
        return design.generator_max_power_w
    if design.rated_power_w is not None and design.crest_factor is not None:
        if not sizing.allow_rated_power_and_crest_factor:
            raise ValueError(
                "rated_power_w and crest_factor given but generator_sizing."
                "allow_rated_power_and_crest_factor is false")
        return design.rated_power_w * design.crest_factor
    if design.generator_specific_power_w_m2_projected is not None:
        return (design.generator_specific_power_w_m2_projected
                * projected_area_m2)
    return sizing.default_specific_power_w_m2_projected * projected_area_m2


# ---------------------------------------------------------------------------
# System file writer
# ---------------------------------------------------------------------------

def update_system_dict(base_system: dict,
                       quantities: ScaledSystemQuantities,
                       settings: ScalingSettings) -> dict:
    """Return a deep copy of ``base_system`` updated with the scaled
    quantities.

    The wing structure mass carries the lumped wing-and-bridle mass
    (V3 convention: bridle mass stays 0), and the control system mass
    carries KCU + sensor mass when sensor inclusion is enabled.
    """
    system = copy.deepcopy(base_system)

    kite = system['components']['kites'][0]
    wingStructure = kite['wing']['structure']
    wingStructure['flat_wing_area'] = quantities.flat_area_m2
    wingStructure['projected_surface_area'] = quantities.projected_area_m2
    wingStructure['mass'] = quantities.wing_and_bridle_mass_kg
    kite['bridle']['structure']['mass'] = 0.0
    kite['control_system']['structure']['mass'] = (
        quantities.control_system_mass_written_kg)

    tetherStructure = system['components']['tethers'][0]['structure']
    tetherStructure['length'] = quantities.tether_length_m
    tetherStructure['diameter'] = quantities.tether_diameter_m
    tetherStructure['density'] = quantities.tether_density_kg_m3
    tetherStructure['max_tether_force'] = quantities.max_tether_force_n
    material = tetherStructure.setdefault('material', {})
    if settings.tether_sizing.write_allowable_stress_to_material:
        material['allowable_stress'] = quantities.allowable_tether_stress_pa
    if settings.tether_sizing.write_breaking_strength_compatibility_field:
        # EcoMo reads material.breaking_strength as the tether maximum
        # stress (life-curve cap and winch drum wall sizing), so this
        # must be the material ultimate strength - NOT the allowable
        # working stress that sizes the diameter (aliasing the two made
        # the drum wall ~4.4x too thin for the V3 defaults).
        material['breaking_strength'] = (
            settings.tether_sizing.material_breaking_strength_pa)

    groundStation = system['components']['ground_station']
    groundStation['drums'][0]['max_tether_speed'] = (
        quantities.max_tether_speed_m_s)
    generator = groundStation['generators'][0]
    generator['max_power'] = quantities.generator_max_power_w
    generator['efficiency'] = quantities.generator_efficiency

    metadata = system.setdefault('metadata', {})
    metadata['name'] = (
        f"TEF scaled soft wing system, S = {quantities.flat_area_m2:g} m2")
    metadata['description'] = 'Generated by TEF system scaling block'
    if quantities.sensor_included_in_control_system_mass:
        metadata['note'] = (
            'Contains sensor mass inside control_system.structure.mass '
            'for AWESPA compatibility')
    else:
        metadata['note'] = (
            'Sensor mass excluded from control_system.structure.mass '
            '(Bredael reproduction mode)')

    return system


# ---------------------------------------------------------------------------
# awesIO validation
# ---------------------------------------------------------------------------

def validate_awesio_file(path: Path) -> None:
    """Validate a generated awesIO file against its schema.

    Silently skips when awesIO is not installed. Validation errors
    propagate: a generated file that fails its schema is a bug in the
    generator, not something to ignore.
    """
    try:
        from awesio.validator import validate
    except ImportError:
        return
    validate(input=str(path))


# ---------------------------------------------------------------------------
# Block API
# ---------------------------------------------------------------------------

def compute_scaled_quantities(design: DesignVariables,
                              settings: ScalingSettings,
                              ) -> ScaledSystemQuantities:
    """Resolve design variables against the settings and evaluate all
    scaling relations for one design point."""
    reference = settings.reference

    flatArea = design.flat_area_m2
    projArea = projected_area(
        flatArea, settings.geometry.projected_to_flat_area_ratio)

    wingAndBridleMass = wing_mass(flatArea, settings)
    kcuMass = kcu_mass(flatArea, settings)
    sensorMass = sensor_mass(settings)
    includeSensor = (
        settings.sensor_mass_model.include_in_control_system_mass)
    controlMassWritten = (kcuMass + sensorMass if includeSensor
                          else kcuMass)

    wingLoading = resolve_wing_loading(design, settings)
    tetherForce = max_tether_force(projArea, wingLoading)
    allowableStress = resolve_allowable_stress(design, settings)
    diameter = tether_diameter(
        tetherForce, settings.tether_sizing.fibre_area_fraction,
        allowableStress)

    generatorPower = generator_max_power(projArea, design, settings)

    tetherLength = (design.tether_length_m
                    if design.tether_length_m is not None
                    else reference.tether_length_m)
    tetherSpeed = (design.max_tether_speed_m_s
                   if design.max_tether_speed_m_s is not None
                   else reference.max_tether_speed_m_s)

    return ScaledSystemQuantities(
        flat_area_m2=flatArea,
        projected_area_m2=projArea,
        wing_and_bridle_mass_kg=wingAndBridleMass,
        kcu_mass_kg=kcuMass,
        sensor_mass_kg=sensorMass,
        control_system_mass_written_kg=controlMassWritten,
        total_airborne_mass_kg=wingAndBridleMass + controlMassWritten,
        max_wing_loading_n_m2_projected=wingLoading,
        max_tether_force_n=tetherForce,
        allowable_tether_stress_pa=allowableStress,
        tether_diameter_m=diameter,
        tether_length_m=tetherLength,
        tether_density_kg_m3=reference.tether_density_kg_m3,
        generator_max_power_w=generatorPower,
        generator_specific_power_w_m2_projected=generatorPower / projArea,
        max_tether_speed_m_s=tetherSpeed,
        generator_efficiency=reference.generator_efficiency,
        mass_model_type=settings.wing_mass_model.type,
        kcu_model_type=settings.kcu_mass_model.type,
        kcu_calibration_mode=settings.kcu_mass_model.calibration_mode,
        sensor_included_in_control_system_mass=includeSensor,
    )


def scaling_summary_section(quantities: ScaledSystemQuantities) -> dict:
    """Scaling section of the case summary: the full mass and sizing
    breakdown (the KCU/sensor split is not recoverable from system.yml,
    where the control system mass is lumped)."""
    return {
        'projected_area_m2': quantities.projected_area_m2,
        'wing_and_bridle_mass_kg': quantities.wing_and_bridle_mass_kg,
        'kcu_mass_kg': quantities.kcu_mass_kg,
        'sensor_mass_kg': quantities.sensor_mass_kg,
        'control_system_mass_written_kg':
            quantities.control_system_mass_written_kg,
        'total_airborne_mass_kg': quantities.total_airborne_mass_kg,
        'max_wing_loading_n_m2_projected':
            quantities.max_wing_loading_n_m2_projected,
        'max_tether_force_n': quantities.max_tether_force_n,
        'allowable_tether_stress_pa': quantities.allowable_tether_stress_pa,
        'tether_diameter_m': quantities.tether_diameter_m,
        'tether_length_m': quantities.tether_length_m,
        'generator_max_power_w': quantities.generator_max_power_w,
        'generator_specific_power_w_m2_projected':
            quantities.generator_specific_power_w_m2_projected,
        'max_tether_speed_m_s': quantities.max_tether_speed_m_s,
        'model_choices': {
            'wing_mass_model': quantities.mass_model_type,
            'kcu_mass_model': quantities.kcu_model_type,
            'kcu_calibration_mode': quantities.kcu_calibration_mode,
            'sensor_included_in_control_system_mass':
                quantities.sensor_included_in_control_system_mass,
        },
    }


def build_scaled_system(base_system_path: Path,
                        scaling_settings_path: Path,
                        design: DesignVariables,
                        output_system_path: Path,
                        validate: bool = True) -> ScaledSystemQuantities:
    """Generate a scaled awesIO system file for one design.

    Loads the base system and scaling settings, resolves missing design
    variables, evaluates the scaling relations, writes the system file
    and returns the derived quantities (the caller records them in the
    case summary via :func:`scaling_summary_section`).
    """
    baseSystem = load_yaml(base_system_path)
    settings = ScalingSettings.load(scaling_settings_path)

    quantities = compute_scaled_quantities(design, settings)
    system = update_system_dict(baseSystem, quantities, settings)

    write_yaml(system, output_system_path)
    if validate and settings.validation.validate_awesio_system:
        validate_awesio_file(output_system_path)
    return quantities
