"""Tests for the generated system file, scaling summary section and
study inputs preparation."""

import pytest

from tef.io import base_config_dir, load_yaml
from tef.pipeline import prepare_study_inputs
from tef.system_scaling import (
    DesignVariables,
    build_scaled_system,
    compute_scaled_quantities,
    scaling_summary_section,
)


@pytest.fixture
def generated_case(tmp_path):
    config = base_config_dir()
    system_path = tmp_path / 'system.yml'
    quantities = build_scaled_system(
        base_system_path=config / 'base_system.yml',
        scaling_settings_path=config / 'scaling_settings.yml',
        design=DesignVariables(flat_area_m2=100.0),
        output_system_path=system_path,
        validate=True,  # exercises awesIO validation of the output
    )
    return quantities, system_path


def test_system_file_fields(generated_case):
    quantities, system_path = generated_case
    system = load_yaml(system_path)

    kite = system['components']['kites'][0]
    assert kite['wing']['structure']['flat_wing_area'] == pytest.approx(100.0)
    assert kite['wing']['structure']['projected_surface_area'] == (
        pytest.approx(79.0))
    assert kite['wing']['structure']['mass'] == pytest.approx(
        quantities.wing_and_bridle_mass_kg)
    assert kite['bridle']['structure']['mass'] == 0.0
    assert kite['control_system']['structure']['mass'] == pytest.approx(
        quantities.kcu_mass_kg + quantities.sensor_mass_kg)

    tether = system['components']['tethers'][0]['structure']
    assert tether['max_tether_force'] == pytest.approx(
        quantities.max_tether_force_n)
    assert tether['diameter'] == pytest.approx(quantities.tether_diameter_m)
    assert tether['material']['allowable_stress'] == pytest.approx(
        quantities.allowable_tether_stress_pa)
    # breaking_strength is the material ultimate (EcoMo max-stress), not
    # the sizing allowable
    assert tether['material']['breaking_strength'] == pytest.approx(1.5e9)

    station = system['components']['ground_station']
    assert station['generators'][0]['max_power'] == pytest.approx(
        quantities.generator_max_power_w)
    assert station['drums'][0]['max_tether_speed'] == pytest.approx(10.0)


def test_base_system_not_mutated(generated_case, base_system):
    """The loaded base system dict must never be modified in place."""
    fresh = load_yaml(base_config_dir() / 'base_system.yml')
    assert fresh == base_system


def test_scaling_summary_section(base_settings):
    """The summary section preserves the mass split that system.yml
    lumps into the control system mass."""
    quantities = compute_scaled_quantities(
        DesignVariables(flat_area_m2=100.0), base_settings)
    section = scaling_summary_section(quantities)

    assert section['total_airborne_mass_kg'] == pytest.approx(
        quantities.total_airborne_mass_kg)
    assert section['kcu_mass_kg'] == pytest.approx(quantities.kcu_mass_kg)
    assert section['sensor_mass_kg'] == pytest.approx(3.4)
    assert section['tether_diameter_m'] == pytest.approx(
        quantities.tether_diameter_m)
    assert section['model_choices'][
        'sensor_included_in_control_system_mass']


def test_prepare_study_inputs(tmp_path):
    """All shared inputs are copied once; existing files are kept."""
    inputs = prepare_study_inputs(base_config_dir(), tmp_path / 'inputs')

    for path in (inputs.base_system, inputs.scaling_settings,
                 inputs.wind_resource, inputs.qsm_settings,
                 inputs.economic_cost_inputs,
                 inputs.economic_settings_template):
        assert path.exists(), path

    # A second call must not overwrite the recorded study inputs.
    inputs.scaling_settings.write_text('marker: true\n', encoding='utf-8')
    prepare_study_inputs(base_config_dir(), tmp_path / 'inputs')
    assert inputs.scaling_settings.read_text(
        encoding='utf-8') == 'marker: true\n'
