"""Tests for the generated per-case economic settings file."""

import pytest

from tef.io import base_config_dir, load_yaml
from tef.pipeline import write_case_economic_settings


def _generate(case_dir, inputs_dir):
    return load_yaml(write_case_economic_settings(
        template_path=base_config_dir() / 'economic_settings_template.yml',
        case_dir=case_dir,
        system_path=case_dir / 'system.yml',
        cost_inputs_path=inputs_dir / 'economic_cost_inputs.yml',
        wind_resource_path=inputs_dir / 'wind_resource.yml',
        power_curves_path=case_dir / 'power_curves.yml',
        aep_results_path=case_dir / 'aep_results.yml',
        output_path=case_dir / 'economic_settings.yml',
    ))


@pytest.fixture
def case_settings(tmp_path):
    """Study layout: case folder beside a shared inputs folder."""
    case_dir = tmp_path / 'area_100'
    case_dir.mkdir()
    (tmp_path / 'inputs').mkdir()
    return _generate(case_dir, tmp_path / 'inputs')


def test_paths_relative_to_case(case_settings):
    """Case outputs are referenced locally, shared inputs via ../inputs
    (EcoMo resolves everything relative to the settings file)."""
    input_files = case_settings['input_files']
    assert input_files['system'] == 'system.yml'
    assert input_files['aep_results'] == 'aep_results.yml'
    assert input_files['power_curves'] == 'power_curves.yml'
    assert input_files['performance'] is None
    assert input_files['cost_inputs'] == '../inputs/economic_cost_inputs.yml'
    assert case_settings['wind_resource']['resource_file'] == (
        '../inputs/wind_resource.yml')


def test_economic_assumptions_preserved(case_settings):
    """Everything except the file references comes from the template."""
    template = load_yaml(
        base_config_dir() / 'economic_settings_template.yml')
    for section in ('topology', 'business', 'operations', 'replacements',
                    'system_extras', 'analysis'):
        assert case_settings[section] == template[section]
    # Hydraulic accumulator capacity is intentionally NOT scaled: it is
    # unused for the soft-wing GG case.
    assert case_settings['system_extras'][
        'hydraulic_accumulator_capacity'] == pytest.approx(1.1377)
