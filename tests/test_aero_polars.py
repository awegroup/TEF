"""Unit tests for the aero polar study helpers (no AWESPA/EcoMo runs)."""

import numpy as np
import pytest

from tef.aero_polars import ParabolicPolar, envelope_power_curves
from tef.io import load_yaml, write_yaml


# -- ParabolicPolar ---------------------------------------------------------

def test_anchored_polar_reproduces_reference_point():
    # The reference point is the L/D_max point of the reference polar.
    polar = ParabolicPolar.anchored(ld_max=0.63 / 0.14, cl_max=1.2,
                                    reference_cl=0.63, reference_cd=0.14)
    assert polar.cl_at_ld_max == pytest.approx(0.63)
    assert polar.cd(0.63) == pytest.approx(0.14)
    assert polar.cd0 == pytest.approx(0.07)


def test_higher_ld_max_lowers_cd0_at_fixed_k():
    low = ParabolicPolar.anchored(4.5, 1.0, 0.63, 0.14)
    high = ParabolicPolar.anchored(6.5, 1.0, 0.63, 0.14)
    assert high.k == pytest.approx(low.k)
    assert high.cd0 < low.cd0
    assert 0.63 / high.cd(0.63) <= 6.5 + 1e-12


def test_operating_points_span_best_ld_to_cl_max():
    polar = ParabolicPolar.anchored(4.5, 1.2, 0.63, 0.14)
    points = polar.operating_points(max_points=4, min_spacing=0.1)
    lifts = [cl for cl, _ in points]
    assert len(points) == 4
    assert lifts[0] == pytest.approx(0.63, abs=1e-3)
    assert lifts[-1] == pytest.approx(1.2)
    for cl, cd in points:
        assert cd == pytest.approx(polar.cd(cl), abs=1e-4)


def test_cl_max_below_best_ld_point_flies_at_cl_max_only():
    polar = ParabolicPolar.anchored(4.5, 0.5, 0.63, 0.14)
    assert polar.operating_points(4, 0.1) == [(0.5, round(polar.cd(0.5), 4))]


def test_min_spacing_limits_point_count():
    polar = ParabolicPolar.anchored(4.5, 0.7, 0.63, 0.14)
    assert len(polar.operating_points(4, 0.1)) == 2


# -- envelope_power_curves --------------------------------------------------

def _write_curve(path, powers, successful=None):
    successful = successful or [True] * len(powers)
    entries = []
    for i, (power, ok) in enumerate(zip(powers, successful)):
        entries.append({
            'wind_speed': 5.0 + i,
            'successful': ok,
            'performance': {
                'power': {'average_cycle_power': power},
                'electrical_power': {'average_cycle_power': power},
            },
        })
    write_yaml({
        'metadata': {'model_config': {}, 'note': ''},
        'reference_wind_speeds': [5.0 + i for i in range(len(powers))],
        'power_curves': [{'profile_id': 1, 'wind_speed_data': entries}],
    }, path)
    np.savez_compressed(path.with_suffix('.npz'), **{
        f'p1_ws{i}_power': np.full(3, power) for i, power in enumerate(powers)})


def test_envelope_takes_best_successful_point(tmp_path):
    a, b = tmp_path / 'a.yml', tmp_path / 'b.yml'
    _write_curve(a, [-10.0, 200.0, 500.0])
    _write_curve(b, [50.0, 100.0, 900.0], successful=[True, True, False])
    out = tmp_path / 'env' / 'power_curves.yml'
    out.parent.mkdir()

    selection = envelope_power_curves([a, b], out)

    assert [s['source'] for s in selection] == [1, 0, 0]
    envelope = load_yaml(out)
    powers = [e['performance']['electrical_power']['average_cycle_power']
              for e in envelope['power_curves'][0]['wind_speed_data']]
    assert powers == [50.0, 200.0, 500.0]
    config = envelope['metadata']['model_config']
    assert config['nominal_electrical_power'] == 500.0
    assert config['cut_in_wind_speed'] == 5.0
    histories = np.load(out.with_suffix('.npz'))
    assert histories['p1_ws0_power'][0] == 50.0
    assert histories['p1_ws2_power'][0] == 500.0


def test_envelope_rejects_mismatched_wind_grid(tmp_path):
    a, b = tmp_path / 'a.yml', tmp_path / 'b.yml'
    _write_curve(a, [1.0, 2.0])
    _write_curve(b, [1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        envelope_power_curves([a, b], tmp_path / 'out.yml')


# -- append_zero_power_point ------------------------------------------------

def test_zero_power_point_appended_past_cut_out(tmp_path):
    from tef.pipeline import append_zero_power_point
    path = tmp_path / 'power_curves.yml'
    _write_curve(path, [100.0, 200.0])
    append_zero_power_point(path)
    data = load_yaml(path)
    assert data['reference_wind_speeds'][-1] == pytest.approx(6.01)
    last = data['power_curves'][0]['wind_speed_data'][-1]
    assert last['successful'] is False
    assert last['performance']['electrical_power']['average_cycle_power'] == 0.0


# -- ScaledPolar -------------------------------------------------------------

def test_scaled_polar_peaks_at_design_cl():
    from tef.aero_polars import ScaledPolar
    polar = ScaledPolar(ld_max=9.0, cl_design=0.8)
    lifts = np.linspace(0.3, 1.4, 111)
    ratios = lifts / np.array([polar.cd(cl) for cl in lifts])
    assert lifts[np.argmax(ratios)] == pytest.approx(0.8, abs=0.01)
    assert ratios.max() == pytest.approx(9.0, rel=1e-3)


def test_scaled_polar_implied_ld_max_round_trip():
    from tef.aero_polars import ScaledPolar
    polar = ScaledPolar(ld_max=6.0, cl_design=0.8)
    assert ScaledPolar.implied_ld_max(1.1, polar.cd(1.1), 0.8) == \
        pytest.approx(6.0)


def test_case_qsm_override_writes_case_local_settings(tmp_path):
    from tef.pipeline import TefCaseRunner, prepare_study_inputs
    from tef.io import base_config_dir
    from tef.system_scaling import DesignVariables
    inputs = prepare_study_inputs(base_config_dir(), tmp_path / 'inputs')
    runner = TefCaseRunner(
        case_dir=tmp_path / 'case', design=DesignVariables(flat_area_m2=25.0),
        inputs_dir=inputs.inputs_dir,
        qsm_override={'aerodynamics': {'kite_lift_coefficient_reel_out': 1.0}})
    path = runner._prepare_case_dir()
    aero = load_yaml(path)['aerodynamics']
    assert path.parent == tmp_path / 'case'
    assert aero['kite_lift_coefficient_reel_out'] == 1.0
    assert aero['kite_drag_coefficient_reel_out'] == 0.14


def test_envelope_primary_source_sets_cut_in(tmp_path):
    # Source 1 (the maximum-CL curve) cannot fly at the first wind speed;
    # source 0 could, but depowering is only allowed from source 1's cut-in.
    a, b = tmp_path / 'a.yml', tmp_path / 'b.yml'
    _write_curve(a, [5.0, 100.0, 900.0])
    _write_curve(b, [0.0, 300.0, 400.0], successful=[False, True, True])
    out = tmp_path / 'env.yml'
    selection = envelope_power_curves([a, b], out, primary_source=1)
    assert [s['source'] for s in selection] == [1, 1, 0]
