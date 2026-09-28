"""Tests for the storage sizing block."""

import pytest

from tef.io import write_yaml
from tef.storage_sizing import (
    _rolling_mean_max,
    required_capacity_cycle_imbalance,
    required_capacity_retraction_fallback,
    storage_capacity_from_power_curves,
    update_storage_capacity_after_power_curves,
)
from tef.system_scaling import StorageSizingSettings


def _point(energies, times=None, successful=True):
    performance = {'electrical_energy': energies}
    if times is not None:
        performance['timing'] = times
    return {'successful': successful, 'performance': performance}


def _energies(reel_out, rori, reel_in, riro):
    return {'reel_out_energy': reel_out, 'transition_rori_energy': rori,
            'reel_in_energy': reel_in, 'transition_riro_energy': riro}


def _times(reel_out, rori, reel_in, riro):
    return {'reel_out_time': reel_out, 'transition_rori_time': rori,
            'reel_in_time': reel_in, 'transition_riro_time': riro,
            'cycle_time': reel_out + rori + reel_in + riro}


def test_cycle_imbalance_simple_two_phase():
    """Reel-out 300 J in 60 s, reel-in -100 J in 40 s: net 200 J over
    100 s -> P_avg = 2 W. Imbalance after reel-out = 300 - 120 = 180 J,
    back to 0 after reel-in, so the range is 180 J."""
    point = _point(_energies(300.0, 0.0, -100.0, 0.0),
                   _times(60.0, 0.0, 40.0, 0.0))
    required = required_capacity_cycle_imbalance(point['performance'])
    assert required == pytest.approx(180.0)


def test_fallback_uses_retraction_energy_only():
    performance = _point(
        _energies(300.0, -5.0, -100.0, -10.0))['performance']
    assert required_capacity_cycle_imbalance(performance) is None
    assert required_capacity_retraction_fallback(performance) == (
        pytest.approx(115.0))


def test_rolling_mean_max_suppresses_lone_spike():
    # window 1 -> raw max; window 3 -> a lone spike is averaged with its
    # neighbours so the highest *local average* governs instead.
    assert _rolling_mean_max([100.0, 1000.0, 100.0], 1) == 1000.0
    assert _rolling_mean_max([100.0, 1000.0, 100.0], 3) == pytest.approx(550.0)
    # a genuinely sustained high region still governs (and beats the spike).
    assert _rolling_mean_max([100.0, 1000.0, 1000.0, 100.0], 3) == (
        pytest.approx(700.0))
    # degenerate inputs fall back to the plain maximum.
    assert _rolling_mean_max([5.0], 3) == 5.0


def test_smoothing_reduces_capacity_from_a_spike():
    # Three wind speeds; the middle cycle is a coincidental high spike.
    def pt(ws, reel_out, reel_in):
        point = _point(_energies(reel_out, 0.0, reel_in, 0.0),
                       _times(60.0, 0.0, 40.0, 0.0))
        point['wind_speed'] = ws
        return point
    pc = {'power_curves': [{'wind_speed_data': [
        pt(10.0, 300.0, -100.0),      # imbalance 180 J
        pt(15.0, 3000.0, -1000.0),    # imbalance 1800 J (spike)
        pt(20.0, 300.0, -100.0),      # imbalance 180 J
    ]}]}
    raw, _ = storage_capacity_from_power_curves(pc, smoothing_window=1)
    smoothed, _ = storage_capacity_from_power_curves(pc, smoothing_window=3)
    assert raw == pytest.approx(1800.0 / 3600.0)
    assert smoothed < raw
    assert smoothed == pytest.approx((180.0 + 1800.0) / 2 / 3600.0)


def _write_case(tmp_path, power_curve_points, generator_power=40000.0):
    system = {
        'metadata': {'name': 'test'},
        'components': {
            'kites': [],
            'tethers': [],
            'ground_station': {
                'generators': [{'max_power': generator_power}],
                'storages': [
                    {'name': 'uc', 'type': 'capacitor_bank',
                     'capacity': 1137.7, 'efficiency': 1.0},
                    {'name': 'bat', 'type': 'battery_bank',
                     'capacity': 1137.7, 'efficiency': 1.0},
                ],
            },
        },
    }
    system_path = tmp_path / 'system.yml'
    write_yaml(system, system_path)
    curves_path = tmp_path / 'power_curves.yml'
    write_yaml({'power_curves': [{'wind_speed_data': power_curve_points}]},
               curves_path)
    return system_path, curves_path


def test_update_from_power_curves(tmp_path):
    """Capacity comes from the worst operating point, in Wh (raw max;
    smoothing pinned off since the points share a wind speed)."""
    points = [
        _point(_energies(300.0, 0.0, -100.0, 0.0),
               _times(60.0, 0.0, 40.0, 0.0)),
        # P_avg = 36 W; after reel-out: 7200 - 36*60 = 5040 J = 1.4 Wh
        _point(_energies(7200.0, 0.0, -3600.0, 0.0),
               _times(60.0, 0.0, 40.0, 0.0)),
        _point(_energies(1e9, 0.0, -1e9, 0.0),
               _times(60.0, 0.0, 40.0, 0.0), successful=False),
    ]
    system_path, curves_path = _write_case(tmp_path, points)
    summary = update_storage_capacity_after_power_curves(
        system_path, curves_path, StorageSizingSettings(smoothing_window=1))

    assert summary['method'] == 'cycle_energy_imbalance'
    assert summary['capacity_wh'] == pytest.approx(1.4)

    from tef.io import load_yaml
    system = load_yaml(system_path)
    for storage in system['components']['ground_station']['storages']:
        assert storage['capacity'] == pytest.approx(1.4)


def test_generator_power_fallback(tmp_path):
    """Without usable energy data, capacity scales with generator power."""
    system_path, curves_path = _write_case(
        tmp_path, [{'successful': False}], generator_power=80000.0)
    summary = update_storage_capacity_after_power_curves(
        system_path, curves_path, StorageSizingSettings())
    assert summary['method'] == 'scale_with_generator_power'
    assert summary['capacity_wh'] == pytest.approx(2 * 1137.7)


def test_safety_factor(tmp_path):
    import dataclasses
    points = [_point(_energies(7200.0, 0.0, -3600.0, 0.0),
                     _times(60.0, 0.0, 40.0, 0.0))]
    system_path, curves_path = _write_case(tmp_path, points)
    settings = dataclasses.replace(StorageSizingSettings(),
                                   safety_factor=1.5)
    summary = update_storage_capacity_after_power_curves(
        system_path, curves_path, settings)
    assert summary['capacity_wh'] == pytest.approx(1.4 * 1.5)
