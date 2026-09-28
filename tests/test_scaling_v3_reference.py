"""V3 reference reproduction and sensor-inclusion tests."""

import pytest

from tef.system_scaling import DesignVariables, compute_scaled_quantities


def test_v3_reference_reproduction(v3_reproduction_settings):
    """With sensor mass disabled and KCU calibrated to V3, the 25 m2
    design reproduces Bredael's original V3 system inputs."""
    quantities = compute_scaled_quantities(
        DesignVariables(flat_area_m2=25.0), v3_reproduction_settings)

    assert quantities.projected_area_m2 == pytest.approx(19.75)
    assert quantities.wing_and_bridle_mass_kg == pytest.approx(10.6)
    assert quantities.kcu_mass_kg == pytest.approx(8.4)
    assert quantities.control_system_mass_written_kg == pytest.approx(8.4)
    assert quantities.max_tether_force_n == pytest.approx(8200.0)
    assert quantities.tether_diameter_m == pytest.approx(0.006)
    assert quantities.generator_max_power_w == pytest.approx(40000.0)
    assert quantities.max_tether_speed_m_s == pytest.approx(10.0)
    assert quantities.tether_length_m == pytest.approx(500.0)
    # 10.6 + 8.4 = 19 kg airborne mass used in AWESPA
    assert quantities.total_airborne_mass_kg == pytest.approx(19.0)


def test_sensor_inclusion(base_settings):
    """With sensor inclusion enabled, the written control system mass
    carries KCU + sensor and the airborne total is consistent."""
    quantities = compute_scaled_quantities(
        DesignVariables(flat_area_m2=25.0), base_settings)

    assert quantities.sensor_mass_kg == pytest.approx(3.4)
    assert quantities.control_system_mass_written_kg == pytest.approx(
        quantities.kcu_mass_kg + 3.4)
    assert quantities.total_airborne_mass_kg == pytest.approx(
        quantities.wing_and_bridle_mass_kg
        + quantities.control_system_mass_written_kg)
