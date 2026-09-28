"""Anchor-size tests for wing mass, tether sizing and generator power."""

import pytest

from tef.system_scaling import DesignVariables, compute_scaled_quantities

ANCHOR_AREAS = [25.0, 50.0, 100.0, 200.0, 400.0]
ANCHOR_WING_MASSES = [10.60, 23.97, 55.79, 133.75, 330.21]
# Constant wing loading + constant allowable stress: d ~ sqrt(A_proj)
EXPECTED_TETHER_DIAMETERS = [0.0060, 0.00849, 0.01200, 0.01697, 0.02400]
EXPECTED_GENERATOR_POWERS_KW = [40.0, 80.0, 160.0, 320.0, 640.0]


@pytest.mark.parametrize(
    'area, wing_mass', list(zip(ANCHOR_AREAS, ANCHOR_WING_MASSES)))
def test_wing_mass_anchors(base_settings, area, wing_mass):
    quantities = compute_scaled_quantities(
        DesignVariables(flat_area_m2=area), base_settings)
    assert quantities.wing_and_bridle_mass_kg == pytest.approx(wing_mass)


def test_wing_mass_extrapolation_disabled(base_settings):
    with pytest.raises(ValueError, match='extrapolation'):
        compute_scaled_quantities(
            DesignVariables(flat_area_m2=500.0), base_settings)


@pytest.mark.parametrize(
    'area, diameter', list(zip(ANCHOR_AREAS, EXPECTED_TETHER_DIAMETERS)))
def test_tether_diameter_anchors(base_settings, area, diameter):
    quantities = compute_scaled_quantities(
        DesignVariables(flat_area_m2=area), base_settings)
    assert quantities.tether_diameter_m == pytest.approx(diameter, rel=1e-3)


def test_tether_force_scales_linearly_with_projected_area(base_settings):
    q25 = compute_scaled_quantities(
        DesignVariables(flat_area_m2=25.0), base_settings)
    q100 = compute_scaled_quantities(
        DesignVariables(flat_area_m2=100.0), base_settings)
    assert q100.max_tether_force_n == pytest.approx(
        4.0 * q25.max_tether_force_n)
    assert q100.tether_diameter_m == pytest.approx(
        2.0 * q25.tether_diameter_m)


@pytest.mark.parametrize(
    'area, power_kw', list(zip(ANCHOR_AREAS, EXPECTED_GENERATOR_POWERS_KW)))
def test_generator_power_anchors(base_settings, area, power_kw):
    quantities = compute_scaled_quantities(
        DesignVariables(flat_area_m2=area), base_settings)
    assert quantities.generator_max_power_w == pytest.approx(
        power_kw * 1e3, rel=1e-6)


def test_generator_sizing_priority(base_settings):
    """Explicit max power wins over rated*crest over specific power."""
    design_explicit = DesignVariables(
        flat_area_m2=25.0, generator_max_power_w=55000.0,
        rated_power_w=20000.0, crest_factor=2.0,
        generator_specific_power_w_m2_projected=1000.0)
    q = compute_scaled_quantities(design_explicit, base_settings)
    assert q.generator_max_power_w == pytest.approx(55000.0)

    design_crest = DesignVariables(
        flat_area_m2=25.0, rated_power_w=20000.0, crest_factor=2.0,
        generator_specific_power_w_m2_projected=1000.0)
    q = compute_scaled_quantities(design_crest, base_settings)
    assert q.generator_max_power_w == pytest.approx(40000.0)

    design_specific = DesignVariables(
        flat_area_m2=25.0,
        generator_specific_power_w_m2_projected=1000.0)
    q = compute_scaled_quantities(design_specific, base_settings)
    assert q.generator_max_power_w == pytest.approx(19750.0)
