"""Unit tests for the model-agnostic design space."""

import numpy as np
import pytest

from tef.design_space import DesignSpace, DesignVariableSpec


def make_space():
    return DesignSpace(
        variables=(
            DesignVariableSpec('flat_area_m2', 25.0, 250.0),
            DesignVariableSpec('allowable_tether_stress_pa', 2.0e8, 6.0e8),
        ),
        fixed={'tether_length_m': 500.0, 'rated_power_w': 100000.0},
    )


def test_spec_rejects_unknown_field():
    with pytest.raises(ValueError):
        DesignVariableSpec('not_a_field', 0.0, 1.0)


def test_spec_rejects_bad_bounds():
    with pytest.raises(ValueError):
        DesignVariableSpec('flat_area_m2', 10.0, 10.0)


def test_space_geometry():
    space = make_space()
    assert space.names == ['flat_area_m2', 'allowable_tether_stress_pa']
    assert space.dimension == 2
    np.testing.assert_allclose(space.lower, [25.0, 2.0e8])
    np.testing.assert_allclose(space.upper, [250.0, 6.0e8])


def test_space_rejects_fixed_variable_overlap():
    with pytest.raises(ValueError):
        DesignSpace(
            variables=(DesignVariableSpec('flat_area_m2', 25.0, 250.0),),
            fixed={'flat_area_m2': 100.0},
        )


def test_to_design_merges_fixed_variables_and_overrides():
    space = make_space()
    design = space.to_design([100.0, 4.0e8], case_name='c',
                             generator_max_power_w=120000.0)
    assert design.flat_area_m2 == 100.0
    assert design.allowable_tether_stress_pa == 4.0e8
    assert design.tether_length_m == 500.0        # from fixed
    assert design.rated_power_w == 100000.0        # from fixed
    assert design.generator_max_power_w == 120000.0  # from override
    assert design.case_name == 'c'


def test_to_design_wrong_length_raises():
    space = make_space()
    with pytest.raises(ValueError):
        space.to_design([100.0])


def test_clip_respects_bounds():
    space = make_space()
    np.testing.assert_allclose(space.clip([10.0, 9.0e8]), [25.0, 6.0e8])


def test_grid_count_and_fastest_axis():
    space = make_space()
    vectors = list(space.grid({'flat_area_m2': 3,
                               'allowable_tether_stress_pa': 2}))
    assert len(vectors) == 6
    # The last variable varies fastest: first two share flat_area_m2.
    assert vectors[0][0] == vectors[1][0]
    assert vectors[0][1] != vectors[1][1]


def test_grid_points_as_int_and_list_agree():
    space = make_space()
    assert len(list(space.grid(4))) == 16
    assert len(list(space.grid([4, 4]))) == 16
