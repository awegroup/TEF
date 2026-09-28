"""Unit tests for the model-agnostic optimisation algorithms.

The objectives here are plain numeric functions, so these tests run
without AWESPA or EcoMo.
"""

import numpy as np
import pytest

from tef.design_space import DesignSpace, DesignVariableSpec
from tef.optimisation import (
    grid_optimise,
    optimise,
    solve_monotonic_input,
)


def make_space():
    return DesignSpace(
        variables=(
            DesignVariableSpec('flat_area_m2', 0.0, 10.0),
            DesignVariableSpec('allowable_tether_stress_pa', 0.0, 10.0),
        ),
        fixed={},
    )


# -- solve_monotonic_input --------------------------------------------------

def test_solve_linear_response():
    # response(x) = 2x, target 10 -> x = 5
    x = solve_monotonic_input(lambda v: 2.0 * v, target=10.0,
                              lower=0.0, upper=100.0, tolerance=1e-6)
    assert x == pytest.approx(5.0, abs=1e-3)


def test_solve_expands_upper_bound():
    # target beyond the initial upper bound forces expansion
    x = solve_monotonic_input(lambda v: v, target=80.0,
                              lower=0.0, upper=10.0, tolerance=1e-6)
    assert x == pytest.approx(80.0, abs=1e-3)


def test_solve_target_below_lower_raises():
    with pytest.raises(ValueError):
        solve_monotonic_input(lambda v: v + 5.0, target=1.0,
                              lower=0.0, upper=10.0, tolerance=1e-6)


# -- grid_optimise ----------------------------------------------------------

def test_grid_finds_minimum_cell():
    space = make_space()
    # Minimum at the (5, 5) grid vertex.
    objective = lambda x: (x[0] - 5.0) ** 2 + (x[1] - 5.0) ** 2
    result = grid_optimise(objective, space, points=11)
    assert result.succeeded
    np.testing.assert_allclose(result.best_vector, [5.0, 5.0])
    assert result.best_value == pytest.approx(0.0)
    assert len(result.evaluations) == 121


def test_grid_records_and_skips_failures():
    space = make_space()

    def flaky(x):
        if x[0] == 0.0:
            raise RuntimeError("boom")
        return float(x[0] + x[1])

    result = grid_optimise(flaky, space, points=3)
    statuses = [e.ok for e in result.evaluations]
    assert statuses.count(False) == 3      # the whole x0 == 0 column
    assert result.succeeded


# -- optimise dispatcher ----------------------------------------------------

def test_optimise_grid_requires_points():
    space = make_space()
    with pytest.raises(ValueError):
        optimise(lambda x: 0.0, space, method='grid')


def test_optimise_unknown_method_raises():
    space = make_space()
    with pytest.raises(ValueError):
        optimise(lambda x: 0.0, space, method='banana', points=3)
