"""Model-agnostic optimisation algorithms.

The optimiser sees only a numeric objective and a :class:`DesignSpace`.
It never imports the models: the caller injects an objective that maps
a design vector to a scalar (LCoE), so any model behind that objective
can be swapped without touching this module.

Two things live here:

1. :func:`solve_monotonic_input` - a bracketed 1-D solver used to size
   the generator so the emergent rated power equals a target. It needs
   no third-party dependency and assumes only monotonicity.
2. :func:`optimise` - a small dispatcher over pluggable methods. The
   default ``grid`` is a full-factorial search: for a low-dimensional,
   possibly noisy objective it is global within its resolution and it
   produces the objective surface that the study plots need. ``scipy``
   is imported lazily and only for the optional local-refinement
   methods, so it is not a hard dependency.

Public interface:
    Evaluation, OptimisationResult
    solve_monotonic_input
    grid_optimise, optimise
"""

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np

from tef.design_space import DesignSpace, GridPoints

# An objective maps a design vector to a scalar to be minimised.
Objective = Callable[[np.ndarray], float]


# ---------------------------------------------------------------------------
# 1-D bracketed solver (generator sizing for a target rated power)
# ---------------------------------------------------------------------------

def solve_monotonic_input(response: Callable[[float], float],
                          target: float,
                          lower: float,
                          upper: float,
                          tolerance: float,
                          max_iterations: int = 20,
                          max_expansions: int = 3,
                          bracket_rtol: float = 0.01) -> float:
    """Find ``x`` such that ``response(x) == target`` by bisection.

    ``response`` is assumed monotonically increasing over the bracket
    (as the emergent rated power is in the generator ``max_power``). The
    upper bound is doubled up to ``max_expansions`` times if the target
    is not yet bracketed, so a loose initial bracket still converges.

    Args:
        response: Monotonically increasing function of the input.
        target: Desired response value.
        lower: Lower bracket for the input.
        upper: Initial upper bracket for the input.
        tolerance: Absolute tolerance on the response.
        max_iterations: Maximum bisection steps.
        max_expansions: Maximum upper-bound doublings.

    Returns:
        The input value whose response matches the target.

    Raises:
        ValueError: If the target lies below ``response(lower)`` or the
            bracket cannot be expanded to contain it.
    """
    lowValue = response(lower)
    if lowValue - target > tolerance:
        raise ValueError(
            f"target {target:g} is below response(lower)={lowValue:g}; "
            f"widen the lower bound")

    highBound = upper
    highValue = response(highBound)
    expansions = 0
    while highValue - target < 0 and expansions < max_expansions:
        highBound *= 2.0
        highValue = response(highBound)
        expansions += 1
    if highValue - target < -tolerance:
        raise ValueError(
            f"could not bracket target {target:g}; response at "
            f"{highBound:g} is only {highValue:g}")

    lowBound = lower
    for _ in range(max_iterations):
        midBound = 0.5 * (lowBound + highBound)
        midValue = response(midBound)
        if abs(midValue - target) <= tolerance:
            return midBound
        # Stop once the input bracket is narrow, even if a noisy response
        # never meets the tolerance (otherwise the loop runs every
        # iteration). Bounds a noisy root-find to ~log2(1/bracket_rtol)
        # evaluations.
        if (highBound - lowBound) <= bracket_rtol * highBound:
            return midBound
        if midValue < target:
            lowBound = midBound
        else:
            highBound = midBound
    return 0.5 * (lowBound + highBound)


# ---------------------------------------------------------------------------
# Optimisation results
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Evaluation:
    """One evaluated design vector and its objective value."""

    vector: Tuple[float, ...]
    value: Optional[float]
    ok: bool = True
    error: Optional[str] = None


@dataclass(frozen=True)
class OptimisationResult:
    """Outcome of an optimisation over a design space."""

    method: str
    best_vector: Optional[np.ndarray]
    best_value: Optional[float]
    evaluations: List[Evaluation] = field(default_factory=list)

    @property
    def succeeded(self) -> bool:
        """True when at least one evaluation produced a finite value."""
        return self.best_vector is not None


# ---------------------------------------------------------------------------
# Methods
# ---------------------------------------------------------------------------

def grid_optimise(objective: Objective, design_space: DesignSpace,
                  points: GridPoints,
                  continue_on_failure: bool = True) -> OptimisationResult:
    """Minimise ``objective`` over a full-factorial grid.

    Every grid vertex is evaluated and recorded, so the result carries
    the full objective surface, not only the optimum. A failed
    evaluation is recorded and skipped when ``continue_on_failure`` is
    set.

    Args:
        objective: Maps a design vector to the scalar to minimise.
        design_space: The bounds and grid geometry.
        points: Grid resolution (see :meth:`DesignSpace.grid`).
        continue_on_failure: Record and skip failed evaluations rather
            than raising.

    Returns:
        The optimisation result with the best vector and all
        evaluations.
    """
    evaluations: List[Evaluation] = []
    bestVector: Optional[np.ndarray] = None
    bestValue: Optional[float] = None

    # Materialise the grid so the total cell count is known for progress.
    grid = list(design_space.grid(points))
    total = len(grid)
    names = [var.name for var in design_space.variables]

    for index, vector in enumerate(grid, start=1):
        label = "  ".join(f"{name}={value:g}"
                          for name, value in zip(names, vector))
        print(f"\n{'=' * 64}\n [cell {index}/{total}]  {label}\n{'=' * 64}",
              flush=True)
        try:
            value = float(objective(vector))
        except Exception as exc:  # noqa: BLE001 - a study records failures
            if not continue_on_failure:
                raise
            evaluations.append(Evaluation(tuple(vector), None, False, str(exc)))
            print(f" [cell {index}/{total}] FAILED: {exc}", flush=True)
            continue
        evaluations.append(Evaluation(tuple(vector), value))
        print(f" [cell {index}/{total}] done: LCoE = {value:.1f} EUR/MWh",
              flush=True)
        if np.isfinite(value) and (bestValue is None or value < bestValue):
            bestValue = value
            bestVector = vector

    return OptimisationResult('grid', bestVector, bestValue, evaluations)


def _scipy_minimise(objective: Objective, design_space: DesignSpace,
                    x0: np.ndarray, method: str,
                    scipy_method: str) -> OptimisationResult:
    """Local refinement through ``scipy.optimize.minimize`` (lazy import)."""
    try:
        from scipy.optimize import minimize
    except ImportError as exc:
        raise ImportError(
            f"method '{method}' needs scipy; install it or use method='grid'"
        ) from exc

    evaluations: List[Evaluation] = []

    def tracked(vector: np.ndarray) -> float:
        clipped = design_space.clip(vector)
        value = float(objective(clipped))
        evaluations.append(Evaluation(tuple(clipped), value))
        return value

    bounds = list(zip(design_space.lower, design_space.upper))
    outcome = minimize(tracked, np.asarray(x0, dtype=float),
                       method=scipy_method, bounds=bounds)
    bestVector = design_space.clip(outcome.x)
    return OptimisationResult(method, bestVector, float(outcome.fun),
                              evaluations)


def optimise(objective: Objective, design_space: DesignSpace,
             method: str = 'grid', points: Optional[GridPoints] = None,
             x0: Optional[np.ndarray] = None,
             continue_on_failure: bool = True) -> OptimisationResult:
    """Minimise ``objective`` over ``design_space`` with a chosen method.

    Methods:
        ``grid`` (default): full-factorial search; ``points`` required.
        ``nelder_mead``: derivative-free local refinement (scipy).
        ``slsqp``: gradient local refinement (scipy).

    The local methods start from ``x0`` (defaulting to the box centre).
    They are intended to refine the best grid vertex, not to be the
    primary search, because the objective wraps inner solvers and its
    finite-difference gradients are noisy.

    Args:
        objective: Maps a design vector to the scalar to minimise.
        design_space: The bounds and grid geometry.
        method: ``grid``, ``nelder_mead`` or ``slsqp``.
        points: Grid resolution for ``method='grid'``.
        x0: Start vector for the local methods.
        continue_on_failure: Passed to the grid method.

    Returns:
        The optimisation result.
    """
    if method == 'grid':
        if points is None:
            raise ValueError("method='grid' requires points")
        return grid_optimise(objective, design_space, points,
                             continue_on_failure)
    if x0 is None:
        x0 = 0.5 * (design_space.lower + design_space.upper)
    if method == 'nelder_mead':
        return _scipy_minimise(objective, design_space, x0, method,
                               'Nelder-Mead')
    if method == 'slsqp':
        return _scipy_minimise(objective, design_space, x0, method, 'SLSQP')
    raise ValueError(f"unknown optimisation method: {method!r}")
