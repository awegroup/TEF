"""Declarative design space for optimisation.

The design space names the design variables that are optimised, their
bounds, and the fixed values that complete a :class:`DesignVariables`
instance. It is deliberately model-agnostic: it knows only the
``DesignVariables`` schema (field names), never the models behind the
framework. The optimiser works on plain numeric vectors and asks the
design space to turn a vector into a ``DesignVariables`` object, so the
mapping between "what is optimised" and "how a design is built" lives
in exactly one place.

Adding or removing a design variable (for example adding wing loading
later) is a change to the study configuration or to the tuple of
:class:`DesignVariableSpec` only; the optimiser and the model pipeline
are untouched.

Public interface:
    DesignVariableSpec, DesignSpace
"""

from dataclasses import dataclass, fields
from itertools import product
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np

from tef.system_scaling import DesignVariables

# Field names of DesignVariables that may be optimised or fixed through
# the design space (everything except the bookkeeping field).
_DESIGN_FIELDS = {f.name for f in fields(DesignVariables)} - {'case_name'}

# Points specification for a grid: one count for all variables, a count
# per variable (in order), or a count per variable name.
GridPoints = Union[int, Sequence[int], Dict[str, int]]


@dataclass(frozen=True)
class DesignVariableSpec:
    """One optimised design variable and its inclusive bounds.

    Attributes:
        name: A ``DesignVariables`` field name (e.g. ``flat_area_m2``).
        lower: Lower bound.
        upper: Upper bound (must exceed ``lower``).
    """

    name: str
    lower: float
    upper: float

    def __post_init__(self) -> None:
        if self.name not in _DESIGN_FIELDS:
            raise ValueError(
                f"'{self.name}' is not an optimisable DesignVariables field; "
                f"choose one of {sorted(_DESIGN_FIELDS)}")
        if not self.upper > self.lower:
            raise ValueError(
                f"upper bound must exceed lower bound for '{self.name}' "
                f"(got lower={self.lower}, upper={self.upper})")


@dataclass(frozen=True)
class DesignSpace:
    """The optimised variables plus the fixed design values.

    Attributes:
        variables: The optimised design variables, in vector order.
        fixed: Fixed ``DesignVariables`` fields shared by every design
            in the space (e.g. ``rated_power_w``, ``tether_length_m``).
    """

    variables: Tuple[DesignVariableSpec, ...]
    fixed: Dict[str, Any]

    def __post_init__(self) -> None:
        if not self.variables:
            raise ValueError("a design space needs at least one variable")
        names = self.names
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate design variables: {names}")
        unknownFixed = set(self.fixed) - _DESIGN_FIELDS
        if unknownFixed:
            raise ValueError(
                f"unknown fixed design fields: {sorted(unknownFixed)}")
        overlap = set(names) & set(self.fixed)
        if overlap:
            raise ValueError(
                f"fields are both optimised and fixed: {sorted(overlap)}")

    # -- geometry -----------------------------------------------------------

    @property
    def names(self) -> List[str]:
        """Design variable names in vector order."""
        return [spec.name for spec in self.variables]

    @property
    def dimension(self) -> int:
        """Number of optimised variables."""
        return len(self.variables)

    @property
    def lower(self) -> np.ndarray:
        """Lower bounds as a vector."""
        return np.array([spec.lower for spec in self.variables], dtype=float)

    @property
    def upper(self) -> np.ndarray:
        """Upper bounds as a vector."""
        return np.array([spec.upper for spec in self.variables], dtype=float)

    def clip(self, vector: Sequence[float]) -> np.ndarray:
        """Clip a vector to the bounds."""
        return np.clip(np.asarray(vector, dtype=float), self.lower, self.upper)

    # -- design construction ------------------------------------------------

    def to_design(self, vector: Sequence[float],
                  case_name: Optional[str] = None,
                  **overrides: Any) -> DesignVariables:
        """Build a :class:`DesignVariables` from a design vector.

        The fixed values, then the optimised variables, then any
        keyword overrides are combined. Overrides let a caller inject a
        value resolved during evaluation (for example a ``max_power``
        solved to hit a target rated power).

        Args:
            vector: Values for the optimised variables, in vector order.
            case_name: Optional case name recorded on the design.
            **overrides: Extra ``DesignVariables`` fields to set.

        Returns:
            The assembled design point.
        """
        vector = np.asarray(vector, dtype=float)
        if vector.shape != (self.dimension,):
            raise ValueError(
                f"expected a vector of length {self.dimension}, "
                f"got shape {vector.shape}")
        unknownOverrides = set(overrides) - _DESIGN_FIELDS
        if unknownOverrides:
            raise ValueError(
                f"unknown design overrides: {sorted(unknownOverrides)}")

        values: Dict[str, Any] = dict(self.fixed)
        values.update(zip(self.names, (float(v) for v in vector)))
        values.update(overrides)
        return DesignVariables(case_name=case_name, **values)

    # -- grid ---------------------------------------------------------------

    def _points_per_variable(self, points: GridPoints) -> List[int]:
        """Resolve a grid points specification to one count per variable."""
        if isinstance(points, int):
            counts = [points] * self.dimension
        elif isinstance(points, dict):
            missing = set(self.names) - set(points)
            if missing:
                raise ValueError(f"grid points missing for {sorted(missing)}")
            counts = [int(points[name]) for name in self.names]
        else:
            counts = [int(p) for p in points]
            if len(counts) != self.dimension:
                raise ValueError(
                    f"expected {self.dimension} grid counts, got {len(counts)}")
        if any(count < 1 for count in counts):
            raise ValueError("grid needs at least one point per variable")
        return counts

    def grid_axes(self, points: GridPoints) -> List[np.ndarray]:
        """Return the per-variable sample points of a full-factorial grid."""
        counts = self._points_per_variable(points)
        return [np.linspace(spec.lower, spec.upper, count)
                for spec, count in zip(self.variables, counts)]

    def grid(self, points: GridPoints) -> Iterator[np.ndarray]:
        """Iterate the vectors of a full-factorial grid.

        The last variable varies fastest.

        Args:
            points: One count for all variables, a count per variable in
                order, or a count per variable name.

        Yields:
            Each grid vertex as a vector.
        """
        for combination in product(*self.grid_axes(points)):
            yield np.array(combination, dtype=float)
