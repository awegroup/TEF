"""Plots for the design-optimisation study.

Reads the study outputs written by
:func:`tef.design_studies.run_design_optimisation` and produces the
figures of the plan:

- LCoE of the optimum versus generator power (the scaling result).
- Optimal wing area versus generator power (the matching law).
- The LCoE surface over the design space at one generator limit, with the
  optimum marked.
- Diagnostics of the optimum versus generator power (crest factors,
  emergent rated power, tether diameter, capacity factor, storage,
  airborne mass, AEP).

The module is model-agnostic: it reads the CSV/YAML the study writes,
not the models. matplotlib is imported at module load because it is the
sole purpose of this module; install it with the ``plots`` extra.

Public interface:
    load_optimum_table, generator_surface_files
    plot_lcoe_vs_generator_power, plot_optimal_area_vs_generator_power
    plot_rated_power_vs_generator_power
    plot_lcoe_surface, plot_lcoe_curve, plot_diagnostics_vs_generator_power
    plot_all
"""

import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
except ImportError as exc:  # pragma: no cover - import-time guard
    raise ImportError(
        "tef.plotting needs matplotlib; install it, e.g. "
        "pip install matplotlib") from exc


# Column -> (axis label, unit label, multiplier applied to the raw value).
_AXIS = {
    'flat_area_m2': ('Wing area', 'm$^2$', 1.0),
    'allowable_tether_stress_pa': ('Allowable tether stress', 'GPa', 1e-9),
    'generator_max_power_kw': ('Generator mechanical power limit', 'kW', 1.0),
    'emergent_rated_power_kw': ('Rated power', 'kW', 1.0),
    'peak_mechanical_power_kw': ('Full-cycle mechanical peak', 'kW', 1.0),
    'lcoe_eur_per_mwh': ('LCoE', 'EUR/MWh', 1.0),
    'crest_peak_over_rated': ('Crest (peak / rated)', '-', 1.0),
    'crest_limit_over_rated': ('Crest (limit / rated)', '-', 1.0),
    'tether_diameter_mm': ('Tether diameter', 'mm', 1.0),
    'capacity_factor': ('Capacity factor', '-', 1.0),
    'storage_capacity_wh': ('Storage capacity', 'Wh', 1.0),
    'total_airborne_mass_kg': ('Airborne mass', 'kg', 1.0),
    'aep_mwh': ('AEP', 'MWh', 1.0),
}

_GENERATOR_DIR = re.compile(r'gen_(\d+)kW$')

# The outer-axis column of the optimum table.
_OUTER = 'generator_max_power_kw'

# grid_surface.csv columns that are NOT design variables.
_NON_VARIABLE_COLUMNS = ('lcoe_eur_per_mwh', 'status', 'error')


def _label(column: str) -> str:
    """Axis label with unit for a known column, else the raw name."""
    if column in _AXIS:
        name, unit, _ = _AXIS[column]
        return f"{name} ({unit})"
    return column


def _scale(column: str) -> float:
    """Multiplier that converts a raw column to its plotted unit."""
    return _AXIS[column][2] if column in _AXIS else 1.0


def _save(fig, output_dir: Optional[Path], name: str) -> None:
    """Save a figure as PNG and PDF when an output directory is given."""
    if output_dir is None:
        return
    outputDir = Path(output_dir)
    outputDir.mkdir(parents=True, exist_ok=True)
    for suffix in ('png', 'pdf'):
        fig.savefig(outputDir / f"{name}.{suffix}", bbox_inches='tight',
                    dpi=200)


# ---------------------------------------------------------------------------
# Data access
# ---------------------------------------------------------------------------

def load_optimum_table(results_dir: Path) -> pd.DataFrame:
    """Load ``optimum_by_generator_power.csv`` sorted by generator power."""
    path = Path(results_dir) / 'optimum_by_generator_power.csv'
    if not path.exists():
        raise FileNotFoundError(
            f"optimum table not found: {path}; run the study first")
    frame = pd.read_csv(path)
    return frame.sort_values(_OUTER).reset_index(drop=True)


def generator_surface_files(results_dir: Path) -> List[Tuple[float, Path]]:
    """Return ``(generator_kW, grid_surface.csv)`` pairs, sorted by power."""
    resultsDir = Path(results_dir)
    found = []
    for child in resultsDir.iterdir():
        match = _GENERATOR_DIR.search(child.name)
        surface = child / 'grid_surface.csv'
        if match and surface.exists():
            found.append((float(match.group(1)), surface))
    return sorted(found)


def _surface_variables(surface_path: Path) -> List[str]:
    """Design-variable columns of a grid_surface.csv (all but lcoe/status)."""
    columns = pd.read_csv(surface_path, nrows=0).columns
    return [c for c in columns if c not in _NON_VARIABLE_COLUMNS]


# ---------------------------------------------------------------------------
# Optimum-versus-generator-power curves
# ---------------------------------------------------------------------------

def plot_lcoe_vs_generator_power(results_dir: Path,
                                 output_dir: Optional[Path] = None,
                                 ax=None):
    """LCoE of the optimal design versus generator power."""
    frame = load_optimum_table(results_dir)
    createdFig = ax is None
    if createdFig:
        fig, ax = plt.subplots(figsize=(5.0, 3.6))
    else:
        fig = ax.figure

    ax.plot(frame[_OUTER], frame['lcoe_eur_per_mwh'], marker='o',
            color='#1f77b4')
    ax.set_xlabel(_label(_OUTER))
    ax.set_ylabel(_label('lcoe_eur_per_mwh'))
    ax.grid(True, alpha=0.3)

    if createdFig:
        fig.tight_layout()
        _save(fig, output_dir, 'lcoe_vs_generator_power')
    return fig, ax


def plot_optimal_area_vs_generator_power(results_dir: Path,
                                         output_dir: Optional[Path] = None,
                                         ax=None):
    """Optimal wing area versus generator power (the matching law)."""
    frame = load_optimum_table(results_dir)
    createdFig = ax is None
    if createdFig:
        fig, ax = plt.subplots(figsize=(5.0, 3.6))
    else:
        fig = ax.figure

    ax.plot(frame[_OUTER], frame['flat_area_m2'], marker='s', color='#1f77b4')
    ax.set_xlabel(_label(_OUTER))
    ax.set_ylabel(_label('flat_area_m2'))
    ax.grid(True, alpha=0.3)

    if createdFig:
        fig.tight_layout()
        _save(fig, output_dir, 'optimal_area_vs_generator_power')
    return fig, ax


def plot_rated_power_vs_generator_power(results_dir: Path,
                                        output_dir: Optional[Path] = None,
                                        ax=None):
    """Cycle power levels versus the generator mechanical power limit, for
    the optimal design at each rating: the generator cap itself (y = x),
    the full-cycle peak mechanical power (the drivetrain/generator cost
    basis), and the emergent rated power (cycle-average electrical).
    """
    frame = load_optimum_table(results_dir)
    createdFig = ax is None
    if createdFig:
        fig, ax = plt.subplots(figsize=(5.0, 3.6))
    else:
        fig = ax.figure

    limit = frame[_OUTER]
    lo, hi = limit.min() * 0.75, limit.max() * 1.04
    ax.plot([lo, hi], [lo, hi], '--', color='0.55', lw=1.3,
            label=f"{_label(_OUTER)}  (y = x)")
    ax.plot(limit, frame['peak_mechanical_power_kw'], marker='^',
            color='#1f4e79', label=_label('peak_mechanical_power_kw'))
    ax.plot(limit, frame['emergent_rated_power_kw'], marker='o',
            color='#4a90d9', label=_label('emergent_rated_power_kw'))
    ax.set_xlabel(_label(_OUTER))
    ax.set_ylabel('Rated power (kW)')
    ax.set_xlim(lo, hi)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, loc='upper left', fontsize=9)

    if createdFig:
        fig.tight_layout()
        _save(fig, output_dir, 'rated_power_vs_generator_power')
    return fig, ax


# ---------------------------------------------------------------------------
# LCoE surface over the design space
# ---------------------------------------------------------------------------

def _surface_for(results_dir: Path, generator_kw: float) -> Path:
    for power, path in generator_surface_files(results_dir):
        if abs(power - generator_kw) < 1e-6:
            return path
    raise FileNotFoundError(
        f"no grid_surface.csv for {generator_kw:g} kW under {results_dir}")


def _isoline_levels(low: float, high: float, count: int,
                    mode: str = 'optimum') -> np.ndarray:
    """LCoE values to draw iso-contours at, between `low` and `high`.

    'linear' gives evenly spaced round numbers -- readable on the steep
    parts of the surface, but nearly blank in the flat basin around the
    optimum. 'optimum' instead spaces them geometrically as
    ``low * (1 + f)`` with f from 1% up to the full range, putting most
    lines close to the minimum so the near-optimal region is legible.
    """
    if mode == 'linear':
        return MaxNLocator(nbins=count).tick_values(low, high)
    if mode != 'optimum':
        raise ValueError(f"unknown isoline_mode {mode!r}")
    if not (low > 0 and high > low):
        return MaxNLocator(nbins=count).tick_values(low, high)
    topFraction = high / low - 1.0
    lowFraction = min(0.01, topFraction / 2)
    fractions = np.geomspace(lowFraction, topFraction, max(count, 2))
    levels = low * (1.0 + fractions)
    # Round to a readable number of significant figures, then drop the
    # duplicates that rounding creates among the closest-in levels.
    decimals = max(0, 2 - int(np.floor(np.log10(max(low, 1e-12)))))
    return np.unique(np.round(levels, decimals))


def _labelled_levels(contour_set, x_values, y_values, mode: str = 'auto'):
    """The contour levels that should carry an inline label.

    'auto' keeps a level only when one of its lines is long enough to hold
    a label without colliding with its neighbours -- length measured as a
    fraction of the axes, so it does not depend on the units. The tight
    rings around an optimum fail this and are left clean; the colourbar
    still carries a white tick for every level.
    """
    if mode == 'none':
        return []
    if mode == 'all':
        return list(contour_set.levels)
    if not (mode == 'auto' or np.isscalar(mode)):
        raise ValueError(f"unknown isoline_labels {mode!r}")

    xSpan = float(np.ptp(x_values)) or 1.0
    ySpan = float(np.ptp(y_values)) or 1.0
    longest = {}
    for level, segments in zip(contour_set.levels, contour_set.allsegs):
        for segment in segments:
            if len(segment) < 2:
                continue
            steps = np.diff(np.asarray(segment), axis=0)
            length = float(np.hypot(steps[:, 0] / xSpan,
                                    steps[:, 1] / ySpan).sum())
            longest[level] = max(longest.get(level, 0.0), length)

    minLength = 0.9   # in axes widths; ~3x a label's own width
    roomy = sorted(level for level, length in longest.items()
                   if length >= minLength)
    if mode == 'auto':
        return roomy
    # An int keeps N of the roomy lines, spread evenly through the value
    # range rather than taking the N longest -- the longest lines are
    # usually neighbours, so their labels would pile up in one corner.
    count = int(mode)
    if count <= 0 or not roomy:
        return []
    if count >= len(roomy):
        return roomy
    picks = np.linspace(0, len(roomy) - 1, count)
    return [roomy[int(round(index))] for index in picks]


def plot_lcoe_surface(results_dir: Path, generator_kw: float,
                      output_dir: Optional[Path] = None, ax=None,
                      levels=None, isolines=None,
                      isoline_mode: str = 'optimum',
                      isoline_labels: str = 'auto',
                      isoline_width: float = 0.7,
                      name_suffix: str = ''):
    """Filled-contour LCoE surface over the two design variables at one
    generator limit, with the optimum marked.

    Only defined for a two-variable design space; a one-variable study
    should use :func:`plot_lcoe_curve`.

    Args:
        levels: Explicit contour levels (e.g. ``np.linspace(lo, hi, 61)``).
            Pass the SAME array across multiple calls (different
            generators) so a given LCoE value gets the same colour in every
            figure -- see :func:`plot_all`, which does this automatically.
            None (default) auto-scales each figure to its own data, as
            before.
        isolines: Draw white LCoE iso-contours on top of the filled
            surface, labelled inline, so a colour can be read back as a
            number. True picks ~8 round values over this surface's own
            range; an int asks for about that many; a sequence gives the
            exact LCoE values (EUR/MWh). None/False (default) draws none.
        isoline_mode: How auto-picked levels are spaced. 'optimum'
            (default) spaces them geometrically upward from this surface's
            minimum, so the flat basin around the optimum gets most of the
            lines -- that is where the design question lives. 'linear'
            spreads round values evenly over the range, which reads well
            on the steep edges but leaves the basin bare. Ignored when
            explicit levels are passed.
        isoline_labels: Which iso-contours carry an inline number.
            'auto' (default) labels only the lines long enough to hold a
            legible label, which drops the tight rings around the optimum
            where labels would collide; 'all' labels every line; 'none'
            labels none and leaves the colourbar ticks to carry the
            values; an int labels that many lines, spread through the
            value range.
        isoline_width: Line width of the iso-contours; raise it for
            projection.
        name_suffix: Appended to the saved figure name, e.g. ``_isolines``,
            so a variant does not overwrite the plain figure.
    """
    frame = pd.read_csv(_surface_for(results_dir, generator_kw))
    frame = frame[frame.get('status', 'ok') == 'ok']
    variables = [c for c in frame.columns
                 if c not in _NON_VARIABLE_COLUMNS]
    if len(variables) != 2:
        raise ValueError(
            f"surface plot needs exactly two design variables, "
            f"found {variables}")

    xName, yName = variables
    grid = frame.pivot_table(index=yName, columns=xName,
                             values='lcoe_eur_per_mwh')
    xValues = grid.columns.to_numpy() * _scale(xName)
    yValues = grid.index.to_numpy() * _scale(yName)

    createdFig = ax is None
    if createdFig:
        fig, ax = plt.subplots(figsize=(5.4, 4.0))
    else:
        fig = ax.figure

    values = grid.to_numpy()
    contour = ax.contourf(xValues, yValues, values,
                          levels=60 if levels is None else levels,
                          cmap='viridis')
    contour.set_edgecolor('face')
    colorbar = fig.colorbar(contour, ax=ax,
                            label=_label('lcoe_eur_per_mwh'))

    # White iso-LCoE lines, so the reader can put a number on a colour.
    if isolines is not None and isolines is not False:
        finite = values[np.isfinite(values)]
        low, high = float(finite.min()), float(finite.max())
        if isolines is True or np.isscalar(isolines):
            count = 12 if isolines is True else int(isolines)
            lineLevels = _isoline_levels(low, high, count, isoline_mode)
        else:
            lineLevels = np.asarray(isolines, dtype=float)
        lineLevels = sorted({float(level) for level in lineLevels
                             if low < level < high})
        if lineLevels:
            span = high - low
            fmt = '%.0f' if span > 20 else ('%.1f' if span > 2 else '%.2f')
            lines = ax.contour(xValues, yValues, values, levels=lineLevels,
                               colors='white', linewidths=isoline_width,
                               alpha=0.9)
            labelled = _labelled_levels(lines, xValues, yValues,
                                        isoline_labels)
            if labelled:
                ax.clabel(lines, levels=labelled, inline=True, fontsize=7,
                          fmt=fmt, colors='white')
            colorbar.add_lines(lines)

    # Overlay the sampled grid nodes.
    ax.plot(frame[xName] * _scale(xName), frame[yName] * _scale(yName),
            linestyle='none', marker='o', markersize=3, color='white',
            markeredgecolor='none')

    # Mark the optimum cell of this surface.
    best = frame.loc[frame['lcoe_eur_per_mwh'].idxmin()]
    ax.plot(best[xName] * _scale(xName), best[yName] * _scale(yName),
            linestyle='none', marker='*', color='gold', markersize=15,
            markeredgecolor='k', label='optimum')
    ax.set_xlabel(_label(xName))
    ax.set_ylabel(_label(yName))
    ax.set_title(f"LCoE surface at {generator_kw:g} kW generator")
    ax.legend(loc='upper right')

    if createdFig:
        fig.tight_layout()
        _save(fig, output_dir,
              f'lcoe_surface_{int(round(generator_kw)):04d}kW{name_suffix}')
    return fig, ax


def plot_lcoe_curve(results_dir: Path, generator_kw: float,
                    output_dir: Optional[Path] = None, ax=None):
    """LCoE versus the single design variable at one generator limit.

    The one-variable counterpart of :func:`plot_lcoe_surface`. The optimum
    is marked.
    """
    frame = pd.read_csv(_surface_for(results_dir, generator_kw))
    frame = frame[frame.get('status', 'ok') == 'ok']
    variables = [c for c in frame.columns
                 if c not in _NON_VARIABLE_COLUMNS]
    if len(variables) != 1:
        raise ValueError(
            f"curve plot needs exactly one design variable, found {variables}")

    xName = variables[0]
    frame = frame.sort_values(xName)
    xValues = frame[xName].to_numpy() * _scale(xName)

    createdFig = ax is None
    if createdFig:
        fig, ax = plt.subplots(figsize=(5.0, 3.6))
    else:
        fig = ax.figure

    ax.plot(xValues, frame['lcoe_eur_per_mwh'], marker='o', color='#1f77b4')
    best = frame.loc[frame['lcoe_eur_per_mwh'].idxmin()]
    ax.plot(best[xName] * _scale(xName), best['lcoe_eur_per_mwh'],
            marker='*', color='crimson', markersize=15, label='optimum')
    ax.set_xlabel(_label(xName))
    ax.set_ylabel(_label('lcoe_eur_per_mwh'))
    ax.set_title(f"LCoE at {generator_kw:g} kW generator")
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')

    if createdFig:
        fig.tight_layout()
        _save(fig, output_dir,
              f'lcoe_curve_{int(round(generator_kw)):04d}kW')
    return fig, ax


# ---------------------------------------------------------------------------
# Diagnostics of the optimum
# ---------------------------------------------------------------------------

_DIAGNOSTICS = (
    'emergent_rated_power_kw',
    'crest_peak_over_rated',
    'crest_limit_over_rated',
    'tether_diameter_mm',
    'capacity_factor',
    'storage_capacity_wh',
    'total_airborne_mass_kg',
    'aep_mwh',
)


def plot_diagnostics_vs_generator_power(results_dir: Path,
                                        output_dir: Optional[Path] = None):
    """A panel grid of optimum diagnostics versus generator power."""
    frame = load_optimum_table(results_dir)
    quantities = [q for q in _DIAGNOSTICS if q in frame.columns]

    columns = 3
    rows = (len(quantities) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(4.2 * columns,
                                                     3.2 * rows))
    axesFlat = axes.flatten()

    for index, quantity in enumerate(quantities):
        ax = axesFlat[index]
        ax.plot(frame[_OUTER], frame[quantity], marker='o', color='#1f77b4')
        ax.set_xlabel(_label(_OUTER))
        ax.set_ylabel(_label(quantity))
        ax.set_title(f"({chr(ord('a') + index)})", loc='left')
        ax.grid(True, alpha=0.3)

    for spare in range(len(quantities), len(axesFlat)):
        axesFlat[spare].axis('off')

    fig.tight_layout()
    _save(fig, output_dir, 'diagnostics_vs_generator_power')
    return fig, axes


# ---------------------------------------------------------------------------
# All figures
# ---------------------------------------------------------------------------

def plot_all(results_dir: Path, output_dir: Path) -> List[str]:
    """Generate and save every study figure. Returns the figure names."""
    plot_lcoe_vs_generator_power(results_dir, output_dir)
    plot_optimal_area_vs_generator_power(results_dir, output_dir)
    plot_rated_power_vs_generator_power(results_dir, output_dir)
    plot_diagnostics_vs_generator_power(results_dir, output_dir)
    names = ['lcoe_vs_generator_power', 'optimal_area_vs_generator_power',
             'rated_power_vs_generator_power',
             'diagnostics_vs_generator_power']
    # Shared contour levels across every two-variable surface, so the same
    # LCoE value gets the same colour in every lcoe_surface_*kW figure
    # (otherwise contourf auto-scales each one to its own min/max).
    surfaceFiles = generator_surface_files(results_dir)
    twoVarSurfaces = [(power, path) for power, path in surfaceFiles
                      if len(_surface_variables(path)) == 2]
    sharedLevels = None
    if twoVarSurfaces:
        allLcoe = pd.concat(
            [pd.read_csv(path)['lcoe_eur_per_mwh'] for _, path in twoVarSurfaces])
        sharedLevels = np.linspace(allLcoe.min(), allLcoe.max(), 61)

    for power, surfacePath in surfaceFiles:
        variables = _surface_variables(surfacePath)
        if len(variables) == 2:
            plot_lcoe_surface(results_dir, power, output_dir,
                              levels=sharedLevels)
            names.append(f'lcoe_surface_{int(round(power)):04d}kW')
        elif len(variables) == 1:
            plot_lcoe_curve(results_dir, power, output_dir)
            names.append(f'lcoe_curve_{int(round(power)):04d}kW')
    plt.close('all')
    return names
