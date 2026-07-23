"""Plots for the design-optimisation study.

Reads the study outputs written by
:func:`tef.design_studies.run_design_optimisation` and produces the
figures of the plan:

- LCoE of the optimum versus rated power (the scaling result).
- Optimal wing area versus rated power.
- The LCoE surface over the design space at one rated power, with the
  optimum marked.
- Diagnostics of the optimum versus rated power (emergent crest factor,
  tether diameter, capacity factor, storage, airborne mass, AEP).

The module is model-agnostic: it reads the CSV/YAML the study writes,
not the models. matplotlib is imported at module load because it is the
sole purpose of this module; install it with the ``plots`` extra.

Public interface:
    load_optimum_table, rated_surface_files
    plot_lcoe_vs_rated_power, plot_optimal_area_vs_rated_power
    plot_lcoe_surface, plot_diagnostics_vs_rated_power
    plot_all
"""

import re
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

try:
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - import-time guard
    raise ImportError(
        "tef.plotting needs matplotlib; install it, e.g. "
        "pip install matplotlib") from exc


# Column -> (axis label, unit label, multiplier applied to the raw value).
_AXIS = {
    'flat_area_m2': ('Wing area', 'm$^2$', 1.0),
    'allowable_tether_stress_pa': ('Allowable tether stress', 'GPa', 1e-9),
    'target_rated_power_kw': ('Rated power', 'kW', 1.0),
    'lcoe_eur_per_mwh': ('LCoE', 'EUR/MWh', 1.0),
    'emergent_crest_factor': ('Emergent crest factor', '-', 1.0),
    'tether_diameter_mm': ('Tether diameter', 'mm', 1.0),
    'capacity_factor': ('Capacity factor', '-', 1.0),
    'storage_capacity_wh': ('Storage capacity', 'Wh', 1.0),
    'total_airborne_mass_kg': ('Airborne mass', 'kg', 1.0),
    'aep_mwh': ('AEP', 'MWh', 1.0),
}

_RATED_DIR = re.compile(r'rated_(\d+)kW$')

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
    """Load ``optimum_by_rated_power.csv`` sorted by rated power."""
    path = Path(results_dir) / 'optimum_by_rated_power.csv'
    if not path.exists():
        raise FileNotFoundError(
            f"optimum table not found: {path}; run the study first")
    frame = pd.read_csv(path)
    return frame.sort_values('target_rated_power_kw').reset_index(drop=True)


def rated_surface_files(results_dir: Path) -> List[Tuple[float, Path]]:
    """Return ``(rated_kW, grid_surface.csv)`` pairs, sorted by rating."""
    resultsDir = Path(results_dir)
    found = []
    for child in resultsDir.iterdir():
        match = _RATED_DIR.search(child.name)
        surface = child / 'grid_surface.csv'
        if match and surface.exists():
            found.append((float(match.group(1)), surface))
    return sorted(found)


def _surface_variables(surface_path: Path) -> List[str]:
    """Design-variable columns of a grid_surface.csv (all but lcoe/status)."""
    columns = pd.read_csv(surface_path, nrows=0).columns
    return [c for c in columns if c not in _NON_VARIABLE_COLUMNS]


# ---------------------------------------------------------------------------
# Optimum-versus-rated-power curves
# ---------------------------------------------------------------------------

def plot_lcoe_vs_rated_power(results_dir: Path,
                             output_dir: Optional[Path] = None,
                             ax=None):
    """LCoE of the optimal design versus rated power."""
    frame = load_optimum_table(results_dir)
    createdFig = ax is None
    if createdFig:
        fig, ax = plt.subplots(figsize=(5.0, 3.6))
    else:
        fig = ax.figure

    ax.plot(frame['target_rated_power_kw'], frame['lcoe_eur_per_mwh'],
            marker='o', color='k')
    ax.set_xlabel(_label('target_rated_power_kw'))
    ax.set_ylabel(_label('lcoe_eur_per_mwh'))
    ax.grid(True, alpha=0.3)

    if createdFig:
        fig.tight_layout()
        _save(fig, output_dir, 'lcoe_vs_rated_power')
    return fig, ax


def plot_optimal_area_vs_rated_power(results_dir: Path,
                                     output_dir: Optional[Path] = None,
                                     ax=None):
    """Optimal wing area versus rated power."""
    frame = load_optimum_table(results_dir)
    createdFig = ax is None
    if createdFig:
        fig, ax = plt.subplots(figsize=(5.0, 3.6))
    else:
        fig = ax.figure

    ax.plot(frame['target_rated_power_kw'], frame['flat_area_m2'],
            marker='s', color='k')
    ax.set_xlabel(_label('target_rated_power_kw'))
    ax.set_ylabel(_label('flat_area_m2'))
    ax.grid(True, alpha=0.3)

    if createdFig:
        fig.tight_layout()
        _save(fig, output_dir, 'optimal_area_vs_rated_power')
    return fig, ax


# ---------------------------------------------------------------------------
# LCoE surface over the design space
# ---------------------------------------------------------------------------

def plot_lcoe_surface(results_dir: Path, rated_kw: float,
                      output_dir: Optional[Path] = None, ax=None):
    """Filled-contour LCoE surface over the two design variables at one
    rated power, with the optimum marked.

    Only defined for a two-variable design space; a one-variable study
    should use :func:`plot_lcoe_vs_rated_power` style line plots.
    """
    surfacePath = None
    for rating, path in rated_surface_files(results_dir):
        if abs(rating - rated_kw) < 1e-6:
            surfacePath = path
    if surfacePath is None:
        raise FileNotFoundError(
            f"no grid_surface.csv for {rated_kw:g} kW under {results_dir}")

    frame = pd.read_csv(surfacePath)
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

    contour = ax.contourf(xValues, yValues, grid.to_numpy(), levels=18,
                          cmap='viridis')
    fig.colorbar(contour, ax=ax, label=_label('lcoe_eur_per_mwh'))

    # Mark the optimum cell of this surface.
    best = frame.loc[frame['lcoe_eur_per_mwh'].idxmin()]
    ax.plot(best[xName] * _scale(xName), best[yName] * _scale(yName),
            marker='*', color='white', markersize=15,
            markeredgecolor='k', label='optimum')
    ax.set_xlabel(_label(xName))
    ax.set_ylabel(_label(yName))
    ax.set_title(f"LCoE surface at {rated_kw:g} kW")
    ax.legend(loc='upper right')

    if createdFig:
        fig.tight_layout()
        _save(fig, output_dir, f'lcoe_surface_{int(round(rated_kw)):04d}kW')
    return fig, ax


def plot_lcoe_curve(results_dir: Path, rated_kw: float,
                    output_dir: Optional[Path] = None, ax=None):
    """LCoE versus the single design variable at one rated power.

    The one-variable counterpart of :func:`plot_lcoe_surface`, used when
    a stage sweeps only one variable (e.g. wing area). The optimum is
    marked.
    """
    surfacePath = None
    for rating, path in rated_surface_files(results_dir):
        if abs(rating - rated_kw) < 1e-6:
            surfacePath = path
    if surfacePath is None:
        raise FileNotFoundError(
            f"no grid_surface.csv for {rated_kw:g} kW under {results_dir}")

    frame = pd.read_csv(surfacePath)
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

    ax.plot(xValues, frame['lcoe_eur_per_mwh'], marker='o', color='k')
    best = frame.loc[frame['lcoe_eur_per_mwh'].idxmin()]
    ax.plot(best[xName] * _scale(xName), best['lcoe_eur_per_mwh'],
            marker='*', color='crimson', markersize=15, label='optimum')
    ax.set_xlabel(_label(xName))
    ax.set_ylabel(_label('lcoe_eur_per_mwh'))
    ax.set_title(f"LCoE at {rated_kw:g} kW")
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')

    if createdFig:
        fig.tight_layout()
        _save(fig, output_dir, f'lcoe_curve_{int(round(rated_kw)):04d}kW')
    return fig, ax


# ---------------------------------------------------------------------------
# Diagnostics of the optimum
# ---------------------------------------------------------------------------

_DIAGNOSTICS = (
    'emergent_crest_factor',
    'tether_diameter_mm',
    'capacity_factor',
    'storage_capacity_wh',
    'total_airborne_mass_kg',
    'aep_mwh',
)


def plot_diagnostics_vs_rated_power(results_dir: Path,
                                    output_dir: Optional[Path] = None):
    """A panel grid of optimum diagnostics versus rated power."""
    frame = load_optimum_table(results_dir)
    quantities = [q for q in _DIAGNOSTICS if q in frame.columns]

    columns = 3
    rows = (len(quantities) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(4.2 * columns,
                                                     3.2 * rows))
    axesFlat = axes.flatten()

    for index, quantity in enumerate(quantities):
        ax = axesFlat[index]
        ax.plot(frame['target_rated_power_kw'], frame[quantity],
                marker='o', color='k')
        ax.set_xlabel(_label('target_rated_power_kw'))
        ax.set_ylabel(_label(quantity))
        ax.set_title(f"({chr(ord('a') + index)})", loc='left')
        ax.grid(True, alpha=0.3)

    for spare in range(len(quantities), len(axesFlat)):
        axesFlat[spare].axis('off')

    fig.tight_layout()
    _save(fig, output_dir, 'diagnostics_vs_rated_power')
    return fig, axes


# ---------------------------------------------------------------------------
# All figures
# ---------------------------------------------------------------------------

def plot_all(results_dir: Path, output_dir: Path) -> List[str]:
    """Generate and save every study figure. Returns the figure names."""
    plot_lcoe_vs_rated_power(results_dir, output_dir)
    plot_optimal_area_vs_rated_power(results_dir, output_dir)
    plot_diagnostics_vs_rated_power(results_dir, output_dir)
    names = ['lcoe_vs_rated_power', 'optimal_area_vs_rated_power',
             'diagnostics_vs_rated_power']
    for rating, surfacePath in rated_surface_files(results_dir):
        variables = _surface_variables(surfacePath)
        if len(variables) == 2:
            plot_lcoe_surface(results_dir, rating, output_dir)
            names.append(f'lcoe_surface_{int(round(rating)):04d}kW')
        elif len(variables) == 1:
            plot_lcoe_curve(results_dir, rating, output_dir)
            names.append(f'lcoe_curve_{int(round(rating)):04d}kW')
    plt.close('all')
    return names
