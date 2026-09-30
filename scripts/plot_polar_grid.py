"""Plot the polar grid study (polar efficiency x fixed CL).

Writes ``<results_dir>/plots/polar_grid.png``:
    (a) LCoE map over fixed CL x L/D_max, best CL per L/D_max boxed and
        the reference kites placed at their implied L/D_max
    (b) LCoE against L/D_max, one line per CL
    (c) AEP against L/D_max, one line per CL
"""

import argparse
from typing import Optional
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from tef.io import load_yaml, project_root, studies_config_dir

# Reference palette: sequential blue ramp (map) and its ordinal steps
# 250..650 for the ordered CL series; chart chrome.
SEQUENTIAL = ['#cde2fb', '#86b6ef', '#3987e5', '#1c5cab', '#0d366b']
ORDINAL = ['#86b6ef', '#5598e7', '#2a78d6', '#1c5cab', '#104281']
SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK_2 = '#52514e'
MUTED = '#898781'
GRID = '#e1e0d9'
AXIS = '#c3c2b7'


def _style() -> None:
    plt.rcParams.update({
        'font.family': ['Segoe UI', 'DejaVu Sans'],
        'mathtext.fontset': 'dejavusans',
        'font.size': 9.5,
        'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE,
        'savefig.facecolor': SURFACE,
        'axes.edgecolor': AXIS, 'axes.labelcolor': INK_2,
        'axes.titlecolor': INK, 'axes.titlesize': 10.5,
        'axes.titleweight': 'bold', 'axes.titlelocation': 'left',
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
        'axes.axisbelow': True,
        'xtick.color': MUTED, 'ytick.color': MUTED,
        'xtick.labelcolor': INK_2, 'ytick.labelcolor': INK_2,
        'legend.frameon': False, 'legend.fontsize': 9,
        'lines.linewidth': 2.0, 'lines.solid_capstyle': 'round',
    })


def _edges(values):
    values = np.asarray(values, dtype=float)
    mids = (values[:-1] + values[1:]) / 2
    return np.concatenate([[values[0] - (mids[0] - values[0])], mids,
                           [values[-1] + (values[-1] - mids[-1])]])


def plot_polar_grid(frame: pd.DataFrame, kites: dict, cl_design: float,
                    title: str, output: Path,
                    area_optimised: bool = False) -> None:
    ok = frame[~frame['error']]
    ldValues = sorted(frame['ld_max'].unique())
    clValues = sorted(frame['cl'].unique())
    lcoe = ok.pivot(index='ld_max', columns='cl',
                    values='lcoe_eur_per_mwh').reindex(
                        index=ldValues, columns=clValues)

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8),
                             gridspec_kw={'width_ratios': [1.25, 1, 1]})

    # (a) LCoE map with row minima boxed and reference kites.
    ax = axes[0]
    cmap = LinearSegmentedColormap.from_list('seq', SEQUENTIAL)
    lo, hi = np.nanmin(lcoe.values), np.nanpercentile(lcoe.values, 90)
    mesh = ax.pcolormesh(_edges(clValues), _edges(ldValues), lcoe.values,
                         cmap=cmap, vmin=lo, vmax=hi, shading='flat')
    for i, ld in enumerate(ldValues):
        row = lcoe.values[i]
        best = int(np.nanargmin(row)) if np.isfinite(row).any() else None
        for j, cl in enumerate(clValues):
            value = row[j]
            if not np.isfinite(value):
                ax.text(cl, ld, 'fail', ha='center', va='center',
                        color=MUTED, fontsize=8)
                continue
            dark = value > lo + 0.55 * (hi - lo)
            label = f"{value:.0f}"
            if area_optimised:
                area = ok[(ok['ld_max'] == ld) & (ok['cl'] == cl)][
                    'flat_area_m2'].iloc[0]
                label += f"\n{area:g} m$^2$"
            # Text sits in the lower part of the cell, clear of the kite
            # markers (their implied L/D_max lies in the upper part).
            ax.text(cl, ld - 0.42, label, ha='center', va='center',
                    fontsize=8.5, fontweight='bold' if j == best else 'normal',
                    color='#ffffff' if dark else INK)
        if best is not None:
            cx, cy = _edges(clValues), _edges(ldValues)
            ax.add_patch(plt.Rectangle((cx[best], cy[i]),
                                       cx[best + 1] - cx[best],
                                       cy[i + 1] - cy[i], fill=False,
                                       edgecolor=INK, linewidth=1.8))
    for name, kite in kites.items():
        ax.scatter([kite['cl']], [kite['implied_ld_max']], s=55,
                   marker='D', color='#ffffff', edgecolor=INK,
                   linewidth=1.4, zorder=5)
        ax.annotate(name, (kite['cl'], kite['implied_ld_max']),
                    xytext=(7, 4), textcoords='offset points',
                    color=INK, fontsize=9, fontweight='bold')
    ax.set_xticks(clValues)
    ax.set_yticks(ldValues)
    ax.set_xlabel('Fixed reel-out lift coefficient $C_L$ [-]')
    ax.set_ylabel('Polar efficiency $(L/D)_{max}$ [-]')
    ax.set_title('Optimum LCoE [EUR/MWh] and wing area  (box: best $C_L$)'
                 if area_optimised
                 else 'LCoE [EUR/MWh]  (box: best $C_L$ per row)')
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    colorbar = fig.colorbar(mesh, ax=ax, fraction=0.05, pad=0.02)
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(color=MUTED, labelcolor=INK_2)

    # (b), (c) trends with L/D_max, one line per CL (ordered ramp).
    third = (('flat_area_m2', 'Optimal wing area [m$^2$]') if area_optimised
             else ('aep_mwh', 'Annual energy production [MWh]'))
    for ax, column, heading in [
            (axes[1], 'lcoe_eur_per_mwh', 'LCoE [EUR/MWh]'),
            (axes[2], *third)]:
        for color, cl in zip(ORDINAL, clValues):
            rows = ok[ok['cl'] == cl].sort_values('ld_max')
            ax.plot(rows['ld_max'], rows[column], color=color, marker='o',
                    markersize=5.5, markeredgecolor=SURFACE,
                    markeredgewidth=1.3, label=f"$C_L$ = {cl:g}")
        ax.set_xlabel('Polar efficiency $(L/D)_{max}$ [-]')
        ax.set_xticks(ldValues)
        ax.set_title(heading)
    axes[1].legend(loc='upper right')

    fig.suptitle(title, x=0.01, ha='left', fontsize=12, fontweight='bold',
                 color=INK)
    fig.text(0.01, 0.905,
             f"Polar family $C_D = C_{{L,d}}/(2E)\\,(1 + (C_L/C_{{L,d}})^2)$, "
             f"$C_{{L,d}}$ = {cl_design:g}; each case flies one fixed $C_L$ "
             "at all wind speeds.  Diamonds: kites at their implied "
             "$(L/D)_{max}$ (not run).", color=INK_2, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(output, dpi=200)
    plt.close(fig)


# Second sequential context (AEP) takes the next categorical hue, orange,
# as its own light-to-dark ramp.
SEQUENTIAL_ORANGE = ['#fde3d6', '#f7b393', '#eb6834', '#b8481f', '#7a2e12']


def _metric_map(fig, ax, ok: pd.DataFrame, ld_values, cl_values,
                column: str, ramp, value_format: str, title: str,
                colorbar_label: str, kites: dict, boxes,
                show_area: bool = True) -> None:
    """One (CL x L/D_max) map of ``column`` with each cell's optimal wing
    area, the best-LCoE cell per row boxed and the reference kites."""
    table = ok.pivot(index='ld_max', columns='cl', values=column).reindex(
        index=ld_values, columns=cl_values)
    areas = ok.pivot(index='ld_max', columns='cl',
                     values='flat_area_m2').reindex(
                         index=ld_values, columns=cl_values)
    cmap = LinearSegmentedColormap.from_list(column, ramp)
    lo = np.nanpercentile(table.values, 5)
    hi = np.nanpercentile(table.values, 95)
    mesh = ax.pcolormesh(_edges(cl_values), _edges(ld_values), table.values,
                         cmap=cmap, vmin=lo, vmax=hi, shading='flat')
    cx, cy = _edges(cl_values), _edges(ld_values)
    for i, ld in enumerate(ld_values):
        for j, cl in enumerate(cl_values):
            value = table.values[i, j]
            if not np.isfinite(value):
                ax.text(cl, ld, 'fail', ha='center', va='center',
                        color=MUTED, fontsize=8)
                continue
            dark = value > lo + 0.55 * (hi - lo)
            ax.text(cl, ld - 0.42,
                    f"{value:{value_format}}"
                    + (f"\n{areas.values[i, j]:g} m$^2$" if show_area else ''),
                    ha='center', va='center', fontsize=8.5,
                    fontweight='bold' if boxes[i] == j else 'normal',
                    color='#ffffff' if dark else INK)
        if boxes[i] is not None:
            ax.add_patch(plt.Rectangle(
                (cx[boxes[i]], cy[i]), cx[boxes[i] + 1] - cx[boxes[i]],
                cy[i + 1] - cy[i], fill=False, edgecolor=INK, linewidth=1.8))
    for name, kite in kites.items():
        ax.scatter([kite['cl']], [kite['implied_ld_max']], s=60,
                   marker='D', color='#ffffff', edgecolor=INK,
                   linewidth=1.4, zorder=5)
        ax.annotate(name, (kite['cl'], kite['implied_ld_max']),
                    xytext=(7, 4), textcoords='offset points',
                    color=INK, fontsize=9.5, fontweight='bold')
    ax.set_xticks(cl_values)
    ax.set_yticks(ld_values)
    ax.set_xlabel('Fixed reel-out lift coefficient $C_L$ [-]')
    ax.set_ylabel('Polar efficiency $(L/D)_{max}$ [-]')
    ax.set_title(title)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    colorbar = fig.colorbar(mesh, ax=ax, fraction=0.05, pad=0.02)
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(color=MUTED, labelcolor=INK_2)
    colorbar.set_label(colorbar_label, color=INK_2)


def plot_heatmaps(frame: pd.DataFrame, kites: dict, cl_design: float,
                  title: str, output: Path,
                  fixed_area: Optional[float] = None) -> None:
    """LCoE and AEP maps side by side, at each cell's optimal wing area."""
    ok = frame[~frame['error']]
    ldValues = sorted(frame['ld_max'].unique())
    clValues = sorted(frame['cl'].unique())
    lcoe = ok.pivot(index='ld_max', columns='cl',
                    values='lcoe_eur_per_mwh').reindex(
                        index=ldValues, columns=clValues)
    boxes = [int(np.nanargmin(row)) if np.isfinite(row).any() else None
             for row in lcoe.values]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6))
    _metric_map(fig, axes[0], ok, ldValues, clValues, 'lcoe_eur_per_mwh',
                SEQUENTIAL, '.0f', 'LCoE [EUR/MWh]  (lower is better)',
                'LCoE [EUR/MWh]', kites, boxes, fixed_area is None)
    _metric_map(fig, axes[1], ok, ldValues, clValues, 'aep_mwh',
                SEQUENTIAL_ORANGE, '.0f',
                'Annual energy production [MWh]  (higher is better)',
                'AEP [MWh]', kites, boxes, fixed_area is None)
    fig.suptitle(title, x=0.01, ha='left', fontsize=12, fontweight='bold',
                 color=INK)
    fig.text(0.01, 0.905,
             (f"All cells at S = {fixed_area:g} m$^2$. Box: " if fixed_area
              else "Values at each cell's optimal wing area (second line). "
                   "Box: ") +
             "lowest-LCoE $C_L$ per $(L/D)_{max}$, marked in both maps.  "
             "Diamonds: V3 / V4 / V5 at their implied $(L/D)_{max}$ on the "
             f"polar family ($C_{{L,d}}$ = {cl_design:g}), not run.",
             color=INK_2, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(output, dpi=200)
    plt.close(fig)


def _continuous_map(fig, ax, table: pd.DataFrame, ramp, title: str,
                    colorbar_label: str, kites: dict, value_format: str,
                    unit: str) -> None:
    """Filled contours of a (L/D_max x CL) table interpolated with a cubic
    spline on the grid (clipped to the computed range), computed points
    shown as dots and the reference kites labelled with their
    interpolated value."""
    from scipy.interpolate import RegularGridInterpolator

    ldValues = table.index.to_numpy(dtype=float)
    clValues = table.columns.to_numpy(dtype=float)
    # Invalid cells (NaN) are filled from their neighbours only so the
    # spline can be built; their area is blanked out afterwards.
    invalid = ~np.isfinite(table.values)
    filled = (table.interpolate(axis=0, limit_direction='both')
              .interpolate(axis=1, limit_direction='both'))
    interp = RegularGridInterpolator((ldValues, clValues), filled.values,
                                     method='cubic')
    ldFine = np.linspace(ldValues[0], ldValues[-1], 200)
    clFine = np.linspace(clValues[0], clValues[-1], 200)
    LD, CL = np.meshgrid(ldFine, clFine, indexing='ij')
    values = np.clip(interp(np.stack([LD, CL], axis=-1)),
                     np.nanmin(table.values), np.nanmax(table.values))
    ldEdges, clEdges = _edges(ldValues), _edges(clValues)
    for i, j in zip(*np.nonzero(invalid)):
        inCell = ((LD >= ldEdges[i]) & (LD <= ldEdges[i + 1])
                  & (CL >= clEdges[j]) & (CL <= clEdges[j + 1]))
        values[inCell] = np.nan

    lo = np.nanpercentile(table.values, 3)
    hi = np.nanpercentile(table.values, 97)
    levels = np.linspace(lo, hi, 13)
    cmap = LinearSegmentedColormap.from_list(colorbar_label, ramp)
    filled = ax.contourf(CL, LD, values, levels=levels, cmap=cmap,
                         extend='both')
    lines = ax.contour(CL, LD, values, levels=levels[::2], colors=SURFACE,
                       linewidths=0.8, alpha=0.9)
    ax.clabel(lines, fmt=lambda v: f"{v:{value_format}}", fontsize=8,
              colors=INK, inline=True)
    for i, j in zip(*np.nonzero(invalid)):
        x0, x1 = max(clEdges[j], clValues[0]), min(clEdges[j + 1], clValues[-1])
        y0, y1 = max(ldEdges[i], ldValues[0]), min(ldEdges[i + 1], ldValues[-1])
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                   facecolor=SURFACE, edgecolor=MUTED,
                                   hatch='////', linewidth=0.8, zorder=3))
        ax.text((x0 + x1) / 2, (y0 + y1) / 2, 'QSM\nfail', ha='center',
                va='center', fontsize=8, color=INK_2, zorder=4)
    CLp, LDp = np.meshgrid(clValues, ldValues)
    ax.scatter(CLp.ravel(), LDp.ravel(), s=9, color=INK, alpha=0.45,
               linewidth=0, zorder=4)
    for name, kite in kites.items():
        point = (kite['implied_ld_max'], kite['cl'])
        value = float(interp([point])[0])
        ax.scatter([kite['cl']], [kite['implied_ld_max']], s=70, marker='D',
                   color='#ffffff', edgecolor=INK, linewidth=1.4, zorder=6,
                   clip_on=False)
        # Label to the left of markers on the right edge of the map.
        onRightEdge = kite['cl'] > clValues[-1] - 0.1 * np.ptp(clValues)
        ax.annotate(f"{name}  {value:{value_format}} {unit}",
                    (kite['cl'], kite['implied_ld_max']),
                    xytext=(-8, 5) if onRightEdge else (8, 5),
                    ha='right' if onRightEdge else 'left',
                    textcoords='offset points', color=INK,
                    fontsize=9, fontweight='bold', zorder=7,
                    bbox={'boxstyle': 'round,pad=0.2', 'fc': SURFACE,
                          'ec': 'none', 'alpha': 0.8})
    ax.set_xticks(clValues)
    ax.set_yticks(ldValues)
    ax.set_xlabel('Fixed reel-out lift coefficient $C_L$ [-]')
    ax.set_ylabel('Polar efficiency $(L/D)_{max}$ [-]')
    ax.set_title(title)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    colorbar = fig.colorbar(filled, ax=ax, fraction=0.05, pad=0.02,
                            ticks=levels[::2], format='{x:.0f}')
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(color=MUTED, labelcolor=INK_2)
    colorbar.set_label(colorbar_label, color=INK_2)


def plot_continuous_maps(frame: pd.DataFrame, kites: dict, cl_design: float,
                         title: str, output: Path,
                         fixed_area: Optional[float] = None) -> None:
    """Continuous LCoE and AEP maps (each cell at its optimal area)."""
    ok = frame[~frame['error']]
    ldValues = sorted(frame['ld_max'].unique())
    clValues = sorted(frame['cl'].unique())

    def table(column):
        return ok.pivot(index='ld_max', columns='cl', values=column).reindex(
            index=ldValues, columns=clValues)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6))
    _continuous_map(fig, axes[0], table('lcoe_eur_per_mwh'), SEQUENTIAL,
                    'LCoE [EUR/MWh]  (lower is better)', 'LCoE [EUR/MWh]',
                    kites, '.0f', 'EUR/MWh')
    _continuous_map(fig, axes[1], table('aep_mwh'), SEQUENTIAL_ORANGE,
                    'Annual energy production [MWh]  (higher is better)',
                    'AEP [MWh]', kites, '.0f', 'MWh')
    fig.suptitle(title, x=0.01, ha='left', fontsize=12, fontweight='bold',
                 color=INK)
    fig.text(0.01, 0.905,
             "Cubic interpolation between the computed points (dots), "
             + (f"all at S = {fixed_area:g} m$^2$" if fixed_area
                else "each at its own optimal wing area")
             + "; values between dots are interpolated, not computed.  "
             "Diamonds: V3 / V4 / V5 at their "
             f"implied $(L/D)_{{max}}$ ($C_{{L,d}}$ = {cl_design:g}).",
             color=INK_2, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(output, dpi=200)
    plt.close(fig)


# A case is shown as failed when the QSM found no cycle at more than this
# many wind speeds above its cut-in (failures below cut-in are real zeros).
MAX_FAILED_POINTS_ABOVE_CUT_IN = 3


def _mark_qsm_failures(frame: pd.DataFrame, results_dir: Path) -> pd.DataFrame:
    """Blank LCoE/AEP of cases whose power curve failed at more than
    MAX_FAILED_POINTS_ABOVE_CUT_IN wind speeds above the first productive
    one; those zeros are solver failures, not physics."""
    frame = frame.copy()
    for idx, row in frame.iterrows():
        caseDir = (results_dir / (f"E{row['ld_max']:04.1f}_CL{row['cl']:.2f}"
                                  .replace('.', 'p'))
                   / f"S{row['flat_area_m2']:05.1f}".replace('.', 'p'))
        path = caseDir / 'power_curves.yml'
        if not path.exists():
            continue
        entries = load_yaml(path)['power_curves'][0]['wind_speed_data']
        # The appended cut-out zero point is not a QSM result.
        entries = [e for e in entries if e.get('performance', {}).get(
            'timing') != {}]
        ok = [e.get('successful', False) for e in entries]
        if True not in ok:
            continue
        failed = sum(not v for v in ok[ok.index(True):])
        if failed > MAX_FAILED_POINTS_ABOVE_CUT_IN:
            frame.loc[idx, ['lcoe_eur_per_mwh', 'aep_mwh']] = np.nan
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--study', type=Path,
        default=studies_config_dir() / 'aero_polar_grid_40kw.yml',
        help='Polar grid study configuration file')
    parser.add_argument(
        '--fixed-area', type=float, default=None,
        help="Plot all cells at this wing area [m2] (must be on the "
             "study's area grid) instead of each cell's optimal area")
    args = parser.parse_args()

    config = load_yaml(args.study)
    resultsDir = project_root() / config['results_dir']
    summary = load_yaml(resultsDir / 'polar_grid_summary.yml')
    areaOptimised = len(summary.get('flat_areas_m2') or []) > 1
    fixedArea = args.fixed_area
    if fixedArea is not None:
        frame = pd.read_csv(resultsDir / 'polar_grid.csv')
        frame = frame[np.isclose(frame['flat_area_m2'], fixedArea)]
        if frame.empty:
            raise SystemExit(f"No cases at {fixedArea:g} m2; areas on the "
                             f"grid: {summary.get('flat_areas_m2')}")
    else:
        frame = pd.read_csv(resultsDir / (
            'polar_grid_optima.csv' if areaOptimised else 'polar_grid.csv'))
    frame = _mark_qsm_failures(frame, resultsDir)
    design = config['design']
    if fixedArea is not None:
        areaText = f"S = {fixedArea:g} m$^2$ (fixed)"
    elif areaOptimised:
        areaText = 'wing area optimised per cell'
    else:
        areaText = f"S = {design['flat_area_m2']:g} m$^2$"
    title = (f"Polar efficiency x fixed $C_L$ - "
             f"{design['generator_max_power_w'] / 1e3:g} kW generator, "
             f"{areaText}, "
             f"$\\sigma$ = {design['allowable_tether_stress_pa'] / 1e9:g} GPa")

    plotDir = resultsDir / 'plots'
    plotDir.mkdir(parents=True, exist_ok=True)
    _style()
    kites = summary.get('reference_kites') or {}
    clDesign = summary['polar']['cl_design']
    suffix = f"_S{fixedArea:03.0f}" if fixedArea is not None else ''
    plot_polar_grid(frame, kites, clDesign, title,
                    plotDir / f'polar_grid{suffix}.png',
                    areaOptimised and fixedArea is None)
    if areaOptimised or fixedArea is not None:
        plot_heatmaps(frame, kites, clDesign, title,
                      plotDir / f'polar_grid_heatmaps{suffix}.png', fixedArea)
        plot_continuous_maps(frame, kites, clDesign, title,
                             plotDir / f'polar_grid_contours{suffix}.png',
                             fixedArea)
    print(f"Plots written to {plotDir}")


if __name__ == '__main__':
    main()
