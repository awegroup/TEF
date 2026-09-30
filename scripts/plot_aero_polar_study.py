"""Plot the aero polar study (Stage A) results.

Writes to ``<results_dir>/plots/``:
    aero_polar_summary.png   LCoE and AEP against L/D_max, one line per
                             CL_max, plus the LCoE map relative to V3
    aero_polar_details.png   per L/D_max: the polar with its operating
                             points, the envelope power curves, and the
                             lift coefficient selected at each wind speed
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from tef.aero_polars import ParabolicPolar, load_aero_polar_study_config
from tef.io import load_yaml, studies_config_dir

# Reference palette: categorical slots 1-3 (validated all-pairs for three
# series), sequential blue ramp, and chart chrome.
SERIES = ['#2a78d6', '#eb6834', '#1baf7a']
SEQUENTIAL = ['#cde2fb', '#86b6ef', '#3987e5', '#1c5cab', '#0d366b']
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
        'figure.facecolor': SURFACE,
        'axes.facecolor': SURFACE,
        'savefig.facecolor': SURFACE,
        'axes.edgecolor': AXIS,
        'axes.labelcolor': INK_2,
        'axes.titlecolor': INK,
        'axes.titlesize': 10.5,
        'axes.titleweight': 'bold',
        'axes.titlelocation': 'left',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.color': GRID,
        'grid.linewidth': 0.6,
        'axes.axisbelow': True,
        'xtick.color': MUTED,
        'ytick.color': MUTED,
        'xtick.labelcolor': INK_2,
        'ytick.labelcolor': INK_2,
        'legend.frameon': False,
        'legend.fontsize': 9,
        'lines.linewidth': 2.0,
        'lines.solid_capstyle': 'round',
    })


def _cl_label(cl_max: float) -> str:
    return f"$C_{{L,max}}$ = {cl_max:g}"


def _power_curve(case_dir: Path):
    data = load_yaml(case_dir / 'power_curves.yml')
    entries = data['power_curves'][0]['wind_speed_data']
    wind = np.array([e['wind_speed'] for e in entries])
    power = np.array([e['performance']['electrical_power']
                      ['average_cycle_power'] for e in entries]) / 1e3
    return wind, power


def _end_labels(ax, x, values, all_values) -> None:
    """Value labels at the line ends, nudged apart so they never collide."""
    gap = 0.045 * (all_values.max() - all_values.min())
    order = np.argsort(values)
    placed = []
    for idx in order:
        y = values[idx]
        if placed and y - placed[-1] < gap:
            y = placed[-1] + gap
        placed.append(y)
        ax.annotate(f"{values[idx]:.0f}", (x, values[idx]), xytext=(x + 0.12, y),
                    textcoords='data', va='center', color=INK_2, fontsize=8.5)


def plot_summary(frame: pd.DataFrame, baseline: pd.Series,
                 output: Path) -> None:
    ldValues = sorted(frame['ld_max'].unique())
    clValues = sorted(frame['cl_max'].unique())
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2),
                             gridspec_kw={'width_ratios': [1, 1, 1.05]})

    for ax, column, title, unit in [
            (axes[0], 'lcoe_eur_per_mwh', 'LCoE', 'EUR/MWh'),
            (axes[1], 'aep_mwh', 'Annual energy production', 'MWh')]:
        ends = []
        for color, clMax in zip(SERIES, clValues):
            rows = frame[frame['cl_max'] == clMax].sort_values('ld_max')
            ax.plot(rows['ld_max'], rows[column], color=color,
                    marker='o', markersize=6.5, markeredgecolor=SURFACE,
                    markeredgewidth=1.5, label=_cl_label(clMax))
            ends.append(rows.iloc[-1][column])
        _end_labels(ax, ldValues[-1], ends, frame[column])
        ax.scatter([baseline['ld_max']], [baseline[column]], s=150,
                   facecolor='none', edgecolor=INK, linewidth=1.2, zorder=5)
        ax.annotate('V3 baseline', (baseline['ld_max'], baseline[column]),
                    xytext=(10, -14 if column == 'lcoe_eur_per_mwh' else 12),
                    textcoords='offset points', color=INK, fontsize=8.5)
        ax.set_title(f"{title} [{unit}]")
        ax.set_xlabel('Maximum lift-to-drag ratio $(L/D)_{max}$ [-]')
        ax.set_xticks(ldValues)
        ax.set_xlim(ldValues[0] - 0.3, ldValues[-1] + 0.6)
    axes[0].legend(loc='upper right')

    # LCoE change relative to the V3 baseline, as a sequential map.
    ax = axes[2]
    change = (frame.pivot(index='ld_max', columns='cl_max',
                          values='lcoe_eur_per_mwh')
              / baseline['lcoe_eur_per_mwh'] - 1.0) * 100.0
    change = change.sort_index(ascending=False)
    cmap = LinearSegmentedColormap.from_list('seq', SEQUENTIAL[::-1])
    image = ax.imshow(change.values, cmap=cmap, aspect='auto',
                      vmin=change.values.min(), vmax=0.0)
    for i in range(change.shape[0]):
        for j in range(change.shape[1]):
            value = change.values[i, j]
            dark = value < change.values.min() / 2
            ax.text(j, i, f"{value:+.0f}%", ha='center', va='center',
                    fontsize=10, fontweight='bold',
                    color='#ffffff' if dark else INK)
    ax.set_xticks(range(change.shape[1]),
                  [f"{c:g}" for c in change.columns])
    ax.set_yticks(range(change.shape[0]),
                  [f"{l:g}" for l in change.index])
    ax.set_xlabel('Maximum lift coefficient $C_{L,max}$ [-]')
    ax.set_ylabel('$(L/D)_{max}$ [-]')
    ax.set_title('LCoE change vs V3 baseline')
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.05, pad=0.03)
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(color=MUTED, labelcolor=INK_2)
    colorbar.set_label('%', color=INK_2)

    fig.suptitle('Stage A: LCoE sensitivity to the reel-out polar '
                 '(40 kW reference design, S = 75 m$^2$, '
                 '$\\sigma$ = 0.25 GPa)', x=0.01, ha='left',
                 fontsize=12, fontweight='bold', color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output, dpi=200)
    plt.close(fig)


def plot_details(frame: pd.DataFrame, config, results_dir: Path,
                 baseline_dir: Path, output: Path) -> None:
    ldValues = sorted(frame['ld_max'].unique())
    clValues = sorted(frame['cl_max'].unique())
    fig, axes = plt.subplots(3, len(ldValues), figsize=(13.5, 10.5),
                             sharey='row')
    baseWind, basePower = _power_curve(baseline_dir)
    clAxisMax = max(clValues) + 0.1

    for col, ldMax in enumerate(ldValues):
        polar = ParabolicPolar.anchored(ldMax, max(clValues),
                                        config.reference_cl,
                                        config.reference_cd)
        # Row 1: the polar, the CL_max caps and the operating points.
        ax = axes[0, col]
        lifts = np.linspace(0.0, clAxisMax, 200)
        ax.plot([polar.cd(c) for c in lifts], lifts, color=MUTED,
                linewidth=1.5)
        for color, clMax in zip(SERIES, clValues):
            ax.axhline(clMax, color=color, linewidth=1.2, linestyle=(0, (4, 3)))
            ax.text(0.01, clMax, _cl_label(clMax), color=INK_2,
                    fontsize=8, ha='left', va='bottom',
                    transform=ax.get_yaxis_transform())
        points = {pt for clMax in clValues for pt in
                  ParabolicPolar.anchored(
                      ldMax, clMax, config.reference_cl,
                      config.reference_cd).operating_points(
                          config.max_operating_points,
                          config.min_cl_spacing)}
        ax.scatter([cd for _, cd in points], [cl for cl, _ in points],
                   s=40, color=INK, edgecolor=SURFACE, linewidth=1.5,
                   zorder=5, label='QSM operating points')
        ax.scatter([config.reference_cd], [config.reference_cl], s=130,
                   facecolor='none', edgecolor=INK, linewidth=1.2, zorder=6,
                   label='V3 reel-out point')
        ax.set_title(f"$(L/D)_{{max}}$ = {ldMax:g}   "
                     f"($C_{{D,0}}$ = {polar.cd0:.3f})")
        ax.set_xlabel('$C_D$ [-]')
        ax.set_xlim(0, 0.36)
        ax.set_ylim(0, clAxisMax + 0.05)

        # Row 2: envelope power curves.
        ax = axes[1, col]
        ax.plot(baseWind, basePower, color=MUTED, linewidth=1.5,
                linestyle=(0, (4, 3)), label='V3 baseline')
        for color, clMax in zip(SERIES, clValues):
            name = frame[(frame['ld_max'] == ldMax)
                         & (frame['cl_max'] == clMax)]['polar'].iloc[0]
            wind, power = _power_curve(results_dir / 'polars' / name)
            ax.plot(wind, power, color=color, label=_cl_label(clMax))
        ax.set_xlabel('Wind speed at 200 m [m/s]')

        # Row 3: lift coefficient selected at each wind speed.
        ax = axes[2, col]
        for color, clMax in zip(SERIES, clValues):
            name = frame[(frame['ld_max'] == ldMax)
                         & (frame['cl_max'] == clMax)]['polar'].iloc[0]
            selection = load_yaml(results_dir / 'polars' / name
                                  / 'aero_polar.yml')['selection']
            wind = [s['wind_speed'] for s in selection]
            lift = [s['cl'] for s in selection]
            ax.step(wind, lift, where='mid', color=color,
                    label=_cl_label(clMax))
        ax.axhline(polar.cl_at_ld_max, color=MUTED, linewidth=1.0,
                   linestyle=(0, (1, 2)))
        ax.text(0.99, polar.cl_at_ld_max, 'best $L/D$', color=MUTED,
                fontsize=8, ha='right', va='top',
                transform=ax.get_yaxis_transform())
        ax.set_xlabel('Wind speed at 200 m [m/s]')
        ax.set_ylim(0.35, clAxisMax + 0.05)

    axes[0, 0].set_ylabel('$C_L$ [-]')
    axes[1, 0].set_ylabel('Cycle-average electrical power [kW]')
    axes[2, 0].set_ylabel('Selected reel-out $C_L$ [-]')
    handles, labels = axes[1, 0].get_legend_handles_labels()
    pointHandles, pointLabels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles + pointHandles, labels + pointLabels,
               loc='upper left', bbox_to_anchor=(0.01, 0.962),
               ncol=6, handlelength=2.2, columnspacing=1.8)
    fig.suptitle('Stage A details: polar, envelope power curve and '
                 'selected lift coefficient per $(L/D)_{max}$',
                 x=0.01, ha='left', fontsize=12, fontweight='bold',
                 color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.935))
    fig.savefig(output, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--study', type=Path,
        default=studies_config_dir() / 'aero_polar_40kw.yml',
        help='Aero polar study configuration file')
    args = parser.parse_args()

    config = load_aero_polar_study_config(args.study)
    resultsDir = config.results_dir
    frame = pd.read_csv(resultsDir / 'aero_polar_summary.csv')
    # The baseline polar flies the reference point only.
    baseline = frame.loc[
        ((frame['ld_max'] - config.reference_cl / config.reference_cd).abs()
         + (frame['cl_max'] - config.reference_cl).abs()).idxmin()]

    plotDir = resultsDir / 'plots'
    plotDir.mkdir(parents=True, exist_ok=True)
    _style()
    plot_summary(frame, baseline, plotDir / 'aero_polar_summary.png')
    plot_details(frame, config, resultsDir,
                 resultsDir / 'polars' / baseline['polar'],
                 plotDir / 'aero_polar_details.png')
    print(f"Plots written to {plotDir}")


if __name__ == '__main__':
    main()
