"""Plot the kite comparison study (V3 / V4 / V5 at a fixed generator).

Writes to ``<results_dir>/plots/``:
    kite_trends.png     LCoE, AEP and peak mechanical power against wing
                        area (best tether stress per area), optimum marked
    kite_surfaces.png   LCoE surface over wing area x tether stress per
                        kite, optimum marked
    kite_optima.png     fixed reference design (75 m2, 0.25 GPa) vs each
                        kite's optimum, and the optima's power curves
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from tef.io import load_yaml, studies_config_dir
from tef.kite_studies import load_kite_study_config

SERIES = ['#2a78d6', '#eb6834', '#1baf7a']
SEQUENTIAL = ['#cde2fb', '#86b6ef', '#3987e5', '#1c5cab', '#0d366b']
SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK_2 = '#52514e'
MUTED = '#898781'
GRID = '#e1e0d9'
AXIS = '#c3c2b7'
REFERENCE_STRESS_GPA = 0.25


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


def _suptitle(fig, text: str) -> None:
    fig.suptitle(text, x=0.01, ha='left', fontsize=12, fontweight='bold',
                 color=INK)


def _kite_label(kite: str, aero: dict) -> str:
    return f"{kite}  ($C_L$ {aero['cl']:g}, $C_D$ {aero['cd']:g})"


def _kite_aero(inputs_dir: Path) -> dict:
    settings = load_yaml(inputs_dir / 'inertiafree-qsm_settings.yml')
    aero = settings['aerodynamics']
    return {'cl': aero['kite_lift_coefficient_reel_out'],
            'cd': aero['kite_drag_coefficient_reel_out']}


def plot_trends(grid: pd.DataFrame, optima: pd.DataFrame, kites, labels,
                generator_kw: float, reference_area: float,
                peak_cap_kw, output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3))
    # Best tether stress per wing area (the stress axis is nearly flat).
    best = grid.loc[grid.groupby(['kite', 'flat_area_m2'])
                    ['lcoe_eur_per_mwh'].idxmin()]
    panels = [('lcoe_eur_per_mwh', 'LCoE [EUR/MWh]'),
              ('aep_mwh', 'Annual energy production [MWh]'),
              ('peak_mech_power_kw', 'Peak mechanical power [kW]')]
    for ax, (column, title) in zip(axes, panels):
        for color, kite in zip(SERIES, kites):
            rows = best[best['kite'] == kite].sort_values('flat_area_m2')
            ax.plot(rows['flat_area_m2'], rows[column], color=color,
                    marker='o', markersize=5.5, markeredgecolor=SURFACE,
                    markeredgewidth=1.3, label=labels[kite])
            opt = optima[optima['kite'] == kite].iloc[0]
            ax.scatter([opt['flat_area_m2']], [opt[column]], marker='*',
                       s=260, color=color, edgecolor=INK, linewidth=0.8,
                       zorder=5)
        ax.axvline(reference_area, color=MUTED, linewidth=1.0,
                   linestyle=(0, (4, 3)))
        ax.set_title(title)
        ax.set_xlabel('Flat wing area $S$ [m$^2$]')
    axes[2].axhline(generator_kw, color=MUTED, linewidth=1.0,
                    linestyle=(0, (1, 2)))
    axes[2].text(0.99, generator_kw, f'{generator_kw:g} kW generator limit',
                 transform=axes[2].get_yaxis_transform(), ha='right',
                 va='bottom', color=MUTED, fontsize=8)
    if peak_cap_kw:
        axes[2].axhline(peak_cap_kw, color=INK_2, linewidth=1.0,
                        linestyle=(0, (4, 3)))
        axes[2].text(0.99, peak_cap_kw,
                     f'{peak_cap_kw:g} kW drivetrain sizing cap',
                     transform=axes[2].get_yaxis_transform(), ha='right',
                     va='bottom', color=INK_2, fontsize=8)
    axes[0].text(reference_area, 1.0, f' reference {reference_area:g} m$^2$', color=MUTED,
                 fontsize=8, va='top',
                 transform=axes[0].get_xaxis_transform())
    handles, legendLabels = axes[0].get_legend_handles_labels()
    handles.append(plt.Line2D([], [], marker='*', color='none',
                              markerfacecolor=MUTED, markeredgecolor=INK,
                              markersize=13))
    legendLabels.append('optimum')
    fig.legend(handles, legendLabels, loc='upper left',
               bbox_to_anchor=(0.01, 0.9), ncol=4)
    _suptitle(fig, f'Kite comparison at {generator_kw:g} kW generator: '
                   'trends with wing area (best tether stress per area)')
    fig.tight_layout(rect=(0, 0, 1, 0.83))
    fig.savefig(output, dpi=200)
    plt.close(fig)


def plot_surfaces(grid: pd.DataFrame, optima: pd.DataFrame, kites, labels,
                  generator_kw: float, output: Path) -> None:
    fig, axes = plt.subplots(1, len(kites), figsize=(13.5, 4.0),
                             sharey=True)
    cmap = LinearSegmentedColormap.from_list('seq', SEQUENTIAL)
    lo = grid['lcoe_eur_per_mwh'].min()
    hi = grid['lcoe_eur_per_mwh'].quantile(0.9)
    for ax, kite in zip(axes, kites):
        rows = grid[grid['kite'] == kite]
        surface = rows.pivot(index='allowable_tether_stress_gpa',
                             columns='flat_area_m2',
                             values='lcoe_eur_per_mwh').sort_index(
                                 ascending=False)
        image = ax.imshow(surface.values, cmap=cmap, vmin=lo, vmax=hi,
                          aspect='auto')
        opt = optima[optima['kite'] == kite].iloc[0]
        for i, stress in enumerate(surface.index):
            for j, area in enumerate(surface.columns):
                value = surface.values[i, j]
                if np.isnan(value):
                    ax.text(j, i, 'fail', ha='center', va='center',
                            color=MUTED, fontsize=8)
                    continue
                isOpt = (area == opt['flat_area_m2'] and np.isclose(
                    stress, opt['allowable_tether_stress_gpa']))
                dark = value > lo + 0.55 * (hi - lo)
                ax.text(j, i, f"{value:.0f}", ha='center', va='center',
                        fontsize=8.5,
                        fontweight='bold' if isOpt else 'normal',
                        color='#ffffff' if dark else INK)
                if isOpt:
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                               fill=False, edgecolor=INK,
                                               linewidth=2))
        ax.set_xticks(range(surface.shape[1]),
                      [f"{a:g}" for a in surface.columns])
        ax.set_yticks(range(surface.shape[0]),
                      [f"{s:.2f}" for s in surface.index])
        ax.set_xlabel('Flat wing area $S$ [m$^2$]')
        ax.set_title(labels[kite])
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[0].set_ylabel('Allowable tether stress [GPa]')
    colorbar = fig.colorbar(image, cax=fig.add_axes([0.915, 0.14, 0.012, 0.66]))
    colorbar.outline.set_visible(False)
    colorbar.set_label('LCoE [EUR/MWh]', color=INK_2)
    colorbar.ax.tick_params(color=MUTED, labelcolor=INK_2)
    _suptitle(fig, f'LCoE surfaces at {generator_kw:g} kW generator '
                   '(boxed: optimum; colour scale capped at 90th percentile)')
    fig.subplots_adjust(left=0.06, right=0.895, top=0.8, bottom=0.14,
                        wspace=0.08)
    fig.savefig(output, dpi=200)
    plt.close(fig)


def _power_curve(case_dir: Path):
    data = load_yaml(case_dir / 'power_curves.yml')
    entries = data['power_curves'][0]['wind_speed_data']
    return (np.array([e['wind_speed'] for e in entries]),
            np.array([e['performance']['electrical_power']
                      ['average_cycle_power'] for e in entries]) / 1e3)


def plot_optima(grid: pd.DataFrame, optima: pd.DataFrame, kites, labels,
                results_dir: Path, generator_kw: float,
                reference_area: float, output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3),
                             gridspec_kw={'width_ratios': [1, 1, 1.25]})
    reference = grid[(grid['flat_area_m2'] == reference_area)
                     & np.isclose(grid['allowable_tether_stress_gpa'],
                                  REFERENCE_STRESS_GPA)]
    y = np.arange(len(kites))[::-1]
    for ax, column, title in [
            (axes[0], 'lcoe_eur_per_mwh', 'LCoE [EUR/MWh]'),
            (axes[1], 'aep_mwh', 'Annual energy production [MWh]')]:
        for yi, color, kite in zip(y, SERIES, kites):
            fixed = reference[reference['kite'] == kite][column].iloc[0]
            opt = optima[optima['kite'] == kite].iloc[0]
            ax.plot([fixed, opt[column]], [yi, yi], color=AXIS,
                    linewidth=2.5, zorder=1)
            ax.scatter([fixed], [yi], s=70, facecolor=SURFACE,
                       edgecolor=color, linewidth=2, zorder=3)
            ax.scatter([opt[column]], [yi], s=90, color=color,
                       edgecolor=SURFACE, linewidth=1.3, zorder=4)
            ax.annotate(f"{opt[column]:.0f}", (opt[column], yi),
                        xytext=(0, 9), textcoords='offset points',
                        ha='center', color=INK, fontsize=8.5)
            ax.annotate(f"{fixed:.0f}", (fixed, yi), xytext=(0, -15),
                        textcoords='offset points', ha='center',
                        color=INK_2, fontsize=8.5)
        ax.set_yticks(y, [labels[k] for k in kites])
        ax.set_ylim(-0.7, len(kites) - 0.3)
        ax.set_title(title)
        ax.grid(axis='y', visible=False)
    axes[1].set_yticklabels([])
    axes[0].scatter([], [], s=70, facecolor=SURFACE, edgecolor=MUTED,
                    linewidth=2, label=f'{reference_area:g} m$^2$, '
                          f'{REFERENCE_STRESS_GPA:g} GPa (fixed)')
    axes[0].scatter([], [], s=90, color=MUTED, label='optimum')
    axes[0].legend(loc='upper left', bbox_to_anchor=(0.0, -0.12), ncol=2)

    ax = axes[2]
    for color, kite in zip(SERIES, kites):
        opt = optima[optima['kite'] == kite].iloc[0]
        wind, power = _power_curve(results_dir / kite / opt['case'])
        ax.plot(wind, power, color=color,
                label=f"{kite}: S = {opt['flat_area_m2']:g} m$^2$, "
                      f"{opt['allowable_tether_stress_gpa']:.2f} GPa")
    ax.set_title('Power curves of the optima [kW]')
    ax.set_xlabel('Wind speed at 200 m [m/s]')
    ax.legend(loc='upper left')
    _suptitle(fig, f'Fixed reference design vs optimum per kite '
                   f'({generator_kw:g} kW generator)')
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--study', type=Path,
        default=studies_config_dir() / 'kite_comparison_40kw.yml',
        help='Kite study configuration file')
    args = parser.parse_args()

    config = load_kite_study_config(args.study)
    resultsDir = config.results_dir
    grid = pd.read_csv(resultsDir / 'grid_surface.csv')
    optima = pd.read_csv(resultsDir / 'optimum_by_kite.csv')
    kites = list(config.kites)
    labels = {k: _kite_label(k, _kite_aero(resultsDir / k / 'inputs'))
              for k in kites}
    generatorKw = config.design_defaults['generator_max_power_w'] / 1e3

    plotDir = resultsDir / 'plots'
    plotDir.mkdir(parents=True, exist_ok=True)
    _style()
    ok = grid[~grid['error']]
    # Reference design: 75 m2 when on the grid (thesis 40 kW optimum),
    # otherwise the middle grid area.
    areas = config.flat_areas_m2
    referenceArea = 75.0 if 75.0 in areas else areas[len(areas) // 2]
    capFactor = (config.case_options or {}).get('peak_power_cap_factor')
    peakCapKw = capFactor * generatorKw if capFactor else None
    plot_trends(ok, optima, kites, labels, generatorKw, referenceArea,
                peakCapKw,
                plotDir / 'kite_trends.png')
    plot_surfaces(grid, optima, kites, labels, generatorKw,
                  plotDir / 'kite_surfaces.png')
    plot_optima(ok, optima, kites, labels, resultsDir, generatorKw,
                referenceArea,
                plotDir / 'kite_optima.png')
    print(f"Plots written to {plotDir}")


if __name__ == '__main__':
    main()
