"""Run the polar grid study: LCoE of a fixed design over polar
efficiency (L/D_max) x fixed reel-out lift coefficient."""

import argparse
import time
from pathlib import Path

from tef.aero_polars import run_polar_grid_study
from tef.io import studies_config_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--study', type=Path,
        default=studies_config_dir() / 'aero_polar_grid_40kw.yml',
        help='Polar grid study configuration file')
    args = parser.parse_args()

    start = time.perf_counter()
    frame = run_polar_grid_study(args.study)
    # One row per (L/D_max, CL): the lowest-LCoE wing area when the study
    # sweeps areas, the single case otherwise.
    frame = frame[~frame['error']]
    frame = frame.loc[frame.groupby(['ld_max', 'cl'])
                      ['lcoe_eur_per_mwh'].idxmin()]
    print("\nOptimal wing area [m2]:")
    print(frame.pivot(index='ld_max', columns='cl',
                      values='flat_area_m2').to_string())
    print("\nLCoE [EUR/MWh] (rows L/D_max, columns CL):")
    print(frame.pivot(index='ld_max', columns='cl',
                      values='lcoe_eur_per_mwh').round(0).to_string())
    print("\nAEP [MWh]:")
    print(frame.pivot(index='ld_max', columns='cl',
                      values='aep_mwh').round(0).to_string())
    print(f"\nTotal wall time: {(time.perf_counter() - start) / 60:.1f} min")


if __name__ == '__main__':
    main()
