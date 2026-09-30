"""Run the depower variant of a finished polar grid study: at each wind
speed the kite flies the best CL up to its maximum (same polar), built
from the existing fixed-CL power curves."""

import argparse
import time
from pathlib import Path

from tef.aero_polars import run_depower_study
from tef.io import studies_config_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--study', type=Path,
        default=studies_config_dir() / 'aero_polar_grid_40kw_depower.yml',
        help='Depower study configuration file')
    args = parser.parse_args()

    start = time.perf_counter()
    frame = run_depower_study(args.study)
    frame = frame[~frame['error']]
    optima = frame.loc[frame.groupby(['ld_max', 'cl'])
                       ['lcoe_eur_per_mwh'].idxmin()]
    for column, heading in [('lcoe_eur_per_mwh', 'LCoE [EUR/MWh]'),
                            ('aep_mwh', 'AEP [MWh]'),
                            ('flat_area_m2', 'Optimal wing area [m2]')]:
        print(f"\n{heading} (rows L/D_max, columns maximum CL):")
        print(optima.pivot(index='ld_max', columns='cl',
                           values=column).round(0).to_string())
    print(f"\nTotal wall time: {(time.perf_counter() - start) / 60:.1f} min")


if __name__ == '__main__':
    main()
