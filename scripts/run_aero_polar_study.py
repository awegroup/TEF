"""Run the aero polar study (Stage A): LCoE of a fixed design over a
grid of reel-out polars (L/D_max x CL_max)."""

import argparse
import time
from pathlib import Path

from tef.aero_polars import run_aero_polar_study
from tef.io import studies_config_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--study', type=Path,
        default=studies_config_dir() / 'aero_polar_40kw.yml',
        help='Aero polar study configuration file')
    args = parser.parse_args()

    start = time.perf_counter()
    frame = run_aero_polar_study(args.study)

    columns = ['ld_max', 'cl_max', 'n_operating_points', 'selected_cl_min',
               'selected_cl_max', 'aep_mwh', 'capacity_factor',
               'rated_power_kw', 'icc_keur', 'omc_keur_per_year',
               'lcoe_eur_per_mwh', 'qsm_runtime_min']
    print("\nAero polar study:")
    print(frame[columns].to_string(index=False, float_format='%.3g'))
    print("\nLCoE [EUR/MWh] (rows L/D_max, columns CL_max):")
    print(frame.pivot(index='ld_max', columns='cl_max',
                      values='lcoe_eur_per_mwh').round(1).to_string())
    print(f"\nTotal wall time: {(time.perf_counter() - start) / 60:.1f} min")


if __name__ == '__main__':
    main()
