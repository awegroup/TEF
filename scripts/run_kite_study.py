"""Run the kite comparison study: grid-optimise each kite at a fixed
generator power limit, in parallel."""

import argparse
import time
from pathlib import Path

from tef.io import studies_config_dir
from tef.kite_studies import run_kite_study


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--study', type=Path,
        default=studies_config_dir() / 'kite_comparison_40kw.yml',
        help='Kite study configuration file')
    args = parser.parse_args()

    start = time.perf_counter()
    optimum = run_kite_study(args.study)
    columns = ['kite', 'flat_area_m2', 'allowable_tether_stress_gpa',
               'area_on_bound', 'stress_on_bound', 'aep_mwh',
               'capacity_factor', 'rated_power_kw', 'peak_mech_power_kw',
               'icc_keur', 'omc_keur_per_year', 'lcoe_eur_per_mwh']
    print("\nOptimum by kite:")
    print(optimum[columns].to_string(index=False, float_format='%.4g'))
    print(f"\nTotal wall time: {(time.perf_counter() - start) / 60:.1f} min")


if __name__ == '__main__':
    main()
