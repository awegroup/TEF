"""Run the wing area sweep study."""

import argparse
from pathlib import Path

from tef.io import studies_config_dir
from tef.studies import run_area_sweep


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--study', type=Path,
        default=studies_config_dir() / 'area_sweep_25_400.yml',
        help='Study configuration file')
    args = parser.parse_args()
    results = run_area_sweep(args.study)
    n_ok = sum(1 for r in results if r is not None)
    print(f"\nArea sweep finished: {n_ok}/{len(results)} cases succeeded")


if __name__ == '__main__':
    main()
