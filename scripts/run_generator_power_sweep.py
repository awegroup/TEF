"""Run the area x generator specific power sweep study."""

import argparse
from pathlib import Path

from tef.io import studies_config_dir
from tef.studies import run_generator_power_sweep


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--study', type=Path,
        default=studies_config_dir() / 'generator_power_sweep.yml',
        help='Study configuration file')
    args = parser.parse_args()
    results = run_generator_power_sweep(args.study)
    n_ok = sum(1 for r in results if r is not None)
    print(f"\nGenerator power sweep finished: "
          f"{n_ok}/{len(results)} cases succeeded")


if __name__ == '__main__':
    main()
