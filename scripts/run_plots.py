"""Generate the design-optimisation figures from a study's results."""

import argparse
from pathlib import Path

from tef.io import results_dir as default_results_dir
from tef.plotting import plot_all


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--results', type=Path,
        default=default_results_dir() / 'design_optimisation',
        help='Study results directory (holds optimum_by_rated_power.csv)')
    parser.add_argument(
        '--output', type=Path, default=None,
        help='Figure output directory (default: <results>/figures)')
    args = parser.parse_args()

    outputDir = args.output or (args.results / 'figures')
    names = plot_all(args.results, outputDir)
    print(f"Wrote {len(names)} figures to {outputDir}:")
    for name in names:
        print(f"  {name}.png / .pdf")


if __name__ == '__main__':
    main()
