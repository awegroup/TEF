"""Run the design-optimisation study: minimise LCoE over the design
space at each fixed generator power limit."""

import argparse
from pathlib import Path

from tef.design_studies import run_design_optimisation
from tef.io import studies_config_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--study', type=Path,
        default=studies_config_dir() / 'design_optimisation.yml',
        help='Design-optimisation study configuration file')
    args = parser.parse_args()

    frame = run_design_optimisation(args.study)
    print("\nOptimum by generator power:")
    print(frame.to_string(index=False))


if __name__ == '__main__':
    main()
