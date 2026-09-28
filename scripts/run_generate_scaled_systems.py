"""Generate scaled system files and summaries only.

Runs the system scaling block for the five anchor areas without
touching AWESPA or EcoMo. This is the first script to test.
"""

import argparse
from pathlib import Path

from tef.io import base_config_dir, results_dir, write_yaml
from tef.studies import area_case_name
from tef.system_scaling import (
    DesignVariables,
    build_scaled_system,
    scaling_summary_section,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--areas', type=float, nargs='+',
                        default=[25.0, 50.0, 100.0, 200.0, 400.0],
                        help='Flat wing areas [m2] to generate')
    parser.add_argument('--output-dir', type=Path,
                        default=results_dir() / 'scaled_systems',
                        help='Directory for the generated files')
    parser.add_argument('--no-validate', action='store_true',
                        help='Skip awesIO validation')
    args = parser.parse_args()

    config = base_config_dir()
    for area in args.areas:
        caseDir = args.output_dir / area_case_name(area)
        design = DesignVariables(flat_area_m2=area)
        quantities = build_scaled_system(
            base_system_path=config / 'base_system.yml',
            scaling_settings_path=config / 'scaling_settings.yml',
            design=design,
            output_system_path=caseDir / 'system.yml',
            validate=not args.no_validate,
        )
        write_yaml({
            'design': {'design_variables': design.to_dict()},
            'scaling': scaling_summary_section(quantities),
        }, caseDir / 'tef_summary.yml')
        print(f"{area_case_name(area)}: "
              f"m_airborne = {quantities.total_airborne_mass_kg:.2f} kg, "
              f"F_t,max = {quantities.max_tether_force_n:.0f} N, "
              f"d_t = {quantities.tether_diameter_m * 1e3:.2f} mm, "
              f"P_gen = {quantities.generator_max_power_w / 1e3:.0f} kW")
    print(f"\nScaled systems written to {args.output_dir}")


if __name__ == '__main__':
    main()
