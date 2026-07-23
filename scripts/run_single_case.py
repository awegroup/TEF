"""Run one full TEF case (default: the 25 m2 V3-size design)."""

import argparse
from pathlib import Path

from tef.io import base_config_dir, results_dir
from tef.pipeline import TefCaseRunner, prepare_study_inputs
from tef.system_scaling import DesignVariables


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--area', type=float, default=25.0,
                        help='Flat wing area [m2]')
    parser.add_argument('--work-dir', type=Path,
                        default=results_dir() / 'single_case',
                        help='Directory holding inputs/ and the case folder')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite an existing case directory')
    parser.add_argument('--no-validate', action='store_true',
                        help='Skip awesIO validation')
    parser.add_argument('--no-storage-sizing', action='store_true',
                        help='Skip the storage sizing block')
    args = parser.parse_args()

    inputs = prepare_study_inputs(base_config_dir(),
                                  args.work_dir / 'inputs')
    result = TefCaseRunner(
        case_dir=args.work_dir / f"area_{int(round(args.area)):03d}",
        design=DesignVariables(flat_area_m2=args.area),
        inputs_dir=inputs.inputs_dir,
        validate=not args.no_validate,
        run_storage_sizing=not args.no_storage_sizing,
        overwrite=args.overwrite,
    ).run()

    print(f"\n=== TEF case {result.case_name} complete ===")
    print(f"AEP  = {result.aep_mwh:.2f} MWh")
    print(f"CF   = {result.capacity_factor:.3f}")
    print(f"LCoE = {result.lcoe_eur_per_mwh:.1f} EUR/MWh")
    print(f"ICC  = {result.icc_eur / 1e3:.1f} kEUR")
    print(f"OMC  = {result.omc_eur_per_year / 1e3:.1f} kEUR/year")
    print(f"Case folder: {result.case_dir}")


if __name__ == '__main__':
    main()
