"""Re-evaluate the thesis 40 kW reference design (S = 75 m2,
sigma_allow = 0.25 GPa, Table 7.2) with modified reel-out aerodynamics
(CL = 1.2, CD = 0.2), side by side with the baseline aero."""

import time

from tef.io import project_root, results_dir
from tef.pipeline import TefCaseRunner, prepare_study_inputs
from tef.system_scaling import DesignVariables

VARIANTS = {
    'baseline_cl0p63_cd0p14': project_root() / 'config' / 'base',
    'aero_cl1p2_cd0p2': project_root() / 'config' / 'base_aero_cl1p2_cd0p2',
}


def main() -> None:
    workDir = results_dir() / 'aero_variant_40kw'
    results = {}
    runtimes = {}
    for name, baseDir in VARIANTS.items():
        inputs = prepare_study_inputs(baseDir, workDir / name / 'inputs')
        design = DesignVariables(
            flat_area_m2=75.0,
            allowable_tether_stress_pa=2.5e8,
            generator_max_power_w=40000.0,
            max_tether_speed_m_s=10.0,
            tether_length_m=500.0,
            case_name='S075_s0p25_40kw',
        )
        start = time.perf_counter()
        results[name] = TefCaseRunner(
            case_dir=workDir / name / design.case_name,
            design=design,
            inputs_dir=inputs.inputs_dir,
            validate=False,
            run_storage_sizing=True,
            overwrite=True,
        ).run()
        runtimes[name] = time.perf_counter() - start
        print(f"  {name}: {runtimes[name] / 60:.1f} min")

    print(f"\n{'':12s}" + ''.join(f"{n:>26s}" for n in results))
    for label, attr, scale, fmt in [
            ('AEP [MWh]', 'aep_mwh', 1.0, '.1f'),
            ('CF [-]', 'capacity_factor', 1.0, '.3f'),
            ('ICC [kEUR]', 'icc_eur', 1e-3, '.1f'),
            ('OMC [kEUR/y]', 'omc_eur_per_year', 1e-3, '.1f'),
            ('LCoE [EUR/MWh]', 'lcoe_eur_per_mwh', 1.0, '.1f')]:
        print(f"{label:12s}" + ''.join(
            f"{getattr(r, attr) * scale:>26{fmt}}" for r in results.values()))
    print(f"{'Runtime [min]':12s}" + ''.join(
        f"{t / 60:>26.1f}" for t in runtimes.values()))
    print(f"Case folders under: {workDir}")


if __name__ == '__main__':
    main()
