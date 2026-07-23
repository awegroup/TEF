# TEF — Techno-Economic Framework for soft-wing AWE systems

TEF is the top-level orchestration framework that couples:

1. An internal **system scaling block** (`tef.system_scaling`) that
   generates physically consistent awesIO system files from design
   variables (wing area, wing loading, tether stress, generator sizing).
2. The **AWESPA** inertia-free QSM performance model (power curves).
3. The **AWESPA AEP** calculation.
4. A **storage sizing block** that sizes the ground-station storage from
   the computed power curves (cycle energy imbalance method).
5. The **EcoMo** economic model in AWESPA-connected mode (LCoE, ICC,
   OMC, ...).

The framework targets conceptual design and trend identification, not
accurate cost prediction. awesIO-style YAML files are the central data
exchange layer, mirroring AWESPA's architecture one level up.

## Installation

AWESPA, EcoMo and awesIO are external dependencies installed in
editable mode from their existing locations (no clones inside TEF):

```
pip install -e <path-to>/projects/AWESPA
pip install -e <path-to>/projects/ecomo
pip install -e <path-to>/projects/TEF
```

awesIO is installed from PyPI/its own repo (`pip install awesio` or an
editable clone).

## Layout

```
config/base/      base system, scaling settings, QSM settings, wind
                  resource, cost inputs, economic settings template
config/studies/   study definitions (area sweep, generator power sweep)
src/tef/          the framework package: one module per framework block
scripts/          entry points
results/          one directory per study, one subdirectory per case
```

The package mirrors AWESPA's style — one clear module per model block
with a small stable interface, re-exported from ``tef``:

```
tef/io.py              YAML helpers and project paths
tef/system_scaling.py  design variables -> scaled awesIO system file
                       (build_scaled_system, DesignVariables,
                       ScalingSettings)
tef/storage_sizing.py  storage capacity from computed power curves
tef/wrappers.py        AWESPA power / AEP / EcoMo interfaces
tef/pipeline.py        full model chain for one case (TefCaseRunner)
tef/studies.py         sweeps and combined summaries (run_area_sweep,
                       run_generator_power_sweep)
tef/design_space.py    declarative optimised variables + bounds
                       (DesignSpace, DesignVariableSpec)
tef/optimisation.py    model-agnostic optimisers (grid, optional scipy
                       refine) and the 1-D rated-power solver
tef/design_studies.py  LCoE optimisation over the design space at each
                       fixed rated power (run_design_optimisation)
```

Each block writes exactly one output file (AWESPA style). The inputs
shared by all cases of a study are written once to
`results/<study>/inputs/`; a case directory holds only the
case-specific files:

```
results/<study>/
    inputs/           base system, scaling settings, QSM settings,
                      wind resource, cost inputs, economics template
    <case>/
        design.yml           the design variables of this case
        system.yml           scaling block output (updated by storage sizing)
        power_curves.yml     AWESPA output (+ .npz)
        aep_results.yml      AEP block output
        economic_settings.yml  generated; references ../inputs/ relatively
        ecomo_results.yml    EcoMo output
        tef_summary.yml      consolidated summary (design, scaling mass
                             split, storage sizing, performance, economics)
    study_summary.yml / .csv
```

## Usage

```
# 1. Scaled system files only (no AWESPA/EcoMo):
python scripts/run_generate_scaled_systems.py

# 2. One full case (default 25 m2):
python scripts/run_single_case.py --area 25

# Optimisation entry point (quiet, no validation, reusable work dir):
#   from tef import evaluate_design, DesignVariables
#   result = evaluate_design(DesignVariables(flat_area_m2=25.0),
#                            work_dir=Path('results/opt'))
#   result.lcoe_eur_per_mwh  # objective

# 3. The five-point area sweep (25..400 m2):
python scripts/run_area_sweep.py

# 4. The area x generator specific power grid:
python scripts/run_generator_power_sweep.py

# 5. Design optimisation: minimise LCoE over {wing area, tether stress}
#    at each fixed rated power (25..300 kW). The generator is sized so
#    the emergent rated power matches each target.
python scripts/run_design_optimisation.py
```

Run the tests with:

```
python -m pytest tests/ -v
```

## Key settings (config/base/scaling_settings.yml)

- `wing_mass_model.type`: `tabulated_log_interpolation` (thesis anchor
  table, default) or `power_law` (smooth, for optimisation).
- `kcu_mass_model.calibration_mode`: `raw_thesis` (default) or
  `match_v3_at_25` (reproduces Bredael's 8.4 kg KCU exactly).
- `sensor_mass_model.include_in_control_system_mass`: `true` for final
  TEF runs; set `false` only to reproduce Bredael's 19 kg airborne mass.
- `storage_sizing.mode` / `method`: sizing from power curves via the
  cycle energy imbalance (smoothing to cycle-average power), with a
  generator-power-scaled fallback.

Reproducing Bredael's V3 inputs exactly: set `calibration_mode:
match_v3_at_25` and `include_in_control_system_mass: false` — this is
covered by `tests/test_scaling_v3_reference.py`.

## Code conventions

TEF follows the AWESPA developer guide: snake_case functions/methods
and parameters, mixedCase local variables and attributes, CamelCase
classes, UPPER_SNAKE_CASE constants, Google-style docstrings.
Dataclass fields that mirror YAML keys or CSV columns stay snake_case
(they are the serialization schema, like the awesIO files).
