"""Wrappers around the external model blocks: AWESPA power curves,
AWESPA AEP, and EcoMo (AWESPA-connected mode).

Each wrapper is a thin, stable interface: paths in, result dict out.
The external packages are imported lazily so a missing dependency
fails with a helpful message only when the block is actually used.
"""

from pathlib import Path
from typing import Optional


class AwespaPowerRunner:
    """Runs AWESPA power curve generation for one TEF case."""

    def __init__(self, system_path: Path, qsm_settings_path: Path,
                 wind_resource_path: Path, output_power_curves_path: Path,
                 validate: bool = True, verbose: bool = True):
        self.systemPath = Path(system_path)
        self.qsmSettingsPath = Path(qsm_settings_path)
        self.windResourcePath = Path(wind_resource_path)
        self.outputPowerCurvesPath = Path(output_power_curves_path)
        self.validate = validate
        self.verbose = verbose

    def run(self) -> dict:
        """Compute the power curves and write them to the case folder.

        Returns:
            dict: The power curve data returned by AWESPA.
        """
        try:
            from awespa.power.inertiafree_qsm_power import (
                InertiaFreeQSMPowerModel)
        except ImportError as exc:
            raise ImportError(
                "AWESPA is not installed. Install it in editable mode, "
                "e.g.: pip install -e <path-to-AWESPA>") from exc

        model = InertiaFreeQSMPowerModel()
        model.load_configuration(
            system_path=self.systemPath,
            simulation_settings_path=self.qsmSettingsPath,
            wind_resource_path=self.windResourcePath,
            validate=self.validate,
        )
        return model.compute_power_curves(
            profile_ids=None,
            output_path=self.outputPowerCurvesPath,
            verbose=self.verbose,
            showplot=False,
            saveplot=False,
            validate=self.validate,
        )


class AepRunner:
    """Runs the AWESPA AEP calculation for one TEF case."""

    def __init__(self, power_curve_path: Path, wind_resource_path: Path,
                 output_path: Path, plot: bool = False,
                 plot_output_dir: Optional[Path] = None):
        self.powerCurvePath = Path(power_curve_path)
        self.windResourcePath = Path(wind_resource_path)
        self.outputPath = Path(output_path)
        self.plot = plot
        self.plotOutputDir = plot_output_dir

    def run(self) -> dict:
        """Compute AEP and capacity factor from the power curves.

        Returns:
            dict: The AEP results returned by AWESPA.
        """
        try:
            from awespa.pipeline.aep import calculate_aep
        except ImportError as exc:
            raise ImportError(
                "AWESPA is not installed. Install it in editable mode, "
                "e.g.: pip install -e <path-to-AWESPA>") from exc

        return calculate_aep(
            power_curve_path=self.powerCurvePath,
            wind_resource_path=self.windResourcePath,
            output_path=self.outputPath,
            plot=self.plot,
            plot_output_dir=self.plotOutputDir,
        )


class EcomoRunner:
    """Runs EcoMo for one TEF case (AWESPA-connected mode)."""

    def __init__(self, economic_settings_path: Path, output_path: Path,
                 validate: bool = True, verbose: bool = True):
        self.economicSettingsPath = Path(economic_settings_path)
        self.outputPath = Path(output_path)
        self.validate = validate
        self.verbose = verbose

    def run(self) -> dict:
        """Compute the economics and write ecomo_results.yml.

        Returns:
            dict: The full EcoMo result dictionary (with a ``metrics``
            section holding LCoE, ICC, OMC, ...).
        """
        try:
            from ecomo import EcoMo
        except ImportError as exc:
            raise ImportError(
                "EcoMo is not installed. Install it in editable mode, "
                "e.g.: pip install -e <path-to-ecomo>") from exc

        model = EcoMo()
        model.load_configuration(
            economic_settings_path=self.economicSettingsPath,
            validate=self.validate,
        )
        return model.compute_economics(
            output_path=self.outputPath,
            verbose=self.verbose,
            validate=self.validate,
        )
