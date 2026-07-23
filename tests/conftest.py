"""Shared fixtures for the TEF test suite."""

import dataclasses

import pytest

from tef.io import base_config_dir, load_yaml
from tef.system_scaling import ScalingSettings


@pytest.fixture(scope='session')
def base_settings() -> ScalingSettings:
    """The repository's scaling settings (raw_thesis, sensor included)."""
    return ScalingSettings.load(base_config_dir() / 'scaling_settings.yml')


@pytest.fixture(scope='session')
def base_system() -> dict:
    return load_yaml(base_config_dir() / 'base_system.yml')


@pytest.fixture
def v3_reproduction_settings(base_settings) -> ScalingSettings:
    """Settings that reproduce Bredael's original V3 inputs exactly:
    sensor mass excluded and KCU calibrated to 8.4 kg at 25 m2."""
    return dataclasses.replace(
        base_settings,
        kcu_mass_model=dataclasses.replace(
            base_settings.kcu_mass_model,
            calibration_mode='match_v3_at_25'),
        sensor_mass_model=dataclasses.replace(
            base_settings.sensor_mass_model,
            include_in_control_system_mass=False),
    )
