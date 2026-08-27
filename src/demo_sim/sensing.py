"""Sparse noisy modal sensing and environmental normalisation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import constants as C
from .frame import FrameModel
from .ground_truth import GroundTruth


@dataclass
class SensorData:
    observations: pd.DataFrame
    measured_shapes: np.ndarray
    sensor_dofs: np.ndarray
    true_reference_freq: np.ndarray
    pristine_freq: np.ndarray
    true_shapes: np.ndarray


@dataclass
class NormalisedModalData:
    reference_freq: np.ndarray
    frequency_sd: np.ndarray
    measured_shapes: np.ndarray
    sensor_dofs: np.ndarray
    pristine_freq: np.ndarray
    regression_table: pd.DataFrame
    contamination_table: pd.DataFrame


def simulate_sensors(frame: FrameModel, truth: GroundTruth, rng: np.random.Generator) -> SensorData:
    damaged_freq, damaged_shapes = frame.modal(truth.alpha)
    pristine_freq, _ = frame.modal(np.ones(len(frame.elements)))
    sensor_dofs = frame.sensor_dofs()
    phase = np.linspace(0.0, 2.0 * np.pi, C.N_ENV_OBSERVATIONS, endpoint=False)
    temp = C.TEMPERATURE_MEAN + C.TEMPERATURE_AMPLITUDE * np.sin(phase) + rng.normal(0.0, C.TEMPERATURE_RANDOM_SD, len(phase))
    humidity = C.HUMIDITY_MEAN + C.HUMIDITY_AMPLITUDE * np.cos(
        phase + C.HUMIDITY_PHASE_OFFSET
    ) + rng.normal(0.0, C.HUMIDITY_RANDOM_SD, len(phase))
    rows: list[dict] = []
    for obs in range(C.N_ENV_OBSERVATIONS):
        dt = temp[obs] - C.TEMPERATURE_MEAN
        dh = humidity[obs] - C.HUMIDITY_MEAN
        for mode in range(C.N_MODES):
            mode_scale = 1.0 + C.ENV_MODE_SCALE_STEP * mode
            env_shift = mode_scale * (
                C.TEMP_FREQUENCY_COEFF * dt
                + C.HUMIDITY_FREQUENCY_COEFF * dh
                + C.TEMP_QUADRATIC_COEFF * dt**2
            )
            noiseless = damaged_freq[mode] * (1.0 + env_shift)
            measured = noiseless * (1.0 + rng.normal(0.0, C.FREQUENCY_NOISE_FRACTION))
            rows.append({
                "observation": obs,
                "mode": mode + 1,
                "temperature_degC": temp[obs],
                "humidity_pct": humidity[obs],
                "measured_frequency_hz": measured,
                "true_environment_shift_fraction": env_shift,
                "source": "illustrative",
            })
    observed_shapes = damaged_shapes[sensor_dofs, :] + rng.normal(
        0.0, C.MODE_SHAPE_NOISE, (len(sensor_dofs), C.N_MODES)
    )
    return SensorData(pd.DataFrame(rows), observed_shapes, sensor_dofs, damaged_freq, pristine_freq, damaged_shapes)


def normalise_environment(data: SensorData) -> NormalisedModalData:
    regression_rows: list[dict] = []
    contamination_rows: list[dict] = []
    reference = np.zeros(C.N_MODES)
    uncertainty = np.zeros(C.N_MODES)
    for mode in range(1, C.N_MODES + 1):
        subset = data.observations[data.observations["mode"] == mode].copy()
        dt = subset["temperature_degC"].to_numpy() - C.TEMPERATURE_MEAN
        dh = subset["humidity_pct"].to_numpy() - C.HUMIDITY_MEAN
        x = np.column_stack([np.ones(len(subset)), dt, dh])
        y = np.log(subset["measured_frequency_hz"].to_numpy())
        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
        residual = y - x @ beta
        reference[mode - 1] = np.exp(beta[0])
        uncertainty[mode - 1] = reference[mode - 1] * max(np.std(residual, ddof=3), C.FREQUENCY_NOISE_FRACTION)
        predicted_shift = x[:, 1:] @ beta[1:]
        true_shift = subset["true_environment_shift_fraction"].to_numpy()
        contamination = predicted_shift - true_shift
        regression_rows.append({
            "mode": mode,
            "reference_frequency_hz": reference[mode - 1],
            "frequency_sd_hz": uncertainty[mode - 1],
            "temperature_log_coefficient": beta[1],
            "humidity_log_coefficient": beta[2],
            "residual_log_sd": np.std(residual, ddof=3),
            "source": "illustrative",
        })
        contamination_rows.append({
            "mode": mode,
            "residual_environment_rms_fraction": float(np.sqrt(np.mean(contamination**2))),
            "residual_environment_max_abs_fraction": float(np.max(np.abs(contamination))),
            "source": "illustrative",
        })
    return NormalisedModalData(
        reference, uncertainty, data.measured_shapes, data.sensor_dofs, data.pristine_freq,
        pd.DataFrame(regression_rows), pd.DataFrame(contamination_rows)
    )
