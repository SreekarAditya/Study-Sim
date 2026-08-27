"""Two-tier state, illustrative transport M, and performance classification N."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import constants as C
from .estimation import StateEstimate
from .frame import FrameModel
from .ground_truth import GroundTruth


@dataclass
class TwoTierState:
    table: pd.DataFrame
    alpha_mean: np.ndarray
    alpha_covariance: np.ndarray
    width_mean: np.ndarray
    width_sd: np.ndarray
    theta_mean: np.ndarray
    theta_sd: np.ndarray
    full_mask: np.ndarray


@dataclass
class TransportState:
    table: pd.DataFrame
    mean: np.ndarray
    sd: np.ndarray
    true_multiplier: np.ndarray


@dataclass
class PerformanceState:
    table: pd.DataFrame
    score_samples: np.ndarray
    conformal_quantile: float
    calibration: dict


def force_all_reduced(state: TwoTierState) -> TwoTierState:
    """Remove every geometry record for the dedicated two-tier ablation."""
    scale = np.where(state.full_mask, C.REDUCED_ALPHA_UNCERTAINTY_FACTOR, 1.0)
    covariance = np.diag(scale) @ state.alpha_covariance @ np.diag(scale)
    table = state.table.copy()
    table["geometry_admissible"] = False
    table["tier"] = "reduced_alpha_only"
    table["alpha_sd_after_tier"] = np.sqrt(np.diag(covariance))
    table["crack_width_mean_mm"] = np.nan
    table["crack_width_sd_mm"] = np.nan
    table["theta_mean_deg"] = np.nan
    table["theta_sd_deg"] = np.nan
    table["expiry_reason"] = "ablation_forced_reduced"
    table["source"] = "illustrative"
    n = len(state.alpha_mean)
    return TwoTierState(
        table,
        state.alpha_mean.copy(),
        covariance,
        np.full(n, np.nan),
        np.full(n, np.nan),
        np.full(n, np.nan),
        np.full(n, np.nan),
        np.zeros(n, dtype=bool),
    )


def make_two_tier_state(
    frame: FrameModel,
    estimate: StateEstimate,
    truth: GroundTruth,
    rng: np.random.Generator,
) -> TwoTierState:
    n = len(frame.elements)
    exists = rng.random(n) < C.GEOMETRY_COVERAGE_RATE
    age = rng.integers(0, C.GEOMETRY_VALID_PERIODS + 3, n)
    new_event = rng.random(n) < C.NEW_DAMAGE_EVENT_RATE
    admissible = exists & (age <= C.GEOMETRY_VALID_PERIODS) & (~new_event)
    measured_w = np.where(
        exists,
        np.maximum(0.0, truth.crack_width_mm + rng.normal(0.0, C.GEOMETRY_WIDTH_SD, n)),
        np.nan,
    )
    measured_theta = np.where(
        exists,
        np.mod(truth.theta_deg + rng.normal(0.0, C.GEOMETRY_ANGLE_SD, n), 180.0),
        np.nan,
    )
    # Widen the full joint covariance by congruence transformation so all
    # cross-member dependencies survive the tier downgrade.
    scale = np.where(admissible, 1.0, C.REDUCED_ALPHA_UNCERTAINTY_FACTOR)
    widened_cov = np.diag(scale) @ estimate.covariance @ np.diag(scale)
    table = frame.member_table()
    table["geometry_exists"] = exists
    table["geometry_age_periods"] = np.where(exists, age, np.nan)
    table["new_damage_event"] = new_event
    table["geometry_admissible"] = admissible
    table["tier"] = np.where(admissible, "full_alpha_w_theta", "reduced_alpha_only")
    table["alpha_mean"] = estimate.mean
    table["alpha_sd_after_tier"] = np.sqrt(np.diag(widened_cov))
    table["crack_width_mean_mm"] = np.where(admissible, measured_w, np.nan)
    table["crack_width_sd_mm"] = np.where(admissible, C.GEOMETRY_WIDTH_SD, np.nan)
    table["theta_mean_deg"] = np.where(admissible, measured_theta, np.nan)
    table["theta_sd_deg"] = np.where(admissible, C.GEOMETRY_ANGLE_SD, np.nan)
    table["expiry_reason"] = np.select(
        [~exists, new_event & exists, exists & (age > C.GEOMETRY_VALID_PERIODS)],
        ["no_record", "new_damage_event", "age_expired"],
        default="valid",
    )
    table["source"] = "illustrative"
    return TwoTierState(
        table=table,
        alpha_mean=estimate.mean,
        alpha_covariance=widened_cov,
        width_mean=np.where(admissible, measured_w, np.nan),
        width_sd=np.where(admissible, C.GEOMETRY_WIDTH_SD, np.nan),
        theta_mean=np.where(admissible, measured_theta, np.nan),
        theta_sd=np.where(admissible, C.GEOMETRY_ANGLE_SD, np.nan),
        full_mask=admissible,
    )


def m_full(alpha: np.ndarray, width_mm: np.ndarray, theta_deg: np.ndarray) -> np.ndarray:
    """Illustrative full transport ratio M(alpha,w,theta), never calibrated."""
    orientation = 1.0 + C.M_ORIENTATION_WEIGHT * np.sin(np.deg2rad(theta_deg)) ** 2
    return (
        1.0
        + C.M_ALPHA_COEFF * np.maximum(1.0 - alpha, 0.0) ** C.M_ALPHA_POWER
        + C.M_WIDTH_COEFF * np.maximum(width_mm, 0.0) ** C.M_WIDTH_POWER * orientation
    )


def m_reduced(alpha: np.ndarray) -> np.ndarray:
    """Illustrative reduced M(alpha) used when geometry is inadmissible."""
    return 1.0 + C.M_REDUCED_ALPHA_COEFF * np.maximum(1.0 - alpha, 0.0) ** C.M_REDUCED_ALPHA_POWER


def _psd_samples(mean: np.ndarray, covariance: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    vals, vecs = np.linalg.eigh((covariance + covariance.T) / 2.0)
    root = vecs @ np.diag(np.sqrt(np.maximum(vals, 1.0e-12)))
    samples = mean + rng.normal(size=(count, len(mean))) @ root.T
    return np.clip(samples, C.ALPHA_LOWER_BOUND, 1.0)


def compute_transport(state: TwoTierState, truth: GroundTruth, rng: np.random.Generator) -> TransportState:
    count = C.PERFORMANCE_MONTE_CARLO_SAMPLES
    alpha_samples = _psd_samples(state.alpha_mean, state.alpha_covariance, count, rng)
    n = len(state.alpha_mean)
    m_samples = np.zeros((count, n))
    forms: list[str] = []
    for i in range(n):
        if state.full_mask[i]:
            w = np.maximum(0.0, rng.normal(state.width_mean[i], state.width_sd[i], count))
            theta = np.mod(rng.normal(state.theta_mean[i], state.theta_sd[i], count), 180.0)
            m_samples[:, i] = m_full(alpha_samples[:, i], w, theta)
            forms.append("M_full(alpha,w,theta); source=illustrative")
        else:
            centre = m_reduced(alpha_samples[:, i])
            m_samples[:, i] = np.maximum(
                1.0,
                centre * np.exp(rng.normal(0.0, C.M_REDUCED_LOG_SD, count)),
            )
            forms.append("M_reduced(alpha) with inflated log uncertainty; source=illustrative")
    true_m = m_full(truth.alpha, truth.crack_width_mm, truth.theta_deg)
    table = state.table[["member_id", "member", "tier"]].copy()
    table["M_mean"] = m_samples.mean(axis=0)
    table["M_sd"] = m_samples.std(axis=0, ddof=1)
    table["M_p05"] = np.quantile(m_samples, 0.05, axis=0)
    table["M_p95"] = np.quantile(m_samples, 0.95, axis=0)
    table["true_M_hidden_until_final_comparison"] = true_m
    table["functional_form"] = forms
    table["source"] = "illustrative"
    return TransportState(table, m_samples.mean(axis=0), m_samples.std(axis=0, ddof=1), true_m)


def acceptance_n_full(alpha: np.ndarray, width: np.ndarray, theta: np.ndarray, transport: np.ndarray) -> np.ndarray:
    """Illustrative acceptance modifier N(alpha,w,theta,transport)."""
    orientation = np.abs(np.sin(np.deg2rad(theta)))
    raw = (
        1.0
        - C.N_ALPHA_WEIGHT * (1.0 - alpha)
        - C.N_WIDTH_WEIGHT * np.maximum(width, 0.0)
        - C.N_TRANSPORT_WEIGHT * np.maximum(transport - 1.0, 0.0)
        - C.N_ORIENTATION_WEIGHT * orientation * np.maximum(width, 0.0)
    )
    return np.clip(raw, C.N_MINIMUM, 1.0)


def performance_label(score: float) -> str:
    a, b, c = C.PERFORMANCE_THRESHOLDS
    if score <= a:
        return "IO"
    if score <= b:
        return "LS"
    if score <= c:
        return "CP"
    return "Fail"


def _labels_intersecting(low: float, high: float) -> str:
    thresholds = [-np.inf, *C.PERFORMANCE_THRESHOLDS, np.inf]
    labels = ["IO", "LS", "CP", "Fail"]
    included = []
    for label, left, right in zip(labels, thresholds[:-1], thresholds[1:]):
        if high >= left and low <= right:
            included.append(label)
    return "|".join(included)


def conformal_calibration(rng: np.random.Generator) -> tuple[float, dict]:
    n_cal = C.CALIBRATION_SIZE
    predicted_cal = rng.uniform(C.CALIBRATION_SCORE_MIN, C.CALIBRATION_SCORE_MAX, n_cal)
    true_cal = predicted_cal + rng.normal(0.0, C.CALIBRATION_NOISE_SD, n_cal)
    scores = np.abs(true_cal - predicted_cal)
    rank = int(np.ceil((n_cal + 1) * (1.0 - C.CONFORMAL_MIS_COVERAGE)))
    q = float(np.sort(scores)[min(rank - 1, n_cal - 1)])
    predicted_test = rng.uniform(C.CALIBRATION_SCORE_MIN, C.CALIBRATION_SCORE_MAX, C.VALIDATION_SIZE)
    true_test = predicted_test + rng.normal(0.0, C.CALIBRATION_NOISE_SD, C.VALIDATION_SIZE)
    covered = np.abs(true_test - predicted_test) <= q
    return q, {
        "method": "split conformal absolute-residual interval on a synthetic exchangeable calibration set",
        "nominal_coverage": 1.0 - C.CONFORMAL_MIS_COVERAGE,
        "achieved_heldout_coverage": float(covered.mean()),
        "calibration_size": n_cal,
        "heldout_size": C.VALIDATION_SIZE,
        "quantile": q,
        "source": "illustrative",
    }


def classify_performance(
    frame: FrameModel,
    state: TwoTierState,
    transport: TransportState,
    rng: np.random.Generator,
) -> PerformanceState:
    count = C.PERFORMANCE_MONTE_CARLO_SAMPLES
    alpha_samples = _psd_samples(state.alpha_mean, state.alpha_covariance, count, rng)
    n = len(state.alpha_mean)
    score_samples = np.zeros((count, n))
    capacities = np.array([
        C.COLUMN_MOMENT_CAPACITY if e.kind == "column" else C.BEAM_MOMENT_CAPACITY
        for e in frame.elements
    ])
    for sample in range(count):
        _, moments = frame.static_response(alpha_samples[sample])
        n_factor = np.zeros(n)
        for i in range(n):
            if state.full_mask[i]:
                w = max(0.0, rng.normal(state.width_mean[i], state.width_sd[i]))
                theta = rng.normal(state.theta_mean[i], state.theta_sd[i]) % 180.0
                m = float(m_full(np.array([alpha_samples[sample, i]]), np.array([w]), np.array([theta]))[0])
                n_factor[i] = acceptance_n_full(
                    np.array([alpha_samples[sample, i]]), np.array([w]), np.array([theta]), np.array([m])
                )[0]
            else:
                m_centre = float(m_reduced(np.array([alpha_samples[sample, i]]))[0])
                m = max(1.0, m_centre * np.exp(rng.normal(0.0, C.M_REDUCED_LOG_SD)))
                reduced = 1.0 - C.N_ALPHA_WEIGHT * (1.0 - alpha_samples[sample, i]) - C.N_TRANSPORT_WEIGHT * max(m - 1.0, 0.0)
                n_factor[i] = np.clip(reduced + rng.normal(0.0, C.REDUCED_N_EXTRA_SD), C.N_MINIMUM, 1.0)
        score_samples[sample] = moments / (capacities * n_factor)
    q, calibration = conformal_calibration(rng)
    mean = score_samples.mean(axis=0)
    table = state.table[["member_id", "member", "kind", "tier"]].copy()
    table["utilisation_mean"] = mean
    table["utilisation_sd"] = score_samples.std(axis=0, ddof=1)
    table["performance_level"] = [performance_label(v) for v in mean]
    table["conformal_low"] = np.maximum(0.0, mean - q)
    table["conformal_high"] = mean + q
    table["conformal_label_set"] = [
        _labels_intersecting(max(0.0, v - q), v + q) for v in mean
    ]
    for label in ["IO", "LS", "CP", "Fail"]:
        table[f"prob_{label}"] = np.mean(
            np.vectorize(performance_label)(score_samples) == label, axis=0
        )
    table["N_functional_form"] = np.where(
        state.full_mask,
        "N(alpha,w,theta,transport); source=illustrative",
        "N_reduced(alpha,transport) with extra uncertainty; source=illustrative",
    )
    table["source"] = "illustrative"
    return PerformanceState(table, score_samples, q, calibration)


def structural_loss(scores: np.ndarray) -> float:
    """Smooth illustrative loss used consistently by the decision layer."""
    return float(np.sum(np.maximum(scores - C.PERFORMANCE_THRESHOLDS[0], 0.0) ** 2))
