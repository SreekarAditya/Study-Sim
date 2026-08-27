"""Joint member-stiffness inversion with flat and hierarchical sparse priors."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import least_squares, minimize_scalar

from . import constants as C
from .frame import FrameModel
from .sensing import NormalisedModalData


@dataclass
class StateEstimate:
    mean: np.ndarray
    covariance: np.ndarray
    table: pd.DataFrame
    correlation: np.ndarray
    tradeoffs: pd.DataFrame
    diagnostics: dict


def alpha_covariance_from_frequency_precision(
    frame: FrameModel,
    alpha_at_linearisation: np.ndarray,
    frequency_sd: np.ndarray,
) -> np.ndarray:
    """Weak-prior frequency-only linear covariance for trace reporting."""
    n = len(alpha_at_linearisation)
    base, _ = frame.modal(alpha_at_linearisation)
    sensitivity = np.zeros((len(base), n))
    step = 1.0e-4  # Numerical differentiation step, not an empirical concrete constant.
    for i in range(n):
        trial = alpha_at_linearisation.copy()
        trial[i] = min(1.0, trial[i] + step)
        actual_step = trial[i] - alpha_at_linearisation[i]
        if actual_step <= 0.0:
            trial[i] = alpha_at_linearisation[i] - step
            actual_step = -step
        freq, _ = frame.modal(trial)
        sensitivity[:, i] = (freq - base) / actual_step
    whitened = sensitivity / frequency_sd[:, None]
    information = whitened.T @ whitened + np.eye(n) / C.ALPHA_PRIOR_SD**2
    return np.linalg.pinv(information, rcond=1.0e-10)


def _align_shapes(predicted: np.ndarray, observed: np.ndarray) -> np.ndarray:
    aligned = predicted.copy()
    for mode in range(predicted.shape[1]):
        if np.dot(aligned[:, mode], observed[:, mode]) < 0.0:
            aligned[:, mode] *= -1.0
    return aligned


def _likelihood_residual(
    frame: FrameModel,
    data: NormalisedModalData,
    alpha: np.ndarray,
) -> np.ndarray:
    freq, shapes = frame.modal(alpha)
    shapes_at_sensors = _align_shapes(shapes[data.sensor_dofs, :], data.measured_shapes)
    freq_resid = (freq - data.reference_freq) / data.frequency_sd
    shape_resid = (shapes_at_sensors - data.measured_shapes).ravel(order="F") / C.MODE_SHAPE_NOISE
    return np.concatenate([freq_resid, shape_resid])


def _assemble_state(
    frame: FrameModel,
    mean: np.ndarray,
    covariance: np.ndarray,
    information: np.ndarray,
    diagnostics: dict,
    prior_name: str,
) -> StateEstimate:
    n = len(mean)
    covariance = (covariance + covariance.T) / 2.0
    sd = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    correlation = covariance / np.outer(np.maximum(sd, 1.0e-12), np.maximum(sd, 1.0e-12))
    correlation = np.clip(correlation, -1.0, 1.0)
    max_other_corr = np.array([
        np.max(np.abs(np.delete(correlation[i], i))) for i in range(n)
    ])
    sd_pass = sd <= C.IDENTIFIABLE_SD_LIMIT
    correlation_pass = max_other_corr <= C.IDENTIFIABLE_CORR_LIMIT
    separable = sd_pass & correlation_pass
    table = frame.member_table()
    table["alpha_mean"] = mean
    table["alpha_sd"] = sd
    table["alpha_p05"] = np.clip(
        mean - C.CREDIBLE_INTERVAL_Z_90 * sd,
        C.ALPHA_LOWER_BOUND,
        1.0,
    )
    table["alpha_p95"] = np.clip(
        mean + C.CREDIBLE_INTERVAL_Z_90 * sd,
        C.ALPHA_LOWER_BOUND,
        1.0,
    )
    table["max_abs_parameter_correlation"] = max_other_corr
    table["sd_threshold_pass"] = sd_pass
    table["correlation_threshold_pass"] = correlation_pass
    table["identifiability"] = np.where(separable, "separable", "trades_off")
    table["prior"] = prior_name
    table["source"] = "illustrative"
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if abs(correlation[i, j]) >= C.TRADEOFF_CORR_LIMIT:
                pairs.append({
                    "member_i": f"M{i:02d}",
                    "member_j": f"M{j:02d}",
                    "correlation": correlation[i, j],
                    "prior": prior_name,
                    "source": "illustrative",
                })
    tradeoffs = pd.DataFrame(
        pairs,
        columns=["member_i", "member_j", "correlation", "prior", "source"],
    )
    diagnostics = {
        **diagnostics,
        "prior": prior_name,
        "information_condition_number": float(np.linalg.cond(information)),
        "separability_rule": (
            f"alpha_sd <= {C.IDENTIFIABLE_SD_LIMIT} AND "
            f"max_abs_parameter_correlation <= {C.IDENTIFIABLE_CORR_LIMIT}"
        ),
        "separable_members": int(separable.sum()),
        "tradeoff_members": int((~separable).sum()),
        "source": "illustrative",
    }
    return StateEstimate(mean, covariance, table, correlation, tradeoffs, diagnostics)


def estimate_state_flat(frame: FrameModel, data: NormalisedModalData) -> StateEstimate:
    """Original weak independent Gaussian prior, retained as the comparator."""
    n = len(frame.elements)

    def residual(alpha: np.ndarray) -> np.ndarray:
        likelihood = _likelihood_residual(frame, data, alpha)
        prior = (alpha - C.ALPHA_PRIOR_MEAN) / C.ALPHA_PRIOR_SD
        return np.concatenate([likelihood, prior])

    fit = least_squares(
        residual,
        np.full(n, C.ALPHA_PRIOR_MEAN),
        bounds=(np.full(n, C.ALPHA_LOWER_BOUND), np.ones(n)),
        method="trf",
        max_nfev=C.STATE_ESTIMATION_MAX_EVALUATIONS,
    )
    information = fit.jac.T @ fit.jac
    covariance = np.linalg.pinv(information, rcond=1.0e-10)
    return _assemble_state(
        frame,
        fit.x,
        covariance,
        information,
        {
            "success": bool(fit.success),
            "message": fit.message,
            "cost": float(fit.cost),
            "n_function_evaluations": int(fit.nfev),
        },
        "weak_gaussian_flat",
    )


def estimate_state_sparse(frame: FrameModel, data: NormalisedModalData) -> StateEstimate:
    """Hierarchical Laplace-prior estimate with a joint Laplace covariance.

    Stiffness loss ``d = 1-alpha`` follows a smoothed Laplace density with
    precision ``tau``. A Gamma hyperprior on ``tau`` equivalently induces an
    inverse-Gamma hyperprior on the Laplace scale ``b=1/tau``; its value is
    alternated with the member state, so sparsity is inferred rather than fixed.
    IRLS exploits the normal-exponential scale-mixture representation.
    The returned covariance is the alpha block of the inverse *joint* Hessian in
    ``(alpha, tau)``, retaining parameter correlations and hyperparameter
    uncertainty for downstream mean-variance planning.
    """
    n = len(frame.elements)
    initial = estimate_state_flat(frame, data)
    alpha = initial.mean.copy()
    epsilon = C.SPARSE_PRIOR_SMOOTHING
    shape = C.SPARSE_PRIOR_HYPER_SHAPE
    rate = C.SPARSE_PRIOR_HYPER_RATE
    loss = np.maximum(1.0 - alpha, 0.0)
    tau = (n + shape - 1.0) / (np.sum(np.sqrt(loss**2 + epsilon**2)) + rate)
    total_evaluations = 0
    converged = False
    fit = None
    for iteration in range(C.SPARSE_PRIOR_MAX_ITERATIONS):
        previous_alpha = alpha.copy()
        previous_tau = tau
        weights = tau / np.sqrt(loss**2 + epsilon**2)

        def residual(candidate_alpha: np.ndarray) -> np.ndarray:
            likelihood = _likelihood_residual(frame, data, candidate_alpha)
            candidate_loss = np.maximum(1.0 - candidate_alpha, 0.0)
            return np.concatenate([likelihood, np.sqrt(weights) * candidate_loss])

        fit = least_squares(
            residual,
            alpha,
            bounds=(np.full(n, C.ALPHA_LOWER_BOUND), np.ones(n)),
            method="trf",
            max_nfev=C.STATE_ESTIMATION_MAX_EVALUATIONS,
        )
        total_evaluations += int(fit.nfev)
        alpha = fit.x
        loss = np.maximum(1.0 - alpha, 0.0)
        smooth_l1 = np.sum(np.sqrt(loss**2 + epsilon**2))
        tau = (n + shape - 1.0) / (smooth_l1 + rate)
        alpha_change = np.linalg.norm(alpha - previous_alpha) / max(np.linalg.norm(previous_alpha), 1.0e-12)
        tau_change = abs(tau - previous_tau) / max(abs(previous_tau), 1.0e-12)
        if max(alpha_change, tau_change) <= C.SPARSE_PRIOR_CONVERGENCE_TOL:
            converged = True
            break
    assert fit is not None

    likelihood_count = C.N_MODES + len(data.sensor_dofs) * C.N_MODES
    likelihood_jacobian = fit.jac[:likelihood_count, :]
    # Conditional Gaussian precision from the normal-exponential mixture. This
    # is the covariance-bearing hierarchical representation of the Laplace
    # prior; using the pointwise second derivative of |d| would incorrectly
    # provide essentially no curvature for active losses.
    final_weights = tau / np.sqrt(loss**2 + epsilon**2)
    h_alpha = likelihood_jacobian.T @ likelihood_jacobian + np.diag(final_weights)
    cross = -loss / np.sqrt(loss**2 + epsilon**2)
    h_tau = (n + shape - 1.0) / tau**2
    conditional_covariance = np.linalg.pinv(h_alpha, rcond=1.0e-10)
    alpha_sensitivity_to_tau = -(conditional_covariance @ cross)
    tau_variance = 1.0 / h_tau
    covariance = conditional_covariance + np.outer(
        alpha_sensitivity_to_tau, alpha_sensitivity_to_tau
    ) * tau_variance
    return _assemble_state(
        frame,
        alpha,
        covariance,
        h_alpha,
        {
            "success": bool(fit.success and converged),
            "message": "hierarchical Laplace empirical-Bayes IRLS converged" if converged else "maximum sparse-prior iterations reached",
            "cost": float(0.5 * np.sum(_likelihood_residual(frame, data, alpha) ** 2)),
            "n_function_evaluations": total_evaluations,
            "sparse_iterations": iteration + 1,
            "hyperprior": "tau ~ Gamma(shape, rate), equivalently Laplace scale b=1/tau ~ inverse-Gamma; source=illustrative",
            "inferred_laplace_precision_tau": float(tau),
            "inferred_laplace_scale": float(1.0 / tau),
            "joint_covariance_includes_hyperparameter": True,
            "reference_formulation": "hierarchical Laplace prior following Huang-Beck-Li and Chen-Zhang-Zheng-Sun (2020); implementation constants illustrative",
        },
        "hierarchical_laplace_sparse",
    )


def estimate_state_horseshoe(frame: FrameModel, data: NormalisedModalData) -> StateEstimate:
    """Empirical-Bayes local-global horseshoe alternative for stiffness loss.

    ``d_i=1-alpha_i`` has a zero-centred Gaussian scale mixture with global
    scale ``tau`` and member-specific half-Cauchy local scales. Alternating MAP
    updates use the exact positive stationary root for each squared local scale
    and a bounded scalar update for ``tau`` under its half-Cauchy hyperprior.
    The covariance is the full conditional Laplace covariance at the inferred
    local/global scales; this approximation is labelled explicitly downstream.
    """
    n = len(frame.elements)
    initial = estimate_state_flat(frame, data)
    alpha = initial.mean.copy()
    loss = np.maximum(1.0 - alpha, 0.0)
    smoothing = C.HORSESHOE_LOCAL_SMOOTHING
    tau = float(np.clip(
        np.median(np.maximum(loss, smoothing)),
        C.HORSESHOE_GLOBAL_SCALE_MIN,
        C.HORSESHOE_GLOBAL_SCALE_MAX,
    ))
    hyper_scale = C.HORSESHOE_GLOBAL_HALF_CAUCHY_SCALE
    total_evaluations = 0
    converged = False
    fit = None
    local_squared = np.ones(n)
    for iteration in range(C.HORSESHOE_MAX_ITERATIONS):
        previous_alpha = alpha.copy()
        previous_tau = tau
        scaled_loss = (loss**2 + smoothing**2) / max(tau**2, 1.0e-12)
        local_squared = (
            scaled_loss - 1.0
            + np.sqrt(scaled_loss**2 + 10.0 * scaled_loss + 1.0)
        ) / 6.0
        local_squared = np.maximum(local_squared, 1.0e-10)
        weights = 1.0 / np.maximum(tau**2 * local_squared, 1.0e-12)

        def residual(candidate_alpha: np.ndarray) -> np.ndarray:
            likelihood = _likelihood_residual(frame, data, candidate_alpha)
            candidate_loss = np.maximum(1.0 - candidate_alpha, 0.0)
            return np.concatenate([likelihood, np.sqrt(weights) * candidate_loss])

        fit = least_squares(
            residual,
            alpha,
            bounds=(np.full(n, C.ALPHA_LOWER_BOUND), np.ones(n)),
            method="trf",
            max_nfev=C.STATE_ESTIMATION_MAX_EVALUATIONS,
        )
        total_evaluations += int(fit.nfev)
        alpha = fit.x
        loss = np.maximum(1.0 - alpha, 0.0)

        def global_negative_log(scale: float) -> float:
            variance = scale**2
            gaussian = np.sum(loss**2 / (2.0 * variance * local_squared))
            normalisation = n * np.log(scale)
            half_cauchy = np.log1p((scale / hyper_scale) ** 2)
            return float(gaussian + normalisation + half_cauchy)

        scale_fit = minimize_scalar(
            global_negative_log,
            bounds=(C.HORSESHOE_GLOBAL_SCALE_MIN, C.HORSESHOE_GLOBAL_SCALE_MAX),
            method="bounded",
        )
        tau = float(scale_fit.x)
        alpha_change = np.linalg.norm(alpha - previous_alpha) / max(
            np.linalg.norm(previous_alpha), 1.0e-12
        )
        tau_change = abs(tau - previous_tau) / max(abs(previous_tau), 1.0e-12)
        if max(alpha_change, tau_change) <= C.HORSESHOE_CONVERGENCE_TOL:
            converged = True
            break
    assert fit is not None

    likelihood_count = C.N_MODES + len(data.sensor_dofs) * C.N_MODES
    likelihood_jacobian = fit.jac[:likelihood_count, :]
    scaled_loss = (loss**2 + smoothing**2) / max(tau**2, 1.0e-12)
    local_squared = (
        scaled_loss - 1.0
        + np.sqrt(scaled_loss**2 + 10.0 * scaled_loss + 1.0)
    ) / 6.0
    local_squared = np.maximum(local_squared, 1.0e-10)
    final_weights = 1.0 / np.maximum(tau**2 * local_squared, 1.0e-12)
    information = likelihood_jacobian.T @ likelihood_jacobian + np.diag(final_weights)
    covariance = np.linalg.pinv(information, rcond=1.0e-10)
    overall_success = bool(fit.success and converged)
    if overall_success:
        horseshoe_message = "hierarchical horseshoe empirical-Bayes IRLS converged"
    elif converged:
        horseshoe_message = (
            "outer horseshoe scales stabilised, but the final nonlinear state "
            f"solve was unsuccessful: {fit.message}"
        )
    else:
        horseshoe_message = "maximum horseshoe iterations reached"
    return _assemble_state(
        frame,
        alpha,
        covariance,
        information,
        {
            "success": overall_success,
            "message": horseshoe_message,
            "cost": float(0.5 * np.sum(_likelihood_residual(frame, data, alpha) ** 2)),
            "n_function_evaluations": total_evaluations,
            "horseshoe_iterations": iteration + 1,
            "hyperprior": "local scales and global tau use half-Cauchy priors; source=illustrative",
            "inferred_horseshoe_global_scale": tau,
            "median_inferred_local_scale": float(np.median(np.sqrt(local_squared))),
            "covariance_scope": "full alpha covariance conditional on empirical-Bayes local/global scales",
            "reference_formulation": "hierarchical local-global horseshoe alternative; implementation constants illustrative",
        },
        "hierarchical_horseshoe",
    )


def estimate_state(frame: FrameModel, data: NormalisedModalData) -> StateEstimate:
    """Default state-estimation interface used by the unchanged pipeline."""
    return estimate_state_sparse(frame, data)
