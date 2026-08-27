"""Side-by-side diagnostics for flat and hierarchical sparse state priors."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import constants as C
from .estimation import StateEstimate
from .ground_truth import GroundTruth


def intervention_set(plan: pd.DataFrame) -> set[tuple[str, str]]:
    if plan.empty:
        return set()
    interventions = plan[plan["action"] != "acquire_geometry"]
    return set(
        interventions[["member", "action"]].itertuples(index=False, name=None)
    )


def plan_jaccard(plan: pd.DataFrame, oracle_plan: pd.DataFrame) -> float:
    selected = intervention_set(plan)
    oracle = intervention_set(oracle_plan)
    return len(selected & oracle) / max(len(selected | oracle), 1)


def member_comparison(
    truth: GroundTruth,
    flat: StateEstimate,
    sparse: StateEstimate,
    case: str,
) -> pd.DataFrame:
    table = flat.table[["member", "kind", "storey", "bay"]].copy()
    table["case"] = case
    table["true_alpha"] = truth.alpha
    table["flat_alpha_mean"] = flat.mean
    table["flat_alpha_sd"] = np.sqrt(np.diag(flat.covariance))
    table["flat_max_abs_correlation"] = flat.table["max_abs_parameter_correlation"].to_numpy()
    table["flat_identifiability"] = flat.table["identifiability"].to_numpy()
    table["sparse_alpha_mean"] = sparse.mean
    table["sparse_alpha_sd"] = np.sqrt(np.diag(sparse.covariance))
    table["sparse_max_abs_correlation"] = sparse.table["max_abs_parameter_correlation"].to_numpy()
    table["sparse_identifiability"] = sparse.table["identifiability"].to_numpy()
    table["sparse_minus_flat_abs_error"] = (
        np.abs(sparse.mean - truth.alpha) - np.abs(flat.mean - truth.alpha)
    )
    table["source"] = "illustrative"
    return table


def prior_summary(
    truth: GroundTruth,
    flat: StateEstimate,
    sparse: StateEstimate,
    flat_plan: pd.DataFrame,
    sparse_plan: pd.DataFrame,
    oracle_plan: pd.DataFrame,
    case: str,
) -> pd.DataFrame:
    rows = []
    for name, estimate, plan in [
        ("weak_gaussian_flat", flat, flat_plan),
        ("hierarchical_laplace_sparse", sparse, sparse_plan),
    ]:
        sd = np.sqrt(np.diag(estimate.covariance))
        lower = np.clip(
            estimate.mean - C.CREDIBLE_INTERVAL_Z_90 * sd,
            C.ALPHA_LOWER_BOUND,
            1.0,
        )
        upper = np.clip(
            estimate.mean + C.CREDIBLE_INTERVAL_Z_90 * sd,
            C.ALPHA_LOWER_BOUND,
            1.0,
        )
        rows.append({
            "case": case,
            "prior": name,
            "alpha_rmse": float(np.sqrt(np.mean((estimate.mean - truth.alpha) ** 2))),
            "alpha_mae": float(np.mean(np.abs(estimate.mean - truth.alpha))),
            "alpha_mean_bias": float(np.mean(estimate.mean - truth.alpha)),
            "mean_posterior_sd": float(np.mean(sd)),
            "median_posterior_sd": float(np.median(sd)),
            "max_abs_parameter_correlation": float(
                estimate.table["max_abs_parameter_correlation"].max()
            ),
            "mean_max_abs_parameter_correlation": float(
                estimate.table["max_abs_parameter_correlation"].mean()
            ),
            "separable_members": int(
                (estimate.table["identifiability"] == "separable").sum()
            ),
            "interval_90_empirical_coverage": float(
                np.mean((truth.alpha >= lower) & (truth.alpha <= upper))
            ),
            "plan_jaccard_vs_ground_truth_oracle": plan_jaccard(plan, oracle_plan),
            "selected_intervention_pairs": len(intervention_set(plan)),
            "inferred_laplace_scale": estimate.diagnostics.get(
                "inferred_laplace_scale", np.nan
            ),
            "source": "illustrative",
        })
    return pd.DataFrame(rows)


def member_bias_table(
    truth: GroundTruth,
    estimates: dict[str, StateEstimate],
    case: str,
) -> pd.DataFrame:
    """Long member-level bias and interval-coverage table for every prior."""
    rows: list[dict] = []
    for prior, estimate in estimates.items():
        sd = np.sqrt(np.diag(estimate.covariance))
        lower = np.clip(
            estimate.mean - C.CREDIBLE_INTERVAL_Z_90 * sd,
            C.ALPHA_LOWER_BOUND,
            1.0,
        )
        upper = np.clip(
            estimate.mean + C.CREDIBLE_INTERVAL_Z_90 * sd,
            C.ALPHA_LOWER_BOUND,
            1.0,
        )
        for member, true_alpha in enumerate(truth.alpha):
            rows.append({
                "case": case,
                "member": f"M{member:02d}",
                "prior": prior,
                "true_alpha": true_alpha,
                "estimated_alpha_mean": estimate.mean[member],
                "estimated_minus_true_alpha": estimate.mean[member] - true_alpha,
                "posterior_sd": sd[member],
                "credible_interval_90_low": lower[member],
                "credible_interval_90_high": upper[member],
                "credible_interval_contains_truth": bool(
                    lower[member] <= true_alpha <= upper[member]
                ),
                "genuinely_damaged": bool(
                    true_alpha < C.GENUINELY_DAMAGED_ALPHA_LIMIT
                ),
                "source": "illustrative",
            })
    return pd.DataFrame(rows)


def bias_coverage_summary(member_bias: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (case, prior), group in member_bias.groupby(["case", "prior"], sort=False):
        damaged = group[group["genuinely_damaged"]]
        rows.append({
            "case": case,
            "prior": prior,
            "member_count": len(group),
            "mean_bias_all_members": group["estimated_minus_true_alpha"].mean(),
            "median_bias_all_members": group["estimated_minus_true_alpha"].median(),
            "interval_90_empirical_coverage_all_members": group[
                "credible_interval_contains_truth"
            ].mean(),
            "genuinely_damaged_member_count": len(damaged),
            "mean_bias_genuinely_damaged_members": damaged[
                "estimated_minus_true_alpha"
            ].mean(),
            "median_bias_genuinely_damaged_members": damaged[
                "estimated_minus_true_alpha"
            ].median(),
            "fraction_damaged_members_biased_toward_undamaged": (
                damaged["estimated_minus_true_alpha"] > 0.0
            ).mean(),
            "source": "illustrative",
        })
    return pd.DataFrame(rows)
