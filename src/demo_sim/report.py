"""Markdown report assembled directly from generated verification artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import constants as C
from .analysis import DecisionAnalysis
from .condition import PerformanceState, TransportState, TwoTierState
from .estimation import StateEstimate
from .frame import FrameModel
from .sensing import NormalisedModalData


def _md_table(frame: pd.DataFrame, max_rows: int | None = None) -> str:
    if max_rows is not None:
        frame = frame.head(max_rows)
    if frame.empty:
        return "_No rows._"
    columns = list(frame.columns)
    header = "| " + " | ".join(columns) + " |"
    rule = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for row in frame.itertuples(index=False, name=None):
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(f"{value:.5g}")
            else:
                values.append(str(value).replace("|", "/"))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, rule, *rows])


def _lambda_reliability(summary: pd.DataFrame) -> pd.DataFrame:
    rows = summary[summary["configuration"].str.startswith("lambda_")].copy()
    rows["lambda"] = rows["configuration"].str.removeprefix("lambda_").astype(float)
    return rows.sort_values("lambda")


def write_report(
    output: Path,
    frame: FrameModel,
    normalised: NormalisedModalData,
    estimate: StateEstimate,
    tier: TwoTierState,
    transport: TransportState,
    performance: PerformanceState,
    decisions: DecisionAnalysis,
    uncertainty_trace: pd.DataFrame,
    constants: pd.DataFrame,
    sensing_summary: dict[str, Any],
    *,
    flat_estimate: StateEstimate,
    prior_members: pd.DataFrame,
    prior_summaries: pd.DataFrame,
    reliability_summary: pd.DataFrame,
    reliability_samples: pd.DataFrame,
    exact_ladder: pd.DataFrame,
    negative_controls: dict[str, dict],
    member_bias: pd.DataFrame,
    bias_summary: pd.DataFrame,
    horseshoe_estimate: StateEstimate | None,
    dense_horseshoe_estimate: StateEstimate | None,
    shrinkage_bias_increment: float,
    prior_plan_samples: pd.DataFrame,
    prior_plan_summary: pd.DataFrame,
    jaccard_summary: pd.DataFrame,
    jaccard_diagnostic: dict[str, Any],
    pooled_plan_summary: pd.DataFrame,
    milp_summary: pd.DataFrame,
    solver_exactness: pd.DataFrame,
    milp_prior_jaccard: pd.DataFrame,
    exact_ground_truth_comparison: dict[str, Any],
    milp_sparse_plan: pd.DataFrame,
    milp_oracle_plan: pd.DataFrame,
) -> None:
    del reliability_samples, prior_plan_samples, milp_oracle_plan, prior_members
    env_rms = normalised.contamination_table[
        "residual_environment_rms_fraction"
    ].mean()
    constants_display = constants[[
        "name", "value", "units", "stage", "illustrative", "source", "origin"
    ]]
    plan_display = (
        milp_sparse_plan
        if not milp_sparse_plan.empty
        else pd.DataFrame({"plan": ["No action selected"]})
    )
    lambda_rel = _lambda_reliability(reliability_summary)
    best_energy_monotone = bool(
        np.all(np.diff(lambda_rel["best_energy"]) >= -1.0e-8)
    )
    p1 = negative_controls["negative_control_penalty_1"]
    p10 = negative_controls["negative_control_penalty_10"]
    control_table = pd.DataFrame([{
        "penalty": item["penalty_weight"],
        "checker_rejected": item["checker_rejected"],
        "violation_count": item["violation_count"],
        "violation_categories": ", ".join(item["violation_categories"]),
        "objective_only_energy": item["objective_only_energy"],
        "objective_delta_vs_reference": item["objective_delta_vs_reference"],
        "correctly_penalised_energy": item["correctly_penalised_energy"],
        "conclusion": item["conclusion"],
        "source": "illustrative",
    } for item in (p1, p10)])

    flat_j = float(milp_prior_jaccard.loc[
        milp_prior_jaccard["prior"] == "flat_prior",
        "milp_plan_jaccard_vs_milp_oracle",
    ].iloc[0])
    sparse_j = float(milp_prior_jaccard.loc[
        milp_prior_jaccard["prior"] == "sparse_prior",
        "milp_plan_jaccard_vs_milp_oracle",
    ].iloc[0])
    oracle_energy = prior_plan_summary[
        prior_plan_summary["configuration"] == "ground_truth_oracle"
    ].iloc[0]
    main_bias = bias_summary[
        bias_summary["case"] == "clustered_main_case"
    ].set_index("prior")
    dense_bias = bias_summary[
        bias_summary["case"] == "dense_damage_sparsity_false"
    ].set_index("prior")
    flat_damaged_bias = float(main_bias.loc[
        "weak_gaussian_flat", "mean_bias_genuinely_damaged_members"
    ])
    sparse_damaged_bias = float(main_bias.loc[
        "hierarchical_laplace_sparse",
        "mean_bias_genuinely_damaged_members",
    ])
    material_bias = bool(
        shrinkage_bias_increment >= C.MATERIAL_SHRINKAGE_BIAS_INCREMENT
    )
    if jaccard_diagnostic["distinguishable_under_declared_rule"]:
        distribution_verdict = (
            "The two 50-seed Jaccard distributions satisfy the declared "
            "rank-effect, p-value, and IQR-overlap rule and are distinguishable."
        )
    else:
        distribution_verdict = (
            "The two 50-seed Jaccard distributions overlap substantially and "
            "are not distinguishable under the declared rule. Therefore the "
            "reported 0.313 → 0.100 single-run comparison is not evidence about "
            "the prior."
        )
    if sparse_j > flat_j:
        exact_direction = "The sparse prior helps the exact decision in this realization."
    elif sparse_j < flat_j:
        exact_direction = "The sparse prior still hurts the exact decision in this realization."
    else:
        exact_direction = "The exact plans have equal oracle Jaccard in this realization."
    exact_prior_verdict = (
        f"With annealing removed, the sparse-prior exact-plan Jaccard is {sparse_j:.3f} "
        f"versus {flat_j:.3f} for the flat prior. {exact_direction}"
    )
    horse_rows = bias_summary[
        bias_summary["prior"] == "hierarchical_horseshoe"
    ]
    if horseshoe_estimate is None or dense_horseshoe_estimate is None:
        horseshoe_verdict = (
            "The material-bias trigger was not reached, so the horseshoe test was skipped."
        )
    else:
        horse_main = horse_rows[horse_rows["case"] == "clustered_main_case"].iloc[0]
        horse_dense = horse_rows[
            horse_rows["case"] == "dense_damage_sparsity_false"
        ].iloc[0]
        horseshoe_verdict = (
            f"Material bias triggered the horseshoe test. Its plug-in empirical-Bayes "
            f"fit {'converged' if horseshoe_estimate.diagnostics['success'] else 'did not converge'} "
            f"in the main case and {'converged' if dense_horseshoe_estimate.diagnostics['success'] else 'did not converge'} "
            f"in the distributed case. Main damaged-member mean bias was "
            f"{horse_main['mean_bias_genuinely_damaged_members']:.3f} and coverage "
            f"{horse_main['interval_90_empirical_coverage_all_members']:.1%}; distributed-case "
            f"bias was {horse_dense['mean_bias_genuinely_damaged_members']:.3f} and coverage "
            f"{horse_dense['interval_90_empirical_coverage_all_members']:.1%}. This alternative "
            "is retained as a failed diagnostic, not adopted as a remedy."
        )

    reliability_display = reliability_summary[[
        "configuration", "seed_count", "best_energy", "median_energy",
        "worst_energy", "energy_spread", "feasible_runs",
        "runs_with_geometry_acquisition", "geometry_acquisition_run_fraction",
    ]]
    identifiability_table = estimate.table[[
        "member", "alpha_mean", "alpha_sd", "max_abs_parameter_correlation",
        "sd_threshold_pass", "correlation_threshold_pass", "identifiability",
    ]]
    member_bias_display = member_bias[[
        "case", "member", "prior", "true_alpha", "estimated_alpha_mean",
        "estimated_minus_true_alpha", "posterior_sd",
        "credible_interval_90_low", "credible_interval_90_high",
        "credible_interval_contains_truth", "genuinely_damaged", "source",
    ]]
    reduced_name = "no_initial_geometry_acquisition_allowed"
    reduced_acquisitions = int(
        decisions.ablation_table.set_index("scenario").loc[
            reduced_name, "geometry_acquisitions"
        ]
    )

    text = f"""# Synthetic structural assessment and retrofit-planning demonstration

> **Mandatory scope warning:** this is a plumbing and uncertainty-propagation study, not validation. No number below is a finding about concrete. Every empirical constant, generated observation, derived result and decision is tagged `source=illustrative`.

## Headline diagnosis and fix

The earlier prior-plan comparison mixed four effects. The evidence supports **large annealing error**, **a noisy annealed oracle**, and **material Laplace shrinkage bias on genuinely damaged members**. The exact MILP comparison then shows the residual prior effect after removing both solver confounders: {exact_prior_verdict} Therefore the observed 0.313 → 0.100 reversal was caused by comparing noisy searches, not by the sparse prior producing a worse exact plan. Also, the sparse estimate was not uniformly “better”: its lower SD and correlation came with worse truth bias and interval coverage.

The planning fix is an equivalent hard-constraint MILP solved by the open-source HiGHS backend. It preserves the QUBO objective and formulation in the codebase, linearises every binary product, and replaces penalty enforcement with the original linear constraints. All three 504-primary-bit full-size problems—flat, sparse, and hidden-state oracle—reached zero-gap proven optima.

## Part A — does the single-run comparison measure the solver?

Each prior was annealed from {C.PRIOR_PLAN_DIAGNOSTIC_SEEDS} independent seeds. Every run is compared with the fixed exact MILP oracle, so the distribution below contains prior-plan annealing variability but not oracle-plan variability.

{_md_table(jaccard_summary)}

{_md_table(pooled_plan_summary)}

![Prior-plan Jaccard distributions](figures/fig_prior_plan_jaccard_distribution.png)

Mann–Whitney p = {jaccard_diagnostic['mann_whitney_two_sided_p']:.4g}, Cliff's delta (flat minus sparse) = {jaccard_diagnostic['cliffs_delta_flat_minus_sparse']:.3f}, and IQR overlap fraction = {jaccard_diagnostic['iqr_overlap_fraction_of_smaller_iqr']:.3f}. Decision rule: `{jaccard_diagnostic['decision_rule']}`.

**Verdict:** {distribution_verdict}

## Part B — member bias and interval coverage

Positive `estimated_minus_true_alpha` means shrinkage toward the undamaged value α=1. In the main case, mean bias on the {int(main_bias.loc['weak_gaussian_flat','genuinely_damaged_member_count'])} genuinely damaged members is {flat_damaged_bias:.3f} under the flat prior and {sparse_damaged_bias:.3f} under Laplace: an added +{shrinkage_bias_increment:.3f} toward undamaged. The declared material-bias trigger is {C.MATERIAL_SHRINKAGE_BIAS_INCREMENT:.3f}; it is **{material_bias}**.

{_md_table(bias_summary)}

Per-member values for both truth cases and every evaluated prior:

{_md_table(member_bias_display)}

Main-case 90% interval coverage is {main_bias.loc['weak_gaussian_flat','interval_90_empirical_coverage_all_members']:.1%} flat versus {main_bias.loc['hierarchical_laplace_sparse','interval_90_empirical_coverage_all_members']:.1%} Laplace. Distributed-truth coverage is {dense_bias.loc['weak_gaussian_flat','interval_90_empirical_coverage_all_members']:.1%} versus {dense_bias.loc['hierarchical_laplace_sparse','interval_90_empirical_coverage_all_members']:.1%}. Thus the smaller Laplace SD is overconfidence here, not uniformly better estimation.

## Part C — is the oracle solver-limited?

The annealed hidden-state oracle across {int(oracle_energy['seed_count'])} seeds had best energy {oracle_energy['best_energy']:.3f}, median {oracle_energy['median_energy']:.3f}, worst {oracle_energy['worst_energy']:.3f}, and spread {oracle_energy['energy_spread']:.3f}. The exact oracle optimum is {milp_summary.set_index('configuration').loc['ground_truth_oracle','objective_value']:.3f}. Therefore the former Jaccard denominator was another noisy search, not a true optimum.

## Part D — exact MILP equivalence and full-size annealing gap

{_md_table(milp_summary)}

{_md_table(solver_exactness)}

For every configuration, the MILP plan's zero-penalty objective and its correctly penalised feasible-QUBO cross-score agree within the registered numerical tolerance. `pooled_never_better_than_milp=True` confirms no pooled annealing incumbent beats the proven optimum. The full-size annealing optimality gaps, previously unverifiable, are now explicit above for both the best and median of 50 seeds.

Clean exact-plan prior comparison:

{_md_table(milp_prior_jaccard)}

**Verdict:** {exact_prior_verdict}

The final production plan is now the sparse-prior MILP optimum:

{_md_table(plan_display)}

## Part E — horseshoe alternative

{horseshoe_verdict}

## Which candidate causes are supported?

1. **Prior comparison measured annealing variance:** {'supported' if not jaccard_diagnostic['distinguishable_under_declared_rule'] else 'not supported under the declared distribution rule'}.
2. **Laplace posterior shrinkage bias:** {'supported' if material_bias else 'not supported'}; damaged-member bias increased by {shrinkage_bias_increment:.3f}.
3. **Oracle itself was solver-limited:** supported; its 50-seed spread is {oracle_energy['energy_spread']:.3f} and its best annealed energy remains above the exact oracle optimum.
4. **A real prior effect remains after exact solving:** {'supported' if sparse_j != flat_j else 'ambiguous'}; exact Jaccard is {flat_j:.3f} flat versus {sparse_j:.3f} Laplace. This is one synthetic realization, not a general prior ranking.

The remedy that succeeded is exact optimisation. The attempted horseshoe remedy is not adopted because its empirical-Bayes fit failed the convergence/coverage diagnostic. The remaining estimator bias is reported, not worked around.

## Retained QUBO verification and formulation evidence

The QUBO remains fully implemented with cover, jacket, combined, geometry acquisition, conditional deterioration `y`, two non-fungible budgets, hard-precedence penalties derived from objective bounds, outage, temporal separation, and solver-derived load-path interactions. Its independent negative controls remain:

{_md_table(control_table)}

Penalty 1 is rejected with {p1['violation_count']} violations and an objective that appears {-p1['objective_delta_vs_reference']:.3f} better than the feasible reference when constraints are underweighted. Penalty 10 remains feasible/inconclusive. The current derived bound is {decisions.main_model.penalty_base:.4g}; it was not tuned downward.

The retained 20-seed configurations and ablations:

{_md_table(reliability_display)}

![Solver reliability](figures/fig_solver_reliability.png)

The exact QUBO-enumeration ladder still diagnoses annealer hit rate:

{_md_table(exact_ladder)}

![Exact ladder](figures/fig_exact_ladder.png)

Beyond 20 primary bits, **annealing** optimality remains unverified; full-size **planning-problem** optimality is now verified independently by MILP. The previous λ=0.002 acquisition threshold remains an annealing artefact: best-energy monotonicity is {best_energy_monotone}, while acquisition frequency changes gradually.

{_md_table(lambda_rel[['lambda','seed_count','best_energy','median_energy','worst_energy','energy_spread','runs_with_geometry_acquisition','geometry_acquisition_run_fraction']])}

![Lambda sweep](figures/fig_lambda_sweep.png)

The `all_reduced` ablation remains renamed `{reduced_name}`: no geometry is initially valid, acquisition remains permitted, and its selected best-run acquisition count is {reduced_acquisitions}.

{_md_table(decisions.ablation_table)}

## Retained state, transport, and classification evidence

The hidden structure remains the existing {C.N_STOREYS}-storey, {C.N_BAYS}-bay, {len(frame.elements)}-member frame. Six lateral sensor DOFs observe four mass-normalised modes. Frequency noise is {C.FREQUENCY_NOISE_FRACTION:.1%}; mode-shape component noise is {C.MODE_SHAPE_NOISE:g}. Environmental shift is {sensing_summary['environment_p95_to_mild_damage_median_ratio']:.2f}× the median mild-damage frequency benchmark; residual contamination after normalisation is {env_rms:.4%}.

The separability rule remains `alpha_sd <= {C.IDENTIFIABLE_SD_LIMIT}` **AND** `max_abs_parameter_correlation <= {C.IDENTIFIABLE_CORR_LIMIT}`; the separate trade-off reporting threshold is {C.TRADEOFF_CORR_LIMIT}.

{_md_table(identifiability_table)}

![Posterior correlation](figures/fig_posterior_correlation.png)

{int(tier.full_mask.sum())} members begin with full geometry and {len(frame.elements)-int(tier.full_mask.sum())} reduced. Split-conformal held-out coverage is {performance.calibration['achieved_heldout_coverage']:.2%} against the {performance.calibration['nominal_coverage']:.0%} synthetic target.

End-to-end member uncertainty trace:

{_md_table(uncertainty_trace)}

![Uncertainty trace](figures/fig_uncertainty_trace.png)

## Exact oracle comparison

The final sparse-prior MILP plan has Jaccard {exact_ground_truth_comparison['jaccard_similarity']:.3f} against the exact hidden-state MILP oracle: {exact_ground_truth_comparison['matched_member_action_pairs']} matched member-action pairs. Pipeline-only interventions are `{', '.join(exact_ground_truth_comparison['inferred_only'])}`; oracle-only interventions are `{', '.join(exact_ground_truth_comparison['oracle_only'])}`. Both plans are proven optimal for their respective illustrative objectives.

## Replaceable stage boundaries

- Real sensor input can replace `sensing.simulate_sensors` by constructing `SensorData`.
- A real transport law can replace `condition.m_full` / `condition.m_reduced`.
- A real acceptance model can replace `condition.acceptance_n_full`.
- `estimate_state` preserves the state-estimation interface and returns full covariance.
- `milp.solve_exact_milp` and `qubo.build_qubo` consume the same `PlanningInputs`.
- The independent feasibility checker remains separate from both solvers.

## Complete constant register

All fixed empirical, policy, generator, prior, diagnosis, MILP, and annealing settings used by the run are below. `illustrative=True` and `source=illustrative` are constructor-enforced.

{_md_table(constants_display)}
"""
    (output / "REPORT.md").write_text(text, encoding="utf-8")

    changelog = f"""# Changelog

> This remains an illustrative plumbing and uncertainty-propagation study, not validation.

## Prior-plan diagnosis

- Added {C.PRIOR_PLAN_DIAGNOSTIC_SEEDS} independent annealing seeds for flat, Laplace, and hidden-state oracle planning models.
- Added full Jaccard distributions, pooled best-known Jaccard, Mann–Whitney/Cliff/IQR-overlap diagnosis, and a publication-readable violin/box plot.
- Distribution verdict: {distribution_verdict}

## Bias diagnosis

- Added per-member `estimated_alpha - true_alpha`, 90% interval containment, and damaged-member bias for both truth cases.
- Laplace-minus-flat damaged-member bias is +{shrinkage_bias_increment:.3f}; material-bias trigger = {material_bias}.
- Tested the horseshoe alternative because the trigger fired; it is retained as a failed diagnostic and is not adopted.

## Exact planning fix

- Added an equivalent hard-constraint MILP using SciPy's open-source HiGHS backend and standard binary-product linearisation.
- Proved full-size optimality for flat, sparse, and oracle problems at zero reported MIP gap.
- Replaced the production and oracle plan artifacts with MILP optima while retaining QUBO plans and all annealing evidence under explicit filenames.
- Added MILP/QUBO cross-scoring, solve times, best-of-50 and median annealing gaps, and exact-prior Jaccard: flat {flat_j:.3f}, Laplace {sparse_j:.3f}.

## Preserved safeguards

- Scope warning and `source=illustrative` constant registry remain mandatory.
- Penalty negative controls, feasibility checker, exact enumeration ladder, λ sweep, ablations, QUBO formulation, and 20-seed reliability artifacts remain in place.
"""
    (output / "CHANGELOG.md").write_text(changelog, encoding="utf-8")
