"""End-to-end orchestration, verification studies, and result emission."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import __version__, constants as C
from .analysis import decision_analysis, make_oracle_state, uncertainty_trace
from .condition import classify_performance, compute_transport, force_all_reduced, make_two_tier_state
from .estimation import (
    estimate_state,
    estimate_state_flat,
    estimate_state_horseshoe,
    estimate_state_sparse,
)
from .frame import FrameModel
from .ground_truth import generate_dense_ground_truth, generate_ground_truth
from .milp import MILPResult, solve_exact_milp
from .planning import build_planning_inputs
from .plotting import generate_figures
from .prior_comparison import (
    bias_coverage_summary,
    member_bias_table,
    member_comparison,
    plan_jaccard,
    prior_summary,
)
from .qubo import (
    AnnealResult,
    QUBOModel,
    build_qubo,
    exact_reduced_check,
    primary_array_from_vector,
    simulated_annealing,
)
from .report import write_report
from .sensing import normalise_environment, simulate_sensors
from .verification import (
    exact_verification_ladder,
    jaccard_distribution_summary,
    penalty_negative_controls,
    run_seed_ensemble,
)


def _json_default(value: Any):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.bool_,)):
        return bool(value)
    raise TypeError(f"Not JSON serialisable: {type(value)}")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, default=_json_default), encoding="utf-8")


def _alias_reliability_configuration(reliability, source: str, alias: str) -> None:
    samples = reliability.samples[reliability.samples["configuration"] == source].copy()
    samples["configuration"] = alias
    summary = reliability.summary[reliability.summary["configuration"] == source].copy()
    summary["configuration"] = alias
    reliability.samples = pd.concat([reliability.samples, samples], ignore_index=True)
    reliability.summary = pd.concat([reliability.summary, summary], ignore_index=True)
    reliability.best_results[alias] = reliability.best_results[source]


def _pool_and_polish(
    model: QUBOModel,
    candidates: list[AnnealResult],
    seed: int,
) -> AnnealResult:
    best = min(candidates, key=lambda item: item.energy)
    return simulated_annealing(
        model,
        np.random.default_rng(seed),
        initial_primary=primary_array_from_vector(model, best.vector).copy(),
    )


def _milp_record(name: str, result: MILPResult) -> dict[str, Any]:
    return {
        "configuration": name,
        "objective_value": result.objective_value,
        "reference_qubo_energy": result.reference_qubo_energy,
        "qubo_cross_score_absolute_difference": abs(
            result.reference_qubo_energy - result.objective_value
        ),
        "solve_time_seconds": result.solve_time_seconds,
        "proven_optimal": result.proven_optimal,
        "mip_gap": result.mip_gap,
        "mip_dual_bound": result.mip_dual_bound,
        "mip_node_count": result.mip_node_count,
        "primary_binary_count": result.model.primary.size,
        "binary_variable_count_including_y": result.binary_variable_count,
        "product_variable_count": result.product_variable_count,
        "linear_constraint_count": result.linear_constraint_count,
        "independent_violation_count": len(result.violations),
        "solver": "SciPy milp with open-source HiGHS backend",
        "source": "illustrative",
    }


def _exact_plan_comparison(plan: pd.DataFrame, oracle_plan: pd.DataFrame) -> dict[str, Any]:
    def selected(frame: pd.DataFrame) -> set[tuple[str, str]]:
        intervention = frame[frame["action"] != "acquire_geometry"]
        return set(
            intervention[["member", "action"]].itertuples(index=False, name=None)
        )

    inferred = selected(plan)
    oracle = selected(oracle_plan)
    return {
        "definition": "exact MILP plans under inferred state and hidden true state; same catalogue, horizon and hard constraints",
        "matched_member_action_pairs": len(inferred & oracle),
        "inferred_member_action_pairs": len(inferred),
        "oracle_member_action_pairs": len(oracle),
        "jaccard_similarity": len(inferred & oracle) / max(len(inferred | oracle), 1),
        "inferred_only": sorted(f"{member}:{action}" for member, action in inferred - oracle),
        "oracle_only": sorted(f"{member}:{action}" for member, action in oracle - inferred),
        "both_plans_proven_optimal": True,
        "source": "illustrative",
    }


def _sensing_summary(frame: FrameModel, sensor) -> dict:
    mild_shifts = []
    for member in range(len(frame.elements)):
        mild_alpha = np.ones(len(frame.elements))
        mild_alpha[member] = C.ALPHA_PRIOR_MEAN
        mild_frequency, _ = frame.modal(mild_alpha)
        mild_shifts.extend(np.abs(mild_frequency / sensor.pristine_freq - 1.0).tolist())
    environment_abs = np.abs(sensor.observations["true_environment_shift_fraction"].to_numpy())
    return {
        "sensor_count": len(sensor.sensor_dofs),
        "member_count": len(frame.elements),
        "environment_abs_shift_p95_fraction": float(np.quantile(environment_abs, 0.95)),
        "environment_abs_shift_max_fraction": float(environment_abs.max()),
        "single_member_mild_damage_alpha": C.ALPHA_PRIOR_MEAN,
        "single_member_mild_damage_median_abs_frequency_shift": float(np.median(mild_shifts)),
        "environment_p95_to_mild_damage_median_ratio": float(
            np.quantile(environment_abs, 0.95) / max(np.median(mild_shifts), 1.0e-12)
        ),
        "source": "illustrative",
    }


def run_pipeline(output_dir: str | Path = "results") -> Path:
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "plans").mkdir(exist_ok=True)
    # Remove only the two generated legacy artifacts whose old name had
    # ambiguous semantics. All other existing user/output files are preserved.
    for legacy in (
        output / "all_reduced_transport_condition.csv",
        output / "plans" / "all_reduced.csv",
    ):
        legacy.unlink(missing_ok=True)

    # Main synthetic case: upstream sensing is shared exactly by both priors.
    rng = np.random.default_rng(C.SEED)
    frame = FrameModel.regular_frame()
    truth = generate_ground_truth(frame, rng)
    sensor = simulate_sensors(frame, truth, rng)
    sensing_summary = _sensing_summary(frame, sensor)
    normalised = normalise_environment(sensor)
    sparse_estimate = estimate_state(frame, normalised)
    flat_estimate = estimate_state_flat(frame, normalised)

    sparse_tier = make_two_tier_state(
        frame, sparse_estimate, truth, np.random.default_rng(C.SEED + 60)
    )
    flat_tier = make_two_tier_state(
        frame, flat_estimate, truth, np.random.default_rng(C.SEED + 60)
    )
    sparse_transport = compute_transport(
        sparse_tier, truth, np.random.default_rng(C.SEED + 61)
    )
    flat_transport = compute_transport(
        flat_tier, truth, np.random.default_rng(C.SEED + 61)
    )
    performance = classify_performance(
        frame, sparse_tier, sparse_transport, np.random.default_rng(C.SEED + 62)
    )
    sparse_inputs = build_planning_inputs(
        frame, sparse_tier, sparse_transport, np.random.default_rng(C.SEED + 63)
    )
    flat_inputs = build_planning_inputs(
        frame, flat_tier, flat_transport, np.random.default_rng(C.SEED + 63)
    )

    # Explicit ablation semantics: no member starts with valid geometry, but the
    # acquire-geometry action remains available and is part of what is measured.
    no_initial_geometry_tier = force_all_reduced(sparse_tier)
    no_initial_geometry_transport = compute_transport(
        no_initial_geometry_tier, truth, np.random.default_rng(C.SEED + 80)
    )
    no_initial_geometry_inputs = build_planning_inputs(
        frame,
        no_initial_geometry_tier,
        no_initial_geometry_transport,
        np.random.default_rng(C.SEED + 81),
    )

    # Second ground-truth case deliberately violates the sparse-damage premise.
    dense_rng = np.random.default_rng(C.SEED + 600)
    dense_truth = generate_dense_ground_truth(frame, dense_rng)
    dense_sensor = simulate_sensors(frame, dense_truth, dense_rng)
    dense_normalised = normalise_environment(dense_sensor)
    dense_sparse_estimate = estimate_state_sparse(frame, dense_normalised)
    dense_flat_estimate = estimate_state_flat(frame, dense_normalised)
    preliminary_bias = pd.concat([
        member_bias_table(
            truth,
            {
                "weak_gaussian_flat": flat_estimate,
                "hierarchical_laplace_sparse": sparse_estimate,
            },
            "clustered_main_case",
        ),
        member_bias_table(
            dense_truth,
            {
                "weak_gaussian_flat": dense_flat_estimate,
                "hierarchical_laplace_sparse": dense_sparse_estimate,
            },
            "dense_damage_sparsity_false",
        ),
    ], ignore_index=True)
    preliminary_bias_summary = bias_coverage_summary(preliminary_bias)
    main_bias = preliminary_bias_summary[
        preliminary_bias_summary["case"] == "clustered_main_case"
    ].set_index("prior")
    shrinkage_bias_increment = float(
        main_bias.loc[
            "hierarchical_laplace_sparse",
            "mean_bias_genuinely_damaged_members",
        ]
        - main_bias.loc[
            "weak_gaussian_flat",
            "mean_bias_genuinely_damaged_members",
        ]
    )
    horseshoe_triggered = bool(
        shrinkage_bias_increment >= C.MATERIAL_SHRINKAGE_BIAS_INCREMENT
    )
    horseshoe_estimate = (
        estimate_state_horseshoe(frame, normalised) if horseshoe_triggered else None
    )
    dense_horseshoe_estimate = (
        estimate_state_horseshoe(frame, dense_normalised)
        if horseshoe_triggered else None
    )
    main_estimates = {
        "weak_gaussian_flat": flat_estimate,
        "hierarchical_laplace_sparse": sparse_estimate,
    }
    dense_estimates = {
        "weak_gaussian_flat": dense_flat_estimate,
        "hierarchical_laplace_sparse": dense_sparse_estimate,
    }
    if horseshoe_estimate is not None and dense_horseshoe_estimate is not None:
        main_estimates["hierarchical_horseshoe"] = horseshoe_estimate
        dense_estimates["hierarchical_horseshoe"] = dense_horseshoe_estimate
    member_bias = pd.concat([
        member_bias_table(truth, main_estimates, "clustered_main_case"),
        member_bias_table(
            dense_truth,
            dense_estimates,
            "dense_damage_sparsity_false",
        ),
    ], ignore_index=True)
    bias_summary = bias_coverage_summary(member_bias)
    dense_sparse_tier = make_two_tier_state(
        frame,
        dense_sparse_estimate,
        dense_truth,
        np.random.default_rng(C.SEED + 660),
    )
    dense_flat_tier = make_two_tier_state(
        frame,
        dense_flat_estimate,
        dense_truth,
        np.random.default_rng(C.SEED + 660),
    )
    dense_sparse_transport = compute_transport(
        dense_sparse_tier, dense_truth, np.random.default_rng(C.SEED + 661)
    )
    dense_flat_transport = compute_transport(
        dense_flat_tier, dense_truth, np.random.default_rng(C.SEED + 661)
    )
    dense_sparse_inputs = build_planning_inputs(
        frame,
        dense_sparse_tier,
        dense_sparse_transport,
        np.random.default_rng(C.SEED + 663),
    )
    dense_flat_inputs = build_planning_inputs(
        frame,
        dense_flat_tier,
        dense_flat_transport,
        np.random.default_rng(C.SEED + 663),
    )

    # Full-size 20-seed reliability protocol. The reference lambda is an alias
    # of main, so it has the same 20 trajectories rather than a contradictory
    # second single-run answer for an identical QUBO.
    reliability_models = {"main": build_qubo(sparse_inputs, C.RISK_LAMBDA_REFERENCE)}
    for lam in C.LAMBDA_SWEEP:
        if not np.isclose(lam, C.RISK_LAMBDA_REFERENCE):
            reliability_models[f"lambda_{lam:g}"] = build_qubo(sparse_inputs, lam)
    reliability_models.update({
        "point_estimate": build_qubo(
            sparse_inputs.point_estimate(), C.RISK_LAMBDA_REFERENCE
        ),
        "no_initial_geometry_acquisition_allowed": build_qubo(
            no_initial_geometry_inputs, C.RISK_LAMBDA_REFERENCE
        ),
        "no_interactions": build_qubo(
            sparse_inputs.without_interactions(), C.RISK_LAMBDA_REFERENCE
        ),
        "flat_prior_main": build_qubo(flat_inputs, C.RISK_LAMBDA_REFERENCE),
        "dense_sparse_prior_main": build_qubo(
            dense_sparse_inputs, C.RISK_LAMBDA_REFERENCE
        ),
        "dense_flat_prior_main": build_qubo(
            dense_flat_inputs, C.RISK_LAMBDA_REFERENCE
        ),
    })
    reliability = run_seed_ensemble(reliability_models)
    _alias_reliability_configuration(
        reliability, "main", f"lambda_{C.RISK_LAMBDA_REFERENCE:g}"
    )

    # Exact checks and production decision analysis use best-of-20 reliable runs.
    exact_reduced = exact_reduced_check(
        sparse_inputs, np.random.default_rng(C.SEED + 90)
    )
    exact_ladder, exact_ladder_samples = exact_verification_ladder(sparse_inputs)
    decisions = decision_analysis(
        frame,
        sparse_inputs,
        sparse_tier,
        performance,
        truth,
        exact_reduced,
        np.random.default_rng(C.SEED + 100),
        no_initial_geometry_inputs,
        reliable_results=reliability.best_results,
    )

    # Checker negative controls are raw full-bit anneals; production annealing
    # remains structured and feasible. Penalty 10 is allowed to be inconclusive.
    negative_controls = penalty_negative_controls(
        sparse_inputs, decisions.main_result
    )
    decisions.feasibility.update(negative_controls)

    # Exact full-size hard-constraint reference for both priors and the hidden
    # state oracle. These solve the same zero-penalty planning objective, with
    # every binary product linearised and every original constraint explicit.
    oracle_state = make_oracle_state(frame, truth)
    oracle_transport = compute_transport(
        oracle_state,
        truth,
        np.random.default_rng(C.SEED + 401),
    )
    oracle_inputs = build_planning_inputs(
        frame,
        oracle_state,
        oracle_transport,
        np.random.default_rng(C.SEED + 402),
    )
    oracle_inputs.action_variance[:] = 0.0
    milp_results = {
        "flat_prior": solve_exact_milp(flat_inputs, C.RISK_LAMBDA_REFERENCE),
        "sparse_prior": solve_exact_milp(sparse_inputs, C.RISK_LAMBDA_REFERENCE),
        "ground_truth_oracle": solve_exact_milp(oracle_inputs, 0.0),
    }
    milp_summary = pd.DataFrame([
        _milp_record(name, result) for name, result in milp_results.items()
    ])

    # Diagnose whether the earlier prior comparison was a draw from the QUBO
    # solver distribution. Every Jaccard value uses the fixed exact MILP oracle.
    prior_plan_models = {
        "flat_prior": build_qubo(flat_inputs, C.RISK_LAMBDA_REFERENCE),
        "sparse_prior": build_qubo(sparse_inputs, C.RISK_LAMBDA_REFERENCE),
        "ground_truth_oracle": build_qubo(oracle_inputs, 0.0),
    }
    prior_plan_reliability = run_seed_ensemble(
        prior_plan_models,
        seed_count=C.PRIOR_PLAN_DIAGNOSTIC_SEEDS,
        jaccard_references={
            "milp_oracle": milp_results["ground_truth_oracle"].plan,
        },
        seed_base=C.PRIOR_PLAN_SEED_BASE,
    )
    jaccard_summary, jaccard_diagnostic = jaccard_distribution_summary(
        prior_plan_reliability.samples,
        "jaccard_vs_milp_oracle",
    )
    pooled_qubo_results = {
        "flat_prior": _pool_and_polish(
            prior_plan_models["flat_prior"],
            [
                prior_plan_reliability.best_results["flat_prior"],
                reliability.best_results["flat_prior_main"],
            ],
            C.PRIOR_PLAN_SEED_BASE + 10000,
        ),
        "sparse_prior": _pool_and_polish(
            prior_plan_models["sparse_prior"],
            [
                prior_plan_reliability.best_results["sparse_prior"],
                decisions.main_result,
            ],
            C.PRIOR_PLAN_SEED_BASE + 10001,
        ),
        "ground_truth_oracle": _pool_and_polish(
            prior_plan_models["ground_truth_oracle"],
            [
                prior_plan_reliability.best_results["ground_truth_oracle"],
            ],
            C.PRIOR_PLAN_SEED_BASE + 10002,
        ),
    }
    pooled_plan_rows = []
    exactness_rows = []
    for name in ("flat_prior", "sparse_prior", "ground_truth_oracle"):
        group = prior_plan_reliability.samples[
            prior_plan_reliability.samples["configuration"] == name
        ]
        optimum = milp_results[name].objective_value
        best_50 = float(group["energy"].min())
        median_50 = float(group["energy"].median())
        pooled_energy = pooled_qubo_results[name].energy
        pooled_plan_rows.append({
            "configuration": name,
            "pooled_qubo_energy": pooled_energy,
            "jaccard_vs_milp_oracle": plan_jaccard(
                pooled_qubo_results[name].plan,
                milp_results["ground_truth_oracle"].plan,
            ),
            "selected_intervention_pairs": len(
                set(
                    pooled_qubo_results[name].plan.loc[
                        pooled_qubo_results[name].plan["action"] != "acquire_geometry",
                        ["member", "action"],
                    ].itertuples(index=False, name=None)
                )
            ),
            "source": "illustrative",
        })
        exactness_rows.append({
            "configuration": name,
            "milp_optimum_energy": optimum,
            "milp_proven_optimal": milp_results[name].proven_optimal,
            "qubo_best_of_50_energy": best_50,
            "qubo_median_of_50_energy": median_50,
            "qubo_pooled_best_known_energy": pooled_energy,
            "best_of_50_absolute_optimality_gap": best_50 - optimum,
            "best_of_50_relative_optimality_gap_percent": (
                100.0 * (best_50 - optimum) / max(abs(optimum), 1.0e-12)
            ),
            "median_absolute_optimality_gap": median_50 - optimum,
            "median_relative_optimality_gap_percent": (
                100.0 * (median_50 - optimum) / max(abs(optimum), 1.0e-12)
            ),
            "pooled_absolute_optimality_gap": pooled_energy - optimum,
            "pooled_never_better_than_milp": bool(
                pooled_energy >= optimum - C.MILP_EQUIVALENCE_ABS_TOL
            ),
            "source": "illustrative",
        })
    pooled_plan_summary = pd.DataFrame(pooled_plan_rows)
    solver_exactness = pd.DataFrame(exactness_rows)
    if not solver_exactness["pooled_never_better_than_milp"].all():
        raise RuntimeError(
            "A pooled QUBO incumbent scored below the proven MILP optimum; "
            "MILP/QUBO equivalence is broken"
        )
    milp_prior_jaccard = pd.DataFrame([{
        "prior": name,
        "milp_plan_jaccard_vs_milp_oracle": plan_jaccard(
            milp_results[name].plan,
            milp_results["ground_truth_oracle"].plan,
        ),
        "milp_prior_objective": milp_results[name].objective_value,
        "milp_oracle_objective": milp_results["ground_truth_oracle"].objective_value,
        "source": "illustrative",
    } for name in ("flat_prior", "sparse_prior")])
    exact_ground_truth_comparison = _exact_plan_comparison(
        milp_results["sparse_prior"].plan,
        milp_results["ground_truth_oracle"].plan,
    )

    # Dense-case oracle and prior-plan comparisons.
    dense_oracle_state = make_oracle_state(frame, dense_truth)
    dense_oracle_transport = compute_transport(
        dense_oracle_state, dense_truth, np.random.default_rng(C.SEED + 701)
    )
    dense_oracle_inputs = build_planning_inputs(
        frame, dense_oracle_state, dense_oracle_transport, np.random.default_rng(C.SEED + 702)
    )
    dense_oracle_inputs.action_variance[:] = 0.0
    dense_oracle_model = build_qubo(dense_oracle_inputs, 0.0)
    dense_oracle_result = simulated_annealing(
        dense_oracle_model, np.random.default_rng(C.SEED + 703)
    )
    flat_plan = milp_results["flat_prior"].plan
    sparse_plan = milp_results["sparse_prior"].plan
    exact_oracle_plan = milp_results["ground_truth_oracle"].plan
    dense_sparse_plan = reliability.best_results["dense_sparse_prior_main"].plan
    dense_flat_plan = reliability.best_results["dense_flat_prior_main"].plan
    prior_members = pd.concat([
        member_comparison(truth, flat_estimate, sparse_estimate, "clustered_main_case"),
        member_comparison(
            dense_truth,
            dense_flat_estimate,
            dense_sparse_estimate,
            "dense_damage_sparsity_false",
        ),
    ], ignore_index=True)
    prior_summaries = pd.concat([
        prior_summary(
            truth,
            flat_estimate,
            sparse_estimate,
            flat_plan,
            sparse_plan,
            exact_oracle_plan,
            "clustered_main_case",
        ),
        prior_summary(
            dense_truth,
            dense_flat_estimate,
            dense_sparse_estimate,
            dense_flat_plan,
            dense_sparse_plan,
            dense_oracle_result.plan,
            "dense_damage_sparsity_false",
        ),
    ], ignore_index=True)
    traced_member, trace = uncertainty_trace(
        frame,
        sensor,
        normalised,
        sparse_estimate,
        sparse_tier,
        sparse_transport,
        performance,
    )

    # Emit artifacts.
    constants = C.constants_frame()
    constants.to_csv(output / "constants.csv", index=False)
    truth.table.to_csv(output / "ground_truth.csv", index=False)
    dense_truth.table.to_csv(output / "ground_truth_dense_damage.csv", index=False)
    sensor.observations.to_csv(output / "sensor_observations.csv", index=False)
    dense_sensor.observations.to_csv(output / "sensor_observations_dense_damage.csv", index=False)
    _write_json(output / "sensing_summary.json", sensing_summary)
    pd.DataFrame(
        sensor.measured_shapes,
        index=[f"sensor_{d}" for d in sensor.sensor_dofs],
        columns=[f"mode_{i+1}" for i in range(C.N_MODES)],
    ).to_csv(output / "measured_mode_shapes.csv")
    normalised.regression_table.to_csv(output / "environment_normalisation.csv", index=False)
    normalised.contamination_table.to_csv(output / "environment_residual_contamination.csv", index=False)

    labels = [f"M{i:02d}" for i in range(len(frame.elements))]
    sparse_estimate.table.to_csv(output / "posterior_alpha.csv", index=False)
    flat_estimate.table.to_csv(output / "posterior_alpha_flat_prior.csv", index=False)
    dense_sparse_estimate.table.to_csv(output / "posterior_alpha_dense_sparse_prior.csv", index=False)
    dense_flat_estimate.table.to_csv(output / "posterior_alpha_dense_flat_prior.csv", index=False)
    if horseshoe_estimate is not None and dense_horseshoe_estimate is not None:
        horseshoe_estimate.table.to_csv(
            output / "posterior_alpha_horseshoe_prior.csv", index=False
        )
        dense_horseshoe_estimate.table.to_csv(
            output / "posterior_alpha_dense_horseshoe_prior.csv", index=False
        )
        pd.DataFrame(
            horseshoe_estimate.covariance, index=labels, columns=labels
        ).to_csv(output / "posterior_covariance_horseshoe_prior.csv")
        _write_json(
            output / "state_estimation_diagnostics_horseshoe_prior.json",
            horseshoe_estimate.diagnostics,
        )
        _write_json(
            output / "state_estimation_diagnostics_dense_horseshoe_prior.json",
            dense_horseshoe_estimate.diagnostics,
        )
    pd.DataFrame(sparse_estimate.covariance, index=labels, columns=labels).to_csv(output / "posterior_covariance.csv")
    pd.DataFrame(flat_estimate.covariance, index=labels, columns=labels).to_csv(output / "posterior_covariance_flat_prior.csv")
    pd.DataFrame(sparse_estimate.correlation, index=labels, columns=labels).to_csv(output / "posterior_correlation.csv")
    pd.DataFrame(flat_estimate.correlation, index=labels, columns=labels).to_csv(output / "posterior_correlation_flat_prior.csv")
    sparse_estimate.tradeoffs.to_csv(output / "parameter_tradeoffs.csv", index=False)
    flat_estimate.tradeoffs.to_csv(output / "parameter_tradeoffs_flat_prior.csv", index=False)
    _write_json(output / "state_estimation_diagnostics.json", sparse_estimate.diagnostics)
    _write_json(output / "state_estimation_diagnostics_flat_prior.json", flat_estimate.diagnostics)
    prior_members.to_csv(output / "prior_comparison_by_member.csv", index=False)
    prior_summaries.to_csv(output / "prior_comparison_summary.csv", index=False)
    member_bias.to_csv(output / "prior_bias_coverage_by_member.csv", index=False)
    bias_summary.to_csv(output / "prior_bias_coverage_summary.csv", index=False)
    _write_json(output / "horseshoe_trigger.json", {
        "triggered": horseshoe_triggered,
        "laplace_minus_flat_damaged_member_mean_bias": shrinkage_bias_increment,
        "material_bias_threshold": C.MATERIAL_SHRINKAGE_BIAS_INCREMENT,
        "reason": (
            "material damaged-member shrinkage bias detected; horseshoe evaluated"
            if horseshoe_triggered
            else "material shrinkage-bias trigger not reached; horseshoe skipped"
        ),
        "source": "illustrative",
    })

    sparse_tier.table.to_csv(output / "two_tier_state.csv", index=False)
    sparse_transport.table.to_csv(output / "transport_condition.csv", index=False)
    no_initial_geometry_transport.table.to_csv(
        output / "no_initial_geometry_transport_condition.csv", index=False
    )
    performance.table.to_csv(output / "performance_classification.csv", index=False)
    _write_json(output / "conformal_calibration.json", performance.calibration)
    sparse_inputs.interactions.to_csv(output / "solver_derived_interactions.csv", index=False)
    pd.DataFrame(
        sparse_inputs.action_mean,
        columns=[f"{a}_mean_benefit" for a in ("cover", "jacket", "combined")],
    ).assign(member=labels, source="illustrative").to_csv(output / "action_benefit_means.csv", index=False)
    pd.DataFrame(
        sparse_inputs.action_variance,
        columns=[f"{a}_benefit_variance" for a in ("cover", "jacket", "combined")],
    ).assign(member=labels, source="illustrative").to_csv(output / "action_benefit_variances.csv", index=False)
    pd.DataFrame({
        "member": labels,
        "one_period_solver_derived_deterioration_cost": sparse_inputs.deterioration_cost,
        "source": "illustrative",
    }).to_csv(output / "deterioration_costs.csv", index=False)

    sparse_plan.to_csv(output / "retrofit_plan.csv", index=False)
    decisions.main_result.plan.to_csv(
        output / "retrofit_plan_qubo_previous_pooled.csv", index=False
    )
    decisions.lambda_table.to_csv(output / "lambda_sweep.csv", index=False)
    decisions.ablation_table.to_csv(output / "ablations.csv", index=False)
    flat_plan.to_csv(output / "retrofit_plan_flat_prior.csv", index=False)
    dense_sparse_plan.to_csv(output / "retrofit_plan_dense_sparse_prior.csv", index=False)
    dense_flat_plan.to_csv(output / "retrofit_plan_dense_flat_prior.csv", index=False)
    dense_oracle_result.plan.to_csv(output / "ground_truth_dense_oracle_plan.csv", index=False)
    for name, plan in decisions.plans.items():
        plan.to_csv(output / "plans" / f"{name}.csv", index=False)
    exact_oracle_plan.to_csv(output / "ground_truth_oracle_plan.csv", index=False)
    decisions.oracle_plan.to_csv(
        output / "ground_truth_oracle_plan_qubo_previous.csv", index=False
    )
    _write_json(
        output / "ground_truth_plan_comparison.json",
        exact_ground_truth_comparison,
    )
    _write_json(
        output / "ground_truth_plan_comparison_qubo_previous.json",
        decisions.oracle_comparison,
    )
    _write_json(output / "naive_plan_comparison.json", decisions.naive_comparison)
    _write_json(output / "feasibility_checks.json", decisions.feasibility)
    _write_json(output / "exact_reduced_check.json", exact_reduced)
    exact_ladder.to_csv(output / "exact_verification_ladder.csv", index=False)
    exact_ladder_samples.to_csv(output / "exact_verification_ladder_seeds.csv", index=False)
    reliability.samples.to_csv(output / "solver_reliability_seeds.csv", index=False)
    reliability.summary.to_csv(output / "solver_reliability_summary.csv", index=False)
    prior_plan_reliability.samples.to_csv(
        output / "prior_plan_jaccard_50_seeds.csv", index=False
    )
    prior_plan_reliability.summary.to_csv(
        output / "prior_plan_solver_50_seed_summary.csv", index=False
    )
    jaccard_summary.to_csv(
        output / "prior_plan_jaccard_distribution_summary.csv", index=False
    )
    _write_json(
        output / "prior_plan_jaccard_distinguishability.json",
        jaccard_diagnostic,
    )
    pooled_plan_summary.to_csv(
        output / "prior_plan_pooled_qubo_summary.csv", index=False
    )
    milp_summary.to_csv(output / "milp_solver_summary.csv", index=False)
    solver_exactness.to_csv(
        output / "milp_qubo_optimality_comparison.csv", index=False
    )
    milp_prior_jaccard.to_csv(
        output / "milp_prior_plan_jaccard.csv", index=False
    )
    milp_plan_dir = output / "milp_plans"
    milp_plan_dir.mkdir(exist_ok=True)
    for name, result in milp_results.items():
        result.plan.to_csv(milp_plan_dir / f"{name}.csv", index=False)
    pooled_plan_dir = output / "prior_plan_pooled_qubo"
    pooled_plan_dir.mkdir(exist_ok=True)
    for name, result in pooled_qubo_results.items():
        result.plan.to_csv(pooled_plan_dir / f"{name}.csv", index=False)
    reliability_plan_dir = output / "reliability_best_plans"
    reliability_plan_dir.mkdir(exist_ok=True)
    for name, result in reliability.best_results.items():
        result.plan.to_csv(reliability_plan_dir / f"{name}.csv", index=False)
    _write_json(output / "annealing_diagnostics.json", decisions.main_result.diagnostics)
    _write_json(output / "penalty_bounds.json", {
        "unconstrained_objective_absolute_coefficient_bound": decisions.main_model.objective_abs_bound,
        "base_penalty": decisions.main_model.penalty_base,
        "strong_domain_penalty": C.STRONG_PENALTY_MULTIPLIER * decisions.main_model.penalty_base,
        "derivation": "base = sum(abs(all unconstrained linear and quadratic coefficients)) + 1; any objective improvement is bounded by the sum",
        "tuned_downward": False,
        "source": "illustrative",
    })
    trace.to_csv(output / "uncertainty_trace.csv", index=False)
    production_feasibility = {
        key: value for key, value in decisions.feasibility.items()
        if not key.startswith("negative_control_")
    }
    _write_json(output / "run_metadata.json", {
        "package_version": __version__,
        "random_seed": C.SEED,
        "traced_member": f"M{traced_member:02d}",
        "claim_scope": "pipeline plumbing and uncertainty propagation only; no concrete validation",
        "all_empirical_constants_illustrative": bool(constants["illustrative"].all()),
        "all_production_plans_independently_feasible": all(
            value["feasible"] for value in production_feasibility.values()
        ),
        "checker_rejected_penalty_1_negative_control": negative_controls[
            "negative_control_penalty_1"
        ]["checker_rejected"],
        "largest_exactly_verified_primary_bits": int(exact_ladder["primary_bits"].max()),
        "full_size_primary_bits": decisions.main_result.diagnostics["primary_variable_count"],
        "full_size_optimality_verified": bool(
            all(result.proven_optimal for result in milp_results.values())
        ),
        "full_size_exact_solver": "SciPy milp with open-source HiGHS backend",
        "all_milp_plans_independently_feasible": bool(
            all(not result.violations for result in milp_results.values())
        ),
        "prior_plan_diagnostic_seeds_per_configuration": C.PRIOR_PLAN_DIAGNOSTIC_SEEDS,
        "jaccard_distributions_distinguishable": jaccard_diagnostic[
            "distinguishable_under_declared_rule"
        ],
        "horseshoe_triggered": horseshoe_triggered,
        "source": "illustrative",
    })

    generate_figures(output)
    write_report(
        output,
        frame,
        normalised,
        sparse_estimate,
        sparse_tier,
        sparse_transport,
        performance,
        decisions,
        trace,
        constants,
        sensing_summary,
        flat_estimate=flat_estimate,
        prior_members=prior_members,
        prior_summaries=prior_summaries,
        reliability_summary=reliability.summary,
        reliability_samples=reliability.samples,
        exact_ladder=exact_ladder,
        negative_controls=negative_controls,
        member_bias=member_bias,
        bias_summary=bias_summary,
        horseshoe_estimate=horseshoe_estimate,
        dense_horseshoe_estimate=dense_horseshoe_estimate,
        shrinkage_bias_increment=shrinkage_bias_increment,
        prior_plan_samples=prior_plan_reliability.samples,
        prior_plan_summary=prior_plan_reliability.summary,
        jaccard_summary=jaccard_summary,
        jaccard_diagnostic=jaccard_diagnostic,
        pooled_plan_summary=pooled_plan_summary,
        milp_summary=milp_summary,
        solver_exactness=solver_exactness,
        milp_prior_jaccard=milp_prior_jaccard,
        exact_ground_truth_comparison=exact_ground_truth_comparison,
        milp_sparse_plan=sparse_plan,
        milp_oracle_plan=exact_oracle_plan,
    )
    return output
