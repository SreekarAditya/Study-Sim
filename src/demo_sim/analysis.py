"""Ablations, naive baseline, oracle comparison, and uncertainty trace."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import constants as C
from .condition import PerformanceState, TransportState, TwoTierState, compute_transport
from .estimation import StateEstimate, alpha_covariance_from_frequency_precision
from .frame import FrameModel
from .ground_truth import GroundTruth
from .planning import PlanningInputs, build_planning_inputs
from .qubo import (
    ACTIONS,
    AnnealResult,
    QUBOModel,
    build_qubo,
    complete_vector,
    independent_feasibility_check,
    primary_array_from_vector,
    simulated_annealing,
)
from .sensing import NormalisedModalData, SensorData


@dataclass
class DecisionAnalysis:
    main_model: QUBOModel
    main_result: AnnealResult
    lambda_table: pd.DataFrame
    ablation_table: pd.DataFrame
    plans: dict[str, pd.DataFrame]
    feasibility: dict[str, dict]
    naive_comparison: dict
    exact_reduced: dict
    oracle_comparison: dict
    oracle_plan: pd.DataFrame


def _bits_from_result(model: QUBOModel, result: AnnealResult) -> np.ndarray:
    return primary_array_from_vector(model, result.vector).copy()


def _evaluate_bits(model: QUBOModel, bits: np.ndarray) -> float:
    vector = complete_vector(model, bits)
    if vector is None:
        return float("inf")
    return model.qubo.energy(vector)


def _selection_set(plan: pd.DataFrame, include_period: bool = True) -> set[tuple]:
    if plan.empty:
        return set()
    intervention = plan[plan["action"] != "acquire_geometry"]
    columns = ["member", "period", "action"] if include_period else ["member", "action"]
    return set(map(tuple, intervention[columns].itertuples(index=False, name=None)))


def _plan_distance(a: pd.DataFrame, b: pd.DataFrame) -> tuple[int, float]:
    sa, sb = _selection_set(a), _selection_set(b)
    union = sa | sb
    return len(sa ^ sb), (len(sa & sb) / len(union) if union else 1.0)


def naive_severity_plan(
    model: QUBOModel,
    alpha_mean: np.ndarray,
    severity: np.ndarray,
) -> tuple[np.ndarray, pd.DataFrame, float]:
    bits = np.zeros(model.primary.shape, dtype=np.int8)
    order = np.argsort(-severity)
    pending = list(map(int, order))
    for t in range(model.periods):
        for i in pending.copy():
            poor_capacity = alpha_mean[i] < C.CAPACITY_DAMAGE_ALPHA_LIMIT
            poor_transport = bool(model.inputs.needs_cover[i])
            if poor_capacity and poor_transport:
                action_index = ACTIONS.index("combined")
            elif poor_capacity:
                action_index = ACTIONS.index("jacket")
            elif poor_transport:
                action_index = ACTIONS.index("cover")
            else:
                continue
            trial = bits.copy()
            trial[i, t, action_index] = 1
            if not independent_feasibility_check(model, trial):
                bits = trial
                pending.remove(i)
    vector = complete_vector(model, bits)
    assert vector is not None
    from .qubo import plan_frame
    return bits, plan_frame(model, vector), model.qubo.energy(vector)


def make_oracle_state(frame: FrameModel, truth: GroundTruth) -> TwoTierState:
    n = len(truth.alpha)
    table = frame.member_table()
    table["geometry_exists"] = True
    table["geometry_age_periods"] = 0
    table["new_damage_event"] = False
    table["geometry_admissible"] = True
    table["tier"] = "full_alpha_w_theta"
    table["alpha_mean"] = truth.alpha
    table["alpha_sd_after_tier"] = 0.0
    table["crack_width_mean_mm"] = truth.crack_width_mm
    table["crack_width_sd_mm"] = 0.0
    table["theta_mean_deg"] = truth.theta_deg
    table["theta_sd_deg"] = 0.0
    table["expiry_reason"] = "oracle_current"
    table["source"] = "illustrative"
    return TwoTierState(
        table,
        truth.alpha.copy(),
        np.zeros((n, n)),
        truth.crack_width_mm.copy(),
        np.zeros(n),
        truth.theta_deg.copy(),
        np.zeros(n),
        np.ones(n, dtype=bool),
    )


def decision_analysis(
    frame: FrameModel,
    inputs: PlanningInputs,
    tier_state: TwoTierState,
    performance: PerformanceState,
    truth: GroundTruth,
    exact_reduced: dict,
    rng: np.random.Generator,
    no_initial_geometry_planning_inputs: PlanningInputs,
    reliable_results: dict[str, AnnealResult] | None = None,
) -> DecisionAnalysis:
    del rng
    main_model = build_qubo(inputs, C.RISK_LAMBDA_REFERENCE)
    main_result = (
        reliable_results["main"]
        if reliable_results is not None
        else simulated_annealing(main_model, np.random.default_rng(C.SEED + 101))
    )
    plans: dict[str, pd.DataFrame] = {"main": main_result.plan}
    feasibility: dict[str, dict] = {
        "main": {"feasible": main_result.feasible, "violations": main_result.violations}
    }

    lambda_rows = []
    lambda_results: dict[float, AnnealResult] = {}
    for index, lam in enumerate(C.LAMBDA_SWEEP):
        model = build_qubo(inputs, lam)
        label = f"lambda_{lam:g}"
        result = (
            reliable_results[label]
            if reliable_results is not None
            else simulated_annealing(model, np.random.default_rng(C.SEED + 200 + index))
        )
        lambda_results[lam] = result
        acquisitions = int((result.plan["action"] == "acquire_geometry").sum()) if not result.plan.empty else 0
        lambda_rows.append({
            "lambda": lam,
            "qubo_energy": result.energy,
            "interventions": len(result.plan) - acquisitions,
            "geometry_acquisitions": acquisitions,
            "feasible": result.feasible,
            "source": "illustrative",
        })
        plans[label] = result.plan
        feasibility[label] = {"feasible": result.feasible, "violations": result.violations}
    lambda_table = pd.DataFrame(lambda_rows)

    variants: dict[str, tuple[PlanningInputs, float]] = {
        "point_estimate": (inputs.point_estimate(), C.RISK_LAMBDA_REFERENCE),
        "no_initial_geometry_acquisition_allowed": (
            no_initial_geometry_planning_inputs,
            C.RISK_LAMBDA_REFERENCE,
        ),
        "no_interactions": (inputs.without_interactions(), C.RISK_LAMBDA_REFERENCE),
    }
    variant_results: dict[str, tuple[QUBOModel, AnnealResult]] = {}
    for index, (name, (variant_inputs, lam)) in enumerate(variants.items()):
        model = build_qubo(variant_inputs, lam)
        result = (
            reliable_results[name]
            if reliable_results is not None
            else simulated_annealing(model, np.random.default_rng(C.SEED + 300 + index))
        )
        variant_results[name] = (model, result)
        plans[name] = result.plan
        feasibility[name] = {"feasible": result.feasible, "violations": result.violations}

    # Candidate pooling is a convergence diagnostic, not an objective change:
    # every candidate is rescored by the reference uncertainty-aware QUBO, and
    # a final reference anneal starts from the best feasible candidate found by
    # any schedule or ablation. This prevents a weak main run from making an
    # ablation look spuriously superior under the main objective.
    candidate_bits = [_bits_from_result(main_model, main_result)]
    candidate_bits.extend(_bits_from_result(main_model, result) for result in lambda_results.values())
    candidate_bits.extend(_bits_from_result(model, result) for model, result in variant_results.values())
    best_start = min(candidate_bits, key=lambda bits: _evaluate_bits(main_model, bits))
    raw_reference_best_energy = main_result.energy
    pooled_start_energy = _evaluate_bits(main_model, best_start)
    main_result = simulated_annealing(
        main_model,
        np.random.default_rng(C.SEED + 350),
        initial_primary=best_start,
    )
    main_result.diagnostics.update({
        "candidate_pooling_used": True,
        "raw_20_seed_reference_best_energy": raw_reference_best_energy,
        "pooled_feasible_start_energy": pooled_start_energy,
        "pooled_candidate_count": len(candidate_bits),
        "interpretation": "best-known feasible reference after cross-configuration candidate rescoring and final polish; not a proven optimum",
    })
    plans["main"] = main_result.plan
    feasibility["main"] = {"feasible": main_result.feasible, "violations": main_result.violations}
    # In a single-run invocation, keep the reference-lambda row aligned with
    # the pooled production result. Under the requested reliability protocol,
    # preserve the raw best-of-20 lambda result so lambda_sweep.csv remains a
    # faithful multi-seed artifact; the additionally pooled plan is `main`.
    if reliable_results is None:
        reference_row = np.isclose(lambda_table["lambda"], C.RISK_LAMBDA_REFERENCE)
        reference_acquisitions = int(
            (main_result.plan["action"] == "acquire_geometry").sum()
        ) if not main_result.plan.empty else 0
        lambda_table.loc[reference_row, "qubo_energy"] = main_result.energy
        lambda_table.loc[reference_row, "interventions"] = len(main_result.plan) - reference_acquisitions
        lambda_table.loc[reference_row, "geometry_acquisitions"] = reference_acquisitions
        lambda_table.loc[reference_row, "feasible"] = main_result.feasible
        plans[f"lambda_{C.RISK_LAMBDA_REFERENCE:g}"] = main_result.plan
        feasibility[f"lambda_{C.RISK_LAMBDA_REFERENCE:g}"] = {
            "feasible": main_result.feasible,
            "violations": main_result.violations,
        }

    naive_bits, naive_plan, naive_native_energy = naive_severity_plan(
        main_model, tier_state.alpha_mean, performance.table["utilisation_mean"].to_numpy()
    )
    plans["naive_severity"] = naive_plan
    naive_violations = independent_feasibility_check(main_model, naive_bits)
    feasibility["naive_severity"] = {"feasible": not naive_violations, "violations": naive_violations}
    naive_diff, naive_jaccard = _plan_distance(main_result.plan, naive_plan)
    naive_comparison = {
        "different_member_period_action_decisions": naive_diff,
        "intervention_jaccard_similarity": naive_jaccard,
        "qubo_reference_objective_naive": naive_native_energy,
        "qubo_reference_objective_main": main_result.energy,
        "naive_objective_disadvantage": naive_native_energy - main_result.energy,
        "source": "illustrative",
    }

    ablation_rows = []
    for name, (model, result) in variant_results.items():
        bits = _bits_from_result(model, result)
        evaluated = _evaluate_bits(main_model, bits)
        diff, jaccard = _plan_distance(main_result.plan, result.plan)
        ablation_rows.append({
            "scenario": name,
            "native_qubo_energy": result.energy,
            "energy_when_scored_by_reference_objective": evaluated,
            "reference_objective_delta_vs_main": evaluated - main_result.energy,
            "different_member_period_action_decisions": diff,
            "intervention_jaccard_similarity": jaccard,
            "geometry_acquisitions": int((result.plan["action"] == "acquire_geometry").sum()) if not result.plan.empty else 0,
            "feasible": result.feasible,
            "source": "illustrative",
        })
    ablation_rows.append({
        "scenario": "naive_severity",
        "native_qubo_energy": naive_native_energy,
        "energy_when_scored_by_reference_objective": naive_native_energy,
        "reference_objective_delta_vs_main": naive_native_energy - main_result.energy,
        "different_member_period_action_decisions": naive_diff,
        "intervention_jaccard_similarity": naive_jaccard,
        "geometry_acquisitions": 0,
        "feasible": not naive_violations,
        "source": "illustrative",
    })
    ablation_table = pd.DataFrame(ablation_rows)

    oracle_state = make_oracle_state(frame, truth)
    oracle_transport = compute_transport(oracle_state, truth, np.random.default_rng(C.SEED + 401))
    oracle_inputs = build_planning_inputs(frame, oracle_state, oracle_transport, np.random.default_rng(C.SEED + 402))
    oracle_inputs.action_variance[:] = 0.0
    oracle_model = build_qubo(oracle_inputs, 0.0)
    oracle_result = (
        reliable_results["ground_truth_oracle"]
        if reliable_results is not None and "ground_truth_oracle" in reliable_results
        else simulated_annealing(oracle_model, np.random.default_rng(C.SEED + 403))
    )
    oracle_plan = oracle_result.plan
    feasibility["ground_truth_oracle"] = {"feasible": oracle_result.feasible, "violations": oracle_result.violations}
    inferred_set = _selection_set(main_result.plan, include_period=False)
    oracle_set = _selection_set(oracle_plan, include_period=False)
    oracle_comparison = {
        "definition": "same catalogue, horizon and budgets, but action moments computed from hidden true alpha,w,theta with zero epistemic variance",
        "matched_member_action_pairs": len(inferred_set & oracle_set),
        "inferred_member_action_pairs": len(inferred_set),
        "oracle_member_action_pairs": len(oracle_set),
        "jaccard_similarity": len(inferred_set & oracle_set) / max(len(inferred_set | oracle_set), 1),
        "inferred_only": sorted([f"{m}:{a}" for m, a in inferred_set - oracle_set]),
        "oracle_only": sorted([f"{m}:{a}" for m, a in oracle_set - inferred_set]),
        "oracle_feasible": oracle_result.feasible,
        "source": "illustrative",
    }
    return DecisionAnalysis(
        main_model, main_result, lambda_table, ablation_table, plans, feasibility,
        naive_comparison, exact_reduced, oracle_comparison, oracle_plan,
    )


def uncertainty_trace(
    frame: FrameModel,
    sensor: SensorData,
    normalised: NormalisedModalData,
    estimate: StateEstimate,
    tier: TwoTierState,
    transport: TransportState,
    performance: PerformanceState,
) -> tuple[int, pd.DataFrame]:
    # Follow the member with the largest tier-widened alpha SD.
    member = int(np.argmax(np.diag(tier.alpha_covariance)))
    raw_env_fraction = sensor.observations.groupby("mode")["true_environment_shift_fraction"].std().to_numpy()
    raw_frequency_sd = sensor.true_reference_freq * np.sqrt(
        C.FREQUENCY_NOISE_FRACTION**2 + raw_env_fraction**2
    )
    raw_cov = alpha_covariance_from_frequency_precision(frame, estimate.mean, raw_frequency_sd)
    normal_cov = alpha_covariance_from_frequency_precision(frame, estimate.mean, normalised.frequency_sd)
    rows = [
        {
            "stage": "raw sensing (frequency only)",
            "state_quantity": "alpha",
            "centre": C.ALPHA_PRIOR_MEAN,
            "uncertainty": np.sqrt(raw_cov[member, member]),
            "uncertainty_definition": "linearised posterior SD; environment still present",
            "classification_or_note": "member state not directly observed",
        },
        {
            "stage": "environment-normalised frequency",
            "state_quantity": "alpha",
            "centre": C.ALPHA_PRIOR_MEAN,
            "uncertainty": np.sqrt(normal_cov[member, member]),
            "uncertainty_definition": "linearised frequency-only posterior SD",
            "classification_or_note": "residual contamination retained in frequency SD",
        },
        {
            "stage": "joint modal state estimation",
            "state_quantity": "alpha",
            "centre": estimate.mean[member],
            "uncertainty": np.sqrt(estimate.covariance[member, member]),
            "uncertainty_definition": "joint Laplace posterior SD with full covariance",
            "classification_or_note": estimate.table.iloc[member]["identifiability"],
        },
        {
            "stage": "two-tier admissibility",
            "state_quantity": "alpha",
            "centre": tier.alpha_mean[member],
            "uncertainty": np.sqrt(tier.alpha_covariance[member, member]),
            "uncertainty_definition": "tier-adjusted alpha SD",
            "classification_or_note": tier.table.iloc[member]["tier"],
        },
        {
            "stage": "transport propagation",
            "state_quantity": "M transport ratio",
            "centre": transport.mean[member],
            "uncertainty": transport.sd[member],
            "uncertainty_definition": "Monte Carlo SD including alpha and geometry tier",
            "classification_or_note": transport.table.iloc[member]["functional_form"],
        },
        {
            "stage": "performance classification",
            "state_quantity": "utilisation",
            "centre": performance.table.iloc[member]["utilisation_mean"],
            "uncertainty": performance.table.iloc[member]["utilisation_sd"],
            "uncertainty_definition": "Monte Carlo SD; conformal interval reported separately",
            "classification_or_note": (
                f"{performance.table.iloc[member]['performance_level']}; conformal set "
                f"{performance.table.iloc[member]['conformal_label_set']}"
            ),
        },
    ]
    table = pd.DataFrame(rows)
    table["member"] = f"M{member:02d}"
    table["source"] = "illustrative"
    return member, table
