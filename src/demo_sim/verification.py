"""Negative controls, multi-seed reliability, and exact verification ladder."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import multiprocessing as mp
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from . import constants as C
from .planning import PlanningInputs
from .prior_comparison import plan_jaccard
from .qubo import (
    AnnealResult,
    QUBOModel,
    build_qubo,
    complete_vector,
    independent_feasibility_check,
    plan_frame,
    primary_array_from_vector,
    raw_simulated_annealing,
    simulated_annealing,
)


@dataclass
class ReliabilityResults:
    samples: pd.DataFrame
    summary: pd.DataFrame
    best_results: dict[str, AnnealResult]


_WORKER_MODEL: QUBOModel | None = None


def _initialise_worker(model: QUBOModel) -> None:
    global _WORKER_MODEL
    _WORKER_MODEL = model


def _anneal_worker(seed: int) -> AnnealResult:
    if _WORKER_MODEL is None:
        raise RuntimeError("Reliability worker model was not initialised")
    return simulated_annealing(
        _WORKER_MODEL,
        np.random.default_rng(seed),
        restarts=C.RELIABILITY_RESTARTS_PER_SEED,
        steps=C.ANNEAL_STEPS,
    )


def run_seed_ensemble(
    configurations: dict[str, QUBOModel],
    seed_count: int = C.RELIABILITY_SEED_COUNT,
    jaccard_references: dict[str, pd.DataFrame] | None = None,
    seed_base: int = C.RELIABILITY_SEED_BASE,
) -> ReliabilityResults:
    """Run every full-size configuration from independent seed trajectories."""
    rows: list[dict] = []
    best_results: dict[str, AnnealResult] = {}
    for config_index, (name, model) in enumerate(configurations.items()):
        seeds = [seed_base + 1000 * config_index + i for i in range(seed_count)]
        with ProcessPoolExecutor(
            max_workers=C.RELIABILITY_WORKERS,
            mp_context=mp.get_context("fork"),
            initializer=_initialise_worker,
            initargs=(model,),
        ) as pool:
            results = list(pool.map(_anneal_worker, seeds))
        best = min(results, key=lambda item: item.energy)
        best_results[name] = best
        for seed, result in zip(seeds, results):
            acquisitions = int(
                (result.plan["action"] == "acquire_geometry").sum()
            ) if not result.plan.empty else 0
            row = {
                "configuration": name,
                "seed": seed,
                "energy": result.energy,
                "feasible": result.feasible,
                "interventions": len(result.plan) - acquisitions,
                "geometry_acquisitions": acquisitions,
                "source": "illustrative",
            }
            if jaccard_references:
                for reference_name, reference_plan in jaccard_references.items():
                    row[f"jaccard_vs_{reference_name}"] = plan_jaccard(
                        result.plan,
                        reference_plan,
                    )
            rows.append(row)
    samples = pd.DataFrame(rows)
    summary_rows = []
    for name, group in samples.groupby("configuration", sort=False):
        energies = group["energy"].to_numpy()
        summary_rows.append({
            "configuration": name,
            "seed_count": len(group),
            "best_energy": float(np.min(energies)),
            "median_energy": float(np.median(energies)),
            "worst_energy": float(np.max(energies)),
            "energy_spread": float(np.max(energies) - np.min(energies)),
            "feasible_runs": int(group["feasible"].sum()),
            "runs_with_geometry_acquisition": int((group["geometry_acquisitions"] > 0).sum()),
            "geometry_acquisition_run_fraction": float((group["geometry_acquisitions"] > 0).mean()),
            "best_run_geometry_acquisitions": int(
                group.loc[group["energy"].idxmin(), "geometry_acquisitions"]
            ),
            "source": "illustrative",
        })
    return ReliabilityResults(samples, pd.DataFrame(summary_rows), best_results)


def jaccard_distribution_summary(
    samples: pd.DataFrame,
    reference_column: str,
) -> tuple[pd.DataFrame, dict]:
    """Summarise flat/sparse seed distributions and a declared overlap rule."""
    selected = samples[samples["configuration"].isin(["flat_prior", "sparse_prior"])]
    rows = []
    arrays: dict[str, np.ndarray] = {}
    for prior in ("flat_prior", "sparse_prior"):
        values = selected.loc[selected["configuration"] == prior, reference_column].to_numpy()
        arrays[prior] = values
        q1, q3 = np.quantile(values, [0.25, 0.75])
        rows.append({
            "prior": prior,
            "seed_count": len(values),
            "mean_jaccard": float(np.mean(values)),
            "median_jaccard": float(np.median(values)),
            "q1_jaccard": float(q1),
            "q3_jaccard": float(q3),
            "iqr_jaccard": float(q3 - q1),
            "min_jaccard": float(np.min(values)),
            "max_jaccard": float(np.max(values)),
            "source": "illustrative",
        })
    flat, sparse = arrays["flat_prior"], arrays["sparse_prior"]
    statistic, p_value = mannwhitneyu(flat, sparse, alternative="two-sided")
    # Positive delta means flat has higher Jaccard more often than sparse.
    cliff = float(
        (np.sum(flat[:, None] > sparse[None, :])
         - np.sum(flat[:, None] < sparse[None, :]))
        / (len(flat) * len(sparse))
    )
    flat_q1, flat_q3 = np.quantile(flat, [0.25, 0.75])
    sparse_q1, sparse_q3 = np.quantile(sparse, [0.25, 0.75])
    overlap = max(0.0, min(flat_q3, sparse_q3) - max(flat_q1, sparse_q1))
    minimum_iqr = max(min(flat_q3 - flat_q1, sparse_q3 - sparse_q1), 1.0e-12)
    overlap_fraction = float(overlap / minimum_iqr)
    substantial_overlap = bool(
        overlap_fraction >= C.JACCARD_SUBSTANTIAL_IQR_OVERLAP
    )
    distinguishable = bool(
        p_value <= C.JACCARD_DISTINGUISH_P
        and abs(cliff) >= C.JACCARD_DISTINGUISH_CLIFF
        and not substantial_overlap
    )
    diagnostic = {
        "reference_column": reference_column,
        "mann_whitney_u": float(statistic),
        "mann_whitney_two_sided_p": float(p_value),
        "cliffs_delta_flat_minus_sparse": cliff,
        "iqr_overlap_fraction_of_smaller_iqr": overlap_fraction,
        "substantial_iqr_overlap": substantial_overlap,
        "distinguishable_under_declared_rule": distinguishable,
        "decision_rule": (
            f"p <= {C.JACCARD_DISTINGUISH_P} AND |Cliff delta| >= "
            f"{C.JACCARD_DISTINGUISH_CLIFF} AND IQR overlap fraction < "
            f"{C.JACCARD_SUBSTANTIAL_IQR_OVERLAP}"
        ),
        "source": "illustrative",
    }
    return pd.DataFrame(rows), diagnostic


def _one_member_inputs(inputs: PlanningInputs, member: int) -> PlanningInputs:
    return PlanningInputs(
        members=inputs.members.iloc[[member]].copy().reset_index(drop=True),
        action_mean=inputs.action_mean[[member]],
        action_variance=inputs.action_variance[[member]],
        interactions=inputs.interactions.iloc[0:0].copy(),
        needs_cover=inputs.needs_cover[[member]],
        full_mask=inputs.full_mask[[member]],
        geometry_remaining_life=inputs.geometry_remaining_life[[member]],
        deterioration_cost=inputs.deterioration_cost[[member]],
    )


def _exact_primary_optimum(model: QUBOModel) -> tuple[float, np.ndarray, int]:
    """Vectorise all primary assignments, then score every feasible completion."""
    primary_count = model.primary.size
    total = 1 << primary_count
    best_energy = np.inf
    best_vector: np.ndarray | None = None
    feasible_count = 0
    bit_positions = np.arange(primary_count, dtype=np.uint64)
    for start in range(0, total, C.EXACT_ENUMERATION_CHUNK_SIZE):
        stop = min(start + C.EXACT_ENUMERATION_CHUNK_SIZE, total)
        masks = np.arange(start, stop, dtype=np.uint64)
        matrix = ((masks[:, None] >> bit_positions[None, :]) & 1).astype(np.int8)
        shaped = matrix.reshape(-1, model.periods, 4)
        intervention = shaped[:, :, :3]
        acquisition = shaped[:, :, 3]
        valid = np.all(intervention.sum(axis=2) <= 1, axis=1)
        valid &= acquisition.sum(axis=1) <= 1
        valid &= np.all(intervention.sum(axis=1) <= 1, axis=1)
        valid &= ~(
            (intervention[:, :, 2].sum(axis=1) > 0)
            & (intervention[:, :, :2].sum(axis=(1, 2)) > 0)
        )
        period_index = np.arange(model.periods)
        cover_first = np.where(intervention[:, :, 0] > 0, period_index, model.periods).min(axis=1)
        jacket_first = np.where(intervention[:, :, 1] > 0, period_index, model.periods).min(axis=1)
        both = (cover_first < model.periods) & (jacket_first < model.periods)
        valid &= (~both) | (cover_first < jacket_first)
        if model.inputs.needs_cover[0]:
            has_jacket = jacket_first < model.periods
            valid &= (~has_jacket) | (cover_first < jacket_first)
        acquisition_first = np.where(acquisition > 0, period_index, model.periods).min(axis=1)
        first_intervention = np.where(
            intervention.sum(axis=2) > 0, period_index, model.periods
        ).min(axis=1)
        has_acquisition = acquisition_first < model.periods
        valid &= (~has_acquisition) | (first_intervention > acquisition_first)
        for row in matrix[valid]:
            bits = row.reshape(model.primary.shape)
            if independent_feasibility_check(model, bits):
                continue
            vector = complete_vector(model, bits)
            if vector is None:
                continue
            feasible_count += 1
            energy = model.qubo.energy(vector)
            if energy < best_energy:
                best_energy, best_vector = energy, vector
    if best_vector is None:
        raise RuntimeError("No feasible exact-ladder assignment")
    return float(best_energy), best_vector, feasible_count


def exact_verification_ladder(inputs: PlanningInputs) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Exact 2^12, 2^16 and 2^20 primary ladders plus 20-seed hit rates."""
    member = int(np.argmax(inputs.needs_cover.astype(int) * 2 + inputs.action_mean[:, 2]))
    reduced = _one_member_inputs(inputs, member)
    summary_rows: list[dict] = []
    sample_rows: list[dict] = []
    for ladder_index, primary_bits in enumerate(C.EXACT_LADDER_PRIMARY_BITS):
        periods = primary_bits // 4
        model = build_qubo(reduced, C.RISK_LAMBDA_REFERENCE, periods=periods)
        exact_energy, exact_vector, feasible_count = _exact_primary_optimum(model)
        hits = 0
        energies = []
        for seed_index in range(C.RELIABILITY_SEED_COUNT):
            seed = C.RELIABILITY_SEED_BASE + 50000 + ladder_index * 1000 + seed_index
            result = simulated_annealing(
                model,
                np.random.default_rng(seed),
                restarts=C.RELIABILITY_RESTARTS_PER_SEED,
                steps=C.EXACT_LADDER_ANNEAL_STEPS,
            )
            hit = bool(np.isclose(result.energy, exact_energy, atol=1.0e-7))
            hits += int(hit)
            energies.append(result.energy)
            sample_rows.append({
                "primary_bits": primary_bits,
                "raw_assignment_count": 1 << primary_bits,
                "seed": seed,
                "annealed_energy": result.energy,
                "exact_energy": exact_energy,
                "hit_exact_optimum": hit,
                "source": "illustrative",
            })
        summary_rows.append({
            "primary_bits": primary_bits,
            "total_qubo_variables": len(model.qubo.names),
            "raw_assignment_count": 1 << primary_bits,
            "feasible_assignments": feasible_count,
            "exact_optimum_energy": exact_energy,
            "annealer_hits": hits,
            "seed_count": C.RELIABILITY_SEED_COUNT,
            "annealer_hit_rate": hits / C.RELIABILITY_SEED_COUNT,
            "best_annealed_energy": float(np.min(energies)),
            "worst_annealed_energy": float(np.max(energies)),
            "exact_plan": str(plan_frame(model, exact_vector).to_dict(orient="records")),
            "source": "illustrative",
        })
    return pd.DataFrame(summary_rows), pd.DataFrame(sample_rows)


def _violation_categories(violations: Iterable[str]) -> list[str]:
    categories = set()
    for text in violations:
        if "intervention budget" in text:
            categories.add("intervention_budget")
        elif "acquisition budget" in text:
            categories.add("acquisition_budget")
        elif "out-of-service" in text:
            categories.add("out_of_service")
        elif "untreated auxiliary" in text:
            categories.add("untreated_y_recurrence")
        elif "slack equality" in text:
            categories.add("slack_equality")
        elif "not at least one period" in text:
            categories.add("acquisition_intervention_separation")
        elif "jacket" in text or "cover is not before" in text:
            categories.add("cover_jacket_precedence")
        elif "repeated" in text or "multiple intervention" in text or "combined mixed" in text:
            categories.add("catalogue_uniqueness")
        else:
            categories.add("other")
    return sorted(categories)


def penalty_negative_controls(
    inputs: PlanningInputs,
    reference_result: AnnealResult,
) -> dict[str, dict]:
    """Deliberately weaken penalties and ask the checker to reject raw solutions."""
    reference_model = build_qubo(inputs, C.RISK_LAMBDA_REFERENCE)
    objective_model = build_qubo(inputs, C.RISK_LAMBDA_REFERENCE, penalty_override=0.0)
    reference_objective = objective_model.qubo.energy(reference_result.vector)
    controls: dict[str, dict] = {}
    for index, penalty in enumerate(C.NEGATIVE_CONTROL_PENALTIES):
        weak_model = build_qubo(
            inputs,
            C.RISK_LAMBDA_REFERENCE,
            penalty_override=penalty,
        )
        raw = raw_simulated_annealing(
            weak_model,
            np.random.default_rng(C.RELIABILITY_SEED_BASE + 90000 + index),
            initial_vector=reference_result.vector,
        )
        objective_energy = objective_model.qubo.energy(raw.vector)
        derived_penalty_energy = reference_model.qubo.energy(raw.vector)
        controls[f"negative_control_penalty_{penalty:g}"] = {
            "penalty_weight": penalty,
            "checker_rejected": not raw.feasible,
            "feasible": raw.feasible,
            "violation_count": len(raw.violations),
            "violation_categories": _violation_categories(raw.violations),
            "violations": raw.violations,
            "weak_penalty_qubo_energy": raw.energy,
            "objective_only_energy": objective_energy,
            "correctly_penalised_energy": derived_penalty_energy,
            "reference_best_known_feasible_objective": reference_objective,
            "objective_delta_vs_reference": objective_energy - reference_objective,
            "correctly_penalised_delta_vs_reference": derived_penalty_energy - reference_result.energy,
            "scores_better_on_objective_than_reference": bool(objective_energy < reference_objective),
            "conclusion": (
                "checker rejected an infeasible weak-penalty solution"
                if not raw.feasible
                else "no violation was induced at this penalty; this negative control is inconclusive, not a checker pass"
            ),
            "source": "illustrative",
        }
    return controls
