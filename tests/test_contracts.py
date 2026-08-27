from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from demo_sim import constants as C
from demo_sim.condition import m_full, m_reduced
from demo_sim.frame import FrameModel
from demo_sim.milp import solve_exact_milp
from demo_sim.planning import PlanningInputs
from demo_sim.qubo import build_qubo, complete_vector, exact_reduced_check, independent_feasibility_check, simulated_annealing


def small_inputs() -> PlanningInputs:
    members = pd.DataFrame({
        "member_id": [0, 1],
        "member": ["M00", "M01"],
        "kind": ["column", "beam"],
        "storey": [1, 1],
        "bay": [0, 0],
        "tier": ["reduced_alpha_only", "full_alpha_w_theta"],
    })
    interactions = pd.DataFrame(columns=[
        "member_i", "member_j", "action_i", "action_j", "solver_pair_benefit",
        "solver_single_benefit_sum", "interaction_utility", "derivation", "source",
    ])
    return PlanningInputs(
        members=members,
        action_mean=np.array([[2.0, 3.0, 5.5], [1.5, 4.0, 5.0]]),
        action_variance=np.array([[0.4, 0.8, 1.0], [0.3, 0.7, 0.9]]),
        interactions=interactions,
        needs_cover=np.array([True, False]),
        full_mask=np.array([False, True]),
        geometry_remaining_life=np.array([0, 1]),
        deterioration_cost=np.array([0.7, 0.6]),
    )


def test_all_registered_constants_are_illustrative() -> None:
    table = C.constants_frame()
    assert len(table) > 70
    assert table["illustrative"].all()
    assert set(table["source"]) == {"illustrative"}
    assert table["origin"].str.len().min() > 10


def test_frame_has_requested_nontrivial_size_and_modal_solution() -> None:
    frame = FrameModel.regular_frame()
    assert len(frame.elements) == 21
    assert len(frame.sensor_dofs()) == 6
    frequency, shapes = frame.modal(np.ones(21))
    assert np.all(frequency > 0.0)
    assert shapes.shape == (frame.ndof, C.N_MODES)


def test_transport_forms_are_bounded_and_monotone_in_damage() -> None:
    alpha = np.array([0.95, 0.75, 0.55])
    width = np.array([0.05, 0.35, 0.75])
    theta = np.array([0.0, 45.0, 90.0])
    full = m_full(alpha, width, theta)
    reduced = m_reduced(alpha)
    assert np.all(full >= 1.0)
    assert np.all(reduced >= 1.0)
    assert np.all(np.diff(full) > 0.0)
    assert np.all(np.diff(reduced) > 0.0)


def test_independent_checker_catches_original_constraints() -> None:
    model = build_qubo(small_inputs(), C.RISK_LAMBDA_REFERENCE, periods=2)
    bits = np.zeros(model.primary.shape, dtype=np.int8)
    bits[0, 0, 1] = 1  # jacket without earlier cover on poor-transport member
    assert any("lacks earlier cover" in item for item in independent_feasibility_check(model, bits))
    bits[:] = 0
    bits[0, 0, 3] = 1
    bits[0, 0, 0] = 1
    assert any("not at least one period" in item for item in independent_feasibility_check(model, bits))
    bits[:] = 0
    bits[0, 0, 2] = 1
    bits[1, 0, 2] = 1
    assert any("intervention budget" in item for item in independent_feasibility_check(model, bits))
    bits[:] = 0
    vector = complete_vector(model, bits)
    assert vector is not None
    vector[model.untreated[0, 0]] = 0
    assert any(
        "untreated auxiliary" in item
        for item in independent_feasibility_check(model, bits, full_vector=vector)
    )


def test_penalty_bound_and_reduced_exact_check() -> None:
    inputs = small_inputs()
    model = build_qubo(inputs, C.RISK_LAMBDA_REFERENCE, periods=2)
    assert model.penalty_base > model.objective_abs_bound
    empty = np.zeros(model.primary.shape, dtype=np.int8)
    vector = complete_vector(model, empty)
    assert vector is not None
    annealed = simulated_annealing(model, np.random.default_rng(7), restarts=2, steps=1200)
    assert annealed.feasible
    exact = exact_reduced_check(inputs, np.random.default_rng(8))
    assert exact["annealer_reached_exact_optimum"]


def test_milp_matches_bruteforce_objective_on_one_member() -> None:
    inputs = small_inputs()
    one = PlanningInputs(
        members=inputs.members.iloc[[0]].reset_index(drop=True),
        action_mean=inputs.action_mean[[0]],
        action_variance=inputs.action_variance[[0]],
        interactions=inputs.interactions.copy(),
        needs_cover=inputs.needs_cover[[0]],
        full_mask=inputs.full_mask[[0]],
        geometry_remaining_life=inputs.geometry_remaining_life[[0]],
        deterioration_cost=inputs.deterioration_cost[[0]],
    )
    model = build_qubo(one, C.RISK_LAMBDA_REFERENCE, periods=2)
    exact_energy = np.inf
    for mask in range(1 << model.primary.size):
        bits = np.array(
            [(mask >> bit) & 1 for bit in range(model.primary.size)],
            dtype=np.int8,
        ).reshape(model.primary.shape)
        vector = complete_vector(model, bits)
        if vector is not None:
            exact_energy = min(exact_energy, model.qubo.energy(vector))
    result = solve_exact_milp(one, C.RISK_LAMBDA_REFERENCE, periods=2)
    assert result.proven_optimal
    assert not result.violations
    assert np.isclose(result.reference_qubo_energy, exact_energy, atol=1.0e-8)


def test_generated_result_contract_if_present() -> None:
    results = Path("results")
    if not results.exists():
        return
    metadata = json.loads((results / "run_metadata.json").read_text())
    assert metadata["all_empirical_constants_illustrative"]
    assert metadata.get(
        "all_production_plans_independently_feasible",
        metadata.get("all_returned_plans_independently_feasible"),
    )
    assert metadata["full_size_optimality_verified"]
    assert metadata["all_milp_plans_independently_feasible"]
    constants = pd.read_csv(results / "constants.csv")
    assert constants["illustrative"].all()
    assert set(constants["source"]) == {"illustrative"}
    report = (results / "REPORT.md").read_text()
    assert report.splitlines()[2].startswith("> **Mandatory scope warning:**")
    assert "not validation" in report
    assert "source=illustrative" in report
    feasibility = json.loads((results / "feasibility_checks.json").read_text())
    assert feasibility["negative_control_penalty_1"]["checker_rejected"]
    assert not feasibility["negative_control_penalty_1"]["feasible"]
    assert "negative_control_penalty_10" in feasibility
    production = {
        name: value for name, value in feasibility.items()
        if not name.startswith("negative_control_")
    }
    assert all(value["feasible"] for value in production.values())
    reliability = pd.read_csv(results / "solver_reliability_summary.csv")
    assert (reliability["seed_count"] >= 20).all()
    assert (reliability["feasible_runs"] == reliability["seed_count"]).all()
    ladder = pd.read_csv(results / "exact_verification_ladder.csv")
    assert ladder["primary_bits"].tolist() == [12, 16, 20]
    ablations = pd.read_csv(results / "ablations.csv")
    assert "no_initial_geometry_acquisition_allowed" in set(ablations["scenario"])
    assert "all_reduced" not in set(ablations["scenario"])
    covariance = pd.read_csv(results / "posterior_covariance.csv", index_col=0).to_numpy()
    assert covariance.shape == (21, 21)
    assert np.linalg.eigvalsh((covariance + covariance.T) / 2.0).min() >= -1.0e-10
    prior_runs = pd.read_csv(results / "prior_plan_jaccard_50_seeds.csv")
    assert (prior_runs.groupby("configuration").size() == 50).all()
    milp_summary = pd.read_csv(results / "milp_solver_summary.csv")
    assert (milp_summary["primary_binary_count"] == 504).all()
    assert milp_summary["proven_optimal"].all()
    assert (milp_summary["independent_violation_count"] == 0).all()
    exactness = pd.read_csv(results / "milp_qubo_optimality_comparison.csv")
    assert exactness["pooled_never_better_than_milp"].all()
    assert "not evidence about the prior" in report
