"""Exact hard-constraint MILP equivalent of the retrofit-planning QUBO.

The planning objective is taken directly from the zero-penalty QUBO. Every
binary product in that objective receives a standard McCormick auxiliary.
Original planning constraints are imposed directly as linear constraints; QUBO
penalty and slack variables are not part of the MILP decision space.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from . import constants as C
from .planning import ACTIONS, PlanningInputs
from .qubo import (
    ACTION_COST,
    QUBOModel,
    build_qubo,
    complete_vector,
    independent_feasibility_check,
    plan_frame,
)


@dataclass
class MILPResult:
    model: QUBOModel
    primary_bits: np.ndarray
    vector: np.ndarray
    plan: object
    objective_value: float
    reference_qubo_energy: float
    solve_time_seconds: float
    proven_optimal: bool
    status: int
    message: str
    mip_gap: float
    mip_dual_bound: float
    mip_node_count: int
    binary_variable_count: int
    product_variable_count: int
    linear_constraint_count: int
    violations: list[str]


class _Rows:
    def __init__(self) -> None:
        self.row: list[int] = []
        self.col: list[int] = []
        self.data: list[float] = []
        self.lower: list[float] = []
        self.upper: list[float] = []

    def add(self, coefficients: dict[int, float], lower: float, upper: float) -> None:
        current = len(self.lower)
        for index, coefficient in coefficients.items():
            if coefficient != 0.0:
                self.row.append(current)
                self.col.append(index)
                self.data.append(float(coefficient))
        self.lower.append(float(lower))
        self.upper.append(float(upper))


def _hard_constraints(rows: _Rows, model: QUBOModel) -> None:
    """Encode exactly the original constraints checked outside the QUBO."""
    n, periods, _ = model.primary.shape
    x = model.primary
    y = model.untreated

    for member in range(n):
        for period in range(periods):
            # At most one intervention type for a member in a period.
            rows.add({int(x[member, period, a]): 1.0 for a in range(3)}, -np.inf, 1.0)

        # Each catalogue action, including acquisition, occurs at most once.
        for action in range(4):
            rows.add(
                {int(x[member, period, action]): 1.0 for period in range(periods)},
                -np.inf,
                1.0,
            )

        # Combined excludes either separate cover or separate jacket anywhere.
        for combined_period in range(periods):
            for separate_period in range(periods):
                for separate_action in (0, 1):
                    rows.add({
                        int(x[member, combined_period, 2]): 1.0,
                        int(x[member, separate_period, separate_action]): 1.0,
                    }, -np.inf, 1.0)

        # If both separate actions occur, cover must be strictly earlier.
        for cover_period in range(periods):
            for jacket_period in range(cover_period + 1):
                rows.add({
                    int(x[member, cover_period, 0]): 1.0,
                    int(x[member, jacket_period, 1]): 1.0,
                }, -np.inf, 1.0)

        # Poor-transport members require earlier cover before any jacket.
        if model.inputs.needs_cover[member]:
            for jacket_period in range(periods):
                coefficients = {int(x[member, jacket_period, 1]): 1.0}
                for earlier in range(jacket_period):
                    coefficients[int(x[member, earlier, 0])] = -1.0
                rows.add(coefficients, -np.inf, 0.0)

        # Acquisition in t prohibits intervention at or before t.
        for acquisition_period in range(periods):
            for intervention_period in range(acquisition_period + 1):
                for action in range(3):
                    rows.add({
                        int(x[member, acquisition_period, 3]): 1.0,
                        int(x[member, intervention_period, action]): 1.0,
                    }, -np.inf, 1.0)

        # y(i,t) is one until the first intervention in a prior period.
        rows.add({int(y[member, 0]): 1.0}, 1.0, 1.0)
        for period in range(1, periods):
            previous_y = int(y[member, period - 1])
            current_y = int(y[member, period])
            previous_actions = [int(x[member, period - 1, action]) for action in range(3)]
            rows.add({current_y: 1.0, previous_y: -1.0}, -np.inf, 0.0)
            rows.add(
                {current_y: 1.0, **{index: 1.0 for index in previous_actions}},
                -np.inf,
                1.0,
            )
            rows.add(
                {current_y: 1.0, previous_y: -1.0,
                 **{index: 1.0 for index in previous_actions}},
                0.0,
                np.inf,
            )

    for period in range(periods):
        intervention_cost: dict[int, float] = {}
        acquisition_count: dict[int, float] = {}
        outage_count: dict[int, float] = {}
        for member in range(n):
            for action_index, action in enumerate(ACTIONS):
                intervention_cost[int(x[member, period, action_index])] = ACTION_COST[action]
                outage_count[int(x[member, period, action_index])] = 1.0
            acquisition_count[int(x[member, period, 3])] = 1.0
        rows.add(intervention_cost, -np.inf, C.INTERVENTION_BUDGET)
        rows.add(acquisition_count, -np.inf, C.ACQUISITION_BUDGET)
        rows.add(outage_count, -np.inf, C.OUT_OF_SERVICE_LIMIT)


def solve_exact_milp(
    inputs: PlanningInputs,
    lambda_risk: float,
    periods: int = C.N_PERIODS,
) -> MILPResult:
    """Solve the planning problem with SciPy's open-source HiGHS MILP backend."""
    reference_model = build_qubo(inputs, lambda_risk, periods=periods)
    objective_model = build_qubo(
        inputs,
        lambda_risk,
        periods=periods,
        penalty_override=0.0,
    )
    base_variable_count = int(objective_model.untreated.max()) + 1
    tolerance = C.MILP_COEFFICIENT_ZERO_TOL
    products = [
        (int(i), int(j), float(value))
        for (i, j), value in objective_model.qubo.quadratic.items()
        if abs(value) > tolerance
    ]
    total_variables = base_variable_count + len(products)
    objective = np.zeros(total_variables)
    for index, value in objective_model.qubo.linear.items():
        if index < base_variable_count and abs(value) > tolerance:
            objective[index] += value
    for product_index, (_i, _j, value) in enumerate(products, start=base_variable_count):
        objective[product_index] = value

    rows = _Rows()
    _hard_constraints(rows, reference_model)
    # Exact binary-product linearisation z=x_i*x_j.
    for product_index, (left, right, _value) in enumerate(products, start=base_variable_count):
        rows.add({product_index: 1.0, left: -1.0}, -np.inf, 0.0)
        rows.add({product_index: 1.0, right: -1.0}, -np.inf, 0.0)
        rows.add({product_index: -1.0, left: 1.0, right: 1.0}, -np.inf, 1.0)

    matrix = coo_matrix(
        (rows.data, (rows.row, rows.col)),
        shape=(len(rows.lower), total_variables),
    ).tocsc()
    constraints = LinearConstraint(
        matrix,
        np.asarray(rows.lower),
        np.asarray(rows.upper),
    )
    integrality = np.zeros(total_variables, dtype=np.int8)
    integrality[:base_variable_count] = 1
    started = perf_counter()
    result = milp(
        c=objective,
        integrality=integrality,
        bounds=Bounds(np.zeros(total_variables), np.ones(total_variables)),
        constraints=constraints,
        options={
            "disp": False,
            "presolve": True,
            "time_limit": C.MILP_TIME_LIMIT_SECONDS,
            "mip_rel_gap": C.MILP_RELATIVE_GAP,
        },
    )
    elapsed = perf_counter() - started
    if result.x is None:
        raise RuntimeError(f"HiGHS returned no feasible MILP solution: {result.message}")
    base = np.rint(result.x[:base_variable_count]).astype(np.int8)
    primary_bits = base[reference_model.primary]
    vector = complete_vector(reference_model, primary_bits)
    if vector is None:
        raise RuntimeError("MILP returned primary decisions that fail exact completion")
    violations = independent_feasibility_check(
        reference_model,
        primary_bits,
        full_vector=vector,
    )
    objective_vector = complete_vector(objective_model, primary_bits)
    if objective_vector is None:
        raise RuntimeError("MILP solution cannot be scored by the zero-penalty QUBO")
    objective_energy = objective_model.qubo.energy(objective_vector)
    reference_energy = reference_model.qubo.energy(vector)
    reported_objective = float(result.fun + objective_model.qubo.constant)
    if not np.isclose(
        reported_objective,
        objective_energy,
        atol=C.MILP_EQUIVALENCE_ABS_TOL,
        rtol=C.MILP_EQUIVALENCE_REL_TOL,
    ):
        raise RuntimeError(
            f"MILP/QUBO objective mismatch: {reported_objective} vs {objective_energy}"
        )
    if violations:
        raise RuntimeError(f"MILP failed independent feasibility check: {violations}")
    return MILPResult(
        model=reference_model,
        primary_bits=primary_bits,
        vector=vector,
        plan=plan_frame(reference_model, vector),
        objective_value=objective_energy,
        reference_qubo_energy=reference_energy,
        solve_time_seconds=elapsed,
        proven_optimal=int(result.status) == 0,
        status=int(result.status),
        message=str(result.message),
        mip_gap=float(getattr(result, "mip_gap", np.nan)),
        mip_dual_bound=float(getattr(result, "mip_dual_bound", np.nan)),
        mip_node_count=int(getattr(result, "mip_node_count", -1)),
        binary_variable_count=base_variable_count,
        product_variable_count=len(products),
        linear_constraint_count=len(rows.lower),
        violations=violations,
    )
