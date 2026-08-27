"""Constrained retrofit QUBO, annealer, and independent feasibility checks."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd

from . import constants as C
from .planning import ACTIONS, PlanningInputs


PRIMARY_ACTIONS = (*ACTIONS, "acquire_geometry")
ACTION_COST = {"cover": C.COVER_COST, "jacket": C.JACKET_COST, "combined": C.COMBINED_COST}


class QUBO:
    def __init__(self) -> None:
        self.linear: dict[int, float] = {}
        self.quadratic: dict[tuple[int, int], float] = {}
        self.constant = 0.0
        self.names: list[str] = []

    def add_variable(self, name: str) -> int:
        self.names.append(name)
        return len(self.names) - 1

    def add_linear(self, i: int, value: float) -> None:
        self.linear[i] = self.linear.get(i, 0.0) + float(value)

    def add_quadratic(self, i: int, j: int, value: float) -> None:
        if i == j:
            self.add_linear(i, value)
            return
        key = (i, j) if i < j else (j, i)
        self.quadratic[key] = self.quadratic.get(key, 0.0) + float(value)

    def add_square(self, constant: float, terms: dict[int, float], weight: float) -> None:
        self.constant += weight * constant**2
        items = list(terms.items())
        for i, a in items:
            self.add_linear(i, weight * (a * a + 2.0 * constant * a))
        for (i, a), (j, b) in combinations(items, 2):
            self.add_quadratic(i, j, weight * 2.0 * a * b)

    def energy(self, vector: np.ndarray) -> float:
        value = self.constant
        value += sum(v * vector[i] for i, v in self.linear.items())
        value += sum(v * vector[i] * vector[j] for (i, j), v in self.quadratic.items())
        return float(value)

    def adjacency(self) -> list[tuple[np.ndarray, np.ndarray]]:
        idx: list[list[int]] = [[] for _ in self.names]
        weights: list[list[float]] = [[] for _ in self.names]
        for (i, j), value in self.quadratic.items():
            idx[i].append(j); weights[i].append(value)
            idx[j].append(i); weights[j].append(value)
        return [
            (np.asarray(ii, dtype=int), np.asarray(ww, dtype=float))
            for ii, ww in zip(idx, weights)
        ]


@dataclass
class QUBOModel:
    qubo: QUBO
    inputs: PlanningInputs
    periods: int
    primary: np.ndarray  # member x period x 4 variable indices
    untreated: np.ndarray
    intervention_slack: list[list[tuple[int, int]]]
    acquisition_slack: list[list[tuple[int, int]]]
    outage_slack: list[list[tuple[int, int]]]
    penalty_base: float
    objective_abs_bound: float
    lambda_risk: float

    @property
    def n_members(self) -> int:
        return self.primary.shape[0]


@dataclass
class AnnealResult:
    vector: np.ndarray
    energy: float
    plan: pd.DataFrame
    feasible: bool
    violations: list[str]
    diagnostics: dict


@dataclass
class RawAnnealResult:
    """Result from unconstrained bitwise SA used only for penalty negative controls."""

    vector: np.ndarray
    energy: float
    plan: pd.DataFrame
    feasible: bool
    violations: list[str]
    diagnostics: dict


def _annuity(period: int, periods: int) -> float:
    return float(sum(C.DISCOUNT_FACTOR**k for k in range(period, periods)))


def build_qubo(
    inputs: PlanningInputs,
    lambda_risk: float,
    periods: int = C.N_PERIODS,
    penalty_override: float | None = None,
) -> QUBOModel:
    q = QUBO()
    n = len(inputs.members)
    primary = np.empty((n, periods, len(PRIMARY_ACTIONS)), dtype=int)
    for i in range(n):
        for t in range(periods):
            for a, action in enumerate(PRIMARY_ACTIONS):
                primary[i, t, a] = q.add_variable(f"x[{i},{t},{action}]")
    untreated = np.empty((n, periods), dtype=int)
    for i in range(n):
        for t in range(periods):
            untreated[i, t] = q.add_variable(f"y[{i},{t}]")

    intervention_slack: list[list[tuple[int, int]]] = []
    acquisition_slack: list[list[tuple[int, int]]] = []
    outage_slack: list[list[tuple[int, int]]] = []
    for t in range(periods):
        intervention_slack.append([
            (q.add_variable(f"s_intervention[{t},{k}]"), weight)
            for k, weight in enumerate((1, 2, 3, 4))
        ])
        acquisition_slack.append([
            (q.add_variable(f"s_acquisition[{t},{k}]"), 1) for k in range(C.ACQUISITION_BUDGET)
        ])
        outage_slack.append([
            (q.add_variable(f"s_outage[{t},{k}]"), 1) for k in range(C.OUT_OF_SERVICE_LIMIT)
        ])

    # Unconstrained utility terms first, enabling an auditable coefficient bound.
    for i in range(n):
        for t in range(periods):
            annuity = _annuity(t, periods)
            for a, action in enumerate(ACTIONS):
                mean = inputs.action_mean[i, a] * annuity
                variance = inputs.action_variance[i, a] * annuity**2
                q.add_linear(primary[i, t, a], ACTION_COST[action] - mean + lambda_risk * variance)
            q.add_linear(primary[i, t, 3], C.ACQUIRE_COST)
            q.add_linear(
                untreated[i, t],
                inputs.deterioration_cost[i] * (t + 1) * C.DISCOUNT_FACTOR**t,
            )
            # Information only reduces risk for a later intervention. Therefore
            # at lambda=0 acquisition has cost and no artificial direct benefit.
            for future in range(t + 1, periods):
                for a, _action in enumerate(ACTIONS):
                    current_geometry_still_valid = (
                        inputs.full_mask[i] and future <= inputs.geometry_remaining_life[i]
                    )
                    acquired_geometry_still_valid = (future - t) <= C.GEOMETRY_VALID_PERIODS
                    variance_reduction = (
                        lambda_risk
                        * C.ACQUISITION_VARIANCE_REDUCTION
                        * inputs.action_variance[i, a]
                        * _annuity(future, periods) ** 2
                    )
                    if current_geometry_still_valid or not acquired_geometry_still_valid:
                        variance_reduction = 0.0
                    q.add_quadratic(primary[i, t, 3], primary[i, future, a], -variance_reduction)

    if not inputs.interactions.empty:
        for row in inputs.interactions.itertuples(index=False):
            ai = ACTIONS.index(row.action_i)
            aj = ACTIONS.index(row.action_j)
            for ti in range(periods):
                for tj in range(periods):
                    realised = max(ti, tj)
                    q.add_quadratic(
                        primary[row.member_i, ti, ai],
                        primary[row.member_j, tj, aj],
                        -row.interaction_utility * _annuity(realised, periods),
                    )

    objective_abs_bound = float(
        sum(abs(v) for v in q.linear.values()) + sum(abs(v) for v in q.quadratic.values())
    )
    penalty = objective_abs_bound + 1.0 if penalty_override is None else float(penalty_override)
    strong = C.STRONG_PENALTY_MULTIPLIER * penalty

    # Catalogue and uniqueness constraints.
    for i in range(n):
        for t in range(periods):
            for a, b in combinations(range(3), 2):
                q.add_quadratic(primary[i, t, a], primary[i, t, b], strong)
        for action_index in range(4):
            for t, s in combinations(range(periods), 2):
                q.add_quadratic(primary[i, t, action_index], primary[i, s, action_index], strong)
        # Combined excludes separate cover or jacket anywhere in the horizon.
        for tc in range(periods):
            for ts in range(periods):
                q.add_quadratic(primary[i, tc, 2], primary[i, ts, 0], strong)
                q.add_quadratic(primary[i, tc, 2], primary[i, ts, 1], strong)
        # When separate cover and jacket actions are both selected, prohibit
        # jacket in the same or an earlier period. This encodes the physical
        # sequencing rule even for a member not initially flagged poor in transport.
        for cover_t in range(periods):
            for jacket_t in range(cover_t + 1):
                q.add_quadratic(primary[i, cover_t, 0], primary[i, jacket_t, 1], penalty)
        if inputs.needs_cover[i]:
            for t in range(periods):
                jacket = primary[i, t, 1]
                q.add_linear(jacket, penalty)
                for earlier in range(t):
                    q.add_quadratic(jacket, primary[i, earlier, 0], -penalty)
        # If acquisition is selected, all intervention on that member must be at least one period later.
        for ta in range(periods):
            for ti in range(ta + 1):
                for action_index in range(3):
                    q.add_quadratic(primary[i, ta, 3], primary[i, ti, action_index], penalty)

    # Conditional deterioration y_t = y_(t-1) AND NOT(any intervention at t-1).
    for i in range(n):
        q.add_square(1.0, {untreated[i, 0]: -1.0}, strong)
        for t in range(1, periods):
            prev, current = untreated[i, t - 1], untreated[i, t]
            q.add_linear(prev, 3.0 * penalty)
            q.add_quadratic(prev, current, -6.0 * penalty)
            q.add_linear(current, 3.0 * penalty)
            for a in range(3):
                intervention = primary[i, t - 1, a]
                q.add_quadratic(prev, intervention, -3.0 * penalty)
                q.add_quadratic(current, intervention, 6.0 * penalty)

    # Separate budgets and outage constraint, each equality-completed by slack bits.
    for t in range(periods):
        intervention_terms: dict[int, float] = {}
        acquisition_terms: dict[int, float] = {}
        outage_terms: dict[int, float] = {}
        for i in range(n):
            for a, action in enumerate(ACTIONS):
                intervention_terms[primary[i, t, a]] = ACTION_COST[action]
                outage_terms[primary[i, t, a]] = 1.0
            acquisition_terms[primary[i, t, 3]] = 1.0
        for index, weight in intervention_slack[t]:
            intervention_terms[index] = weight
        for index, weight in acquisition_slack[t]:
            acquisition_terms[index] = weight
        for index, weight in outage_slack[t]:
            outage_terms[index] = weight
        q.add_square(-C.INTERVENTION_BUDGET, intervention_terms, penalty)
        q.add_square(-C.ACQUISITION_BUDGET, acquisition_terms, penalty)
        q.add_square(-C.OUT_OF_SERVICE_LIMIT, outage_terms, penalty)

    return QUBOModel(
        q, inputs, periods, primary, untreated, intervention_slack, acquisition_slack,
        outage_slack, penalty, objective_abs_bound, lambda_risk,
    )


def raw_simulated_annealing(
    model: QUBOModel,
    rng: np.random.Generator,
    restarts: int = C.RAW_ANNEAL_RESTARTS,
    steps: int = C.RAW_ANNEAL_STEPS,
    initial_vector: np.ndarray | None = None,
) -> RawAnnealResult:
    """Anneal all QUBO bits without feasibility filtering or auxiliary repair.

    This is intentionally *not* the production solver.  It exists to demonstrate
    that weak penalty encodings can win the QUBO energy while violating the
    original constraints, after which the independent checker must reject them.
    """

    q = model.qubo
    adjacency = q.adjacency()
    best_vector: np.ndarray | None = None
    best_energy = np.inf
    accepted = 0
    proposed = 0
    for restart in range(restarts):
        if restart == 0 and initial_vector is not None:
            vector = initial_vector.copy()
        elif restart == 0:
            vector = np.zeros(len(q.names), dtype=np.int8)
        else:
            vector = rng.integers(0, 2, len(q.names), dtype=np.int8)
        energy = q.energy(vector)
        if energy < best_energy:
            best_energy, best_vector = energy, vector.copy()
        for step in range(steps):
            proposed += 1
            index = int(rng.integers(len(vector)))
            sign = 1.0 - 2.0 * vector[index]
            neighbours, weights = adjacency[index]
            field = q.linear.get(index, 0.0)
            if len(neighbours):
                field += float(np.dot(weights, vector[neighbours]))
            delta = sign * field
            temperature = C.RAW_ANNEAL_START_TEMP * (
                C.RAW_ANNEAL_END_TEMP / C.RAW_ANNEAL_START_TEMP
            ) ** (step / max(steps - 1, 1))
            if delta <= 0.0 or rng.random() < np.exp(-delta / max(temperature, 1.0e-12)):
                vector[index] ^= 1
                energy += delta
                accepted += 1
                if energy < best_energy:
                    best_energy, best_vector = energy, vector.copy()
    assert best_vector is not None
    primary = primary_array_from_vector(model, best_vector)
    violations = independent_feasibility_check(model, primary, full_vector=best_vector)
    return RawAnnealResult(
        vector=best_vector,
        energy=float(q.energy(best_vector)),
        plan=plan_frame(model, best_vector),
        feasible=not violations,
        violations=violations,
        diagnostics={
            "solver": "raw unconstrained full-bit simulated annealing negative control",
            "restarts": restarts,
            "steps_per_restart": steps,
            "proposals": proposed,
            "accepted": accepted,
            "acceptance_rate": accepted / max(proposed, 1),
            "variable_count": len(q.names),
            "source": "illustrative",
        },
    )


def primary_array_from_vector(model: QUBOModel, vector: np.ndarray) -> np.ndarray:
    return vector[model.primary]


def independent_feasibility_check(
    model: QUBOModel,
    primary_bits: np.ndarray,
    full_vector: np.ndarray | None = None,
) -> list[str]:
    """Check original hard constraints without reading QUBO coefficients."""
    violations: list[str] = []
    n, periods, _ = primary_bits.shape
    intervention = primary_bits[:, :, :3]
    acquisition = primary_bits[:, :, 3]
    for t in range(periods):
        spend = sum(
            ACTION_COST[action] * intervention[:, t, a].sum() for a, action in enumerate(ACTIONS)
        )
        if spend > C.INTERVENTION_BUDGET + 1.0e-9:
            violations.append(f"period {t}: intervention budget {spend}>{C.INTERVENTION_BUDGET}")
        if acquisition[:, t].sum() > C.ACQUISITION_BUDGET:
            violations.append(f"period {t}: acquisition budget exceeded")
        if intervention[:, t, :].sum() > C.OUT_OF_SERVICE_LIMIT:
            violations.append(f"period {t}: out-of-service limit exceeded")
    for i in range(n):
        if np.any(intervention[i].sum(axis=1) > 1):
            violations.append(f"member {i}: multiple intervention types in one period")
        if acquisition[i].sum() > 1:
            violations.append(f"member {i}: geometry acquired more than once")
        for a, action in enumerate(ACTIONS):
            if intervention[i, :, a].sum() > 1:
                violations.append(f"member {i}: {action} repeated")
        if intervention[i, :, 2].any() and intervention[i, :, :2].any():
            violations.append(f"member {i}: combined mixed with separate action")
        cover_periods = np.where(intervention[i, :, 0] == 1)[0]
        jacket_periods = np.where(intervention[i, :, 1] == 1)[0]
        if len(cover_periods) and len(jacket_periods) and not np.all(cover_periods < jacket_periods):
            violations.append(f"member {i}: separate cover is not before jacket")
        if model.inputs.needs_cover[i]:
            for jt in jacket_periods:
                if not np.any(cover_periods < jt):
                    violations.append(f"member {i}: jacket lacks earlier cover")
        acquisition_periods = np.where(acquisition[i] == 1)[0]
        intervention_periods = np.where(intervention[i].sum(axis=1) > 0)[0]
        for at in acquisition_periods:
            if np.any(intervention_periods <= at):
                violations.append(f"member {i}: intervention not at least one period after acquisition")
    if full_vector is not None:
        for i in range(n):
            expected_untreated = 1
            for t in range(periods):
                actual = int(full_vector[model.untreated[i, t]])
                if actual != expected_untreated:
                    violations.append(
                        f"member {i} period {t}: untreated auxiliary y={actual}, expected {expected_untreated}"
                    )
                if intervention[i, t, :].any():
                    expected_untreated = 0
        for t in range(periods):
            spend = int(sum(
                ACTION_COST[action] * intervention[:, t, a].sum()
                for a, action in enumerate(ACTIONS)
            ))
            acquisition_count = int(acquisition[:, t].sum())
            outage_count = int(intervention[:, t, :].sum())
            intervention_slack = sum(
                weight * int(full_vector[index]) for index, weight in model.intervention_slack[t]
            )
            acquisition_slack = sum(
                weight * int(full_vector[index]) for index, weight in model.acquisition_slack[t]
            )
            outage_slack = sum(
                weight * int(full_vector[index]) for index, weight in model.outage_slack[t]
            )
            if spend + intervention_slack != C.INTERVENTION_BUDGET:
                violations.append(
                    f"period {t}: intervention slack equality {spend}+{intervention_slack}!={C.INTERVENTION_BUDGET}"
                )
            if acquisition_count + acquisition_slack != C.ACQUISITION_BUDGET:
                violations.append(
                    f"period {t}: acquisition slack equality {acquisition_count}+{acquisition_slack}!={C.ACQUISITION_BUDGET}"
                )
            if outage_count + outage_slack != C.OUT_OF_SERVICE_LIMIT:
                violations.append(
                    f"period {t}: outage slack equality {outage_count}+{outage_slack}!={C.OUT_OF_SERVICE_LIMIT}"
                )
    return violations


def _encode_slack(target: int, slack: list[tuple[int, int]], vector: np.ndarray) -> bool:
    for mask in range(1 << len(slack)):
        if sum(weight for bit, (_, weight) in enumerate(slack) if (mask >> bit) & 1) == target:
            for bit, (index, _weight) in enumerate(slack):
                vector[index] = (mask >> bit) & 1
            return True
    return False


def complete_vector(model: QUBOModel, primary_bits: np.ndarray) -> np.ndarray | None:
    if independent_feasibility_check(model, primary_bits):
        return None
    vector = np.zeros(len(model.qubo.names), dtype=np.int8)
    vector[model.primary] = primary_bits
    n, periods, _ = primary_bits.shape
    for i in range(n):
        untreated = 1
        for t in range(periods):
            vector[model.untreated[i, t]] = untreated
            if primary_bits[i, t, :3].any():
                untreated = 0
    for t in range(periods):
        spend = int(sum(
            ACTION_COST[action] * primary_bits[:, t, a].sum() for a, action in enumerate(ACTIONS)
        ))
        acquisitions = int(primary_bits[:, t, 3].sum())
        outages = int(primary_bits[:, t, :3].sum())
        ok = _encode_slack(C.INTERVENTION_BUDGET - spend, model.intervention_slack[t], vector)
        ok &= _encode_slack(C.ACQUISITION_BUDGET - acquisitions, model.acquisition_slack[t], vector)
        ok &= _encode_slack(C.OUT_OF_SERVICE_LIMIT - outages, model.outage_slack[t], vector)
        if not ok:
            return None
    return vector


def plan_frame(model: QUBOModel, vector: np.ndarray) -> pd.DataFrame:
    primary = primary_array_from_vector(model, vector)
    rows = []
    for i in range(model.n_members):
        for t in range(model.periods):
            for a, action in enumerate(PRIMARY_ACTIONS):
                if primary[i, t, a]:
                    rows.append({
                        "member_id": i,
                        "member": model.inputs.members.iloc[i]["member"],
                        "period": t + 1,
                        "action": action,
                        "source": "illustrative",
                    })
    return pd.DataFrame(rows, columns=["member_id", "member", "period", "action", "source"])


def simulated_annealing(
    model: QUBOModel,
    rng: np.random.Generator,
    restarts: int = C.ANNEAL_RESTARTS,
    steps: int = C.ANNEAL_STEPS,
    initial_primary: np.ndarray | None = None,
) -> AnnealResult:
    adjacency = model.qubo.adjacency()

    def transition_delta(vector: np.ndarray, trial_vector: np.ndarray) -> float:
        changed = np.flatnonzero(vector != trial_vector)
        work = vector.copy()
        delta = 0.0
        for index in changed:
            sign = 1.0 - 2.0 * work[index]
            neighbours, weights = adjacency[index]
            field = model.qubo.linear.get(int(index), 0.0)
            if len(neighbours):
                field += float(np.dot(weights, work[neighbours]))
            delta += sign * field
            work[index] ^= 1
        return delta

    best_vector: np.ndarray | None = None
    best_energy = np.inf
    accepted = 0
    proposed = 0
    for restart in range(restarts):
        bits = (
            initial_primary.copy()
            if initial_primary is not None
            else np.zeros(model.primary.shape, dtype=np.int8)
        )
        # Diversify later restarts with feasible random toggles.
        for _ in range(restart * max(1, model.n_members // 3)):
            trial = bits.copy()
            flat = rng.integers(trial.size)
            trial.reshape(-1)[flat] ^= 1
            if not independent_feasibility_check(model, trial):
                bits = trial
        vector = complete_vector(model, bits)
        assert vector is not None
        energy = model.qubo.energy(vector)
        if energy < best_energy:
            best_energy, best_vector = energy, vector.copy()
        for step in range(steps):
            proposed += 1
            trial_bits = bits.copy()
            proposal = rng.random()
            if proposal < C.SINGLE_FLIP_PROBABILITY:
                flat = int(rng.integers(trial_bits.size))
                trial_bits.reshape(-1)[flat] ^= 1
            elif proposal < C.SINGLE_FLIP_PROBABILITY + C.MOVE_ACTION_PROBABILITY:
                selected = np.argwhere(trial_bits == 1)
                if len(selected):
                    i, old_t, action = selected[int(rng.integers(len(selected)))]
                    new_t = int(rng.integers(model.periods))
                    trial_bits[i, old_t, action] = 0
                    trial_bits[i, new_t, action] = 1
                else:
                    trial_bits.reshape(-1)[int(rng.integers(trial_bits.size))] ^= 1
            elif proposal < (
                C.SINGLE_FLIP_PROBABILITY + C.MOVE_ACTION_PROBABILITY + C.ACQUIRE_PAIR_PROBABILITY
            ) and model.periods > 1:
                i = int(rng.integers(model.n_members))
                acquisition_t = int(rng.integers(model.periods - 1))
                intervention_t = int(rng.integers(acquisition_t + 1, model.periods))
                action = int(rng.integers(3))
                trial_bits[i, acquisition_t, 3] ^= 1
                trial_bits[i, intervention_t, action] ^= 1
            elif model.periods > 1:
                i = int(rng.integers(model.n_members))
                cover_t = int(rng.integers(model.periods - 1))
                jacket_t = int(rng.integers(cover_t + 1, model.periods))
                trial_bits[i, cover_t, 0] ^= 1
                trial_bits[i, jacket_t, 1] ^= 1
            else:
                trial_bits.reshape(-1)[int(rng.integers(trial_bits.size))] ^= 1
            if independent_feasibility_check(model, trial_bits):
                continue
            trial_vector = complete_vector(model, trial_bits)
            if trial_vector is None:
                continue
            delta = transition_delta(vector, trial_vector)
            temperature = C.ANNEAL_START_TEMP * (
                C.ANNEAL_END_TEMP / C.ANNEAL_START_TEMP
            ) ** (step / max(steps - 1, 1))
            if delta <= 0.0 or rng.random() < np.exp(-delta / max(temperature, 1.0e-12)):
                bits, vector, energy = trial_bits, trial_vector, energy + delta
                accepted += 1
                if energy < best_energy:
                    best_energy, best_vector = energy, vector.copy()
    assert best_vector is not None
    # Deterministic feasible cleanup guarantees, among other things, that a
    # cost-only epistemic bit cannot survive in the point-estimate ablation.
    best_bits = primary_array_from_vector(model, best_vector).copy()
    polish_improvements = 0
    for _pass in range(C.LOCAL_POLISH_PASSES):
        best_delta = 0.0
        best_trial_bits = None
        best_trial_vector = None
        for flat in range(best_bits.size):
            trial_bits = best_bits.copy()
            trial_bits.reshape(-1)[flat] ^= 1
            if independent_feasibility_check(model, trial_bits):
                continue
            trial_vector = complete_vector(model, trial_bits)
            if trial_vector is None:
                continue
            delta = transition_delta(best_vector, trial_vector)
            if delta < best_delta - 1.0e-9:
                best_delta = delta
                best_trial_bits = trial_bits
                best_trial_vector = trial_vector
        if best_trial_vector is None:
            break
        best_bits = best_trial_bits
        best_vector = best_trial_vector
        best_energy += best_delta
        polish_improvements += 1
    primary = primary_array_from_vector(model, best_vector)
    violations = independent_feasibility_check(model, primary, full_vector=best_vector)
    return AnnealResult(
        best_vector,
        float(model.qubo.energy(best_vector)),
        plan_frame(model, best_vector),
        not violations,
        violations,
        {
            "restarts": restarts,
            "steps_per_restart": steps,
            "proposals": proposed,
            "accepted": accepted,
            "acceptance_rate": accepted / max(proposed, 1),
            "variable_count": len(model.qubo.names),
            "primary_variable_count": int(model.primary.size),
            "quadratic_term_count": len(model.qubo.quadratic),
            "local_polish_improvements": polish_improvements,
            "source": "illustrative",
        },
    )


def exact_reduced_check(inputs: PlanningInputs, rng: np.random.Generator) -> dict:
    # One member and two periods retain all four requested decision variables,
    # acquisition delay, deterioration y, budgets and precedence, while keeping
    # exact enumeration transparent (2^8 primary assignments).
    chosen = int(np.argmax(inputs.needs_cover.astype(int) * 2 + inputs.action_mean[:, 2]))
    reduced_members = inputs.members.iloc[[chosen]].copy().reset_index(drop=True)
    reduced_interactions = inputs.interactions.iloc[0:0].copy()
    reduced = PlanningInputs(
        reduced_members,
        inputs.action_mean[[chosen]],
        inputs.action_variance[[chosen]],
        reduced_interactions,
        inputs.needs_cover[[chosen]],
        inputs.full_mask[[chosen]],
        inputs.geometry_remaining_life[[chosen]],
        inputs.deterioration_cost[[chosen]],
    )
    model = build_qubo(reduced, C.RISK_LAMBDA_REFERENCE, periods=2)
    best_energy = np.inf
    best_vector = None
    feasible_count = 0
    for mask in range(1 << model.primary.size):
        bits = np.array([(mask >> k) & 1 for k in range(model.primary.size)], dtype=np.int8).reshape(model.primary.shape)
        vector = complete_vector(model, bits)
        if vector is None:
            continue
        feasible_count += 1
        energy = model.qubo.energy(vector)
        if energy < best_energy:
            best_energy, best_vector = energy, vector
    annealed = simulated_annealing(
        model,
        rng,
        restarts=C.EXACT_CHECK_RESTARTS,
        steps=C.EXACT_CHECK_STEPS,
    )
    match = bool(np.isclose(annealed.energy, best_energy, atol=1.0e-7))
    return {
        "member_from_full_instance": chosen,
        "primary_assignments_enumerated": 1 << model.primary.size,
        "feasible_assignments": feasible_count,
        "exact_optimum_energy": float(best_energy),
        "annealed_energy": float(annealed.energy),
        "annealer_reached_exact_optimum": match,
        "exact_plan": plan_frame(model, best_vector).to_dict(orient="records") if best_vector is not None else [],
        "annealed_plan": annealed.plan.to_dict(orient="records"),
        "source": "illustrative",
    }
