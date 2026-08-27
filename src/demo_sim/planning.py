"""Solver-derived benefit, uncertainty and load-path interaction calculations."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from . import constants as C
from .condition import (
    TwoTierState,
    TransportState,
    _psd_samples,
    acceptance_n_full,
    m_full,
    m_reduced,
    structural_loss,
)
from .frame import FrameModel


ACTIONS = ("cover", "jacket", "combined")
CAPACITY_ACTIONS = ("jacket", "combined")


@dataclass
class PlanningInputs:
    members: pd.DataFrame
    action_mean: np.ndarray  # member x action
    action_variance: np.ndarray
    interactions: pd.DataFrame
    needs_cover: np.ndarray
    full_mask: np.ndarray
    geometry_remaining_life: np.ndarray
    deterioration_cost: np.ndarray
    source: str = "illustrative"

    def point_estimate(self) -> "PlanningInputs":
        return replace(self, action_variance=np.zeros_like(self.action_variance))

    def without_interactions(self) -> "PlanningInputs":
        return replace(self, interactions=self.interactions.iloc[0:0].copy())


def _capacity(frame: FrameModel) -> np.ndarray:
    return np.array([
        C.COLUMN_MOMENT_CAPACITY if element.kind == "column" else C.BEAM_MOMENT_CAPACITY
        for element in frame.elements
    ])


def _representative_geometry(state: TwoTierState, alpha: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    width = np.where(state.full_mask, state.width_mean, C.CRACK_WIDTH_SCALE * np.maximum(1.0 - alpha, 0.0))
    theta = np.where(state.full_mask, state.theta_mean, 90.0)
    transport = np.where(state.full_mask, m_full(alpha, width, theta), m_reduced(alpha))
    return width, theta, transport


def scores_for_state(
    frame: FrameModel,
    alpha: np.ndarray,
    width: np.ndarray,
    theta: np.ndarray,
    transport: np.ndarray,
    full_mask: np.ndarray,
) -> np.ndarray:
    _, moments = frame.static_response(alpha)
    n_factor = np.empty(len(alpha))
    full = full_mask
    n_factor[full] = acceptance_n_full(alpha[full], width[full], theta[full], transport[full])
    reduced = ~full
    n_factor[reduced] = np.clip(
        1.0
        - C.N_ALPHA_WEIGHT * (1.0 - alpha[reduced])
        - C.N_TRANSPORT_WEIGHT * np.maximum(transport[reduced] - 1.0, 0.0),
        C.N_MINIMUM,
        1.0,
    )
    return moments / (_capacity(frame) * n_factor)


def apply_action(
    action: str,
    member: int,
    alpha: np.ndarray,
    width: np.ndarray,
    theta: np.ndarray,
    transport: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    a, w, th, m = alpha.copy(), width.copy(), theta.copy(), transport.copy()
    if action == "cover":
        w[member] = 0.0
        m[member] = 1.0
    elif action == "jacket":
        a[member] = min(1.0, a[member] + C.JACKET_RESTORE_FRACTION * (1.0 - a[member]))
    elif action == "combined":
        # The bundled action is internally sequenced: cover first, jacket second.
        w[member] = 0.0
        m[member] = 1.0
        a[member] = min(1.0, a[member] + C.COMBINED_RESTORE_FRACTION * (1.0 - a[member]))
    else:
        raise ValueError(f"Unknown action: {action}")
    return a, w, th, m


def build_planning_inputs(
    frame: FrameModel,
    state: TwoTierState,
    transport_state: TransportState,
    rng: np.random.Generator,
) -> PlanningInputs:
    del transport_state  # state samples are regenerated jointly below.
    n = len(frame.elements)
    count = C.PLANNING_MONTE_CARLO_SAMPLES
    alpha_samples = _psd_samples(state.alpha_mean, state.alpha_covariance, count, rng)
    benefits = np.zeros((count, n, len(ACTIONS)))
    for sample in range(count):
        alpha = alpha_samples[sample]
        width, theta, transport = _representative_geometry(state, alpha)
        # Sample admissible geometry; reduced geometry remains marginalised by its surrogate.
        full_idx = np.where(state.full_mask)[0]
        if len(full_idx):
            width[full_idx] = np.maximum(
                0.0, rng.normal(state.width_mean[full_idx], state.width_sd[full_idx])
            )
            theta[full_idx] = np.mod(
                rng.normal(state.theta_mean[full_idx], state.theta_sd[full_idx]), 180.0
            )
            transport[full_idx] = m_full(alpha[full_idx], width[full_idx], theta[full_idx])
        reduced_idx = np.where(~state.full_mask)[0]
        if len(reduced_idx):
            transport[reduced_idx] = np.maximum(
                1.0,
                m_reduced(alpha[reduced_idx]) * np.exp(
                    rng.normal(0.0, C.M_REDUCED_LOG_SD, len(reduced_idx))
                ),
            )
        base_loss = structural_loss(scores_for_state(frame, alpha, width, theta, transport, state.full_mask))
        for member in range(n):
            for action_index, action in enumerate(ACTIONS):
                modified = apply_action(action, member, alpha, width, theta, transport)
                new_loss = structural_loss(scores_for_state(frame, *modified, state.full_mask))
                benefits[sample, member, action_index] = C.BENEFIT_SCALE * (base_loss - new_loss)

    action_mean = benefits.mean(axis=0)
    action_variance = benefits.var(axis=0, ddof=1)

    # Pair effects are finite differences of actual frame solver calls at the
    # posterior centre. Nothing is inserted merely because members are adjacent.
    alpha = state.alpha_mean.copy()
    width, theta, transport = _representative_geometry(state, alpha)
    base_loss = structural_loss(scores_for_state(frame, alpha, width, theta, transport, state.full_mask))
    single: dict[tuple[int, str], float] = {}
    for member in range(n):
        for action in CAPACITY_ACTIONS:
            modified = apply_action(action, member, alpha, width, theta, transport)
            single[(member, action)] = C.BENEFIT_SCALE * (
                base_loss - structural_loss(scores_for_state(frame, *modified, state.full_mask))
            )
    rows: list[dict] = []
    for i in range(n):
        for j in range(i + 1, n):
            for action_i in CAPACITY_ACTIONS:
                for action_j in CAPACITY_ACTIONS:
                    state_i = apply_action(action_i, i, alpha, width, theta, transport)
                    state_ij = apply_action(action_j, j, *state_i)
                    pair_benefit = C.BENEFIT_SCALE * (
                        base_loss - structural_loss(scores_for_state(frame, *state_ij, state.full_mask))
                    )
                    interaction = pair_benefit - single[(i, action_i)] - single[(j, action_j)]
                    if abs(interaction) >= C.LOAD_PATH_INTERACTION_KEEP:
                        rows.append({
                            "member_i": i,
                            "member_j": j,
                            "action_i": action_i,
                            "action_j": action_j,
                            "solver_pair_benefit": pair_benefit,
                            "solver_single_benefit_sum": single[(i, action_i)] + single[(j, action_j)],
                            "interaction_utility": interaction,
                            "derivation": "paired frame-solver finite difference",
                            "source": "illustrative",
                        })
    interactions = pd.DataFrame(rows, columns=[
        "member_i", "member_j", "action_i", "action_j", "solver_pair_benefit",
        "solver_single_benefit_sum", "interaction_utility", "derivation", "source",
    ])
    needs_cover = m_reduced(state.alpha_mean) > C.TRANSPORT_POOR_LIMIT
    needs_cover[state.full_mask] = m_full(
        state.alpha_mean[state.full_mask], state.width_mean[state.full_mask], state.theta_mean[state.full_mask]
    ) > C.TRANSPORT_POOR_LIMIT
    age = state.table["geometry_age_periods"].fillna(C.GEOMETRY_VALID_PERIODS + 1).to_numpy()
    remaining = np.where(state.full_mask, np.maximum(C.GEOMETRY_VALID_PERIODS - age, 0), 0).astype(int)
    deterioration_cost = np.zeros(n)
    for member in range(n):
        deteriorated_alpha = alpha.copy()
        deteriorated_alpha[member] = max(
            C.ALPHA_LOWER_BOUND,
            deteriorated_alpha[member] - C.DETERIORATION_ALPHA_PER_PERIOD,
        )
        deteriorated_width, deteriorated_theta, deteriorated_transport = _representative_geometry(
            state, deteriorated_alpha
        )
        deterioration_cost[member] = max(
            C.DETERIORATION_OBJECTIVE_PENALTY,
            C.BENEFIT_SCALE * (
                structural_loss(scores_for_state(
                    frame,
                    deteriorated_alpha,
                    deteriorated_width,
                    deteriorated_theta,
                    deteriorated_transport,
                    state.full_mask,
                )) - base_loss
            ),
        )
    return PlanningInputs(
        members=state.table[["member_id", "member", "kind", "storey", "bay", "tier"]].copy(),
        action_mean=action_mean,
        action_variance=action_variance,
        interactions=interactions,
        needs_cover=needs_cover,
        full_mask=state.full_mask.copy(),
        geometry_remaining_life=remaining,
        deterioration_cost=deterioration_cost,
    )

