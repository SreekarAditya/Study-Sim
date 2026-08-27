"""Hidden synthetic ground-truth state generation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import constants as C
from .frame import FrameModel


@dataclass
class GroundTruth:
    alpha: np.ndarray
    crack_width_mm: np.ndarray
    theta_deg: np.ndarray
    table: pd.DataFrame


def generate_ground_truth(frame: FrameModel, rng: np.random.Generator) -> GroundTruth:
    n = len(frame.elements)
    # A mild background field plus deliberately clustered severe regions creates
    # load-path interactions without hard-coding an intervention answer.
    alpha = rng.uniform(C.ALPHA_MILD_MIN, 1.0, n)
    severe = np.array(C.SEVERE_MEMBER_IDS)
    alpha[severe] = rng.uniform(C.ALPHA_MIN_DRAW, C.ALPHA_SEVERE_MAX, len(severe))
    moderate = np.array(C.MODERATE_MEMBER_IDS)
    alpha[moderate] = rng.uniform(C.ALPHA_MODERATE_MIN, C.ALPHA_MILD_MIN, len(moderate))
    width = np.maximum(0.0, C.CRACK_WIDTH_SCALE * (1.0 - alpha) + rng.normal(0.0, C.CRACK_WIDTH_NOISE, n))
    # Mixed orientations ensure stiffness and transport do not rank members identically.
    theta = np.mod(
        C.THETA_BASE + C.THETA_MEMBER_STEP * np.arange(n) + rng.normal(0.0, C.THETA_TRUE_SCATTER, n),
        180.0,
    )
    table = frame.member_table()
    table["true_alpha"] = alpha
    table["true_crack_width_mm"] = width
    table["true_theta_deg"] = theta
    table["source"] = "illustrative"
    return GroundTruth(alpha, width, theta, table)


def generate_dense_ground_truth(frame: FrameModel, rng: np.random.Generator) -> GroundTruth:
    """Deliberately violate sparsity by damaging essentially every member."""
    n = len(frame.elements)
    alpha = rng.uniform(C.DENSE_DAMAGE_ALPHA_MIN, C.DENSE_DAMAGE_ALPHA_MAX, n)
    # Add a storey-wide pattern while keeping damage distributed over the rest.
    second_storey = np.array([e.storey == 2 for e in frame.elements])
    alpha[second_storey] = rng.uniform(
        C.DENSE_DAMAGE_ALPHA_MIN,
        (C.DENSE_DAMAGE_ALPHA_MIN + C.DENSE_DAMAGE_ALPHA_MAX) / 2.0,
        int(second_storey.sum()),
    )
    width = np.maximum(
        0.0,
        C.CRACK_WIDTH_SCALE * (1.0 - alpha) + rng.normal(0.0, C.CRACK_WIDTH_NOISE, n),
    )
    theta = np.mod(
        C.THETA_BASE + C.THETA_MEMBER_STEP * np.arange(n) + rng.normal(0.0, C.THETA_TRUE_SCATTER, n),
        180.0,
    )
    table = frame.member_table()
    table["true_alpha"] = alpha
    table["true_crack_width_mm"] = width
    table["true_theta_deg"] = theta
    table["ground_truth_case"] = "dense_damage_sparsity_false"
    table["source"] = "illustrative"
    return GroundTruth(alpha, width, theta, table)
