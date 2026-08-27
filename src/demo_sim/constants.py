"""Single registry for every empirical constant used by the demonstration.

The values in this module are deliberately synthetic.  ``source`` is fixed to
``illustrative`` by the constructor, so an empirical value cannot be registered
without carrying the required warning into code and outputs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class IllustrativeConstant:
    name: str
    value: Any
    units: str
    stage: str
    origin: str
    source: str = "illustrative"
    illustrative: bool = True

    def __post_init__(self) -> None:
        if self.source != "illustrative" or not self.illustrative:
            raise ValueError("Every empirical constant must be tagged illustrative")


REGISTRY: dict[str, IllustrativeConstant] = {}


def _i(name: str, value: Any, units: str, stage: str, origin: str) -> Any:
    REGISTRY[name] = IllustrativeConstant(name, value, units, stage, origin)
    return value


# Reproducibility and frame definition.
SEED = _i("random_seed", 4817, "-", "global", "Fixed for reproducible synthetic output; engineering judgement.")
N_STOREYS = _i("number_of_storeys", 3, "-", "ground_truth", "Chosen to give a non-trivial but inspectable frame.")
N_BAYS = _i("number_of_bays", 3, "-", "ground_truth", "Chosen with three storeys to produce 21 members.")
BAY_WIDTH = _i("bay_width", 4.5, "m", "ground_truth", "Plausible demonstration geometry; engineering judgement.")
STOREY_HEIGHT = _i("storey_height", 3.2, "m", "ground_truth", "Plausible demonstration geometry; engineering judgement.")
E_CONCRETE = _i("elastic_modulus", 28.0e9, "Pa", "ground_truth", "Plausible RC elastic modulus from common design ranges; not calibrated.")
COLUMN_AREA = _i("column_area", 0.16, "m2", "ground_truth", "Equivalent 0.4 m square column; engineering judgement.")
COLUMN_INERTIA = _i("column_second_moment", 0.00213, "m4", "ground_truth", "Equivalent 0.4 m square column; engineering judgement.")
BEAM_AREA = _i("beam_area", 0.15, "m2", "ground_truth", "Equivalent 0.3 x 0.5 m beam; engineering judgement.")
BEAM_INERTIA = _i("beam_second_moment", 0.003125, "m4", "ground_truth", "Equivalent 0.3 x 0.5 m beam; engineering judgement.")
FLOOR_NODE_MASS = _i("floor_node_mass", 24000.0, "kg", "ground_truth", "Plausible lumped tributary floor mass; engineering judgement.")
ROTATIONAL_MASS = _i("rotational_mass", 1200.0, "kg m2", "ground_truth", "Numerical rotational inertia with plausible scale; engineering judgement.")
ALPHA_MIN_DRAW = _i("true_alpha_minimum", 0.52, "-", "ground_truth", "Synthetic damage range selected to span mild to severe states.")
ALPHA_MILD_MIN = _i("true_alpha_mild_minimum", 0.86, "-", "ground_truth", "Synthetic mild-damage band; engineering judgement.")
ALPHA_SEVERE_MAX = _i("true_alpha_severe_maximum", 0.78, "-", "ground_truth", "Synthetic severe-damage band upper edge.")
ALPHA_MODERATE_MIN = _i("true_alpha_moderate_minimum", 0.74, "-", "ground_truth", "Synthetic moderate-damage band lower edge.")
SEVERE_MEMBER_IDS = _i("severe_member_ids", [1, 4, 6, 9, 13, 17, 19], "member ids", "ground_truth", "Fixed clustered synthetic damage scenario.")
MODERATE_MEMBER_IDS = _i("moderate_member_ids", [2, 7, 11, 15, 20], "member ids", "ground_truth", "Fixed distributed synthetic damage scenario.")
CRACK_WIDTH_SCALE = _i("crack_width_damage_scale", 1.45, "mm", "ground_truth", "Synthetic mapping from stiffness loss to crack width; engineering judgement.")
CRACK_WIDTH_NOISE = _i("true_crack_width_scatter", 0.06, "mm", "ground_truth", "Synthetic member-to-member scatter; engineering judgement.")
THETA_BASE = _i("crack_orientation_base", 18.0, "degrees", "ground_truth", "Synthetic orientation pattern offset.")
THETA_MEMBER_STEP = _i("crack_orientation_member_step", 31.0, "degrees/member", "ground_truth", "Synthetic pattern chosen to span orientations.")
THETA_TRUE_SCATTER = _i("true_crack_orientation_scatter", 8.0, "degrees", "ground_truth", "Synthetic member-to-member scatter.")

# Sensing and environmental response.
N_MODES = _i("number_of_measured_modes", 4, "-", "sensing", "Four modes provide sparse modal features without resolving all members.")
SENSOR_BAYS = _i("lateral_sensor_bays", [1, 3], "bay ids", "sensing", "Two lateral sensors per elevated floor; sparse layout choice.")
N_ENV_OBSERVATIONS = _i("environmental_observations", 48, "-", "sensing", "Synthetic monitoring window; engineering judgement.")
FREQUENCY_NOISE_FRACTION = _i("frequency_noise_fraction", 0.01, "fraction", "sensing", "Specified by the task as the starting frequency noise.")
MODE_SHAPE_NOISE = _i("mode_shape_component_noise", 0.02, "mass-normalised component", "sensing", "Specified by the task as the starting mode-shape noise.")
TEMPERATURE_MEAN = _i("temperature_mean", 27.0, "degC", "sensing", "Plausible warm-climate monitoring centre; engineering judgement.")
TEMPERATURE_AMPLITUDE = _i("temperature_amplitude", 8.0, "degC", "sensing", "Environmental swing chosen to rival mild damage effects.")
HUMIDITY_MEAN = _i("relative_humidity_mean", 68.0, "%", "sensing", "Plausible monitoring centre; engineering judgement.")
HUMIDITY_AMPLITUDE = _i("relative_humidity_amplitude", 18.0, "%", "sensing", "Environmental swing chosen to rival mild damage effects.")
TEMP_FREQUENCY_COEFF = _i("temperature_frequency_coefficient", -0.00125, "fraction/degC", "sensing", "Plausible illustrative environmental sensitivity; engineering judgement.")
HUMIDITY_FREQUENCY_COEFF = _i("humidity_frequency_coefficient", 0.00042, "fraction/%", "sensing", "Plausible illustrative environmental sensitivity; engineering judgement.")
TEMP_QUADRATIC_COEFF = _i("temperature_quadratic_coefficient", -0.000035, "fraction/degC2", "sensing", "Small omitted nonlinearity added to leave residual contamination.")
ENV_MODE_SCALE_STEP = _i("environment_mode_scale_step", 0.08, "fraction/mode", "sensing", "Creates slightly mode-dependent environmental sensitivity.")
TEMPERATURE_RANDOM_SD = _i("temperature_random_scatter", 0.8, "degC", "sensing", "Synthetic short-term sensor/environment scatter.")
HUMIDITY_RANDOM_SD = _i("humidity_random_scatter", 2.0, "%", "sensing", "Synthetic short-term sensor/environment scatter.")
HUMIDITY_PHASE_OFFSET = _i("humidity_cycle_phase_offset", 0.55, "radians", "sensing", "Synthetic phase lag relative to temperature.")

# Inverse problem and two-tier state.
ALPHA_PRIOR_MEAN = _i("alpha_prior_mean", 0.90, "-", "state_estimation", "Weak regularising prior centred on modest damage; engineering judgement.")
ALPHA_PRIOR_SD = _i("alpha_prior_sd", 0.18, "-", "state_estimation", "Broad prior used to regularise the sparse inverse problem.")
ALPHA_LOWER_BOUND = _i("alpha_estimation_lower_bound", 0.35, "-", "state_estimation", "Bound avoids nonphysical singular trial frames.")
IDENTIFIABLE_SD_LIMIT = _i("identifiable_sd_limit", 0.17, "-", "state_estimation", "Operational synthetic identifiability flag, not a physical threshold.")
IDENTIFIABLE_CORR_LIMIT = _i("identifiable_correlation_limit", 0.82, "-", "state_estimation", "Operational synthetic separability flag.")
TRADEOFF_CORR_LIMIT = _i("tradeoff_correlation_limit", 0.70, "-", "state_estimation", "Operational threshold for reporting parameter trade-offs.")
CREDIBLE_INTERVAL_Z_90 = _i("credible_interval_normal_z_90", 1.645, "standard deviations", "state_estimation", "Normal/Laplace-approximation multiplier for illustrative central 90 percent member intervals.")
STATE_ESTIMATION_MAX_EVALUATIONS = _i("state_estimation_max_function_evaluations", 260, "evaluations", "state_estimation", "Nonlinear solver compute limit for the demonstration.")
SPARSE_PRIOR_HYPER_SHAPE = _i("sparse_prior_precision_hyper_shape", 1.0, "-", "state_estimation", "Weak Gamma hyperprior shape for the inferred Laplace precision; illustrative hierarchical-prior choice.")
SPARSE_PRIOR_HYPER_RATE = _i("sparse_prior_precision_hyper_rate", 0.50, "stiffness loss", "state_estimation", "Weak Gamma hyperprior rate preventing a degenerate infinite precision; illustrative hierarchical-prior choice.")
SPARSE_PRIOR_SMOOTHING = _i("sparse_prior_laplace_smoothing", 0.015, "stiffness loss", "state_estimation", "Differentiable approximation scale for the hierarchical Laplace prior.")
SPARSE_PRIOR_MAX_ITERATIONS = _i("sparse_prior_max_iterations", 30, "iterations", "state_estimation", "Maximum empirical-Bayes IRLS updates.")
SPARSE_PRIOR_CONVERGENCE_TOL = _i("sparse_prior_convergence_tolerance", 0.0005, "relative change", "state_estimation", "Stopping tolerance for joint stiffness loss and precision updates.")
HORSESHOE_GLOBAL_HALF_CAUCHY_SCALE = _i("horseshoe_global_half_cauchy_scale", 0.15, "stiffness loss", "state_estimation", "Illustrative weak global-scale hyperprior for the local-global horseshoe alternative.")
HORSESHOE_LOCAL_SMOOTHING = _i("horseshoe_local_scale_smoothing", 0.01, "stiffness loss", "state_estimation", "Numerical regularisation of the horseshoe spike at zero loss.")
HORSESHOE_GLOBAL_SCALE_MIN = _i("horseshoe_global_scale_minimum", 0.005, "stiffness loss", "state_estimation", "Numerical lower bound for inferred global horseshoe scale.")
HORSESHOE_GLOBAL_SCALE_MAX = _i("horseshoe_global_scale_maximum", 1.0, "stiffness loss", "state_estimation", "Numerical upper bound for inferred global horseshoe scale.")
HORSESHOE_MAX_ITERATIONS = _i("horseshoe_max_iterations", 30, "iterations", "state_estimation", "Maximum empirical-Bayes local-global horseshoe updates.")
HORSESHOE_CONVERGENCE_TOL = _i("horseshoe_convergence_tolerance", 0.0005, "relative change", "state_estimation", "Stopping tolerance for horseshoe state and scale updates.")
DENSE_DAMAGE_ALPHA_MIN = _i("dense_case_alpha_minimum", 0.64, "-", "ground_truth_dense_case", "Lower edge of the deliberately non-sparse damage case.")
DENSE_DAMAGE_ALPHA_MAX = _i("dense_case_alpha_maximum", 0.86, "-", "ground_truth_dense_case", "Upper edge of the deliberately non-sparse damage case spanning most members.")
GEOMETRY_VALID_PERIODS = _i("geometry_expiry_periods", 2, "periods", "two_tier_state", "Requested finite geometry lifetime; illustrative policy choice.")
GEOMETRY_COVERAGE_RATE = _i("geometry_record_probability", 0.57, "fraction", "two_tier_state", "Creates a mixed full/reduced demonstration population.")
GEOMETRY_WIDTH_SD = _i("geometry_crack_width_sd", 0.08, "mm", "two_tier_state", "Illustrative field geometry measurement uncertainty.")
GEOMETRY_ANGLE_SD = _i("geometry_orientation_sd", 7.0, "degrees", "two_tier_state", "Illustrative field orientation measurement uncertainty.")
REDUCED_ALPHA_UNCERTAINTY_FACTOR = _i("reduced_tier_alpha_sd_multiplier", 1.65, "-", "two_tier_state", "Inflation for members lacking admissible geometry.")
NEW_DAMAGE_EVENT_RATE = _i("new_damage_event_probability", 0.14, "fraction", "two_tier_state", "Synthetic invalidation rate for geometry records.")

# Transport multiplier M.
M_ALPHA_COEFF = _i("M_full_alpha_coefficient", 2.2, "-", "transport", "Illustrative monotone stiffness-damage contribution.")
M_ALPHA_POWER = _i("M_full_alpha_power", 1.35, "-", "transport", "Illustrative nonlinear stiffness-damage exponent.")
M_WIDTH_COEFF = _i("M_full_width_coefficient", 1.75, "1/mm", "transport", "Illustrative crack-width contribution informed by engineering judgement.")
M_WIDTH_POWER = _i("M_full_width_power", 1.18, "-", "transport", "Illustrative crack-width nonlinearity.")
M_ORIENTATION_WEIGHT = _i("M_full_orientation_weight", 0.65, "-", "transport", "Illustrative orientation modulation of transport.")
M_REDUCED_ALPHA_COEFF = _i("M_reduced_alpha_coefficient", 5.0, "-", "transport", "Synthetic reduced-form surrogate after marginalising missing geometry.")
M_REDUCED_ALPHA_POWER = _i("M_reduced_alpha_power", 1.10, "-", "transport", "Synthetic reduced-form exponent.")
M_REDUCED_LOG_SD = _i("M_reduced_log_sd", 0.42, "log-ratio", "transport", "Inflated irreducible uncertainty for missing geometry.")
TRANSPORT_POOR_LIMIT = _i("poor_transport_multiplier_limit", 1.75, "ratio", "transport", "Illustrative decision threshold for durability intervention.")
CAPACITY_DAMAGE_ALPHA_LIMIT = _i("capacity_damage_alpha_limit", 0.80, "alpha", "retrofit", "Illustrative two-axis logic threshold for stiffness intervention.")

# Performance modifier N and acceptance model.
DESIGN_STOREY_FORCE = _i("design_storey_force", 155000.0, "N/storey", "performance", "Fixed synthetic lateral demand; engineering judgement.")
COLUMN_MOMENT_CAPACITY = _i("column_nominal_moment_capacity", 315000.0, "N m", "performance", "Illustrative nominal capacity, not a tested RC value.")
BEAM_MOMENT_CAPACITY = _i("beam_nominal_moment_capacity", 245000.0, "N m", "performance", "Illustrative nominal capacity, not a tested RC value.")
N_ALPHA_WEIGHT = _i("N_alpha_weight", 0.52, "-", "performance", "Illustrative acceptance reduction from stiffness damage.")
N_WIDTH_WEIGHT = _i("N_width_weight", 0.12, "1/mm", "performance", "Illustrative acceptance reduction from crack width.")
N_TRANSPORT_WEIGHT = _i("N_transport_weight", 0.055, "-", "performance", "Illustrative acceptance reduction from transport condition.")
N_ORIENTATION_WEIGHT = _i("N_orientation_weight", 0.08, "-", "performance", "Illustrative orientation contribution to acceptance reduction.")
N_MINIMUM = _i("N_minimum", 0.42, "-", "performance", "Floor prevents negative illustrative acceptance factors.")
PERFORMANCE_THRESHOLDS = _i("performance_level_thresholds", [0.65, 0.90, 1.15], "utilisation", "performance", "Illustrative IO/LS/CP boundaries for pipeline testing.")
CONFORMAL_MIS_COVERAGE = _i("conformal_miscoverage", 0.10, "fraction", "performance", "Nominal 90 percent split-conformal target.")
CALIBRATION_SIZE = _i("conformal_calibration_size", 240, "cases", "performance", "Synthetic calibration sample size.")
VALIDATION_SIZE = _i("conformal_validation_size", 600, "cases", "performance", "Synthetic held-out coverage-check sample size.")
CALIBRATION_NOISE_SD = _i("calibration_prediction_noise", 0.115, "utilisation", "performance", "Synthetic score error for distribution-free calibration.")
CALIBRATION_SCORE_MIN = _i("calibration_score_minimum", 0.35, "utilisation", "performance", "Synthetic calibration support lower edge.")
CALIBRATION_SCORE_MAX = _i("calibration_score_maximum", 1.35, "utilisation", "performance", "Synthetic calibration support upper edge.")
PERFORMANCE_MONTE_CARLO_SAMPLES = _i("performance_monte_carlo_samples", 320, "samples", "performance", "Compute-precision choice for propagated classification uncertainty.")
REDUCED_N_EXTRA_SD = _i("reduced_N_extra_uncertainty", 0.055, "acceptance factor", "performance", "Additional uncertainty when crack geometry is unavailable.")

# Retrofit catalogue, horizon and deterioration.
N_PERIODS = _i("planning_periods", 6, "periods", "retrofit", "Requested approximate six-period horizon.")
DISCOUNT_FACTOR = _i("period_discount_factor", 0.94, "-", "retrofit", "Illustrative time preference.")
COVER_COST = _i("cover_action_cost", 3.0, "cost units", "retrofit", "Illustrative relative intervention cost.")
JACKET_COST = _i("jacket_action_cost", 5.0, "cost units", "retrofit", "Illustrative relative intervention cost.")
COMBINED_COST = _i("combined_action_cost", 7.0, "cost units", "retrofit", "Illustrative bundle cost with sequencing internal to action.")
ACQUIRE_COST = _i("geometry_acquisition_cost", 0.55, "cost units", "retrofit", "Illustrative epistemic action cost.")
INTERVENTION_BUDGET = _i("intervention_budget_per_period", 10, "cost units", "retrofit", "Separate illustrative per-period intervention budget.")
ACQUISITION_BUDGET = _i("acquisition_budget_per_period", 2, "acquisitions", "retrofit", "Separate illustrative per-period acquisition budget.")
OUT_OF_SERVICE_LIMIT = _i("simultaneous_out_of_service_limit", 2, "members", "retrofit", "Illustrative operational constraint.")
JACKET_RESTORE_FRACTION = _i("jacket_stiffness_restore_fraction", 0.72, "fraction of loss", "retrofit", "Illustrative partial stiffness restoration.")
COMBINED_RESTORE_FRACTION = _i("combined_stiffness_restore_fraction", 0.82, "fraction of loss", "retrofit", "Illustrative combined-action stiffness restoration.")
DETERIORATION_ALPHA_PER_PERIOD = _i("untreated_alpha_loss_per_period", 0.018, "alpha/period", "retrofit", "Illustrative conditional deterioration increment.")
DETERIORATION_OBJECTIVE_PENALTY = _i("untreated_state_cost_per_period", 0.48, "utility units", "retrofit", "Illustrative cost assigned to continued untreated exposure.")
BENEFIT_SCALE = _i("retrofit_benefit_scale", 10.0, "utility/loss", "retrofit", "Scales structural loss reduction into planning utility.")
RISK_LAMBDA_REFERENCE = _i("reference_risk_lambda", 0.002, "utility/variance", "retrofit", "Illustrative reference risk preference scaled to the synthetic benefit variance.")
LAMBDA_SWEEP = _i("risk_lambda_sweep", [0.0, 0.0005, 0.001, 0.002, 0.005, 0.01], "utility/variance", "retrofit", "Illustrative sweep spanning risk-neutral to risk-averse planning.")
LOAD_PATH_INTERACTION_KEEP = _i("interaction_absolute_keep_threshold", 0.004, "utility", "retrofit", "Numerical reporting threshold for solver-derived pair effects.")
ACQUISITION_VARIANCE_REDUCTION = _i("geometry_acquisition_variance_reduction", 0.68, "fraction", "retrofit", "Illustrative expected variance reduction after new geometry.")
PLANNING_MONTE_CARLO_SAMPLES = _i("planning_monte_carlo_samples", 72, "samples", "retrofit", "Compute-precision choice for action benefit moments.")

# Annealing controls are algorithm settings but kept in the same table so every
# fixed value used by a run is visible. They remain tagged illustrative.
ANNEAL_RESTARTS = _i("annealing_restarts", 3, "-", "solver", "Compute-quality choice for each candidate-generating anneal.")
ANNEAL_STEPS = _i("annealing_steps_per_restart", 7000, "proposals", "solver", "Compute-quality choice for each candidate-generating anneal.")
ANNEAL_START_TEMP = _i("annealing_start_temperature", 6.0, "energy", "solver", "Initial temperature for structured feasible QUBO annealing.")
ANNEAL_END_TEMP = _i("annealing_end_temperature", 0.015, "energy", "solver", "Final temperature for structured feasible QUBO annealing.")
STRONG_PENALTY_MULTIPLIER = _i("strong_penalty_multiplier", 5.0, "base penalties", "solver", "Extra separation for Boolean-domain constraints that guard other reductions.")
EXACT_CHECK_RESTARTS = _i("exact_check_annealing_restarts", 8, "-", "solver", "Reduced-instance verification compute setting.")
EXACT_CHECK_STEPS = _i("exact_check_steps_per_restart", 5000, "proposals", "solver", "Reduced-instance verification compute setting.")
SINGLE_FLIP_PROBABILITY = _i("annealing_single_flip_probability", 0.70, "fraction", "solver", "Proposal-mixture setting for feasible structured annealing.")
MOVE_ACTION_PROBABILITY = _i("annealing_move_action_probability", 0.15, "fraction", "solver", "Proposal-mixture setting for temporal action moves.")
ACQUIRE_PAIR_PROBABILITY = _i("annealing_acquisition_pair_probability", 0.10, "fraction", "solver", "Proposal-mixture setting for acquisition plus later action.")
LOCAL_POLISH_PASSES = _i("annealing_local_polish_passes", 14, "passes", "solver", "Deterministic feasible one-flip cleanup after annealing.")
NEGATIVE_CONTROL_PENALTIES = _i("negative_control_penalties", [1.0, 10.0], "energy", "solver_verification", "Deliberately under-weighted hard-constraint penalties requested for checker negative controls.")
RAW_ANNEAL_RESTARTS = _i("raw_negative_control_annealing_restarts", 6, "-", "solver_verification", "Independent raw-bit annealing effort for each negative control.")
RAW_ANNEAL_STEPS = _i("raw_negative_control_steps_per_restart", 45000, "single-bit proposals", "solver_verification", "Independent raw-bit annealing effort for each negative control.")
RAW_ANNEAL_START_TEMP = _i("raw_negative_control_start_temperature", 30.0, "energy", "solver_verification", "High initial temperature for unconstrained full-bit negative-control search.")
RAW_ANNEAL_END_TEMP = _i("raw_negative_control_end_temperature", 0.01, "energy", "solver_verification", "Final temperature for unconstrained full-bit negative-control search.")
RELIABILITY_SEED_COUNT = _i("full_size_reliability_seed_count", 20, "independent seeds", "solver_verification", "Requested minimum multi-seed reliability protocol.")
RELIABILITY_SEED_BASE = _i("full_size_reliability_seed_base", 12000, "-", "solver_verification", "Reproducible seed-series origin for full-size solver verification.")
RELIABILITY_RESTARTS_PER_SEED = _i("full_size_reliability_restarts_per_seed", 1, "restart/seed", "solver_verification", "One independent restart per reported seed; the 20 seeds form the restart ensemble.")
RELIABILITY_WORKERS = _i("full_size_reliability_worker_processes", 4, "processes", "solver_verification", "Parallel compute setting; does not change an annealing trajectory.")
EXACT_LADDER_PRIMARY_BITS = _i("exact_verification_ladder_primary_bits", [12, 16, 20], "primary bits", "solver_verification", "Requested intermediate brute-force ladder sizes.")
EXACT_LADDER_ANNEAL_STEPS = _i("exact_ladder_annealing_steps", 5000, "proposals/seed", "solver_verification", "Per-seed annealing effort for exact ladder comparisons.")
EXACT_ENUMERATION_CHUNK_SIZE = _i("exact_enumeration_chunk_size", 65536, "assignments", "solver_verification", "Memory-bounded vectorised brute-force chunk size.")
PRIOR_PLAN_DIAGNOSTIC_SEEDS = _i("prior_plan_diagnostic_seed_count", 50, "independent seeds", "solver_diagnosis", "Requested minimum seed count for flat, sparse and oracle plan-distribution diagnosis.")
PRIOR_PLAN_SEED_BASE = _i("prior_plan_diagnostic_seed_base", 31000, "-", "solver_diagnosis", "Reproducible seed-series origin for prior and oracle plan comparisons.")
JACCARD_DISTINGUISH_P = _i("jaccard_distribution_p_threshold", 0.05, "p value", "solver_diagnosis", "Illustrative Mann-Whitney decision threshold for distribution distinguishability.")
JACCARD_DISTINGUISH_CLIFF = _i("jaccard_distribution_cliffs_delta_threshold", 0.33, "absolute effect", "solver_diagnosis", "Illustrative minimum rank-effect magnitude for distinguishability.")
JACCARD_SUBSTANTIAL_IQR_OVERLAP = _i("jaccard_substantial_iqr_overlap_fraction", 0.25, "fraction", "solver_diagnosis", "Illustrative threshold for calling interquartile-range overlap substantial.")
GENUINELY_DAMAGED_ALPHA_LIMIT = _i("genuinely_damaged_alpha_limit", 0.86, "alpha", "state_estimation_diagnosis", "Matches the synthetic mild-damage lower edge for reporting damaged-member shrinkage bias.")
MATERIAL_SHRINKAGE_BIAS_INCREMENT = _i("material_shrinkage_bias_increment", 0.05, "alpha", "state_estimation_diagnosis", "Illustrative trigger for testing the horseshoe alternative when sparse-minus-flat damaged-member bias exceeds this value.")
MILP_TIME_LIMIT_SECONDS = _i("milp_time_limit", 600.0, "seconds/solve", "solver_verification", "Compute limit for full-size HiGHS exactness checks; failure to prove optimality is reported.")
MILP_RELATIVE_GAP = _i("milp_requested_relative_gap", 0.0, "fraction", "solver_verification", "Requests a proof of global optimality from HiGHS rather than an approximate gap.")
MILP_COEFFICIENT_ZERO_TOL = _i("milp_coefficient_zero_tolerance", 1.0e-12, "objective coefficient", "solver_verification", "Numerical threshold for omitting algebraic zero coefficients from the MILP linearisation.")
MILP_EQUIVALENCE_ABS_TOL = _i("milp_qubo_equivalence_absolute_tolerance", 1.0e-5, "energy", "solver_verification", "Numerical acceptance tolerance when cross-scoring the MILP solution in the zero-penalty QUBO.")
MILP_EQUIVALENCE_REL_TOL = _i("milp_qubo_equivalence_relative_tolerance", 1.0e-8, "fraction", "solver_verification", "Relative numerical acceptance tolerance for MILP/QUBO objective equivalence.")


def constants_frame() -> pd.DataFrame:
    """Return the complete constant registry in report-ready form."""
    rows = []
    for item in REGISTRY.values():
        row = asdict(item)
        if isinstance(row["value"], list):
            row["value"] = "; ".join(map(str, row["value"]))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["stage", "name"]).reset_index(drop=True)
