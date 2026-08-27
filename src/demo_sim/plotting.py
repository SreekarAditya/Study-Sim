"""Publication-readable, data-driven figures for generated synthetic results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from . import constants as C


def _style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.15,
    })


def _save(fig: plt.Figure, output: Path, name: str) -> None:
    fig.savefig(output / f"{name}.pdf")
    fig.savefig(output / f"{name}.png", dpi=300)
    plt.close(fig)


def generate_figures(results_dir: Path | str) -> None:
    results = Path(results_dir)
    output = results / "figures"
    output.mkdir(parents=True, exist_ok=True)
    _style()
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]  # Okabe-Ito subset.

    corr = pd.read_csv(results / "posterior_correlation.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(6.75, 5.5))
    sns.heatmap(
        corr,
        cmap="vlag",
        vmin=-1.0,
        vmax=1.0,
        center=0.0,
        square=True,
        xticklabels=True,
        yticklabels=True,
        cbar_kws={"label": "Posterior correlation", "shrink": 0.75},
        ax=ax,
    )
    ax.set_title("Joint stiffness posterior: trade-offs are retained")
    ax.tick_params(axis="x", labelrotation=90, labelsize=6)
    ax.tick_params(axis="y", labelrotation=0, labelsize=6)
    _save(fig, output, "fig_posterior_correlation")

    reliability = pd.read_csv(results / "solver_reliability_summary.csv")
    sweep = reliability[reliability["configuration"].str.startswith("lambda_")].copy()
    sweep["lambda"] = sweep["configuration"].str.replace("lambda_", "", regex=False).astype(float)
    sweep = sweep.sort_values("lambda")
    fig, ax = plt.subplots(figsize=(6.75, 2.9))
    ax.fill_between(
        sweep["lambda"], sweep["best_energy"], sweep["worst_energy"],
        color=colors[0], alpha=0.15, label="best–worst energy",
    )
    ax.plot(sweep["lambda"], sweep["median_energy"], marker="o", color=colors[0], label="median energy")
    ax.set_xlabel("Risk preference λ (illustrative units)")
    ax.set_ylabel("QUBO energy across 20 seeds")
    second = ax.twinx()
    second.plot(
        sweep["lambda"], sweep["geometry_acquisition_run_fraction"],
        marker="s", color=colors[1], label="acquisition run fraction",
    )
    second.set_ylabel("Fraction selecting acquisition")
    second.set_ylim(-0.03, 1.03)
    ax.set_title("Multi-seed λ sweep: energy spread and epistemic action")
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = second.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, ncol=3, loc="upper left")
    _save(fig, output, "fig_lambda_sweep")

    ablation = pd.read_csv(results / "ablations.csv")
    fig, ax = plt.subplots(figsize=(6.75, 2.9))
    values = ablation["reference_objective_delta_vs_main"].to_numpy()
    bars = ax.barh(ablation["scenario"], values, color=colors[2])
    ax.axvline(0.0, color="#555555", linewidth=0.8)
    ax.set_xlabel("Reference QUBO energy increase vs main plan (lower is better)")
    ax.set_title("Ablations scored under the same uncertainty-aware objective")
    for bar, value in zip(bars, values):
        ax.text(value, bar.get_y() + bar.get_height() / 2, f" {value:.2f}", va="center", fontsize=7)
    _save(fig, output, "fig_ablations")

    trace = pd.read_csv(results / "uncertainty_trace.csv")
    alpha_trace = trace[trace["state_quantity"] == "alpha"]
    fig, ax = plt.subplots(figsize=(6.75, 2.9))
    x = np.arange(len(alpha_trace))
    ax.errorbar(
        x,
        alpha_trace["centre"],
        yerr=alpha_trace["uncertainty"],
        marker="o",
        capsize=4,
        color=colors[3],
    )
    ax.set_xticks(x)
    ax.set_xticklabels(alpha_trace["stage"], rotation=18, ha="right")
    ax.set_ylabel("α centre ± one SD")
    ax.set_title(f"Member {alpha_trace.iloc[0]['member']}: uncertainty through the inverse stages")
    _save(fig, output, "fig_uncertainty_trace")

    prior = pd.read_csv(results / "prior_comparison_by_member.csv")
    main_prior = prior[prior["case"] == "clustered_main_case"]
    x = np.arange(len(main_prior))
    fig, ax = plt.subplots(figsize=(6.75, 3.1))
    width = 0.38
    ax.bar(x - width / 2, main_prior["flat_alpha_sd"], width, color=colors[0], label="Flat prior")
    ax.bar(x + width / 2, main_prior["sparse_alpha_sd"], width, color=colors[1], label="Hierarchical Laplace")
    ax.axhline(C.IDENTIFIABLE_SD_LIMIT, color="#555555", linestyle="--", linewidth=1, label="SD separability threshold")
    ax.set_xticks(x)
    ax.set_xticklabels(main_prior["member"], rotation=90)
    ax.set_ylabel("Posterior SD of α")
    ax.set_title("Per-member uncertainty: flat versus sparse prior")
    ax.legend(ncol=3)
    _save(fig, output, "fig_prior_comparison")

    ordered = reliability.sort_values("energy_spread", ascending=True)
    y = np.arange(len(ordered))
    fig, ax = plt.subplots(figsize=(6.75, max(3.2, 0.34 * len(ordered))))
    ax.hlines(y, ordered["best_energy"], ordered["worst_energy"], color="#B0BEC5", linewidth=4)
    ax.scatter(ordered["median_energy"], y, color=colors[0], s=24, label="Median")
    ax.scatter(ordered["best_energy"], y, color=colors[2], s=20, marker="|", label="Best")
    ax.scatter(ordered["worst_energy"], y, color=colors[1], s=20, marker="|", label="Worst")
    ax.set_yticks(y)
    ax.set_yticklabels(ordered["configuration"])
    ax.set_xlabel("QUBO energy")
    ax.set_title("Full-size annealing reliability across 20 independent seeds")
    ax.legend(ncol=3)
    _save(fig, output, "fig_solver_reliability")

    ladder = pd.read_csv(results / "exact_verification_ladder.csv")
    fig, ax = plt.subplots(figsize=(6.75, 2.9))
    bars = ax.bar(
        ladder["primary_bits"].astype(str),
        ladder["annealer_hit_rate"],
        color=[colors[2], colors[0], colors[1]],
    )
    ax.set_ylim(0.0, 1.08)
    ax.set_xlabel("Exactly enumerated primary bits")
    ax.set_ylabel("Exact-optimum hit rate (20 seeds)")
    ax.set_title("Annealer verification degrades as exact instances grow")
    for bar, hits, seeds in zip(bars, ladder["annealer_hits"], ladder["seed_count"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.025, f"{hits}/{seeds}", ha="center")
    _save(fig, output, "fig_exact_ladder")

    prior_runs = pd.read_csv(results / "prior_plan_jaccard_50_seeds.csv")
    prior_runs = prior_runs[
        prior_runs["configuration"].isin(["flat_prior", "sparse_prior"])
    ].copy()
    prior_runs["Prior"] = prior_runs["configuration"].map({
        "flat_prior": "Flat Gaussian",
        "sparse_prior": "Laplace sparse",
    })
    exact = pd.read_csv(results / "milp_prior_plan_jaccard.csv")
    exact["Prior"] = exact["prior"].map({
        "flat_prior": "Flat Gaussian",
        "sparse_prior": "Laplace sparse",
    })
    fig, ax = plt.subplots(figsize=(6.75, 3.1))
    sns.violinplot(
        data=prior_runs,
        x="Prior",
        y="jaccard_vs_milp_oracle",
        hue="Prior",
        palette={"Flat Gaussian": colors[0], "Laplace sparse": colors[1]},
        inner=None,
        cut=0,
        alpha=0.35,
        legend=False,
        ax=ax,
    )
    sns.boxplot(
        data=prior_runs,
        x="Prior",
        y="jaccard_vs_milp_oracle",
        width=0.22,
        showfliers=False,
        color="white",
        linecolor="#333333",
        ax=ax,
    )
    positions = {"Flat Gaussian": 0, "Laplace sparse": 1}
    for row in exact.itertuples(index=False):
        ax.scatter(
            positions[row.Prior],
            row.milp_plan_jaccard_vs_milp_oracle,
            marker="D",
            s=42,
            color=colors[2],
            edgecolor="white",
            linewidth=0.6,
            zorder=5,
            label="Exact MILP plan" if row.Prior == "Flat Gaussian" else None,
        )
    ax.set_ylim(-0.03, 1.03)
    ax.set_ylabel("Intervention Jaccard vs exact MILP oracle")
    ax.set_xlabel("")
    ax.set_title("Prior comparison: 50 annealing seeds and exact-plan result")
    ax.legend(loc="upper right")
    _save(fig, output, "fig_prior_plan_jaccard_distribution")
