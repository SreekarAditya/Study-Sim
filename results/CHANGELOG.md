# Changelog

> This remains an illustrative plumbing and uncertainty-propagation study, not validation.

## Prior-plan diagnosis

- Added 50 independent annealing seeds for flat, Laplace, and hidden-state oracle planning models.
- Added full Jaccard distributions, pooled best-known Jaccard, Mann–Whitney/Cliff/IQR-overlap diagnosis, and a publication-readable violin/box plot.
- Distribution verdict: The two 50-seed Jaccard distributions overlap substantially and are not distinguishable under the declared rule. Therefore the reported 0.313 → 0.100 single-run comparison is not evidence about the prior.

## Bias diagnosis

- Added per-member `estimated_alpha - true_alpha`, 90% interval containment, and damaged-member bias for both truth cases.
- Laplace-minus-flat damaged-member bias is +0.079; material-bias trigger = True.
- Tested the horseshoe alternative because the trigger fired; it is retained as a failed diagnostic and is not adopted.

## Exact planning fix

- Added an equivalent hard-constraint MILP using SciPy's open-source HiGHS backend and standard binary-product linearisation.
- Proved full-size optimality for flat, sparse, and oracle problems at zero reported MIP gap.
- Replaced the production and oracle plan artifacts with MILP optima while retaining QUBO plans and all annealing evidence under explicit filenames.
- Added MILP/QUBO cross-scoring, solve times, best-of-50 and median annealing gaps, and exact-prior Jaccard: flat 0.250, Laplace 0.429.

## Preserved safeguards

- Scope warning and `source=illustrative` constant registry remain mandatory.
- Penalty negative controls, feasibility checker, exact enumeration ladder, λ sweep, ablations, QUBO formulation, and 20-seed reliability artifacts remain in place.
