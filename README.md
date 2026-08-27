# Study-Sim

An end-to-end **synthetic** demonstration of structural sensing, environmental
normalisation, joint state estimation, two-tier durability state, performance
classification, and uncertainty-aware retrofit planning with a constrained
QUBO.

This repository is a plumbing and uncertainty-propagation study. It is not a
validation study and none of its constants are calibrated to real concrete.
Every empirical constant is registered with `source="illustrative"` and emitted
to `results/constants.csv`.

Run the complete deterministic demonstration. The retained QUBO ensembles,
50-seed prior/oracle diagnosis, and three full-size HiGHS MILP proofs can take
roughly 10–20 minutes on a typical laptop:

```bash
PYTHONPATH=src python3 -m demo_sim.cli --output results
```

Run the tests:

```bash
pytest -q
```

The generated narrative is `results/REPORT.md`; `results/CHANGELOG.md` records
the negative controls, seed-distribution diagnosis, state-bias comparison,
exact MILP equivalence check, horseshoe diagnostic, and reporting changes. The
QUBO and annealer remain available; `results/retrofit_plan.csv` is now the
proven-optimal sparse-prior MILP plan.
