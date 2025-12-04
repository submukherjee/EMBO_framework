"""
EMBO — Empirical Model-Based Optimization framework.

This package provides:
- LASSO-based surrogate modelling (`LassoSurrogateModel`)
- Flowrate optimization (`optimize_flowrates_lasso`)
- A driver script (`run_optimization_single.main`) for one-shot runs.
"""

from .lasso_flowrate_module import (
    LassoSurrogateModel,
    optimize_flowrates_lasso,
    generate_initial_setpoints,
    generate_perturbed_setpoints,
    generate_perturbed_setpoints_1,
    align_tariff_to_experiment,
    save_rounded_perturbed_setpoints,
)