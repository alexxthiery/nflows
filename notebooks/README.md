# Notebooks Overview

- `real_nvp_double_well_dim16.ipynb`: Trains and evaluates a RealNVP flow on a 16-dimensional double-well target distribution. Includes data sampling, flow architecture setup, training loop, and diagnostics (loss, samples, and contour/summary plots).

- `real_nvp_neals_funnel.ipynb`: Applies RealNVP to Neal's funnel distribution (a challenging hierarchical target). Demonstrates stability considerations, training configuration, and visualization of learned density and samples versus ground truth.

- `spline_real_nvp_double_well_dim16.ipynb`: Uses a spline-coupling variant of RealNVP on the 16D double-well. Compares spline vs standard affine coupling, shows training curves, qualitative samples, and quantitative metrics.
