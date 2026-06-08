"""
main.py
-------
Entry point for the h2aircraft-uq project.

Runs all three DASAL pillars in sequence:
  Pillar 1 — Monte Carlo uncertainty propagation
  Pillar 2 — Sobol global sensitivity analysis
  Pillar 3 — Bayesian model-data update

Author: Gamar Ismayilova
Project: h2aircraft-uq — DASAL PhD Preparation
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from aircraft_model import size_aircraft
from propagation import run_monte_carlo, print_summary, plot_range_distribution
from sensitivity import run_sobol, print_sobol_results, plot_sobol_indices
from bayesian_update import run_bayesian_analysis, print_bayesian_summary
import matplotlib.pyplot as plt


def main():
    print("\n" + "=" * 60)
    print("  H2AIRCRAFT-UQ")
    print("  Uncertainty-Aware Conceptual Design of a")
    print("  Medium-Haul Hydrogen Fuel Cell Aircraft")
    print("  DASAL PhD Preparation — Gamar Ismayilova")
    print("=" * 60)

    # ── Deterministic design point ────────────────────────────────────────────
    print("\n[1/4] Deterministic design point...")
    result = size_aircraft()
    print(f"  L/D = {result['LD']:.2f}  |  "
          f"eta_total = {result['eta_total']:.3f}  |  "
          f"MTOW = {result['MTOW']:,.0f} kg  |  "
          f"Range = {result['range_km']:.0f} km")

    # ── Pillar 1: Monte Carlo propagation ─────────────────────────────────────
    print("\n[2/4] DASAL Pillar 1 — Monte Carlo uncertainty propagation...")
    mc_results = run_monte_carlo(n_samples=2000, seed=42)
    print_summary(mc_results)
    plot_range_distribution(
        mc_results,
        save_path="results/01_range_distribution.png"
    )

    # ── Pillar 2: Sobol sensitivity analysis ──────────────────────────────────
    print("\n[3/4] DASAL Pillar 2 — Sobol sensitivity analysis...")
    Si, problem = run_sobol(n_base=1024, seed=42)
    dominant_name, dominant_s1 = print_sobol_results(Si, problem)
    plot_sobol_indices(
        Si, problem,
        save_path="results/02_sobol_indices.png"
    )

    # ── Pillar 3: Bayesian update ─────────────────────────────────────────────
    print("\n[4/4] DASAL Pillar 3 — Bayesian model-data update...")
    fig, posteriors, range_stds = run_bayesian_analysis(
        save_path="results/03_bayesian_update.png"
    )
    print_bayesian_summary(posteriors, range_stds)

    print("\n" + "=" * 60)
    print("  All results saved to results/")
    print("  Run complete.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
