"""
propagation.py
--------------
Monte Carlo uncertainty propagation for the hydrogen aircraft range model.

Uses Latin Hypercube Sampling (LHS) instead of pure random Monte Carlo
for better coverage of the input space with fewer model evaluations.

Why LHS over pure Monte Carlo?
  Pure Monte Carlo draws inputs completely randomly — by chance, some regions
  of the input space get many samples and others get few. LHS divides each
  input distribution into N equal-probability intervals and draws exactly one
  sample from each interval. This guarantees uniform coverage of the full
  input range, giving more accurate statistics with fewer model evaluations.
  For N=2000 samples, LHS typically achieves the accuracy of ~5000-10000
  pure Monte Carlo samples.

DASAL Pillar 1: Uncertainty Propagation
  Given uncertain inputs → what is the distribution of aircraft range?
  How wide is our confidence interval on the 3000 km target?

Author: Gamar Ismayilova
Project: h2aircraft-uq — DASAL PhD Preparation
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import norm

from aircraft_model import analyse_aircraft
from uncertainty_model import PARAMETERS, get_salib_problem

# SALib for Latin Hypercube Sampling
from SALib.sample import latin


# ── Monte Carlo propagation ───────────────────────────────────────────────────

def run_monte_carlo(n_samples=2000, seed=42):
    """
    Run Monte Carlo uncertainty propagation using Latin Hypercube Sampling.

    For each sample:
      1. Draw random parameter values from their distributions
      2. Run the aircraft sizing model
      3. Record the output range

    Parameters
    ----------
    n_samples : int  — number of Monte Carlo samples
    seed      : int  — random seed for reproducibility

    Returns
    -------
    results : dict with keys:
        ranges_km    — array of range values [km] for all samples
        params       — dict of sampled parameter arrays
        n_samples    — number of samples run
        n_valid      — number of converged samples
        mean_range   — mean range [km]
        std_range    — standard deviation of range [km]
        p05          — 5th percentile range [km]
        p50          — median range [km]
        p90          — 90th percentile range [km]
        p95          — 95th percentile range [km]
    """
    problem = get_salib_problem()

    # Latin Hypercube Sampling — draws samples in [0,1] mapped to bounds
    param_names = problem["names"]
    bounds      = problem["bounds"]

    # Use SALib LHS sampler
    lhs_samples = latin.sample(problem, n_samples, seed=seed)

    # Run model for each sample
    ranges_km  = np.zeros(n_samples)
    converged  = np.zeros(n_samples, dtype=bool)
    param_vals = {name: lhs_samples[:, i]
                  for i, name in enumerate(param_names)}

    for j in range(n_samples):
        params = {
            name: float(lhs_samples[j, i])
            for i, name in enumerate(param_names)
        }
        ranges_km[j] = analyse_aircraft(params)
        converged[j] = True

    # Filter to physically valid solutions only
    valid_mask = (ranges_km > 0) & (ranges_km < 20000)
    ranges_valid = ranges_km[valid_mask]
    n_valid = valid_mask.sum()

    return {
        "ranges_km":  ranges_valid,
        "params":     param_vals,
        "n_samples":  n_samples,
        "n_valid":    n_valid,
        "mean_range": np.mean(ranges_valid),
        "std_range":  np.std(ranges_valid),
        "p05":        np.percentile(ranges_valid, 5),
        "p50":        np.percentile(ranges_valid, 50),
        "p90":        np.percentile(ranges_valid, 90),
        "p95":        np.percentile(ranges_valid, 95),
    }


def print_summary(results):
    """Print a clean summary of Monte Carlo results."""
    print("\n" + "=" * 55)
    print("  Monte Carlo Uncertainty Propagation — Results")
    print("=" * 55)
    print(f"  Samples run:        {results['n_samples']:,}")
    print(f"  Valid samples:      {results['n_valid']:,}")
    print(f"")
    print(f"  Mean range:         {results['mean_range']:>8.1f} km")
    print(f"  Std deviation:      {results['std_range']:>8.1f} km")
    print(f"  CV (range):         {results['std_range']/results['mean_range']*100:>7.1f} %")
    print(f"")
    print(f"  5th  percentile:    {results['p05']:>8.1f} km")
    print(f"  50th percentile:    {results['p50']:>8.1f} km")
    print(f"  90th percentile:    {results['p90']:>8.1f} km")
    print(f"  95th percentile:    {results['p95']:>8.1f} km")
    print(f"")
    print(f"  P90 range achieves target (3000 km): "
          f"{'YES' if results['p90'] >= 3000 else 'NO'}")
    print("=" * 55)


def plot_range_distribution(results, save_path=None):
    """
    Plot the probability distribution of aircraft range.

    Shows:
      - Histogram of Monte Carlo range samples
      - Fitted normal distribution
      - Confidence interval markers (P5, P50, P90)
      - Target range line (3000 km)
    """
    ranges = results["ranges_km"]
    mu     = results["mean_range"]
    sigma  = results["std_range"]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Histogram
    n_bins = 60
    counts, bin_edges, patches = ax.hist(
        ranges, bins=n_bins, density=True,
        color="#4C72B0", alpha=0.7, edgecolor="white", linewidth=0.5,
        label=f"Monte Carlo samples (N={results['n_valid']:,})"
    )

    # Fitted normal distribution
    x = np.linspace(ranges.min(), ranges.max(), 500)
    ax.plot(x, norm.pdf(x, mu, sigma),
            color="#C44E52", linewidth=2.5,
            label=f"Normal fit  μ={mu:.0f} km,  σ={sigma:.0f} km")

    # Percentile markers
    for pct, label, color in [
        (results["p05"],  "P5",     "#2196F3"),
        (results["p50"],  "P50",    "#4CAF50"),
        (results["p90"],  "P90",    "#FF9800"),
    ]:
        ax.axvline(pct, color=color, linewidth=1.8, linestyle="--", alpha=0.9)
        ax.text(pct, ax.get_ylim()[1] * 0.02, f" {label}\n {pct:.0f} km",
                color=color, fontsize=9, va="bottom")

    # Target range line
    ax.axvline(3000, color="black", linewidth=2.0, linestyle="-",
               label="Target range: 3,000 km")

    # Shaded confidence interval (P5 to P95)
    x_ci = np.linspace(results["p05"], results["p95"], 300)
    ax.fill_between(x_ci, norm.pdf(x_ci, mu, sigma),
                    alpha=0.15, color="#4C72B0",
                    label=f"P5–P95 interval")

    ax.set_xlabel("Aircraft Range [km]", fontsize=12)
    ax.set_ylabel("Probability Density", fontsize=12)
    ax.set_title(
        "Uncertainty Propagation — Hydrogen Aircraft Range\n"
        "Monte Carlo with Latin Hypercube Sampling  |  DASAL Pillar 1",
        fontsize=13, fontweight="bold"
    )
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Plot saved: {save_path}")

    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Running Monte Carlo uncertainty propagation...")
    print("(N=2000 Latin Hypercube samples — this takes a few seconds)")

    results = run_monte_carlo(n_samples=2000, seed=42)
    print_summary(results)

    fig = plot_range_distribution(
        results,
        save_path="/home/claude/h2aircraft-uq/results/01_range_distribution.png"
    )
    plt.show()
