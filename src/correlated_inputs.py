"""
correlated_inputs.py
--------------------
Demonstrates the effect of input parameter CORRELATION on the output
uncertainty — and why assuming independence can be misleading.

The baseline UQ assumed all parameters are independent. In reality,
several are physically linked:
  - AR and CD0: higher aspect ratio wings have different drag build-up
  - eta_fc and eta_prop: both improve with overall powertrain maturity

This module compares range uncertainty under:
  1. Independent inputs (the baseline assumption)
  2. Positively correlated inputs
  3. Negatively correlated inputs

Why this matters for DASAL:
  Correlations between component models are common in a coupled digital
  thread (shared technology assumptions, common operating conditions).
  Ignoring them can either over- or under-estimate system uncertainty.
  A rigorous UQ framework must handle dependence structure explicitly.

Method:
  Gaussian copula — sample correlated standard normals, then map each
  to its marginal distribution. This separates the dependence structure
  (correlation matrix) from the marginals (each parameter's distribution).

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from aircraft_model import analyse_aircraft
from uncertainty_model import PARAMETERS


def sample_correlated(corr_AR_CD0=0.0, n=2000, seed=42):
    """
    Sample parameters with a specified correlation between AR and CD0,
    using a Gaussian copula. All other parameters remain independent.

    Parameters
    ----------
    corr_AR_CD0 : float — correlation coefficient between AR and CD0 [-1, 1]
    n           : int   — number of samples
    seed        : int   — random seed

    Returns
    -------
    samples : dict — {param_name: array}
    """
    rng = np.random.default_rng(seed)
    names = list(PARAMETERS.keys())

    # Build correlation matrix — identity except AR-CD0 block
    n_params = len(names)
    corr = np.eye(n_params)
    i_ar  = names.index("AR")
    i_cd0 = names.index("CD0")
    corr[i_ar, i_cd0] = corr_AR_CD0
    corr[i_cd0, i_ar] = corr_AR_CD0

    # Sample correlated standard normals via Cholesky
    L = np.linalg.cholesky(corr)
    z = rng.standard_normal((n, n_params))
    z_corr = z @ L.T

    # Map each correlated normal to its marginal distribution
    samples = {}
    for j, name in enumerate(names):
        p = PARAMETERS[name]
        # Uniform in [0,1] via normal CDF, then to the marginal
        u = norm.cdf(z_corr[:, j])
        # All marginals here are normal — map back with inverse CDF
        vals = norm.ppf(u, loc=p["mean"], scale=p["std"])
        lo, hi = p["bounds"]
        vals = np.clip(vals, lo, hi)
        samples[name] = vals

    return samples


def propagate(samples):
    """Run the model for each correlated sample, return range array."""
    n = len(next(iter(samples.values())))
    ranges = np.zeros(n)
    for k in range(n):
        params = {name: float(samples[name][k]) for name in samples}
        ranges[k] = analyse_aircraft(params)
    return ranges


def run_analysis(save_path=None):
    """Compare range uncertainty under different AR-CD0 correlations."""
    cases = {
        "Independent (ρ=0)":        0.0,
        "Positive (ρ=+0.7)":       0.7,
        "Negative (ρ=−0.7)":      -0.7,
    }

    results = {}
    for label, rho in cases.items():
        samples = sample_correlated(corr_AR_CD0=rho)
        ranges  = propagate(samples)
        results[label] = {
            "rho":  rho,
            "ranges": ranges,
            "mean": ranges.mean(),
            "std":  ranges.std(),
        }

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {"Independent (ρ=0)": "#4C72B0",
              "Positive (ρ=+0.7)": "#C44E52",
              "Negative (ρ=−0.7)": "#55A868"}

    for label, r in results.items():
        ax.hist(r["ranges"], bins=50, density=True, alpha=0.5,
                color=colors[label],
                label=f"{label}: σ={r['std']:.0f} km")

    ax.set_xlabel("Aircraft Range [km]", fontsize=12)
    ax.set_ylabel("Probability Density", fontsize=12)
    ax.set_title(
        "Effect of Input Correlation on Range Uncertainty\n"
        "Independence is an assumption — correlation changes the spread",
        fontsize=12, fontweight="bold"
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Plot saved: {save_path}")

    return fig, results


def print_summary(results):
    print("\n" + "=" * 55)
    print("  Effect of AR-CD0 Correlation on Range Uncertainty")
    print("=" * 55)
    print(f"  {'Case':22s}  {'Mean [km]':>10s}  {'Std [km]':>10s}")
    print(f"  {'-'*22}  {'-'*10}  {'-'*10}")
    for label, r in results.items():
        print(f"  {label:22s}  {r['mean']:>10.1f}  {r['std']:>10.1f}")
    print("=" * 55)

    indep_std = results["Independent (ρ=0)"]["std"]
    pos_std   = results["Positive (ρ=+0.7)"]["std"]
    neg_std   = results["Negative (ρ=−0.7)"]["std"]
    print(f"\n  KEY INSIGHT:")
    print(f"  Assuming independence gives σ = {indep_std:.0f} km.")
    print(f"  Positive AR-CD0 correlation: σ = {pos_std:.0f} km "
          f"({(pos_std/indep_std-1)*100:+.0f}%)")
    print(f"  Negative AR-CD0 correlation: σ = {neg_std:.0f} km "
          f"({(neg_std/indep_std-1)*100:+.0f}%)")
    print(f"  Ignoring real correlations mis-estimates system uncertainty.")
    print(f"  A coupled digital thread must model dependence explicitly.")
    print("=" * 55)


if __name__ == "__main__":
    print("Running correlated input analysis...")
    fig, results = run_analysis(
        save_path="/home/claude/h2aircraft-uq/results/07_correlated_inputs.png"
    )
    print_summary(results)
    plt.show()
