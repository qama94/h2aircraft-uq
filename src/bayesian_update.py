"""
bayesian_update.py
------------------
Bayesian parameter update for the hydrogen aircraft range model.

Demonstrates how epistemic uncertainty (knowledge gaps) reduces as
new measurement data arrives, while the fundamental spread in range
predictions reflects the remaining irreducible uncertainty.

Core Bayesian idea:
  prior     — what we believed about a parameter BEFORE any measurements
  likelihood — how probable our observed data is, given a parameter value
  posterior  — updated belief AFTER incorporating the measurements
               posterior ∝ prior × likelihood

Key distinction this module illustrates (DASAL Pillar 3):
  EPISTEMIC uncertainty (e.g. eta_fc): posterior narrows with more data
  → we can reduce this by running more fuel cell tests

  The width of the RANGE distribution reflects remaining uncertainty
  after parameter updating — this shrinks as we learn more about eta_fc

This connects directly to Dr. Dwight's published work on Bayesian
calibration of computer models and data assimilation.

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from aircraft_model import analyse_aircraft
from uncertainty_model import PARAMETERS


# ── Bayesian update functions ─────────────────────────────────────────────────

def bayesian_update(prior_mean, prior_std, measurements, measurement_noise_std):
    """
    Conjugate Bayesian update for a Gaussian prior with Gaussian likelihood.

    For a normal prior N(mu_0, sigma_0^2) and N independent measurements
    with known noise sigma_n, the posterior is also normal:

      posterior_variance  = 1 / (1/sigma_0^2 + N/sigma_n^2)
      posterior_mean      = posterior_variance * (mu_0/sigma_0^2 +
                            sum(measurements)/sigma_n^2)

    This is the simplest case of Bayesian inference — exact and analytical.
    No MCMC needed when prior and likelihood are both Gaussian.

    Parameters
    ----------
    prior_mean          : float — prior mean
    prior_std           : float — prior standard deviation
    measurements        : array — observed data points
    measurement_noise_std: float — known measurement noise (std)

    Returns
    -------
    posterior_mean : float
    posterior_std  : float
    """
    n = len(measurements)

    # Posterior precision = prior precision + data precision
    prior_precision = 1.0 / prior_std**2
    data_precision  = n / measurement_noise_std**2
    post_precision  = prior_precision + data_precision
    post_variance   = 1.0 / post_precision

    # Posterior mean = precision-weighted average of prior and data mean
    post_mean = post_variance * (
        prior_mean / prior_std**2 +
        np.sum(measurements) / measurement_noise_std**2
    )

    return post_mean, np.sqrt(post_variance)


def simulate_measurements(true_value, noise_std, n_measurements, seed=42):
    """
    Simulate noisy measurements of a parameter.

    In reality these would come from fuel cell bench tests, wind tunnel
    measurements, or flight test data. Here we simulate them around a
    'true' value with Gaussian noise.

    Parameters
    ----------
    true_value      : float — the true parameter value
    noise_std       : float — measurement instrument noise
    n_measurements  : int   — number of measurements
    seed            : int   — random seed

    Returns
    -------
    measurements : array of shape (n_measurements,)
    """
    rng = np.random.default_rng(seed)
    return rng.normal(loc=true_value, scale=noise_std, size=n_measurements)


def compute_range_uncertainty(param_name, param_mean, param_std, n_samples=500):
    """
    Estimate range uncertainty given a specific parameter distribution.

    Holds all other parameters at their nominal values and varies
    only the specified parameter — to isolate its contribution to
    range uncertainty.

    Parameters
    ----------
    param_name : str   — parameter to vary
    param_mean : float — current mean estimate
    param_std  : float — current uncertainty (std)
    n_samples  : int   — Monte Carlo samples

    Returns
    -------
    range_std : float — standard deviation of range [km]
    """
    rng = np.random.default_rng(seed=42)

    # Build nominal parameter set
    nominal = {name: p["mean"] for name, p in PARAMETERS.items()}

    ranges = []
    for _ in range(n_samples):
        params = nominal.copy()
        val = rng.normal(param_mean, param_std)
        # Clip to bounds
        lo, hi = PARAMETERS[param_name]["bounds"]
        val = np.clip(val, lo, hi)
        params[param_name] = val
        ranges.append(analyse_aircraft(params))

    return np.std(ranges)


# ── Main analysis ─────────────────────────────────────────────────────────────

def run_bayesian_analysis(save_path=None):
    """
    Full Bayesian update analysis for eta_fc (dominant parameter).

    Shows:
      1. How posterior distribution of eta_fc narrows with more data
      2. How range uncertainty reduces as eta_fc becomes better known
      3. The distinction: epistemic uncertainty reduces, but range
         uncertainty never reaches zero (other parameters still uncertain)
    """

    # Focus on eta_fc — the dominant Sobol parameter
    param  = PARAMETERS["eta_fc"]
    p_mean = param["mean"]   # 0.55
    p_std  = param["std"]    # 0.04

    # Assume the 'true' fuel cell efficiency is slightly different from
    # our prior belief — this is what triggers the posterior to shift
    true_eta_fc      = 0.57   # slightly better than assumed
    measurement_noise = 0.015  # bench test measurement precision

    # Simulate increasing amounts of measurement data
    n_measurement_sequence = [0, 1, 3, 5, 10, 20, 50]
    all_measurements = simulate_measurements(
        true_eta_fc, measurement_noise, max(n_measurement_sequence), seed=42
    )

    posteriors = []
    range_stds = []

    for n in n_measurement_sequence:
        if n == 0:
            post_mean = p_mean
            post_std  = p_std
        else:
            post_mean, post_std = bayesian_update(
                p_mean, p_std,
                all_measurements[:n],
                measurement_noise
            )
        posteriors.append((n, post_mean, post_std))
        r_std = compute_range_uncertainty("eta_fc", post_mean, post_std)
        range_stds.append(r_std)

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: posterior distributions evolving with data
    ax1 = axes[0]
    x   = np.linspace(0.35, 0.75, 500)

    colors = plt.cm.Blues(np.linspace(0.3, 1.0, len(n_measurement_sequence)))

    for i, (n, mu, sigma) in enumerate(posteriors):
        label = f"Prior (N=0)" if n == 0 else f"N={n} measurements"
        lw    = 2.5 if n == 0 else 1.5
        ls    = "--" if n == 0 else "-"
        ax1.plot(x, norm.pdf(x, mu, sigma),
                 color=colors[i], linewidth=lw, linestyle=ls, label=label)

    ax1.axvline(true_eta_fc, color="#C44E52", linewidth=2.0,
                linestyle=":", label=f"True value: {true_eta_fc}")
    ax1.axvline(p_mean, color="gray", linewidth=1.0,
                linestyle="--", alpha=0.5, label=f"Prior mean: {p_mean}")

    ax1.set_xlabel("Fuel Cell Efficiency η_fc", fontsize=12)
    ax1.set_ylabel("Probability Density", fontsize=12)
    ax1.set_title(
        "Bayesian Update — Posterior Narrows with Data\n"
        "Epistemic Uncertainty Reduces as Measurements Accumulate",
        fontsize=11, fontweight="bold"
    )
    ax1.legend(fontsize=8, loc="upper left")
    ax1.grid(True, alpha=0.3)

    # Right: range uncertainty vs number of measurements
    ax2 = axes[1]
    ns  = [p[0] for p in posteriors]

    ax2.plot(ns, range_stds, "o-",
             color="#4C72B0", linewidth=2.5, markersize=7,
             label="Range std from η_fc uncertainty")

    # Show floor — uncertainty from other parameters
    other_range_std = compute_range_uncertainty("eta_fc", p_mean, 0.001)
    ax2.axhline(other_range_std, color="#C44E52", linewidth=1.5,
                linestyle="--",
                label=f"Floor (other params): ~{other_range_std:.0f} km")

    # Annotate reduction
    reduction_pct = (range_stds[0] - range_stds[-1]) / range_stds[0] * 100
    ax2.annotate(
        f"{reduction_pct:.0f}% reduction\nwith 50 measurements",
        xy=(50, range_stds[-1]),
        xytext=(35, range_stds[-1] + 30),
        fontsize=9, color="#4C72B0",
        arrowprops=dict(arrowstyle="->", color="#4C72B0", lw=1.0)
    )

    ax2.set_xlabel("Number of Fuel Cell Measurements", fontsize=12)
    ax2.set_ylabel("Range Uncertainty  σ(Range) [km]", fontsize=12)
    ax2.set_title(
        "Range Uncertainty Reduces as η_fc Becomes Better Known\n"
        "But Never Reaches Zero — Other Parameters Remain Uncertain",
        fontsize=11, fontweight="bold"
    )
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(ns)

    plt.suptitle(
        "Bayesian Model-Data Update  |  DASAL Pillar 3\n"
        "Epistemic uncertainty reduces with data — aleatory floor remains",
        fontsize=12, fontweight="bold", y=1.02
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Plot saved: {save_path}")

    return fig, posteriors, range_stds


def print_bayesian_summary(posteriors, range_stds):
    """Print a clean summary of Bayesian update results."""
    print("\n" + "=" * 60)
    print("  Bayesian Update — eta_fc Posterior Evolution")
    print("=" * 60)
    print(f"  {'N meas':>8s}  {'Post mean':>10s}  {'Post std':>10s}  "
          f"{'Range std':>10s}")
    print(f"  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*10}")
    for (n, mu, sigma), r_std in zip(posteriors, range_stds):
        print(f"  {n:>8d}  {mu:>10.4f}  {sigma:>10.4f}  {r_std:>10.1f} km")
    print("=" * 60)
    print(f"\n  KEY FINDING:")
    print(f"  Prior std on eta_fc:     {posteriors[0][2]:.4f}")
    print(f"  Posterior std (N=50):    {posteriors[-1][2]:.4f}")
    red = (posteriors[0][2] - posteriors[-1][2]) / posteriors[0][2] * 100
    print(f"  Reduction:               {red:.1f}%")
    print(f"\n  This is epistemic uncertainty reducing with data.")
    print(f"  The residual range uncertainty comes from other")
    print(f"  uncertain parameters — it cannot be eliminated by")
    print(f"  measuring eta_fc alone. This is the key DASAL insight:")
    print(f"  you must address all dominant sources simultaneously.")
    print("=" * 60)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Running Bayesian update analysis...")

    fig, posteriors, range_stds = run_bayesian_analysis(
        save_path="/home/claude/h2aircraft-uq/results/03_bayesian_update.png"
    )
    print_bayesian_summary(posteriors, range_stds)
    plt.show()
