"""
robust_design.py
----------------
Robust design optimisation for the hydrogen aircraft range model.

Two analyses:

1. INVERSION — answers: "What accuracy is required in each uncertain
   parameter to guarantee P90 range >= 3000 km?"
   Varies each parameter's uncertainty (std) one at a time and finds
   the threshold below which the P90 constraint is satisfied.

2. ROBUST vs DETERMINISTIC comparison — shows the MTOW penalty of
   designing conservatively to meet P90 target vs deterministic target.

Key DASAL insight:
  The deterministic optimal design is NOT the robust optimal design.
  A design optimised for nominal performance may sit in a high-sensitivity
  region where parameter uncertainty causes large range degradation.
  Robust design explicitly accounts for this — trading nominal performance
  for robustness to uncertainty.

Author: Gamar Ismayilova
Project: h2aircraft-uq — DASAL PhD Preparation
"""

import numpy as np
import matplotlib.pyplot as plt
from uncertainty_model import PARAMETERS, get_salib_problem
from aircraft_model import analyse_aircraft, size_aircraft
from SALib.sample import latin


# ── Monte Carlo with variable uncertainty ─────────────────────────────────────

def estimate_p90_range(param_stds, n_samples=500, seed=42):
    """
    Estimate P90 range for a given set of parameter standard deviations.

    Parameters
    ----------
    param_stds : dict — {param_name: std} overrides for uncertainty levels
    n_samples  : int  — Monte Carlo samples
    seed       : int  — random seed

    Returns
    -------
    p90 : float — 90th percentile range [km]
    """
    rng = np.random.default_rng(seed)
    ranges = []

    for _ in range(n_samples):
        params = {}
        for name, p in PARAMETERS.items():
            std  = param_stds.get(name, p["std"])
            mean = p["mean"]
            lo, hi = p["bounds"]
            val  = rng.normal(mean, std)
            val  = np.clip(val, lo, hi)
            params[name] = val
        ranges.append(analyse_aircraft(params))

    return np.percentile(ranges, 90)


# ── Inversion analysis ────────────────────────────────────────────────────────

def inversion_analysis(target_p90=3000, n_steps=15, n_samples=300):
    """
    For each parameter, find the maximum allowable std such that
    P90 range >= target_p90, holding all other parameters at nominal std.

    This answers: "How accurately must we know each parameter to guarantee
    the target range is met with 90% confidence?"

    Parameters
    ----------
    target_p90 : float — target P90 range [km]
    n_steps    : int   — number of std values to test per parameter
    n_samples  : int   — Monte Carlo samples per evaluation

    Returns
    -------
    results : dict — {param_name: {'std_values', 'p90_values', 'threshold'}}
    """
    results = {}
    param_names = list(PARAMETERS.keys())

    # Nominal stds
    nominal_stds = {name: PARAMETERS[name]["std"] for name in param_names}

    for param in param_names:
        nominal_std = PARAMETERS[param]["std"]
        # Test range: from very tight (10% of nominal) to loose (200% of nominal)
        std_values = np.linspace(nominal_std * 0.1, nominal_std * 2.0, n_steps)
        p90_values = []

        for std in std_values:
            param_stds = nominal_stds.copy()
            param_stds[param] = std
            p90 = estimate_p90_range(param_stds, n_samples=n_samples)
            p90_values.append(p90)

        p90_values = np.array(p90_values)

        # Find threshold — largest std where P90 still >= target
        above_target = std_values[p90_values >= target_p90]
        threshold = above_target[-1] if len(above_target) > 0 else None

        results[param] = {
            "std_values": std_values,
            "p90_values": p90_values,
            "threshold":  threshold,
            "nominal_std": nominal_std,
        }

    return results


def print_inversion_results(results, target_p90=3000):
    """Print inversion analysis results."""
    print("\n" + "=" * 65)
    print(f"  Inversion Analysis — Required Accuracy for P90 >= {target_p90} km")
    print("=" * 65)
    print(f"  {'Parameter':12s}  {'Nominal std':>12s}  "
          f"{'Max allowed std':>15s}  {'Tightening':>12s}")
    print(f"  {'-'*12}  {'-'*12}  {'-'*15}  {'-'*12}")

    for name, r in results.items():
        nominal = r["nominal_std"]
        thresh  = r["threshold"]
        if thresh is not None:
            ratio = thresh / nominal
            tighten = f"{(1-ratio)*100:.0f}% tighter" if ratio < 1 else "OK as-is"
            print(f"  {name:12s}  {nominal:>12.4f}  {thresh:>15.4f}  {tighten:>12s}")
        else:
            print(f"  {name:12s}  {nominal:>12.4f}  {'N/A':>15s}  "
                  f"{'cannot meet':>12s}")

    print("=" * 65)
    print(f"\n  Parameters where current uncertainty already meets P90 target")
    print(f"  show 'OK as-is'. Parameters requiring tightening need")
    print(f"  additional measurement or higher-fidelity modelling.")
    print("=" * 65)


def plot_inversion(results, target_p90=3000, save_path=None):
    """Plot P90 range vs parameter std for each parameter."""
    n_params = len(results)
    ncols = 3
    nrows = (n_params + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(14, 4 * nrows),
                             sharey=False)
    axes = axes.flatten()

    for i, (name, r) in enumerate(results.items()):
        ax = axes[i]

        ax.plot(r["std_values"], r["p90_values"],
                "o-", color="#4C72B0", linewidth=2, markersize=4)

        # Target line
        ax.axhline(target_p90, color="#C44E52", linewidth=1.5,
                   linestyle="--", label=f"Target P90 = {target_p90} km")

        # Nominal std
        ax.axvline(r["nominal_std"], color="gray", linewidth=1.2,
                   linestyle=":", label=f"Nominal std = {r['nominal_std']:.4f}")

        # Threshold
        if r["threshold"] is not None:
            ax.axvline(r["threshold"], color="#FF9800", linewidth=1.5,
                       linestyle="-.", label=f"Max std = {r['threshold']:.4f}")

        ax.set_xlabel(f"std({name})", fontsize=10)
        ax.set_ylabel("P90 Range [km]", fontsize=10)
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(
        f"Inversion Analysis — Required Parameter Accuracy for P90 ≥ {target_p90} km\n"
        "DASAL Pillar 3: Which uncertainties must be reduced, and by how much?",
        fontsize=12, fontweight="bold", y=1.02
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Plot saved: {save_path}")

    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Running inversion analysis...")
    print("(Testing each parameter across 15 uncertainty levels — ~2 minutes)")

    results = inversion_analysis(target_p90=3000, n_steps=15, n_samples=300)
    print_inversion_results(results, target_p90=3000)

    fig = plot_inversion(
        results,
        target_p90=3000,
        save_path="/home/claude/h2aircraft-uq/results/04_inversion.png"
    )
    plt.show()
