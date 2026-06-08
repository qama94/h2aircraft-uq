"""
uncertainty_model.py
--------------------
Uncertainty characterisation for the hydrogen aircraft range model.

For each uncertain input parameter we define:
  - Distribution type (normal or uniform)
  - Mean / nominal value
  - Standard deviation or bounds
  - Uncertainty type: ALEATORY or EPISTEMIC
  - Physical justification

Key distinction:
  ALEATORY  — irreducible randomness inherent to the physical process.
               More data does not shrink this uncertainty.
  EPISTEMIC — knowledge gap. Better modelling or measurement can reduce it.
               This is the uncertainty DASAL aims to characterise and manage.

References:
  Hoelzen 2022 — hydrogen aviation parameter ranges
  Raymer 2018  — aircraft design: a conceptual approach (aerodynamic parameters)
  SALib docs   — parameter bounds format for Sobol sampling

"""

import numpy as np


# ── Parameter definitions ─────────────────────────────────────────────────────
#
# Each parameter is a dict with:
#   mean   : nominal / best-estimate value
#   std    : standard deviation (for normal distributions)
#   bounds : [lower, upper] (for uniform distributions, used by SALib)
#   dist   : 'normal' or 'uniform'
#   type   : 'aleatory' or 'epistemic'
#   unit   : physical unit string
#   note   : physical justification for the uncertainty level

PARAMETERS = {

    # ── Aerodynamic parameters ────────────────────────────────────────────────

    "AR": {
        "mean":   9.0,
        "std":    0.5,
        "bounds": [7.5, 10.5],
        "dist":   "normal",
        "type":   "epistemic",
        "unit":   "-",
        "note":   (
            "Aspect ratio uncertainty reflects conceptual-stage wing design "
            "freedom. At early design stage, AR is a decision variable with "
            "structural and aerodynamic trade-offs not yet fully resolved. "
            "Reducible with detailed structural analysis."
        ),
    },

    "CD0": {
        "mean":   0.020,
        "std":    0.003,
        "bounds": [0.014, 0.026],
        "dist":   "normal",
        "type":   "epistemic",
        "unit":   "-",
        "note":   (
            "Zero-lift drag uncertainty from surface finish, interference drag, "
            "and Reynolds number effects not captured at conceptual stage. "
            "Reducible with higher-fidelity CFD or wind tunnel testing."
        ),
    },

    "e": {
        "mean":   0.80,
        "std":    0.06,
        "bounds": [0.65, 0.95],
        "dist":   "normal",
        "type":   "epistemic",
        "unit":   "-",
        "note":   (
            "Oswald efficiency factor is an empirical correlation with wide "
            "scatter across aircraft types. For a novel hydrogen aircraft "
            "planform, no validated empirical data exists yet. This is a "
            "STRUCTURAL MODELLING ASSUMPTION — not a tuneable parameter. "
            "The correlation itself may not hold for this configuration. "
            "Reducible only with dedicated aerodynamic testing or high-fidelity "
            "CFD across the full flight envelope."
        ),
    },

    # ── Propulsion parameters ─────────────────────────────────────────────────

    "eta_fc": {
        "mean":   0.55,
        "std":    0.04,
        "bounds": [0.45, 0.65],
        "dist":   "normal",
        "type":   "epistemic",
        "unit":   "-",
        "note":   (
            "Fuel cell stack efficiency at cruise conditions. Uncertainty from "
            "degradation over flight cycles, temperature sensitivity, and "
            "limited operational data for aviation-grade stacks. "
            "Reducible with stack testing and degradation modelling."
        ),
    },

    "eta_motor": {
        "mean":   0.95,
        "std":    0.01,
        "bounds": [0.92, 0.98],
        "dist":   "normal",
        "type":   "epistemic",
        "unit":   "-",
        "note":   (
            "Electric motor efficiency is well-characterised technology. "
            "Narrow uncertainty band reflects mature motor design knowledge. "
            "Small residual uncertainty from thermal effects at altitude."
        ),
    },

    "eta_prop": {
        "mean":   0.85,
        "std":    0.03,
        "bounds": [0.78, 0.92],
        "dist":   "normal",
        "type":   "epistemic",
        "unit":   "-",
        "note":   (
            "Propeller efficiency uncertainty from blade design optimisation "
            "and off-design operating conditions. Moderate uncertainty for "
            "a novel hydrogen aircraft propeller not yet tested in flight."
        ),
    },

    # ── Weight parameters ─────────────────────────────────────────────────────

    "eta_grav": {
        "mean":   0.35,
        "std":    0.04,
        "bounds": [0.25, 0.50],
        "dist":   "normal",
        "type":   "epistemic",
        "unit":   "-",
        "note":   (
            "Cryogenic tank gravimetric efficiency: m_H2 / (m_H2 + m_tank). "
            "Large uncertainty because liquid hydrogen tank technology for "
            "aviation is immature. Values range 0.25-0.50 across studies "
            "(Hoelzen 2022, Brewer 1991). This drives the dominant hydrogen "
            "weight penalty and is highly reducible with tank development. "
            "Strongly epistemic — almost no flight-validated data exists."
        ),
    },
}


# ── Aleatory / epistemic summary ──────────────────────────────────────────────

def print_uncertainty_taxonomy():
    """Print a summary of uncertainty types for all parameters."""
    print("\n" + "=" * 60)
    print("  Uncertainty Taxonomy — H2 Aircraft Parameters")
    print("=" * 60)

    aleatory = {k: v for k, v in PARAMETERS.items() if v["type"] == "aleatory"}
    epistemic = {k: v for k, v in PARAMETERS.items() if v["type"] == "epistemic"}

    print(f"\n  EPISTEMIC ({len(epistemic)} parameters) — reducible with better data:")
    for name, p in epistemic.items():
        cv = p["std"] / abs(p["mean"]) * 100  # coefficient of variation
        print(f"    {name:12s}  mean={p['mean']:>8.4f}  std={p['std']:>7.4f}  "
              f"CV={cv:5.1f}%")

    if aleatory:
        print(f"\n  ALEATORY ({len(aleatory)} parameters) — irreducible randomness:")
        for name, p in aleatory.items():
            cv = p["std"] / abs(p["mean"]) * 100
            print(f"    {name:12s}  mean={p['mean']:>8.4f}  std={p['std']:>7.4f}  "
                  f"CV={cv:5.1f}%")
    else:
        print("\n  ALEATORY: none — all uncertainty is epistemic at conceptual stage.")
        print("  (Aleatory uncertainty, e.g. atmospheric turbulence, is averaged")
        print("   out over the mission profile in the Breguet formulation.)")

    print("=" * 60)


def get_salib_problem():
    """
    Format parameter bounds for SALib Sobol sampling.

    SALib expects:
        problem = {
            'num_vars': N,
            'names': [...],
            'bounds': [[lo, hi], ...]
        }

    Returns
    -------
    problem : dict — SALib-compatible problem definition
    """
    names  = list(PARAMETERS.keys())
    bounds = [PARAMETERS[n]["bounds"] for n in names]

    return {
        "num_vars": len(names),
        "names":    names,
        "bounds":   bounds,
    }


def sample_parameters(n_samples=1, rng=None):
    """
    Draw random samples from parameter distributions (for Monte Carlo).

    Parameters
    ----------
    n_samples : int — number of samples to draw
    rng       : numpy RandomGenerator (for reproducibility)

    Returns
    -------
    samples : dict — {param_name: array of shape (n_samples,)}
    """
    if rng is None:
        rng = np.random.default_rng(seed=42)

    samples = {}
    for name, p in PARAMETERS.items():
        if p["dist"] == "normal":
            vals = rng.normal(loc=p["mean"], scale=p["std"], size=n_samples)
            # Clip to physical bounds
            lo, hi = p["bounds"]
            vals = np.clip(vals, lo, hi)
        elif p["dist"] == "uniform":
            lo, hi = p["bounds"]
            vals = rng.uniform(lo, hi, size=n_samples)
        samples[name] = vals

    return samples


if __name__ == "__main__":
    print_uncertainty_taxonomy()

    # Show a single random sample
    s = sample_parameters(n_samples=5)
    print("\n  Example random samples (first 5 draws):")
    print(f"  {'Parameter':12s} {'Sample values':>40s}")
    print(f"  {'-'*12} {'-'*40}")
    for name, vals in s.items():
        val_str = "  ".join([f"{v:.4f}" for v in vals])
        print(f"  {name:12s} {val_str}")
