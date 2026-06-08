"""
design_optimisation.py
----------------------
Deterministic optimal design for the hydrogen aircraft range model.

Finds the combination of design variables (AR, e target, eta_fc target)
that maximises range at the nominal (deterministic) parameter values,
subject to physical constraints.

This is the baseline against which the robust design is compared.
The key result: optimal deterministic design is NOT the same as optimal
robust design — because the deterministic optimum may sit in a region
of high sensitivity, making it fragile to parameter uncertainty.

"""

import numpy as np
from scipy.optimize import minimize
from aircraft_model import analyse_aircraft, compute_LD, compute_eta_total
from uncertainty_model import PARAMETERS


# ── Objective and constraints ─────────────────────────────────────────────────

def negative_range(x):
    """
    Objective function: negative range (we minimise, so this maximises range).

    Design variables:
      x[0] = AR    (aspect ratio)
      x[1] = e     (Oswald efficiency factor)
      x[2] = eta_fc (fuel cell efficiency target)
    """
    AR, e, eta_fc = x

    params = {name: p["mean"] for name, p in PARAMETERS.items()}
    params["AR"]     = AR
    params["e"]      = e
    params["eta_fc"] = eta_fc

    return -analyse_aircraft(params)


def run_deterministic_optimisation():
    """
    Find the deterministic optimal design point.

    Design variables: AR, e, eta_fc
    Fixed parameters: CD0, eta_motor, eta_prop, OEW at nominal values

    Returns
    -------
    result : dict with optimal design and range
    """
    # Initial guess — nominal values
    x0 = [
        PARAMETERS["AR"]["mean"],
        PARAMETERS["e"]["mean"],
        PARAMETERS["eta_fc"]["mean"],
    ]

    # Bounds — physical limits
    bounds = [
        PARAMETERS["AR"]["bounds"],
        PARAMETERS["e"]["bounds"],
        PARAMETERS["eta_fc"]["bounds"],
    ]

    result = minimize(
        negative_range,
        x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"ftol": 1e-12, "gtol": 1e-8, "maxiter": 1000}
    )

    AR_opt, e_opt, eta_fc_opt = result.x
    range_opt = -result.fun

    # Build full params at optimal point
    params_opt = {name: p["mean"] for name, p in PARAMETERS.items()}
    params_opt["AR"]     = AR_opt
    params_opt["e"]      = e_opt
    params_opt["eta_fc"] = eta_fc_opt

    LD_opt       = compute_LD(AR_opt, PARAMETERS["CD0"]["mean"], e_opt)
    eta_total_opt = compute_eta_total(
        eta_fc_opt,
        PARAMETERS["eta_motor"]["mean"],
        PARAMETERS["eta_prop"]["mean"]
    )

    return {
        "AR":        AR_opt,
        "e":         e_opt,
        "eta_fc":    eta_fc_opt,
        "LD":        LD_opt,
        "eta_total": eta_total_opt,
        "range_km":  range_opt,
        "success":   result.success,
    }


def print_optimisation_results(nominal_range, opt_result):
    """Print comparison of nominal vs optimal deterministic design."""
    print("\n" + "=" * 55)
    print("  Deterministic Design Optimisation")
    print("=" * 55)
    print(f"  {'Parameter':15s}  {'Nominal':>10s}  {'Optimal':>10s}")
    print(f"  {'-'*15}  {'-'*10}  {'-'*10}")
    print(f"  {'AR':15s}  {PARAMETERS['AR']['mean']:>10.2f}  "
          f"{opt_result['AR']:>10.2f}")
    print(f"  {'e':15s}  {PARAMETERS['e']['mean']:>10.3f}  "
          f"{opt_result['e']:>10.3f}")
    print(f"  {'eta_fc':15s}  {PARAMETERS['eta_fc']['mean']:>10.3f}  "
          f"{opt_result['eta_fc']:>10.3f}")
    print(f"  {'L/D':15s}  "
          f"{compute_LD(PARAMETERS['AR']['mean'], PARAMETERS['CD0']['mean'], PARAMETERS['e']['mean']):>10.2f}  "
          f"{opt_result['LD']:>10.2f}")
    print(f"  {'-'*15}  {'-'*10}  {'-'*10}")
    print(f"  {'Range [km]':15s}  {nominal_range:>10.1f}  "
          f"{opt_result['range_km']:>10.1f}")
    gain = opt_result['range_km'] - nominal_range
    print(f"  {'Gain [km]':15s}  {'':>10s}  {gain:>+10.1f}")
    print("=" * 55)
    print(f"\n  Optimiser converged: {opt_result['success']}")
    print(f"  The optimal design pushes AR and e to their upper bounds")
    print(f"  — this maximises L/D. But high AR, high e designs may sit")
    print(f"  in regions of higher sensitivity. The robust design will")
    print(f"  trade some nominal range for lower uncertainty penalty.")
    print("=" * 55)


if __name__ == "__main__":
    # Nominal range at default parameters
    nominal_params = {name: p["mean"] for name, p in PARAMETERS.items()}
    nominal_range  = analyse_aircraft(nominal_params)

    print("Running deterministic optimisation...")
    opt_result = run_deterministic_optimisation()
    print_optimisation_results(nominal_range, opt_result)
