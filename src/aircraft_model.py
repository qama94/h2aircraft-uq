"""
aircraft_model.py
-----------------
Physics-based range model for a medium-haul hydrogen fuel cell aircraft.

Two modes:
  1. size_aircraft()   — design mode: find MTOW/fuel needed for target range
  2. analyse_aircraft() — analysis mode: given fixed design, compute range
                          from uncertain parameters. Used for UQ.

The circular dependency in sizing:
  MTOW = OEW + m_fuel
  m_fuel depends on range, which depends on MTOW
  → resolved by iterative convergence

Author: Gamar Ismayilova
Project: h2aircraft-uq — DASAL PhD Preparation
"""

import numpy as np

# ── Physical constants ────────────────────────────────────────────────────────
G    = 9.81     # gravitational acceleration [m/s²]
E_H2 = 120e6   # specific energy of hydrogen [J/kg]

# ── Nominal design point (from sizing at deterministic parameters) ─────────────
# Fixed design: MTOW and OEW are set at the deterministic design point.
# In analysis mode, only aerodynamic and propulsion parameters vary.
NOMINAL_MTOW   = 46502.0   # kg — from deterministic sizing
NOMINAL_OEW    = 45000.0   # kg — operating empty weight (fixed structure)

# ── Default design parameters ─────────────────────────────────────────────────
DEFAULT_PARAMS = {
    "AR":        9.0,
    "CD0":       0.020,
    "e":         0.80,
    "eta_fc":    0.55,
    "eta_motor": 0.95,
    "eta_prop":  0.85,
    "OEW":       45000,
    "mf_frac":   0.20,
}

RANGE_TARGET = 3_000_000   # m


def compute_LD(AR, CD0, e):
    """Maximum lift-to-drag ratio from drag polar."""
    return 0.5 * np.sqrt(np.pi * AR * e / CD0)


def compute_eta_total(eta_fc, eta_motor, eta_prop):
    """Total propulsive efficiency."""
    return eta_fc * eta_motor * eta_prop


def breguet_range(LD, eta_total, MTOW, OEW):
    """Modified Breguet range equation for hydrogen propulsion [m]."""
    if np.any(MTOW <= OEW):
        return 0.0
    return LD * eta_total * (E_H2 / G) * np.log(MTOW / OEW)


def size_aircraft(params=None, tol=1e-6, max_iter=100):
    """
    DESIGN MODE: iterative MTOW sizing to achieve RANGE_TARGET.
    Returns converged design point.
    """
    if params is None:
        params = DEFAULT_PARAMS.copy()

    AR        = params["AR"]
    CD0       = params["CD0"]
    e         = params["e"]
    eta_fc    = params["eta_fc"]
    eta_motor = params["eta_motor"]
    eta_prop  = params["eta_prop"]
    OEW       = params["OEW"]
    mf_frac   = params.get("mf_frac", 0.20)

    LD        = compute_LD(AR, CD0, e)
    eta_total = compute_eta_total(eta_fc, eta_motor, eta_prop)

    converged = False
    for i in range(max_iter):
        MTOW = OEW / (1.0 - mf_frac)
        log_ratio = RANGE_TARGET / (LD * eta_total * (E_H2 / G))
        MTOW_new  = OEW * np.exp(log_ratio)
        m_fuel    = MTOW_new - OEW
        mf_new    = m_fuel / MTOW_new

        if abs(mf_new - mf_frac) < tol:
            mf_frac   = mf_new
            MTOW      = MTOW_new
            converged = True
            break
        mf_frac = mf_new

    range_m = breguet_range(LD, eta_total, MTOW, OEW)

    return {
        "range_m":    range_m,
        "range_km":   range_m / 1000,
        "MTOW":       MTOW,
        "m_fuel":     MTOW - OEW,
        "mf_frac":    mf_frac,
        "LD":         LD,
        "eta_total":  eta_total,
        "converged":  converged,
        "iterations": i + 1,
    }


def analyse_aircraft(params):
    """
    ANALYSIS MODE: given uncertain parameters and FIXED design (MTOW, OEW),
    compute the range this aircraft actually achieves.

    This is the correct mode for uncertainty propagation:
    - The aircraft was DESIGNED for 3000 km at the nominal parameter values
    - Now we ask: if the real parameters differ from nominal, what range do we get?

    Parameters
    ----------
    params : dict with keys AR, CD0, e, eta_fc, eta_motor, eta_prop, OEW

    Returns
    -------
    range_km : float — achieved range [km]
    """
    AR        = params["AR"]
    CD0       = params["CD0"]
    e         = params["e"]
    eta_fc    = params["eta_fc"]
    eta_motor = params["eta_motor"]
    eta_prop  = params["eta_prop"]
    OEW       = params.get("OEW", NOMINAL_OEW)

    # Fixed fuel mass from nominal design
    m_fuel = NOMINAL_MTOW - NOMINAL_OEW
    MTOW   = OEW + m_fuel   # OEW varies but fuel is fixed from design

    LD        = compute_LD(AR, CD0, e)
    eta_total = compute_eta_total(eta_fc, eta_motor, eta_prop)

    range_m = breguet_range(LD, eta_total, MTOW, OEW)
    return range_m / 1000


if __name__ == "__main__":
    # Design mode
    result = size_aircraft()
    print("=" * 50)
    print("  H2 Aircraft — Deterministic Design Point")
    print("=" * 50)
    print(f"  Converged:       {result['converged']} ({result['iterations']} iters)")
    print(f"  L/D ratio:       {result['LD']:.2f}")
    print(f"  eta_total:       {result['eta_total']:.3f}")
    print(f"  MTOW:            {result['MTOW']:,.0f} kg")
    print(f"  Fuel mass:       {result['m_fuel']:,.0f} kg")
    print(f"  Fuel fraction:   {result['mf_frac']:.3f}")
    print(f"  Range:           {result['range_km']:.1f} km")
    print("=" * 50)

    # Analysis mode check — should give ~3000 km at nominal params
    range_check = analyse_aircraft(DEFAULT_PARAMS)
    print(f"\n  Analysis mode check: {range_check:.1f} km (should be ~3000 km)")
