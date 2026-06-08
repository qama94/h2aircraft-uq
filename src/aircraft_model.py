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
from mass_model import compute_OEW, tank_mass, fuel_cell_mass, ETA_GRAV_DEFAULT

# ── Physical constants ────────────────────────────────────────────────────────
G    = 9.81     # gravitational acceleration [m/s²]
E_H2 = 120e6   # specific energy of hydrogen [J/kg]

# ── Nominal design point ───────────────────────────────────────────────────────
# Computed once by sizing at deterministic parameters (set at bottom of file).
# In analysis mode, the hydrogen mass is held fixed at this nominal value.
NOMINAL_M_H2 = None   # set after size_aircraft is defined

# ── Default design parameters ─────────────────────────────────────────────────
DEFAULT_PARAMS = {
    "AR":        9.0,
    "CD0":       0.020,
    "e":         0.80,
    "eta_fc":    0.55,
    "eta_motor": 0.95,
    "eta_prop":  0.85,
    "eta_grav":  0.35,   # tank gravimetric efficiency (replaces fixed OEW)
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


def size_aircraft(params=None, tol=1e-3, max_iter=200):
    """
    DESIGN MODE: iterative sizing to achieve RANGE_TARGET.

    Now with a hydrogen-specific mass model: OEW is computed from the
    hydrogen mass each iteration, creating a strong circular dependency:
        more fuel -> bigger tank -> heavier OEW -> needs more fuel

    Algorithm:
      1. Guess hydrogen mass
      2. Compute OEW = structure + tank(m_h2) + fuel cell system
      3. Compute MTOW = OEW + m_h2
      4. Invert Breguet to find m_h2 required for target range
      5. Repeat until m_h2 converges

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
    eta_grav  = params.get("eta_grav", ETA_GRAV_DEFAULT)

    LD        = compute_LD(AR, CD0, e)
    eta_total = compute_eta_total(eta_fc, eta_motor, eta_prop)

    # Initial guess for hydrogen mass
    m_h2 = 4000.0
    converged = False

    for i in range(max_iter):
        # Compute OEW from current hydrogen mass (mass model)
        OEW  = compute_OEW(m_h2, eta_grav=eta_grav)
        MTOW = OEW + m_h2

        # Invert Breguet: required ln(MTOW/OEW) for target range
        log_ratio = RANGE_TARGET / (LD * eta_total * (E_H2 / G))
        # MTOW/OEW = exp(log_ratio), so MTOW = OEW * exp(log_ratio)
        # But OEW itself depends on m_h2 — so we solve iteratively
        MTOW_req  = OEW * np.exp(log_ratio)
        m_h2_new  = MTOW_req - OEW

        # Relaxation for stable convergence
        m_h2_next = 0.5 * m_h2 + 0.5 * m_h2_new

        if abs(m_h2_next - m_h2) < tol:
            m_h2 = m_h2_next
            converged = True
            break
        m_h2 = m_h2_next

    # Final converged values
    OEW     = compute_OEW(m_h2, eta_grav=eta_grav)
    MTOW    = OEW + m_h2
    range_m = breguet_range(LD, eta_total, MTOW, OEW)

    return {
        "range_m":    range_m,
        "range_km":   range_m / 1000,
        "MTOW":       MTOW,
        "OEW":        OEW,
        "m_fuel":     m_h2,
        "m_tank":     tank_mass(m_h2, eta_grav),
        "m_fc":       fuel_cell_mass(),
        "mf_frac":    m_h2 / MTOW,
        "LD":         LD,
        "eta_total":  eta_total,
        "converged":  converged,
        "iterations": i + 1,
    }


def analyse_aircraft(params):
    """
    ANALYSIS MODE: given uncertain parameters and a FIXED physical aircraft,
    compute the range it achieves.

    The aircraft is DESIGNED once at nominal parameters (fixing m_h2 and the
    physical airframe). In analysis mode the hydrogen mass and structure are
    fixed — only the performance parameters (aerodynamics, propulsion) and
    the tank gravimetric efficiency vary, changing the achieved range.

    Parameters
    ----------
    params : dict with keys AR, CD0, e, eta_fc, eta_motor, eta_prop, eta_grav

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
    eta_grav  = params.get("eta_grav", ETA_GRAV_DEFAULT)

    # Fixed hydrogen mass from nominal design
    m_h2 = NOMINAL_M_H2

    # OEW depends on tank gravimetric efficiency (which can vary)
    OEW  = compute_OEW(m_h2, eta_grav=eta_grav)
    MTOW = OEW + m_h2

    LD        = compute_LD(AR, CD0, e)
    eta_total = compute_eta_total(eta_fc, eta_motor, eta_prop)

    range_m = breguet_range(LD, eta_total, MTOW, OEW)
    return range_m / 1000


# ── Set nominal design point at module load ───────────────────────────────────
_nominal_design = size_aircraft(DEFAULT_PARAMS.copy())
NOMINAL_M_H2 = _nominal_design["m_fuel"]


if __name__ == "__main__":
    result = size_aircraft()
    print("=" * 55)
    print("  H2 Aircraft — Deterministic Design Point")
    print("=" * 55)
    print(f"  Converged:       {result['converged']} ({result['iterations']} iters)")
    print(f"  L/D ratio:       {result['LD']:.2f}")
    print(f"  eta_total:       {result['eta_total']:.3f}")
    print(f"  MTOW:            {result['MTOW']:,.0f} kg")
    print(f"  OEW:             {result['OEW']:,.0f} kg")
    print(f"  Hydrogen mass:   {result['m_fuel']:,.0f} kg")
    print(f"  Tank mass:       {result['m_tank']:,.0f} kg")
    print(f"  Fuel cell mass:  {result['m_fc']:,.0f} kg")
    print(f"  Fuel fraction:   {result['mf_frac']*100:.1f} %")
    print(f"  Range:           {result['range_km']:.1f} km")
    print("=" * 55)

    range_check = analyse_aircraft(DEFAULT_PARAMS)
    print(f"\n  Analysis mode check: {range_check:.1f} km (should be ~3000 km)")
