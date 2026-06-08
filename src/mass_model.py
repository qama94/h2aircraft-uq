"""
mass_model.py
-------------
Hydrogen-specific mass model for the aircraft sizing.

The dominant weight penalty of hydrogen aircraft comes from two sources:
  1. Cryogenic tank mass — liquid hydrogen needs heavy insulated tanks.
     Expressed via gravimetric efficiency:
         eta_grav = m_H2 / (m_H2 + m_tank)
     Typical values 0.30-0.50 (Hoelzen 2022, Brewer 1991).
     eta_grav = 0.35 means the tank weighs ~1.86x the hydrogen it holds.

  2. Fuel cell + powertrain mass — scales with required power.
     Expressed via system specific power:
         m_fc = P_required / specific_power
     Typical specific power 2 kW/kg for current aviation fuel cell systems.

This replaces the fixed OEW assumption with a computed structural mass
that responds to the hydrogen system size — making the iterative sizing
loop physically meaningful (strong circular dependency).

"""

import numpy as np

# ── Hydrogen system constants ─────────────────────────────────────────────────
ETA_GRAV_DEFAULT     = 0.35    # tank gravimetric efficiency [-]
SPECIFIC_POWER_FC    = 2.0     # fuel cell system specific power [kW/kg]
STRUCTURE_PAYLOAD    = 38000   # structure + payload mass [kg] (true empty)
CRUISE_POWER_FRACTION = 0.40   # cruise power as fraction of installed power
INSTALLED_POWER_KW   = 12000   # installed fuel cell power [kW] for medium-haul


def tank_mass(m_h2, eta_grav=ETA_GRAV_DEFAULT):
    """
    Cryogenic tank mass from hydrogen mass and gravimetric efficiency.

    eta_grav = m_H2 / (m_H2 + m_tank)
    => m_tank = m_H2 * (1 - eta_grav) / eta_grav

    Parameters
    ----------
    m_h2     : float — hydrogen mass [kg]
    eta_grav : float — tank gravimetric efficiency [-]

    Returns
    -------
    m_tank : float — tank mass [kg]
    """
    return m_h2 * (1.0 - eta_grav) / eta_grav


def fuel_cell_mass(power_kw=INSTALLED_POWER_KW, specific_power=SPECIFIC_POWER_FC):
    """
    Fuel cell system mass from installed power and specific power.

    m_fc = P_installed / specific_power

    Parameters
    ----------
    power_kw       : float — installed power [kW]
    specific_power : float — system specific power [kW/kg]

    Returns
    -------
    m_fc : float — fuel cell system mass [kg]
    """
    return power_kw / specific_power


def compute_OEW(m_h2, eta_grav=ETA_GRAV_DEFAULT,
                power_kw=INSTALLED_POWER_KW,
                specific_power=SPECIFIC_POWER_FC,
                structure_payload=STRUCTURE_PAYLOAD):
    """
    Compute operating empty weight as a function of hydrogen system size.

    OEW = structure + payload + tank mass + fuel cell system mass

    This makes OEW DEPEND on the hydrogen mass, creating a genuine
    circular dependency: more fuel -> bigger tank -> heavier aircraft
    -> needs more fuel. This is the physically correct behaviour for
    hydrogen aircraft, unlike the fixed-OEW simplification.

    Parameters
    ----------
    m_h2              : float — hydrogen mass [kg]
    eta_grav          : float — tank gravimetric efficiency
    power_kw          : float — installed power [kW]
    specific_power    : float — system specific power [kW/kg]
    structure_payload : float — structure + payload mass [kg]

    Returns
    -------
    OEW : float — operating empty weight [kg]
    """
    m_tank = tank_mass(m_h2, eta_grav)
    m_fc   = fuel_cell_mass(power_kw, specific_power)
    return structure_payload + m_tank + m_fc


if __name__ == "__main__":
    print("=" * 55)
    print("  Hydrogen Mass Model — Component Breakdown")
    print("=" * 55)

    # Example with a realistic hydrogen mass
    m_h2 = 4000  # kg

    m_tank = tank_mass(m_h2)
    m_fc   = fuel_cell_mass()
    OEW    = compute_OEW(m_h2)

    print(f"  Hydrogen mass:        {m_h2:>8,.0f} kg")
    print(f"  Tank mass:            {m_tank:>8,.0f} kg  "
          f"(eta_grav = {ETA_GRAV_DEFAULT})")
    print(f"  Fuel cell system:     {m_fc:>8,.0f} kg  "
          f"({SPECIFIC_POWER_FC} kW/kg)")
    print(f"  Structure + payload:  {STRUCTURE_PAYLOAD:>8,.0f} kg")
    print(f"  {'-'*40}")
    print(f"  OEW (computed):       {OEW:>8,.0f} kg")
    print(f"  MTOW:                 {OEW + m_h2:>8,.0f} kg")
    print(f"  Fuel fraction:        {m_h2/(OEW + m_h2)*100:>7.1f} %")
    print("=" * 55)
