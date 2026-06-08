"""
model_form_error.py
-------------------
Demonstrates the distinction between PARAMETRIC uncertainty and
MODEL-FORM (structural) error — the intellectual core of DASAL.

The Oswald efficiency factor e is typically estimated from an empirical
correlation, e.g. (Raymer):
    e_correlation = 1.78 * (1 - 0.045 * AR**0.68) - 0.64

There are two fundamentally different ways this can be "wrong":

CASE 1 — PARAMETRIC UNCERTAINTY:
    The correlation is structurally correct, but the true value scatters
    around it. We model e ~ Normal(e_correlation, sigma).
    KEY PROPERTY: measuring e more precisely SHRINKS this uncertainty
    toward the true value. More data fixes it.

CASE 2 — MODEL-FORM ERROR:
    The correlation ITSELF is biased for a novel hydrogen aircraft —
    different wing loading, fuselage-tank integration, and cryogenic
    tank placement violate the assumptions the correlation was fit on.
    We model e_true = e_correlation * (1 - bias) + scatter.
    KEY PROPERTY: measuring e on CONVENTIONAL aircraft does NOT fix this.
    The structure is wrong, not the value. You must change the model.

This module shows that these two cases:
  (a) propagate to DIFFERENT range distributions
  (b) respond DIFFERENTLY to added measurement data
     — parametric uncertainty shrinks; structural bias persists

This is exactly the DASAL problem: in a coupled digital thread, a
structural error in one component looks like parametric uncertainty
in the system KPIs until a framework explicitly separates them.

Author: Gamar Ismayilova
Project: h2aircraft-uq — DASAL PhD Preparation
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from aircraft_model import analyse_aircraft, DEFAULT_PARAMS


def oswald_correlation(AR):
    """
    Raymer empirical correlation for Oswald efficiency factor.

    e = 1.78 * (1 - 0.045 * AR^0.68) - 0.64

    This is fit on CONVENTIONAL aircraft data. The question DASAL asks:
    does it hold for a novel hydrogen aircraft configuration?

    Parameters
    ----------
    AR : float — aspect ratio

    Returns
    -------
    e : float — Oswald efficiency factor from correlation
    """
    return 1.78 * (1 - 0.045 * AR**0.68) - 0.64


def sample_parametric(AR, n, scatter=0.06, rng=None):
    """
    CASE 1 — parametric uncertainty.

    The correlation is correct; true e scatters around it.
    e ~ Normal(e_correlation, scatter)
    """
    if rng is None:
        rng = np.random.default_rng(42)
    e_corr = oswald_correlation(AR)
    return rng.normal(e_corr, scatter, n)


def sample_model_form(AR, n, bias=0.12, scatter=0.06, rng=None):
    """
    CASE 2 — model-form error.

    The correlation is structurally biased for a hydrogen aircraft.
    The TRUE e is systematically lower (worse) than the correlation predicts,
    because the tank-fuselage integration adds interference drag the
    correlation does not capture.
    e_true ~ Normal(e_correlation * (1 - bias), scatter)
    """
    if rng is None:
        rng = np.random.default_rng(42)
    e_corr = oswald_correlation(AR)
    e_biased_mean = e_corr * (1 - bias)
    return rng.normal(e_biased_mean, scatter, n)


def propagate_to_range(e_samples):
    """Propagate e samples through the aircraft model to get range distribution."""
    ranges = []
    for e_val in e_samples:
        params = DEFAULT_PARAMS.copy()
        params["e"] = np.clip(e_val, 0.3, 1.0)
        ranges.append(analyse_aircraft(params))
    return np.array(ranges)


def demonstrate_data_response(AR=9.0, n_mc=2000):
    """
    Show how each case responds to added measurement data.

    For parametric uncertainty: as we measure e, the estimate converges
    to the true value and range uncertainty shrinks around the TRUE range.

    For model-form error: measuring e on conventional aircraft converges
    to the BIASED correlation value — so the range prediction remains
    systematically wrong. The gap does not close.
    """
    rng = np.random.default_rng(42)
    e_corr = oswald_correlation(AR)

    # The TRUE Oswald factor for the hydrogen aircraft (unknown to designer)
    e_true_parametric = e_corr            # parametric: correlation is right
    e_true_modelform  = e_corr * (1-0.12) # model-form: true value is biased low

    n_measurements = [0, 1, 3, 5, 10, 20, 50]

    # Range computed from the BELIEF after n measurements
    param_range_belief = []
    param_range_truth  = []
    mf_range_belief    = []
    mf_range_truth     = []

    for n in n_measurements:
        # PARAMETRIC: the correlation is correct. Measurements of the actual
        # aircraft converge the belief to the true value. Belief -> truth.
        if n == 0:
            belief_param = e_corr
        else:
            rng_p = np.random.default_rng(100 + n)
            meas = rng_p.normal(e_true_parametric, 0.06, n)
            belief_param = (e_corr/0.06**2 + meas.sum()/0.015**2) / \
                           (1/0.06**2 + n/0.015**2)
        p = DEFAULT_PARAMS.copy(); p["e"] = belief_param
        param_range_belief.append(analyse_aircraft(p))
        pt = DEFAULT_PARAMS.copy(); pt["e"] = e_true_parametric
        param_range_truth.append(analyse_aircraft(pt))

        # MODEL-FORM: the designer measures CONVENTIONAL aircraft (which the
        # correlation describes correctly) and the belief converges to e_corr.
        # But the TRUE hydrogen aircraft value is e_corr*(1-bias). So the
        # belief converges to the correlation value while the truth sits
        # systematically lower — the gap never closes.
        if n == 0:
            belief_mf = e_corr
        else:
            rng_m = np.random.default_rng(200 + n)
            meas = rng_m.normal(e_corr, 0.06, n)  # conventional aircraft data
            belief_mf = (e_corr/0.06**2 + meas.sum()/0.015**2) / \
                        (1/0.06**2 + n/0.015**2)
        pmf = DEFAULT_PARAMS.copy(); pmf["e"] = belief_mf
        mf_range_belief.append(analyse_aircraft(pmf))
        pmft = DEFAULT_PARAMS.copy(); pmft["e"] = e_true_modelform
        mf_range_truth.append(analyse_aircraft(pmft))

    return {
        "n_measurements":     n_measurements,
        "param_range_belief": np.array(param_range_belief),
        "param_range_truth":  np.array(param_range_truth),
        "mf_range_belief":    np.array(mf_range_belief),
        "mf_range_truth":     np.array(mf_range_truth),
        "e_corr":             e_corr,
        "e_true_modelform":   e_true_modelform,
    }


def run_analysis(save_path=None):
    """Full model-form vs parametric error analysis with plots."""
    AR = 9.0

    # Part 1: range distributions for each case
    e_param = sample_parametric(AR, 2000)
    e_mf    = sample_model_form(AR, 2000)
    range_param = propagate_to_range(e_param)
    range_mf    = propagate_to_range(e_mf)

    # Part 2: data response
    data = demonstrate_data_response(AR)

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: range distributions
    ax1 = axes[0]
    ax1.hist(range_param, bins=50, density=True, alpha=0.6,
             color="#4C72B0", label=f"Parametric (μ={range_param.mean():.0f} km)")
    ax1.hist(range_mf, bins=50, density=True, alpha=0.6,
             color="#C44E52", label=f"Model-form (μ={range_mf.mean():.0f} km)")
    ax1.axvline(range_param.mean(), color="#4C72B0", linewidth=2, linestyle="--")
    ax1.axvline(range_mf.mean(), color="#C44E52", linewidth=2, linestyle="--")
    ax1.set_xlabel("Aircraft Range [km]", fontsize=12)
    ax1.set_ylabel("Probability Density", fontsize=12)
    ax1.set_title(
        "Parametric vs Model-Form Error — Range Distributions\n"
        "Same scatter, but model-form error shifts the whole distribution",
        fontsize=11, fontweight="bold"
    )
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Right: data response — the key DASAL plot
    ax2 = axes[1]
    ns = data["n_measurements"]

    ax2.plot(ns, data["param_range_belief"], "o-", color="#4C72B0",
             linewidth=2, markersize=6, label="Parametric: belief")
    ax2.plot(ns, data["param_range_truth"], "--", color="#4C72B0",
             linewidth=1.5, alpha=0.6, label="Parametric: truth")
    ax2.plot(ns, data["mf_range_belief"], "s-", color="#C44E52",
             linewidth=2, markersize=6, label="Model-form: belief")
    ax2.plot(ns, data["mf_range_truth"], "--", color="#C44E52",
             linewidth=1.5, alpha=0.6, label="Model-form: truth")

    # Highlight the persistent gap
    final_gap = data["mf_range_belief"][-1] - data["mf_range_truth"][-1]
    ax2.annotate(
        f"Persistent gap: {final_gap:.0f} km\n(structural error\nnever closes)",
        xy=(50, (data["mf_range_belief"][-1] + data["mf_range_truth"][-1])/2),
        xytext=(25, data["mf_range_belief"][-1] + 80),
        fontsize=9, color="#C44E52",
        arrowprops=dict(arrowstyle="->", color="#C44E52", lw=1.2)
    )

    ax2.set_xlabel("Number of Measurements (conventional aircraft)", fontsize=12)
    ax2.set_ylabel("Predicted Range [km]", fontsize=12)
    ax2.set_title(
        "Response to Data — The DASAL Distinction\n"
        "Parametric belief → truth; model-form belief → WRONG value",
        fontsize=11, fontweight="bold"
    )
    ax2.legend(fontsize=8, loc="center right")
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(ns)

    plt.suptitle(
        "Parametric Uncertainty vs Model-Form Error\n"
        "Why measuring harder does not close a structural gap — the core DASAL problem",
        fontsize=12, fontweight="bold", y=1.03
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Plot saved: {save_path}")

    return fig, range_param, range_mf, data


def print_summary(range_param, range_mf, data):
    print("\n" + "=" * 62)
    print("  Parametric Uncertainty vs Model-Form Error")
    print("=" * 62)
    print(f"  Oswald correlation value (AR=9):  {data['e_corr']:.3f}")
    print(f"  True value if model-form biased:  {data['e_true_modelform']:.3f}")
    print(f"")
    print(f"  PARAMETRIC case:")
    print(f"    Mean range:        {range_param.mean():.0f} km")
    print(f"    Std range:         {range_param.std():.0f} km")
    print(f"  MODEL-FORM case:")
    print(f"    Mean range:        {range_mf.mean():.0f} km")
    print(f"    Std range:         {range_mf.std():.0f} km")
    print(f"    Systematic shift:  {range_param.mean()-range_mf.mean():.0f} km")
    print(f"")
    print(f"  RESPONSE TO 50 MEASUREMENTS:")
    print(f"    Parametric belief vs truth gap:  "
          f"{abs(data['param_range_belief'][-1]-data['param_range_truth'][-1]):.0f} km")
    print(f"    Model-form belief vs truth gap:  "
          f"{abs(data['mf_range_belief'][-1]-data['mf_range_truth'][-1]):.0f} km")
    print(f"")
    print(f"  KEY INSIGHT:")
    print(f"  Parametric uncertainty closes with data — belief reaches truth.")
    print(f"  Model-form error does NOT close — measuring conventional")
    print(f"  aircraft converges the belief to the WRONG value for the")
    print(f"  novel hydrogen configuration. The gap persists regardless")
    print(f"  of how much data is collected. This is why DASAL needs a")
    print(f"  framework that explicitly separates the two.")
    print("=" * 62)


if __name__ == "__main__":
    print("Running parametric vs model-form error analysis...")
    fig, range_param, range_mf, data = run_analysis(
        save_path="/home/claude/h2aircraft-uq/results/05_model_form_error.png"
    )
    print_summary(range_param, range_mf, data)
    plt.show()
