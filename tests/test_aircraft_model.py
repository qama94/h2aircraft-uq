"""
test_aircraft_model.py
----------------------
Unit tests for the hydrogen aircraft UQ project.

Tests cover:
  - Physical model correctness
  - Uncertainty model consistency
  - Monte Carlo propagation validity
  - Sobol index properties
  - Bayesian update correctness

Run with:
  cd h2aircraft-uq
  python -m pytest tests/ -v

Author: Gamar Ismayilova
Project: h2aircraft-uq — DASAL PhD Preparation
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

from aircraft_model import (
    compute_LD, compute_eta_total, breguet_range,
    size_aircraft, analyse_aircraft, DEFAULT_PARAMS
)
from uncertainty_model import PARAMETERS, get_salib_problem, sample_parameters
from propagation import run_monte_carlo
from bayesian_update import bayesian_update
from mass_model import tank_mass, fuel_cell_mass, compute_OEW, ETA_GRAV_DEFAULT
from model_form_error import oswald_correlation, sample_parametric, sample_model_form


# ── Aircraft model tests ──────────────────────────────────────────────────────

class TestPhysicalModel:

    def test_LD_positive(self):
        """L/D ratio must be positive for physical parameter values."""
        LD = compute_LD(AR=9.0, CD0=0.020, e=0.80)
        assert LD > 0

    def test_LD_increases_with_AR(self):
        """Higher aspect ratio should give higher L/D."""
        LD_low  = compute_LD(AR=7.0, CD0=0.020, e=0.80)
        LD_high = compute_LD(AR=12.0, CD0=0.020, e=0.80)
        assert LD_high > LD_low

    def test_LD_decreases_with_CD0(self):
        """Higher zero-lift drag should reduce L/D."""
        LD_low  = compute_LD(AR=9.0, CD0=0.030, e=0.80)
        LD_high = compute_LD(AR=9.0, CD0=0.015, e=0.80)
        assert LD_high > LD_low

    def test_eta_total_product(self):
        """Total efficiency must equal product of component efficiencies."""
        eta = compute_eta_total(0.55, 0.95, 0.85)
        assert abs(eta - 0.55 * 0.95 * 0.85) < 1e-10

    def test_eta_total_bounded(self):
        """Total efficiency must be between 0 and 1."""
        eta = compute_eta_total(0.55, 0.95, 0.85)
        assert 0 < eta < 1

    def test_breguet_range_positive(self):
        """Range must be positive when MTOW > OEW."""
        R = breguet_range(LD=16.0, eta_total=0.44, MTOW=46500, OEW=45000)
        assert R > 0

    def test_breguet_range_zero_fuel(self):
        """Range must be zero when MTOW equals OEW (no fuel)."""
        R = breguet_range(LD=16.0, eta_total=0.44, MTOW=45000, OEW=45000)
        assert R == 0.0

    def test_breguet_range_increases_with_LD(self):
        """Higher L/D should give longer range."""
        R_low  = breguet_range(LD=12.0, eta_total=0.44, MTOW=46500, OEW=45000)
        R_high = breguet_range(LD=20.0, eta_total=0.44, MTOW=46500, OEW=45000)
        assert R_high > R_low

    def test_sizing_converges(self):
        """Iterative sizing must converge."""
        result = size_aircraft()
        assert result["converged"] is True

    def test_sizing_achieves_target_range(self):
        """Sized aircraft must achieve target range of 3000 km."""
        result = size_aircraft()
        assert abs(result["range_km"] - 3000.0) < 1.0

    def test_sizing_positive_fuel_mass(self):
        """Fuel mass must be positive after sizing."""
        result = size_aircraft()
        assert result["m_fuel"] > 0

    def test_analyse_aircraft_nominal(self):
        """Analysis mode at nominal params must give ~3000 km."""
        range_km = analyse_aircraft(DEFAULT_PARAMS)
        assert abs(range_km - 3000.0) < 10.0

    def test_analyse_aircraft_increases_with_eta_fc(self):
        """Higher fuel cell efficiency should give longer range."""
        params_low  = DEFAULT_PARAMS.copy(); params_low["eta_fc"]  = 0.45
        params_high = DEFAULT_PARAMS.copy(); params_high["eta_fc"] = 0.65
        assert analyse_aircraft(params_high) > analyse_aircraft(params_low)

    def test_analyse_aircraft_increases_with_AR(self):
        """Higher aspect ratio should give longer range."""
        params_low  = DEFAULT_PARAMS.copy(); params_low["AR"]  = 7.0
        params_high = DEFAULT_PARAMS.copy(); params_high["AR"] = 11.0
        assert analyse_aircraft(params_high) > analyse_aircraft(params_low)


# ── Uncertainty model tests ───────────────────────────────────────────────────

class TestUncertaintyModel:

    def test_all_parameters_defined(self):
        """All required parameters must be present."""
        required = {"AR", "CD0", "e", "eta_fc", "eta_motor", "eta_prop", "eta_grav"}
        assert required.issubset(set(PARAMETERS.keys()))

    def test_bounds_consistent_with_mean(self):
        """Mean must lie within bounds for all parameters."""
        for name, p in PARAMETERS.items():
            lo, hi = p["bounds"]
            assert lo <= p["mean"] <= hi, \
                f"{name}: mean {p['mean']} not in bounds [{lo}, {hi}]"

    def test_std_positive(self):
        """Standard deviation must be positive for all parameters."""
        for name, p in PARAMETERS.items():
            assert p["std"] > 0, f"{name}: std must be positive"

    def test_salib_problem_format(self):
        """SALib problem dict must have correct structure."""
        problem = get_salib_problem()
        assert "num_vars" in problem
        assert "names"    in problem
        assert "bounds"   in problem
        assert problem["num_vars"] == len(PARAMETERS)
        assert len(problem["names"])  == len(PARAMETERS)
        assert len(problem["bounds"]) == len(PARAMETERS)

    def test_sample_parameters_shape(self):
        """Sampled parameters must have correct shape."""
        n = 50
        samples = sample_parameters(n_samples=n)
        for name, vals in samples.items():
            assert len(vals) == n, f"{name}: expected {n} samples"

    def test_sample_parameters_within_bounds(self):
        """All sampled values must lie within parameter bounds."""
        samples = sample_parameters(n_samples=200)
        for name, vals in samples.items():
            lo, hi = PARAMETERS[name]["bounds"]
            assert np.all(vals >= lo), f"{name}: sample below lower bound"
            assert np.all(vals <= hi), f"{name}: sample above upper bound"

    def test_all_uncertainty_epistemic(self):
        """All parameters should be epistemic at conceptual design stage."""
        for name, p in PARAMETERS.items():
            assert p["type"] == "epistemic", \
                f"{name}: expected epistemic, got {p['type']}"


# ── Monte Carlo tests ─────────────────────────────────────────────────────────

class TestMonteCarlo:

    def test_mc_returns_correct_sample_count(self):
        """Monte Carlo must return the requested number of valid samples."""
        results = run_monte_carlo(n_samples=100, seed=42)
        assert results["n_valid"] == 100

    def test_mc_mean_near_nominal(self):
        """Monte Carlo mean range must be near nominal 3000 km."""
        results = run_monte_carlo(n_samples=500, seed=42)
        assert abs(results["mean_range"] - 3000.0) < 200.0

    def test_mc_std_positive(self):
        """Monte Carlo range std must be positive — uncertainty exists."""
        results = run_monte_carlo(n_samples=500, seed=42)
        assert results["std_range"] > 0

    def test_mc_percentile_ordering(self):
        """P5 < P50 < P90 < P95 must hold."""
        results = run_monte_carlo(n_samples=500, seed=42)
        assert results["p05"] < results["p50"]
        assert results["p50"] < results["p90"]
        assert results["p90"] < results["p95"]

    def test_mc_reproducible(self):
        """Same seed must give same results."""
        r1 = run_monte_carlo(n_samples=100, seed=99)
        r2 = run_monte_carlo(n_samples=100, seed=99)
        assert abs(r1["mean_range"] - r2["mean_range"]) < 1e-6


# ── Bayesian update tests ─────────────────────────────────────────────────────

class TestBayesianUpdate:

    def test_posterior_mean_shifts_toward_data(self):
        """Posterior mean must shift toward the data mean."""
        prior_mean = 0.55
        prior_std  = 0.04
        measurements = np.array([0.60, 0.61, 0.59, 0.60, 0.61])
        post_mean, post_std = bayesian_update(
            prior_mean, prior_std, measurements, 0.015
        )
        assert post_mean > prior_mean  # data is above prior

    def test_posterior_std_decreases_with_data(self):
        """Posterior std must decrease as more data is added."""
        prior_mean = 0.55
        prior_std  = 0.04
        noise_std  = 0.015
        measurements = np.random.default_rng(42).normal(0.57, noise_std, 50)

        stds = []
        for n in [1, 5, 10, 20, 50]:
            _, post_std = bayesian_update(
                prior_mean, prior_std, measurements[:n], noise_std
            )
            stds.append(post_std)

        # Each std must be smaller than the previous
        for i in range(1, len(stds)):
            assert stds[i] < stds[i-1], \
                f"Posterior std did not decrease at N={[1,5,10,20,50][i]}"

    def test_posterior_std_less_than_prior(self):
        """Posterior std must always be less than prior std."""
        prior_mean = 0.55
        prior_std  = 0.04
        measurements = np.array([0.57])
        _, post_std = bayesian_update(prior_mean, prior_std, measurements, 0.015)
        assert post_std < prior_std

    def test_posterior_std_approaches_zero_with_many_measurements(self):
        """With many measurements, posterior std should become very small."""
        prior_mean = 0.55
        prior_std  = 0.04
        noise_std  = 0.015
        measurements = np.random.default_rng(42).normal(0.57, noise_std, 1000)
        _, post_std = bayesian_update(prior_mean, prior_std, measurements, noise_std)
        assert post_std < 0.001


# ── Mass model tests ──────────────────────────────────────────────────────────

class TestMassModel:

    def test_tank_mass_positive(self):
        """Tank mass must be positive."""
        assert tank_mass(4000) > 0

    def test_tank_heavier_than_hydrogen(self):
        """At eta_grav=0.35, tank must be heavier than the hydrogen it holds."""
        m_h2 = 4000
        assert tank_mass(m_h2, eta_grav=0.35) > m_h2

    def test_tank_mass_scales_with_hydrogen(self):
        """More hydrogen requires a bigger tank."""
        assert tank_mass(8000) > tank_mass(4000)

    def test_better_gravimetric_efficiency_lighter_tank(self):
        """Higher gravimetric efficiency must give a lighter tank."""
        assert tank_mass(4000, eta_grav=0.50) < tank_mass(4000, eta_grav=0.30)

    def test_fuel_cell_mass_positive(self):
        """Fuel cell system mass must be positive."""
        assert fuel_cell_mass() > 0

    def test_OEW_increases_with_hydrogen(self):
        """OEW must increase with hydrogen mass (heavier tank)."""
        assert compute_OEW(8000) > compute_OEW(4000)

    def test_OEW_components_sum(self):
        """OEW must equal structure + tank + fuel cell mass."""
        m_h2 = 4000
        expected = 38000 + tank_mass(m_h2) + fuel_cell_mass()
        assert abs(compute_OEW(m_h2) - expected) < 1.0


# ── Model-form error tests ────────────────────────────────────────────────────

class TestModelFormError:

    def test_oswald_correlation_reasonable(self):
        """Oswald correlation must give a physically reasonable value."""
        e = oswald_correlation(9.0)
        assert 0.6 < e < 0.95

    def test_oswald_decreases_with_AR(self):
        """Raymer correlation: Oswald factor decreases with aspect ratio."""
        assert oswald_correlation(12.0) < oswald_correlation(7.0)

    def test_model_form_lower_than_parametric(self):
        """Model-form biased samples must have a lower mean than parametric."""
        e_param = sample_parametric(9.0, 2000)
        e_mf    = sample_model_form(9.0, 2000)
        assert e_mf.mean() < e_param.mean()

    def test_parametric_centered_on_correlation(self):
        """Parametric samples must center on the correlation value."""
        e_corr  = oswald_correlation(9.0)
        e_param = sample_parametric(9.0, 5000)
        assert abs(e_param.mean() - e_corr) < 0.01

    def test_model_form_bias_magnitude(self):
        """Model-form bias should shift mean down by roughly the bias fraction."""
        e_corr = oswald_correlation(9.0)
        e_mf   = sample_model_form(9.0, 5000, bias=0.12)
        expected = e_corr * (1 - 0.12)
        assert abs(e_mf.mean() - expected) < 0.01


# ── Run tests ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
