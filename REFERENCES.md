# References

This project draws on the following published literature for physical model parameters, uncertainty characterisation, and methodological foundations.

---

## Aircraft Design and Hydrogen Aviation Parameters

**[R1]** Hoelzen, J., Silberhorn, D., Zill, T., Bensmann, B., & Hanke-Rauschenbach, R. (2022).
*Hydrogen-powered aviation and its reliance on green hydrogen infrastructure — Review and research gaps.*
International Journal of Hydrogen Energy, 47(7), 3108–3130.
https://doi.org/10.1016/j.ijhydene.2021.10.248

> **Used for:** fuel cell efficiency range (η_fc = 0.45–0.65), hydrogen specific energy (120 MJ/kg), operating empty weight estimates for medium-haul hydrogen aircraft, and overall system architecture assumptions.

**[R2]** Raymer, D.P. (2018).
*Aircraft Design: A Conceptual Approach* (6th ed.).
American Institute of Aeronautics and Astronautics (AIAA).
ISBN: 978-1-62410-490-9

> **Used for:** Oswald efficiency factor empirical range (e = 0.65–0.95), drag polar formulation, aspect ratio typical values for medium-haul commercial aircraft, and zero-lift drag coefficient estimates.

---

## Uncertainty Quantification and Sensitivity Analysis

**[R3]** Saltelli, A., Tarantola, S., Campolongo, F., & Ratto, M. (2004).
*Sensitivity Analysis in Practice: A Guide to Assessing Scientific Models.*
John Wiley & Sons. ISBN: 978-0-470-87093-8

> **Used for:** theoretical foundation of Sobol variance decomposition, distinction between first-order (S₁) and total-order (S_T) sensitivity indices, Latin Hypercube Sampling rationale, and guidance on choosing sensitivity analysis methods for non-linear models.

**[R4]** Herman, J., & Usher, W. (2017).
*SALib: An open-source Python library for sensitivity analysis.*
Journal of Open Source Software, 2(9), 97.
https://doi.org/10.21105/joss.00097

> **Used for:** computational implementation of Sobol indices via the SALib library (Saltelli sampling scheme, Jansen estimator for S_T). All sensitivity results in `sensitivity.py` were computed using SALib v1.4+.

---

## Bayesian Calibration and Model-Data Comparison

**[R5]** Dwight, R.P., & Han, Z.H. (2009).
*Efficient uncertainty quantification using gradient-enhanced Kriging.*
AIAA Paper 2009-2276, 50th AIAA/ASME/ASCE/AHS/ASC Structures, Structural Dynamics, and Materials Conference.
https://doi.org/10.2514/6.2009-2276

> **Used for:** conceptual foundation of Bayesian model-data comparison and posterior updating as applied to computational engineering models. The `bayesian_update.py` module implements a conjugate Gaussian update consistent with Dr. Dwight's published framework for parameter calibration from observed data.

**[R6]** Kennedy, M.C., & O'Hagan, A. (2001).
*Bayesian calibration of computer models.*
Journal of the Royal Statistical Society: Series B, 63(3), 425–464.
https://doi.org/10.1111/1467-9868.00294

> **Used for:** theoretical basis of the conjugate Gaussian Bayesian update implemented in `bayesian_update.py`. This is the foundational paper for Bayesian calibration of computational models, on which Dr. Dwight's data assimilation work builds.

---

## Parameter Justification Summary

The table below maps each uncertain parameter to its primary literature source:

| Parameter | Value | Source | Justification |
|---|---|---|---|
| `E_H2` | 120 MJ/kg | [R1] | Lower heating value of hydrogen, widely reported |
| `eta_fc` | 0.55 ± 0.04 | [R1] | Aviation-grade PEM stack efficiency at cruise conditions |
| `AR` | 9.0 ± 0.5 | [R2] | Typical medium-haul transport aircraft aspect ratio |
| `CD0` | 0.020 ± 0.003 | [R2] | Clean configuration zero-lift drag, conceptual estimate |
| `e` | 0.80 ± 0.06 | [R2] | Oswald efficiency factor — empirical correlation, wide scatter |
| `eta_motor` | 0.95 ± 0.01 | [R1] | Mature electric motor technology, narrow uncertainty |
| `eta_prop` | 0.85 ± 0.03 | [R2] | Propeller efficiency at cruise, novel design uncertainty |
| `OEW` | 45,000 ± 2,000 kg | [R1] | Medium-haul hydrogen aircraft operating empty weight |

---

## SALib and Python Tools

- **SALib** — `pip install SALib` — https://salib.readthedocs.io
- **NumPy** — https://numpy.org
- **SciPy** — https://scipy.org
- **Matplotlib** — https://matplotlib.org
