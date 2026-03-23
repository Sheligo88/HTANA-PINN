HTANA-PINN
Physics-Informed Neural Network for directional anisotropy in the Hubble diagram
Iván Sheligo — Independent Researcher, Colombia
AI Audit: Claude (Anthropic) · GPT (OpenAI) · Grok (xAI)
---
What this is
HTANA fits a Tsallis-THDE cosmology with a bulk flow dipole to Type Ia supernova data (Pantheon+SH0ES). It tests whether the locally inferred expansion rate contains a preferred directional component.
The model parameterizes:
```
H_obs(z, θ) = H_iso(z) · [1 + B(z) · (vp/c) · cos θ]
```
where `B(z) = -(1+z)/(E·Dc)` is the correct relativistic dipolar factor (Bonvin 2006, Peterson et al. 2022), and `Δ` is the Tsallis index (Δ=2 corresponds to ΛCDM).
---
Key results (v3.9, corrected dipole factor)
Quantity	Value
Tsallis index Δ	1.655
Bulk flow (CMB pole)	55 km/s
Directional hotspot	(l, b) = (302°, −11°)
Hotspot amplitude	131.9 km/s
Null sky scan (43 mocks)	0/43 exceed real · p < 0.023
Stress test	✅ PASS · Δ varies ±0.011
ΛCDM bias test	−0.011 (negligible)
The directional hotspot lies in the southern galactic hemisphere, consistent with previously identified bulk flow directions in the literature.
---
Files
File	Description
`HTANA_v39_fixes.py`	Core PINN model (v3.9) with corrected dipole factor B
`HTANA_v4_audit_scaffold.py`	Full audit pipeline: preflight, injection-recovery, null ΛCDM bias test, diagnostics
`HTANA_v41_stress_tests.py`	Stress tests: z_col sensitivity and error inflation
---
Requirements
```
python >= 3.11
torch
numpy
pandas
scipy
astropy
```
---
Data
Requires `Pantheon+SH0ES.dat` from the official Pantheon+ data release:
https://github.com/PantheonPlusSH0ES/DataRelease
---
How to run
```python
# 1. Load data and run core model
# See HTANA_v39_fixes.py — run cells in order

# 2. Full audit pipeline
report = run_v4_audit_pipeline(
    z_arr, mu_arr, mu_err_arr, cos_th_arr,
    label="Pantheon+",
    null_mocks=10,
    inj_seeds=3
)

# 3. Stress tests
stress = stress_test_zcol_and_errors_v41(
    z_arr, mu_arr, mu_err_arr, cos_th_arr,
    label="Pantheon+"
)
gate = summarize_stress_gate_v41(stress)
print(gate["verdict"])
```
---
Notes
All results use diagonal errors only (no full covariance matrix). Full covmat validation is a planned next step.
The dipole factor B was corrected in v3.9 from an earlier incorrect implementation. Previous reported amplitudes (vp ≈ 208–440 km/s) used the uncorrected factor and are superseded by v3.9 results.
Code and results available upon request for independent replication.
---
Contact
Iván Sheligo · sheligo88@gmail.com · Colombia
