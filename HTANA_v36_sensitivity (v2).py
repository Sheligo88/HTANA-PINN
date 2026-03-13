# =============================================================================
# H-TANA PINN v3.6 — Sensitivity Matrix
# Author: Iván Sheligo (Independent Researcher, Colombia)
# AI Assistance: Claude (Anthropic), GPT (OpenAI), Gemini (Google)
#
# PURPOSE: Test robustness of recovery against pipeline choices.
#
# MATRIX:
#   lambda_fried ∈ {0.1, 0.3, 1.0}  × z_col_max ∈ {0.6, 1.4}
#   = 6 configurations
#   Each runs Mock 1 (ΛCDM + vp=450) and Mock 3 (Tsallis + vp=450)
#
# If Δ_rec and vp_rec are stable across all 6 → ROBUST
# If they vary a lot → pipeline is sensitive to recipe → must report
# =============================================================================

# -----------------------------------------------------------------------
# CELL 1-4: Same setup as v3.5
# -----------------------------------------------------------------------
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["PYTHONHASHSEED"] = "42"

import random, hashlib, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from scipy.integrate import quad
from google.colab import drive

drive.mount('/content/drive')
torch.set_default_dtype(torch.float64)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    print(f"🟢 GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("🟡 CPU mode")

C_LIGHT     = 299792.458
H0_FIDUCIAL = 70.0
DH          = C_LIGHT / H0_FIDUCIAL
DATA_PATH   = "/content/drive/MyDrive/HTANA_data/"
EPOCHS      = 6000   # slightly fewer — 6 configs × 2 mocks = 12 runs
LR_NET      = 1e-3
LR_VP       = 1e-2
Om_m        = torch.tensor(0.315, device=device, dtype=torch.float64)
Om_de       = 1.0 - Om_m
L_CMB, B_CMB = 264.0, 48.3

print(f"PyTorch {torch.__version__} | Device: {device}")
print("✅ Setup ready.")

# -----------------------------------------------------------------------
# CELL 5: Utilities
# -----------------------------------------------------------------------
def galactic_to_unitvec(l_deg, b_deg):
    l = np.deg2rad(np.asarray(l_deg, dtype=float))
    b = np.deg2rad(np.asarray(b_deg, dtype=float))
    return np.stack([np.cos(b)*np.cos(l),
                     np.cos(b)*np.sin(l),
                     np.sin(b)], axis=-1)

def radec_to_galactic(ra_deg, dec_deg):
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    c = SkyCoord(ra=ra_deg*u.deg, dec=dec_deg*u.deg, frame='icrs')
    return c.galactic.l.deg, c.galactic.b.deg

def cumulative_trapz(y, x):
    dx   = x[1:] - x[:-1]
    trap = 0.5 * (y[1:] + y[:-1]) * dx
    out  = torch.zeros_like(y)
    out[1:] = torch.cumsum(trap, dim=0)
    return out

# -----------------------------------------------------------------------
# CELL 6: Load Pantheon+ footprint
# -----------------------------------------------------------------------
d_hat = galactic_to_unitvec(L_CMB, B_CMB)
d_hat = d_hat / np.linalg.norm(d_hat)

df_p     = pd.read_csv(DATA_PATH+"Pantheon+SH0ES.dat", sep=r'\s+', comment='#')
z_raw    = df_p['zHD'].values
mu_raw   = df_p['MU_SH0ES'].values
me_raw   = df_p['MU_SH0ES_ERR_DIAG'].values
l_sn, b_sn = radec_to_galactic(df_p['RA'].values, df_p['DEC'].values)
cos_th   = galactic_to_unitvec(l_sn, b_sn) @ d_hat

mask = (z_raw > 0.005) & (z_raw < 0.6) \
     & np.isfinite(mu_raw) & np.isfinite(me_raw) & np.isfinite(cos_th)
idx  = np.argsort(z_raw[mask])
z_fp  = z_raw[mask][idx]
me_fp = me_raw[mask][idx]
ct_fp = cos_th[mask][idx]

print(f"✅ Footprint: {len(z_fp)} SNe")

z_t  = torch.tensor(z_fp,  dtype=torch.float64, device=device).view(-1,1)
ct_t = torch.tensor(ct_fp, dtype=torch.float64, device=device)
me_t = torch.tensor(me_fp, dtype=torch.float64, device=device)

# -----------------------------------------------------------------------
# CELL 7: Mock generator
# -----------------------------------------------------------------------
def E_tsallis_numpy(z, Om_m_val=0.315, delta=2.0):
    Om_de_val = 1.0 - Om_m_val
    if abs(delta - 2.0) < 1e-6:
        return np.sqrt(Om_m_val*(1+z)**3 + Om_de_val)
    E = np.sqrt(Om_m_val*(1+z)**3 + Om_de_val)
    for _ in range(50):
        E_new = np.sqrt(Om_m_val*(1+z)**3 + Om_de_val*E**(4-2*delta))
        if abs(E_new - E) < 1e-10: break
        E = E_new
    return E

def generate_mock(z_arr, cos_theta, sigma_obs, delta_true, vp_true, seed):
    np.random.seed(seed)
    Dc_arr = np.zeros_like(z_arr)
    for i, z in enumerate(z_arr):
        if z > 1e-6:
            Dc_arr[i], _ = quad(
                lambda zp: 1.0/E_tsallis_numpy(zp, delta=delta_true),
                0, z, limit=200)
    E_arr  = np.array([E_tsallis_numpy(z, delta=delta_true) for z in z_arr])
    Dc_safe = np.maximum(Dc_arr, 1e-10)
    EDc     = E_arr * Dc_safe
    z_safe  = np.maximum(z_arr, 1e-8)
    B       = -1.0/z_safe + (1.0+z_safe)*(EDc-z_safe)/(z_safe*EDc)
    delta_frac = B * (vp_true/C_LIGHT) * cos_theta
    DL      = np.maximum((1.0+z_arr)*Dc_safe*DH*(1.0+delta_frac), 1e-6)
    mu_th   = 5.0*np.log10(DL) + 25.0
    noise   = np.random.normal(0, sigma_obs)  # no double-counting
    return mu_th + noise

print("✅ Mock generator ready.")

# Generate mocks once — reuse across all configs
print("\n📊 Generating mocks...")
mu_m1 = generate_mock(z_fp, ct_fp, me_fp,
                       delta_true=2.0,  vp_true=450.0, seed=42)
mu_m3 = generate_mock(z_fp, ct_fp, me_fp,
                       delta_true=1.52, vp_true=450.0, seed=44)
print("✅ Mock 1 (ΛCDM+vp=450) and Mock 3 (Tsallis+vp=450) ready.")

# -----------------------------------------------------------------------
# CELL 8: PINN model
# -----------------------------------------------------------------------
class HTANA_v36(nn.Module):
    def __init__(self):
        super().__init__()
        self.delta_raw = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        self.vp_raw    = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        self.mlp = nn.Sequential(
            nn.Linear(1, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1),
        ).double()

    def get_params(self):
        delta = 2.0 + 0.8 * torch.tanh(self.delta_raw)
        vp    = 3000.0 * torch.tanh(self.vp_raw)
        return delta, vp

    def E_of_z(self, z):
        f = self.mlp(z).squeeze(-1)
        return torch.exp(z.squeeze(-1) * f)

    def forward(self, z_sorted, cos_theta):
        z1 = z_sorted.squeeze(-1)
        z_wz  = torch.cat([torch.zeros(1,dtype=torch.float64,device=device), z1])
        E_wz  = self.E_of_z(z_wz.unsqueeze(-1))
        Dc_wz = cumulative_trapz(1.0/(E_wz+1e-10), z_wz)
        Dc    = torch.clamp(Dc_wz[1:], min=1e-10)
        E     = E_wz[1:]
        z_s   = torch.clamp(z1, min=1e-8)
        EDc   = torch.clamp(E*Dc, min=1e-12)
        B     = -1.0/z_s + (1.0+z_s)*(EDc-z_s)/(z_s*EDc)
        _, vp = self.get_params()
        DL    = torch.clamp((1.0+z1)*Dc*DH*(1.0+B*(vp/C_LIGHT)*cos_theta),
                            min=1e-6)
        return 5.0*torch.log10(DL) + 25.0

    def E_only(self, z):
        return self.E_of_z(z)

print("✅ PINN v3.6 defined. Init: Δ=2.0, vp=0 (blind start)")

# -----------------------------------------------------------------------
# CELL 9: Single run function
# -----------------------------------------------------------------------
def single_run(mu_mock, lambda_fried, z_col_max):
    torch.manual_seed(SEED)
    model = HTANA_v36().to(device)
    mu_t  = torch.tensor(mu_mock, dtype=torch.float64, device=device)

    opt   = torch.optim.Adam([
        {'params': model.mlp.parameters(), 'lr': LR_NET},
        {'params': [model.delta_raw],       'lr': 5e-3},
        {'params': [model.vp_raw],          'lr': LR_VP},
    ])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=EPOCHS, eta_min=1e-5)

    g = torch.Generator(device=device).manual_seed(SEED+10)

    for epoch in range(EPOCHS):
        opt.zero_grad(set_to_none=True)

        # Data loss
        L_data = torch.mean(((model(z_t, ct_t) - mu_t) / me_t)**2)

        # Friedmann colocation in [0.01, z_col_max]
        z_col = torch.rand(3000, generator=g, device=device,
                           dtype=torch.float64) * (z_col_max - 0.01) + 0.01
        z_col = z_col.view(-1,1)
        E     = model.E_only(z_col)
        delta, _ = model.get_params()
        z1    = z_col.squeeze(-1)
        R     = E**2 - Om_m*(1+z1)**3 - Om_de*torch.pow(E, 4.0-2.0*delta)
        den   = Om_m*(1+z1)**3 + Om_de*torch.pow(E.detach(),
                4.0-2.0*delta) + 1e-8
        L_fried = torch.mean((R/den)**2)

        loss = L_data + lambda_fried * L_fried
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        opt.step()
        sched.step()

    delta, vp = model.get_params()
    return delta.item(), vp.item()

# -----------------------------------------------------------------------
# CELL 10: Run sensitivity matrix
# -----------------------------------------------------------------------
lambda_frieds = [0.1, 0.3, 1.0]
z_col_maxes   = [0.6, 1.4]

results = {}   # key: (mock_name, lambda_f, z_max) → (delta_rec, vp_rec)

mocks_to_run = [
    ("Mock1_LCDM_vp450",   mu_m1, 2.0,  450.0),
    ("Mock3_Tsallis_vp450", mu_m3, 1.52, 450.0),
]

total = len(mocks_to_run) * len(lambda_frieds) * len(z_col_maxes)
run   = 0
t_total = time.time()

print(f"\n🚀 Sensitivity matrix: {total} runs × {EPOCHS} epochs")
print(f"   lambda_fried ∈ {lambda_frieds}")
print(f"   z_col_max    ∈ {z_col_maxes}")
print(f"   Mocks: Mock1 (ΛCDM+vp) and Mock3 (Tsallis+vp)")
print()

for mock_name, mu_mock, delta_true, vp_true in mocks_to_run:
    for lf in lambda_frieds:
        for zmax in z_col_maxes:
            run += 1
            label = f"{mock_name} | λ={lf} | zmax={zmax}"
            t0 = time.time()
            dr, vr = single_run(mu_mock, lf, zmax)
            elapsed = time.time() - t0
            results[(mock_name, lf, zmax)] = (dr, vr)

            d_err = abs(dr - delta_true)
            v_err = abs(abs(vr) - abs(vp_true))
            print(f"  [{run:2d}/{total}] {label}")
            print(f"         Δ: {delta_true:.2f}→{dr:.3f} (err={d_err:.3f}) | "
                  f"vp: {vp_true:.0f}→{vr:.0f} km/s (err={v_err:.0f}) | "
                  f"⏱️ {elapsed:.0f}s")

print(f"\n⏱️  Total: {time.time()-t_total:.0f}s")

# -----------------------------------------------------------------------
# CELL 11: Summary table
# -----------------------------------------------------------------------
print(f"\n{'='*75}")
print(f"📋 SENSITIVITY MATRIX RESULTS")
print(f"{'='*75}")

for mock_name, mu_mock, delta_true, vp_true in mocks_to_run:
    print(f"\n  {mock_name} (Δ_true={delta_true}, vp_true={vp_true} km/s)")
    print(f"  {'λ_fried':>8} | {'z_max':>6} | {'Δ_rec':>7} | "
          f"{'Δ_bias':>8} | {'vp_rec':>8} | {'vp_bias':>8}")
    print(f"  {'-'*60}")
    for lf in lambda_frieds:
        for zmax in z_col_maxes:
            dr, vr = results[(mock_name, lf, zmax)]
            print(f"  {lf:>8.1f} | {zmax:>6.1f} | {dr:>7.3f} | "
                  f"{dr-delta_true:>+8.3f} | {vr:>8.1f} | "
                  f"{vr-vp_true:>+8.1f}")

# Stability assessment
print(f"\n🔬 STABILITY ASSESSMENT:")
for mock_name, mu_mock, delta_true, vp_true in mocks_to_run:
    delta_recs = [results[(mock_name,lf,zmax)][0]
                  for lf in lambda_frieds for zmax in z_col_maxes]
    # vp in AMPLITUDE — sign is convention-dependent (GPT fix)
    vp_amps    = [abs(results[(mock_name,lf,zmax)][1])
                  for lf in lambda_frieds for zmax in z_col_maxes]
    d_std  = np.std(delta_recs)
    v_std  = np.std(vp_amps)
    d_bias = np.mean(delta_recs) - delta_true
    v_bias = np.mean(vp_amps) - abs(vp_true)

    print(f"\n  {mock_name}:")
    print(f"    Δ_rec:  mean={np.mean(delta_recs):.3f} ± {d_std:.3f} "
          f"(bias={d_bias:+.3f})")
    print(f"    |vp|:   mean={np.mean(vp_amps):.1f} ± {v_std:.1f} km/s "
          f"(amp bias={v_bias:+.1f} km/s)")
    print(f"    Note: amplitude recovery only — sign is convention-dependent")

    if d_std < 0.08:
        print(f"    Δ stability: ✅ ROBUST (σ={d_std:.3f} < 0.08)")
    else:
        print(f"    Δ stability: ⚠️  SENSITIVE (σ={d_std:.3f} ≥ 0.08)")

    if v_std < 150:
        print(f"    vp stability: ✅ ROBUST (σ={v_std:.1f} < 150 km/s)")
    else:
        print(f"    vp stability: ⚠️  SENSITIVE (σ={v_std:.1f} ≥ 150 km/s)")

# Save to CSV — protect against Colab crashes
rows = []
for (mock_name, lf, zmax), (dr, vr) in results.items():
    mock_info = {m[0]: (m[2], m[3]) for m in mocks_to_run}
    dt, vt = mock_info[mock_name]
    rows.append({
        "mock": mock_name, "lambda_fried": lf, "z_col_max": zmax,
        "delta_true": dt, "vp_true": vt,
        "delta_rec": dr, "vp_rec": vr,
        "delta_bias": dr-dt, "vp_amp_bias": abs(vr)-abs(vt),
        "vp_bias_signed": vr-vt
    })
pd.DataFrame(rows).to_csv(
    DATA_PATH+"HTANA_v36_sensitivity_results.csv", index=False)
print(f"\n✅ Results saved to HTANA_v36_sensitivity_results.csv")

# -----------------------------------------------------------------------
# CELL 12: Figures
# -----------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('H-TANA v3.6 — Sensitivity Matrix\n'
             'Robustness of recovery vs pipeline choices',
             fontsize=13, fontweight='bold')

markers = ['o', 's', '^']
line_styles = ['-', '--']

for ax_row, (mock_name, mu_mock, delta_true, vp_true) in \
        enumerate(mocks_to_run):

    # Δ recovery
    ax = axes[ax_row, 0]
    for j, zmax in enumerate(z_col_maxes):
        d_vals = [results[(mock_name,lf,zmax)][0] for lf in lambda_frieds]
        ax.plot(lambda_frieds, d_vals,
                marker=markers[j], linestyle=line_styles[j],
                label=f'z_max={zmax}', lw=2, markersize=8)
    ax.axhline(delta_true, linestyle=':', color='red', lw=2,
               label=f'Δ_true={delta_true}')
    ax.axhline(1.52, linestyle='--', color='coral', alpha=0.5,
               label='Δ=1.52 (real data)')
    ax.set_xlabel('λ_fried'); ax.set_ylabel('Δ_recovered')
    ax.set_title(f'{mock_name}\nΔ Recovery vs λ_fried')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_ylim(1.3, 2.1)

    # vp recovery
    ax = axes[ax_row, 1]
    for j, zmax in enumerate(z_col_maxes):
        v_vals = [abs(results[(mock_name,lf,zmax)][1])
                  for lf in lambda_frieds]
        ax.plot(lambda_frieds, v_vals,
                marker=markers[j], linestyle=line_styles[j],
                label=f'z_max={zmax}', lw=2, markersize=8)
    ax.axhline(abs(vp_true), linestyle=':', color='red', lw=2,
               label=f'vp_true={vp_true} km/s')
    ax.axhline(370, linestyle='--', color='green', alpha=0.5,
               label='CMB 370 km/s')
    ax.axhline(455, linestyle='--', color='orange', alpha=0.5,
               label='CF4 455 km/s')
    ax.set_xlabel('λ_fried'); ax.set_ylabel('|vp_recovered| (km/s)')
    ax.set_title(f'{mock_name}\nvp Recovery vs λ_fried')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(DATA_PATH+'HTANA_v36_sensitivity.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Sensitivity figure saved.")
print(f"✅ All done. Results in {DATA_PATH}")
