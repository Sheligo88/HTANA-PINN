# =============================================================================
# H-TANA PINN — v4 AUDIT SCAFFOLD
# Autores: Iván Sheligo + Claude (Anthropic) + GPT (OpenAI) + Grok (xAI)
# Fecha: 2026-03-18
#
# INSTRUCCIONES:
#   1. Correr primero HTANA_v39_fixes.py (los 3 fixes)
#   2. Pegar este bloque DEBAJO en el mismo notebook
#   3. Correr: run_v4_audit_pipeline(...)
#
# REQUISITOS (ya deben existir de v3.9):
#   device, cumulative_trapz, DH, C_LIGHT
#   Om_m, Om_de
#   SEED, EPOCHS, Z_COL_MAX, LAMBDA_FRIED, DATA_PATH
#   clase HTANA_v39
#   run_on_catalog(...) de v3.9
#
# FILOSOFÍA:
#   Si no pasa preflight + injection recovery + null ΛCDM
#   → NO se justifica un sky scan pesado.
#   Primero motor. Luego spoilers.
# =============================================================================

import copy
import math
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ---------------------------------------------------------------------
# v4 helpers — μ fiducial ΛCDM y factor B correcto
# ---------------------------------------------------------------------

def build_mu_fid_lcdm_v4(z_arr, Om_m_v=0.315):
    """μ fiducial ΛCDM puro (Δ=2, vp=0)."""
    Om_de_v = 1.0 - Om_m_v
    z_t = torch.tensor(z_arr, dtype=torch.float64, device=device).view(-1, 1)
    z1  = z_t.squeeze(-1)
    with torch.no_grad():
        z_full  = torch.cat([torch.zeros(1, dtype=torch.float64, device=device), z1])
        E_lcdm  = torch.sqrt(Om_m_v * (1.0 + z_full)**3 + Om_de_v)
        Dc_full = cumulative_trapz(1.0 / (E_lcdm + 1e-10), z_full)
        Dc_lcdm = torch.clamp(Dc_full[1:], min=1e-10)
        DL_lcdm = (1.0 + z1) * Dc_lcdm * DH
        mu_fid  = (5.0 * torch.log10(DL_lcdm) + 25.0).cpu().numpy()
    return mu_fid


def build_B_v4(z_arr, Om_m_v=0.315):
    """Factor B dipolar correcto: B = -(1+z)/(E·Dc)"""
    Om_de_v = 1.0 - Om_m_v
    z_t = torch.tensor(z_arr, dtype=torch.float64, device=device).view(-1, 1)
    z1  = z_t.squeeze(-1)
    with torch.no_grad():
        z_full  = torch.cat([torch.zeros(1, dtype=torch.float64, device=device), z1])
        E_full  = torch.sqrt(Om_m_v * (1.0 + z_full)**3 + Om_de_v)
        Dc_full = cumulative_trapz(1.0 / (E_full + 1e-10), z_full)
        Dc      = torch.clamp(Dc_full[1:], min=1e-10)
        E       = E_full[1:]
        EDc     = torch.clamp(E * Dc, min=1e-12)
        B       = (-(1.0 + z1) / EDc).cpu().numpy()
    return B


# ---------------------------------------------------------------------
# v4 — modelo wrapper con nuisance lineal opcional
# ---------------------------------------------------------------------

class HTANA_v4(nn.Module):
    """
    Envuelve HTANA_v39 sin tocar su física.
    Añade opcionalmente nuisance lineal: δM(z) = a0 + a1*z
    """
    def __init__(self, use_linear_nuisance=False):
        super().__init__()
        self.core = HTANA_v39()
        self.use_linear_nuisance = use_linear_nuisance
        if self.use_linear_nuisance:
            self.a0_raw = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
            self.a1_raw = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))

    def get_params(self):
        return self.core.get_params()

    def get_nuisance(self):
        if not self.use_linear_nuisance:
            return 0.0, 0.0
        a0 = 0.30 * torch.tanh(self.a0_raw)
        a1 = 0.30 * torch.tanh(self.a1_raw)
        return a0, a1

    def E_only(self, z):
        return self.core.E_only(z)

    def forward(self, z_sorted, cos_theta):
        mu = self.core(z_sorted, cos_theta)
        if not self.use_linear_nuisance:
            return mu
        a0, a1 = self.get_nuisance()
        z1 = z_sorted.squeeze(-1)
        return mu + a0 + a1 * z1

print("✅ HTANA_v4 definido.")


# ---------------------------------------------------------------------
# v4 — preflight sanity checks
# ---------------------------------------------------------------------

def preflight_sanity_checks_v4(z_probe=None, cos_probe=None, vp_probe_kms=300.0):
    """
    Verifica antes de correr nada pesado:
    1) forward estable (sin NaN/Inf)
    2) signo correcto al invertir cosθ
    3) simetría aproximada vp → -vp
    4) sin explosión en low-z
    """
    if z_probe is None:
        z_probe = np.array([0.003, 0.005, 0.010, 0.020, 0.050, 0.100], dtype=float)
    if cos_probe is None:
        cos_probe = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=float)

    model = HTANA_v4(use_linear_nuisance=False).to(device)
    model.eval()

    results = {
        'forward_finite': True,
        'sign_flip_ok': True,
        'vp_symmetry_ok': True,
        'lowz_stable': True,
        'messages': []
    }

    z_grid = np.repeat(z_probe, len(cos_probe))
    c_grid = np.tile(cos_probe, len(z_probe))
    z_t = torch.tensor(z_grid, dtype=torch.float64, device=device).view(-1, 1)
    c_t = torch.tensor(c_grid, dtype=torch.float64, device=device)

    with torch.no_grad():
        mu0 = model(z_t, c_t)
        if not torch.isfinite(mu0).all():
            results['forward_finite'] = False
            results['messages'].append("forward devuelve NaN/Inf")

        raw_pos = np.arctanh(np.clip(vp_probe_kms / 3000.0, -0.999999, 0.999999))
        raw_neg = np.arctanh(np.clip(-vp_probe_kms / 3000.0, -0.999999, 0.999999))

        model.core.vp_raw.data.fill_(float(raw_pos))
        mu_pos         = model(z_t,  c_t).detach().cpu().numpy()
        mu_pos_flipcos = model(z_t, -c_t).detach().cpu().numpy()

        model.core.vp_raw.data.fill_(float(raw_neg))
        mu_neg = model(z_t, c_t).detach().cpu().numpy()

    sign_err = np.max(np.abs(mu_pos_flipcos - mu_neg))
    if sign_err > 5e-4:
        results['sign_flip_ok'] = False
        results['messages'].append(f"sign flip inconsistente: {sign_err:.3e}")

    mu0_np  = mu0.detach().cpu().numpy()
    sym_err = np.max(np.abs((mu_pos + mu_neg) / 2.0 - mu0_np))
    if sym_err > 5e-4:
        results['vp_symmetry_ok'] = False
        results['messages'].append(f"simetría vp↔-vp pobre: {sym_err:.3e}")

    lowz_mask = z_grid <= 0.01
    if np.any(~np.isfinite(mu_pos[lowz_mask])) or np.max(np.abs(mu_pos[lowz_mask])) > 100:
        results['lowz_stable'] = False
        results['messages'].append("comportamiento raro en low-z")

    passed = (results['forward_finite'] and results['sign_flip_ok']
              and results['vp_symmetry_ok'] and results['lowz_stable'])
    results['passed'] = passed

    print(f"\n{'='*70}")
    print("🧪 v4 PREFLIGHT SANITY CHECKS")
    print(f"{'='*70}")
    print(f"  forward finite: {'✅' if results['forward_finite'] else '❌'}")
    print(f"  sign flip ok:   {'✅' if results['sign_flip_ok'] else '❌'}")
    print(f"  vp symmetry:    {'✅' if results['vp_symmetry_ok'] else '❌'}")
    print(f"  low-z stable:   {'✅' if results['lowz_stable'] else '❌'}")
    if results['messages']:
        print("  Notas:")
        for msg in results['messages']:
            print(f"   - {msg}")
    print(f"\n  Veredicto: {'✅ PASS' if passed else '❌ FAIL'}")
    return results


# ---------------------------------------------------------------------
# v4 — run_on_catalog extendido
# ---------------------------------------------------------------------

def run_on_catalog_v4(z_arr, mu_arr, mu_err_arr, cos_theta_arr,
                      catalog_name="?", verbose=True,
                      use_linear_nuisance=False,
                      seed=None, epochs=None,
                      lambda_fried=None, z_col_max=None):
    """
    Igual que v3.9, pero:
    - usa HTANA_v4 (nuisance lineal opcional)
    - devuelve dict completo con chi2_red_data y state_dict
    """
    if seed         is None: seed         = SEED
    if epochs       is None: epochs       = EPOCHS
    if lambda_fried is None: lambda_fried = LAMBDA_FRIED
    if z_col_max    is None: z_col_max    = Z_COL_MAX

    torch.manual_seed(seed)
    np.random.seed(seed)

    z_t  = torch.tensor(z_arr,         dtype=torch.float64, device=device).view(-1,1)
    mu_t = torch.tensor(mu_arr,        dtype=torch.float64, device=device)
    me_t = torch.tensor(mu_err_arr,    dtype=torch.float64, device=device)
    ct_t = torch.tensor(cos_theta_arr, dtype=torch.float64, device=device)

    model = HTANA_v4(use_linear_nuisance=use_linear_nuisance).to(device)

    param_groups = [
        {'params': model.core.mlp.parameters(), 'lr': 1e-3},
        {'params': [model.core.delta_raw],      'lr': 5e-3},
        {'params': [model.core.vp_raw],         'lr': 1e-2},
    ]
    if use_linear_nuisance:
        param_groups.append({'params': [model.a0_raw, model.a1_raw], 'lr': 5e-3})

    opt   = torch.optim.Adam(param_groups)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    g     = torch.Generator(device=device).manual_seed(seed + 10)

    history = {'epoch': [], 'loss': [], 'L_data': [], 'L_fried': [], 'delta': [], 'vp': []}
    if use_linear_nuisance:
        history['a0'] = []; history['a1'] = []

    t0 = time.time()
    for epoch in range(epochs):
        opt.zero_grad(set_to_none=True)

        mu_th    = model(z_t, ct_t)
        inv_sig2 = 1.0 / (me_t ** 2)
        delta_M  = (torch.sum((mu_t - mu_th) * inv_sig2) / torch.sum(inv_sig2)).detach()
        L_data   = torch.mean(((mu_th + delta_M - mu_t) / me_t)**2)

        z_col = (torch.rand(3000, generator=g, device=device, dtype=torch.float64)
                 * (z_col_max - 0.01) + 0.01).view(-1, 1)
        E     = model.E_only(z_col)
        delta, _ = model.get_params()
        z1    = z_col.squeeze(-1)
        R     = E**2 - Om_m*(1+z1)**3 - Om_de*torch.pow(E, 4.0 - 2.0*delta)
        den   = Om_m*(1+z1)**3 + Om_de*torch.pow(E.detach(), 4.0 - 2.0*delta) + 1e-8
        L_fried = torch.mean((R/den)**2)

        loss = L_data + lambda_fried * L_fried
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        opt.step()
        sched.step()

        if (epoch + 1) % 250 == 0 or epoch == 0 or epoch == epochs - 1:
            d, v = model.get_params()
            history['epoch'].append(epoch + 1)
            history['loss'].append(loss.item())
            history['L_data'].append(L_data.item())
            history['L_fried'].append(L_fried.item())
            history['delta'].append(d.item())
            history['vp'].append(v.item())
            if use_linear_nuisance:
                a0, a1 = model.get_nuisance()
                history['a0'].append(float(a0.detach().cpu()))
                history['a1'].append(float(a1.detach().cpu()))

        if verbose and ((epoch + 1) % 2000 == 0 or epoch == epochs - 1):
            d, v = model.get_params()
            msg = (f"    [{catalog_name}] epoch {epoch+1} | "
                   f"Δ={d.item():.3f} | vp={v.item():.0f} km/s | loss={loss.item():.4f}")
            if use_linear_nuisance:
                a0, a1 = model.get_nuisance()
                msg += f" | a0={a0.item():+.4f} | a1={a1.item():+.4f}"
            print(msg)

    delta, vp = model.get_params()
    elapsed   = time.time() - t0

    with torch.no_grad():
        mu_fit = model(z_t, ct_t).cpu().numpy()

    inv_s2     = 1.0 / np.asarray(mu_err_arr)**2
    delta_M_np = np.sum((np.asarray(mu_arr) - mu_fit) * inv_s2) / np.sum(inv_s2)
    resid      = np.asarray(mu_arr) - (mu_fit + delta_M_np)

    out = {
        'catalog_name':        catalog_name,
        'delta':               float(delta.detach().cpu()),
        'vp':                  float(vp.detach().cpu()),
        'final_loss':          float(history['loss'][-1]),
        'elapsed_sec':         elapsed,
        'state_dict':          copy.deepcopy(model.state_dict()),
        'use_linear_nuisance': use_linear_nuisance,
        'delta_M_global':      float(delta_M_np),
        'resid_rms':           float(np.sqrt(np.mean(resid**2))),
        'chi2_red_data':       float(np.mean((resid / np.asarray(mu_err_arr))**2)),
        'history':             history,
    }
    if use_linear_nuisance:
        a0, a1 = model.get_nuisance()
        out['a0'] = float(a0.detach().cpu())
        out['a1'] = float(a1.detach().cpu())

    if verbose:
        tail = (f" | a0={out['a0']:+.4f} | a1={out['a1']:+.4f}"
                if use_linear_nuisance else "")
        print(f"  ✅ {catalog_name}: Δ={out['delta']:.3f} | "
              f"vp={out['vp']:.0f} km/s | χ²_red={out['chi2_red_data']:.4f} | "
              f"⏱️ {elapsed:.0f}s{tail}")
    return out

print("✅ run_on_catalog_v4 definido.")


# ---------------------------------------------------------------------
# v4 — diagnostics con forward() real
# ---------------------------------------------------------------------

def run_diagnostics_v4(z_arr, mu_arr, mu_err_arr, cos_th_arr, fit_result, label="?"):
    """
    Carga state_dict real del modelo entrenado.
    NO usa proxy analítico — bug corregido de v3.8.
    """
    model = HTANA_v4(
        use_linear_nuisance=fit_result.get('use_linear_nuisance', False)
    ).to(device)
    model.load_state_dict(fit_result['state_dict'])
    model.eval()

    z_t  = torch.tensor(z_arr,      dtype=torch.float64, device=device).view(-1,1)
    ct_t = torch.tensor(cos_th_arr, dtype=torch.float64, device=device)

    with torch.no_grad():
        mu_th = model(z_t, ct_t).cpu().numpy()

    mu_obs = np.asarray(mu_arr)
    mu_err = np.asarray(mu_err_arr)
    z_np   = np.asarray(z_arr)
    cos_th = np.asarray(cos_th_arr)

    inv_s2  = 1.0 / mu_err**2
    dM_glob = np.sum((mu_obs - mu_th) * inv_s2) / np.sum(inv_s2)
    resid   = mu_obs - (mu_th + dM_glob)

    print(f"\n{'='*70}")
    print(f"🔬 v4 DIAGNOSTICS — {label}  (forward() real)")
    print(f"{'='*70}")

    # TEST 1 — tomography z-bins
    print(f"\n  TEST 1 — Zero-point tomography")
    z_edges = [0.005, 0.10, 0.30, 0.60]
    dM_bins = []
    for i in range(len(z_edges)-1):
        mask = (z_np >= z_edges[i]) & (z_np < z_edges[i+1])
        if mask.sum() < 5:
            continue
        inv_s2_b = 1.0 / mu_err[mask]**2
        dM_b     = np.sum((mu_obs[mask] - mu_th[mask]) * inv_s2_b) / np.sum(inv_s2_b)
        dM_bins.append(dM_b)
        print(f"    z=[{z_edges[i]:.2f},{z_edges[i+1]:.2f}]  N={mask.sum():4d}  δM*={dM_b:+.4f}")

    dM_range = float(max(dM_bins) - min(dM_bins)) if len(dM_bins) >= 2 else np.nan
    test1_ok = (np.isfinite(dM_range) and dM_range < 0.03)
    if np.isfinite(dM_range):
        print(f"    Range: {dM_range:.4f} mag  →  {'✅ FLAT' if test1_ok else '⚠️ DRIFTS'}")

    # TEST 2 — residual angular slope (WLS cerrado)
    print(f"\n  TEST 2 — Residual dipole leakage")
    W   = 1.0 / mu_err**2
    X   = cos_th
    A00 = np.sum(W);      A01 = np.sum(W * X);      A11 = np.sum(W * X**2)
    b0  = np.sum(W * resid); b1 = np.sum(W * resid * X)
    det = A00 * A11 - A01**2

    if abs(det) < 1e-15:
        slope = slope_err = p_like = np.nan; test2_ok = False
        print("    ⚠️ sistema degenerado")
    else:
        intercept = (A11*b0 - A01*b1) / det
        slope     = (A00*b1 - A01*b0) / det
        resid2    = resid - (intercept + slope * X)
        dof       = max(len(resid) - 2, 1)
        sigma2    = np.sum(W * resid2**2) / dof / np.mean(W)
        cov11     = A00 / det
        slope_err = np.sqrt(max(cov11 * sigma2, 0.0))
        zscore    = 0.0 if slope_err < 1e-15 else abs(slope) / slope_err
        p_like    = math.erfc(zscore / math.sqrt(2.0))
        test2_ok  = (p_like > 0.05)
        print(f"    slope={slope:+.4e} ± {slope_err:.4e} | p~{p_like:.4f} "
              f"→ {'✅ OK' if test2_ok else '⚠️ LEAK'}")

    # TEST 3 — linear nuisance a0+a1*z
    print(f"\n  TEST 3 — Linear nuisance sensitivity")
    Z   = z_np
    Y   = mu_obs - mu_th
    A00 = np.sum(W);    A01 = np.sum(W * Z);    A11 = np.sum(W * Z**2)
    b0  = np.sum(W * Y);   b1 = np.sum(W * Y * Z)
    det = A00 * A11 - A01**2

    if abs(det) < 1e-15:
        a0_fit = a1_fit = np.nan; test3_ok = False
        print("    ⚠️ no se pudo ajustar")
    else:
        a0_fit = (A11*b0 - A01*b1) / det
        a1_fit = (A00*b1 - A01*b0) / det
        test3_ok = abs(a1_fit) < 0.05
        print(f"    a0={a0_fit:+.4f} | a1={a1_fit:+.4f} mag/z "
              f"→ {'✅ OK' if test3_ok else '⚠️ z-structure'}")

    out = {
        'dM_global':       float(dM_glob),
        'resid_rms':       float(np.sqrt(np.mean(resid**2))),
        'chi2_red':        float(np.mean((resid / mu_err)**2)),
        'test1_dM_range':  dM_range,
        'test1_ok':        bool(test1_ok),
        'test2_slope':     float(slope)     if np.isfinite(slope)     else np.nan,
        'test2_slope_err': float(slope_err) if np.isfinite(slope_err) else np.nan,
        'test2_p_like':    float(p_like)    if np.isfinite(p_like)    else np.nan,
        'test2_ok':        bool(test2_ok),
        'test3_a0':        float(a0_fit)    if np.isfinite(a0_fit)    else np.nan,
        'test3_a1':        float(a1_fit)    if np.isfinite(a1_fit)    else np.nan,
        'test3_ok':        bool(test3_ok),
    }

    print(f"\n  χ²_red={out['chi2_red']:.4f} | RMS resid={out['resid_rms']:.4f}")
    print(f"  Flags: T1 {'✅' if out['test1_ok'] else '⚠️'} | "
          f"T2 {'✅' if out['test2_ok'] else '⚠️'} | "
          f"T3 {'✅' if out['test3_ok'] else '⚠️'}")
    return out

print("✅ run_diagnostics_v4 definido.")


# ---------------------------------------------------------------------
# v4 — injection recovery
# ---------------------------------------------------------------------

def run_injection_recovery_v4(z_arr, mu_err_arr, cos_th_arr,
                              vp_truth_list=(0.0, 200.0, 370.0, 450.0),
                              n_seeds=3, seed_base=12000):
    """
    Inyecta vp conocido en datos ΛCDM + ruido.
    Verifica que el PINN recupera lo correcto con B corregido.
    """
    print(f"\n{'='*70}")
    print("🚀 v4 INJECTION-RECOVERY")
    print(f"{'='*70}")

    z_arr      = np.asarray(z_arr)
    mu_err_arr = np.asarray(mu_err_arr)
    cos_th_arr = np.asarray(cos_th_arr)

    mu_iso = build_mu_fid_lcdm_v4(z_arr)
    B      = build_B_v4(z_arr)
    rows   = []

    for vp_true in vp_truth_list:
        for j in range(n_seeds):
            seed_i = seed_base + int(1000*(j+1) + 10*abs(vp_true))
            rng    = np.random.default_rng(seed_i)

            # inyección con forma dipolar correcta (B ya corregido)
            dmu     = (5.0 / np.log(10.0)) * B * (vp_true / C_LIGHT) * cos_th_arr
            noise   = rng.normal(0.0, mu_err_arr)
            mu_mock = mu_iso + dmu + noise

            fit = run_on_catalog_v4(
                z_arr, mu_mock, mu_err_arr, cos_th_arr,
                catalog_name=f"inj_vp{int(vp_true)}_seed{j+1}",
                verbose=False, seed=seed_i
            )

            rows.append({
                'vp_true':      float(vp_true),
                'seed':         int(seed_i),
                'vp_fit':       fit['vp'],
                'delta_fit':    fit['delta'],
                'chi2_red':     fit['chi2_red_data'],
                'vp_error':     fit['vp'] - float(vp_true),
                'delta_minus_2':fit['delta'] - 2.0
            })

            print(f"  vp_true={vp_true:>5.0f} | seed={seed_i} | "
                  f"vp_fit={fit['vp']:+7.1f} | Δ={fit['delta']:.3f} | "
                  f"err={fit['vp']-vp_true:+7.1f}")

    df = pd.DataFrame(rows)
    summary = (
        df.groupby('vp_true')
          .agg(
              vp_fit_mean=('vp_fit', 'mean'),
              vp_fit_std=('vp_fit', 'std'),
              vp_err_mean=('vp_error', 'mean'),
              vp_err_std=('vp_error', 'std'),
              delta_mean=('delta_fit', 'mean'),
              delta_std=('delta_fit', 'std'),
              chi2_mean=('chi2_red', 'mean'),
          ).reset_index()
    )

    max_abs_bias = float(np.max(np.abs(summary['vp_err_mean'].values)))
    passed = max_abs_bias < 120.0

    print(f"\n{'-'*70}")
    print(summary.to_string(index=False))
    print(f"\nGate injection-recovery: {'✅ PASS' if passed else '❌ FAIL'}")
    print(f"  max |bias medio vp| = {max_abs_bias:.1f} km/s")

    return {'rows': df, 'summary': summary,
            'passed': passed, 'max_abs_bias_kms': max_abs_bias}

print("✅ run_injection_recovery_v4 definido.")


# ---------------------------------------------------------------------
# v4 — null ΛCDM bias test
# ---------------------------------------------------------------------

def run_lcdm_bias_test_v4(z_arr, mu_err_arr, cos_th_arr,
                          n_mocks=10, seed_base=77777):
    """
    Pregunta clave: si los datos vienen de ΛCDM puro (Δ=2, vp=0),
    ¿el PINN recupera Δ=2.0 o se sesga hacia 1.52?

    Si bias < 0.05 → ✅ Δ≈1.52 en datos reales es física real.
    Si bias ≥ 0.15 → ❌ Δ no confiable — artifact del optimizer.
    """
    print(f"\n{'='*70}")
    print("🧪 v4 NULL ΛCDM BIAS TEST")
    print(f"{'='*70}")
    print(f"  Fiducial: Δ=2.0 | vp=0 km/s | N mocks={n_mocks}")

    z_arr      = np.asarray(z_arr)
    mu_err_arr = np.asarray(mu_err_arr)
    cos_th_arr = np.asarray(cos_th_arr)

    mu_fid = build_mu_fid_lcdm_v4(z_arr)
    rows   = []

    for i in range(n_mocks):
        seed_i = seed_base + i * 1000
        rng    = np.random.default_rng(seed_i)
        noise  = rng.normal(0.0, mu_err_arr)
        mu_mock = mu_fid + noise

        fit = run_on_catalog_v4(
            z_arr, mu_mock, mu_err_arr, cos_th_arr,
            catalog_name=f"nullLCDM_{i+1:02d}",
            verbose=False, seed=seed_i
        )

        rows.append({
            'mock':       i + 1,
            'seed':       seed_i,
            'delta_fit':  fit['delta'],
            'vp_fit':     fit['vp'],
            'chi2_red':   fit['chi2_red_data'],
            'bias_delta': fit['delta'] - 2.0
        })

        print(f"  mock {i+1:02d} | Δ_fit={fit['delta']:.3f} "
              f"(bias={fit['delta']-2.0:+.3f}) | vp_fit={fit['vp']:+7.1f}")

    df = pd.DataFrame(rows)

    delta_mean = float(df['delta_fit'].mean())
    delta_std  = float(df['delta_fit'].std())
    vp_mean    = float(df['vp_fit'].mean())
    vp_std     = float(df['vp_fit'].std())
    bias_mean  = delta_mean - 2.0

    if   abs(bias_mean) < 0.05:  verdict = "✅ SIN BIAS SEVERO"; passed = True
    elif abs(bias_mean) < 0.15:  verdict = "⚠️ BIAS MODERADO";  passed = False
    else:                         verdict = "❌ BIAS SEVERO";     passed = False

    print(f"\n{'-'*70}")
    print(f"Δ_fit mean={delta_mean:.3f} ± {delta_std:.3f}")
    print(f"vp_fit mean={vp_mean:.1f} ± {vp_std:.1f} km/s")
    print(f"Bias medio en Δ: {bias_mean:+.3f}")
    print(f"Veredicto: {verdict}")

    df.to_csv(DATA_PATH + "HTANA_v4_null_lcdm_rows.csv", index=False)

    return {
        'rows': df,
        'summary': {
            'delta_mean': delta_mean, 'delta_std': delta_std,
            'vp_mean': vp_mean, 'vp_std': vp_std,
            'bias_mean_delta': bias_mean,
            'verdict': verdict, 'passed': passed
        }
    }

print("✅ run_lcdm_bias_test_v4 definido.")


# ---------------------------------------------------------------------
# v4 — comparar base vs nuisance lineal
# ---------------------------------------------------------------------

def compare_baseline_vs_linear_nuisance_v4(z_arr, mu_arr, mu_err_arr, cos_th_arr,
                                           label="catalog"):
    print(f"\n{'='*70}")
    print(f"⚖️ v4 BASE vs LINEAR NUISANCE — {label}")
    print(f"{'='*70}")

    base = run_on_catalog_v4(z_arr, mu_arr, mu_err_arr, cos_th_arr,
                             catalog_name=f"{label}_base", verbose=False,
                             use_linear_nuisance=False)
    lin  = run_on_catalog_v4(z_arr, mu_arr, mu_err_arr, cos_th_arr,
                             catalog_name=f"{label}_linNuis", verbose=False,
                             use_linear_nuisance=True)

    summary = {
        'base_delta': base['delta'], 'base_vp': base['vp'],
        'base_chi2': base['chi2_red_data'],
        'lin_delta': lin['delta'],  'lin_vp': lin['vp'],
        'lin_chi2': lin['chi2_red_data'],
        'lin_a0': lin.get('a0', np.nan), 'lin_a1': lin.get('a1', np.nan),
        'delta_shift': lin['delta'] - base['delta'],
        'vp_shift':    lin['vp']   - base['vp'],
        'chi2_improvement': base['chi2_red_data'] - lin['chi2_red_data'],
    }

    print(f"  Base   : Δ={summary['base_delta']:.3f} | vp={summary['base_vp']:.1f} | χ²={summary['base_chi2']:.4f}")
    print(f"  Linear : Δ={summary['lin_delta']:.3f} | vp={summary['lin_vp']:.1f} | χ²={summary['lin_chi2']:.4f}")
    print(f"  a0={summary['lin_a0']:+.4f} | a1={summary['lin_a1']:+.4f}")
    print(f"  Shift Δ={summary['delta_shift']:+.4f} | Shift vp={summary['vp_shift']:+.1f}")
    print(f"  Δχ² improvement={summary['chi2_improvement']:+.4f}")

    return {'base': base, 'linear': lin, 'summary': summary}

print("✅ compare_baseline_vs_linear_nuisance_v4 definido.")


# ---------------------------------------------------------------------
# v4 — pipeline maestro con compuertas
# ---------------------------------------------------------------------

def run_v4_audit_pipeline(z_arr, mu_arr, mu_err_arr, cos_th_arr,
                          label="Pantheon+",
                          null_mocks=10,
                          inj_seeds=3):
    """
    Orden correcto de falsación:
    1) preflight
    2) injection recovery
    3) null ΛCDM bias test
    4) fit real base
    5) diagnostics con forward() real
    6) comparación base vs nuisance lineal
    7) veredicto final: GO / HOLD / STOP

    Empezar con null_mocks=5, inj_seeds=2.
    Si pasa → subir a null_mocks=10, inj_seeds=3.
    """
    report = {'label': label, 'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")}

    # 1) preflight
    preflight = preflight_sanity_checks_v4()
    report['preflight'] = preflight
    if not preflight['passed']:
        report['allow_heavy_scan'] = False
        report['verdict'] = "STOP — preflight falló"
        print("\n❌ Se corta aquí. No se justifica continuar.")
        return report

    # 2) injection recovery
    inj = run_injection_recovery_v4(
        z_arr, mu_err_arr, cos_th_arr, n_seeds=inj_seeds
    )
    report['injection_recovery'] = {
        'summary': inj['summary'].to_dict(orient='records'),
        'passed': inj['passed'],
        'max_abs_bias_kms': inj['max_abs_bias_kms']
    }

    # 3) null ΛCDM
    nulls = run_lcdm_bias_test_v4(
        z_arr, mu_err_arr, cos_th_arr, n_mocks=null_mocks
    )
    report['null_lcdm'] = nulls['summary']

    if not inj['passed'] or not nulls['summary']['passed']:
        report['allow_heavy_scan'] = False
        report['verdict'] = "STOP — injection/null tests no pasaron"
        print("\n❌ Injection-recovery o null ΛCDM no pasaron.")
        print("   No se justifica sky scan pesado.")
        return report

    # 4) fit real base
    fit_real = run_on_catalog_v4(
        z_arr, mu_arr, mu_err_arr, cos_th_arr,
        catalog_name=f"{label}_real_base", verbose=True
    )
    report['fit_real_base'] = {
        k: v for k, v in fit_real.items() if k not in ['state_dict', 'history']
    }

    # 5) diagnostics reales
    diag = run_diagnostics_v4(z_arr, mu_arr, mu_err_arr, cos_th_arr,
                               fit_real, label=label)
    report['diagnostics'] = diag

    # 6) sensibilidad nuisance lineal
    cmp = compare_baseline_vs_linear_nuisance_v4(
        z_arr, mu_arr, mu_err_arr, cos_th_arr, label=label
    )
    report['linear_nuisance_compare'] = cmp['summary']

    # 7) veredicto
    diag_ok     = diag['test1_ok'] and diag['test2_ok'] and diag['test3_ok']
    shift_small = (abs(cmp['summary']['vp_shift']) < 80.0
                   and abs(cmp['summary']['delta_shift']) < 0.08)

    allow_heavy      = bool(diag_ok and shift_small)
    report['allow_heavy_scan'] = allow_heavy
    report['verdict'] = (
        "GO — merece null sky scan pesado"
        if allow_heavy else
        "HOLD — señal interesante, pero aún no blindada"
    )

    print(f"\n{'='*70}")
    print("🏁 v4 FINAL VERDICT")
    print(f"{'='*70}")
    print(f"  Diagnostics all-pass:            {'✅' if diag_ok else '⚠️'}")
    print(f"  Small shift under linear nuisance: {'✅' if shift_small else '⚠️'}")
    print(f"  Allow heavy sky scan:             {'✅ YES' if allow_heavy else '❌ NO'}")
    print(f"  Verdict: {report['verdict']}")

    # guardar reporte plano
    flat = [
        {'block': 'preflight',           'passed': preflight['passed']},
        {'block': 'injection_recovery',  'passed': inj['passed'],
         'max_abs_bias_kms': inj['max_abs_bias_kms']},
        {'block': 'null_lcdm',           'passed': nulls['summary']['passed'],
         'bias_mean_delta': nulls['summary']['bias_mean_delta'],
         'delta_mean': nulls['summary']['delta_mean']},
        {'block': 'fit_real_base',       'delta': fit_real['delta'],
         'vp': fit_real['vp'], 'chi2_red': fit_real['chi2_red_data']},
        {'block': 'diagnostics',         'test1_ok': diag['test1_ok'],
         'test2_ok': diag['test2_ok'],   'test3_ok': diag['test3_ok'],
         'test2_p_like': diag['test2_p_like'], 'test3_a1': diag['test3_a1']},
        {'block': 'linear_nuisance',     'delta_shift': cmp['summary']['delta_shift'],
         'vp_shift': cmp['summary']['vp_shift'],
         'chi2_improvement': cmp['summary']['chi2_improvement']},
        {'block': 'final',               'allow_heavy_scan': allow_heavy,
         'verdict': report['verdict']},
    ]
    pd.DataFrame(flat).to_csv(DATA_PATH + "HTANA_v4_audit_report.csv", index=False)
    print(f"\n✅ Reporte guardado: HTANA_v4_audit_report.csv")

    return report


print("✅ run_v4_audit_pipeline definido.")

# =============================================================================
# USO — corre esto cuando tengas el catálogo cargado:
#
# z_arr, mu_arr, mu_err_arr, cos_th_arr = catalogs['Pantheon+']
#
# Primera corrida (rápida — para verificar que todo corre):
# report_v4 = run_v4_audit_pipeline(
#     z_arr, mu_arr, mu_err_arr, cos_th_arr,
#     label="Pantheon+",
#     null_mocks=5,
#     inj_seeds=2
# )
#
# Si pasa → corrida completa:
# report_v4 = run_v4_audit_pipeline(
#     z_arr, mu_arr, mu_err_arr, cos_th_arr,
#     label="Pantheon+",
#     null_mocks=10,
#     inj_seeds=3
# )
#
# print(report_v4['verdict'])
# =============================================================================
