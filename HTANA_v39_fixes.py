# =============================================================================
# H-TANA PINN — FIXES v3.9
# Author: Iván Sheligo
# AI Audit: Claude (Anthropic) + GPT (OpenAI) + Grok (xAI)
# Fecha: 2026-03-18
#
# DOS FIXES respecto a v3.8:
#
# FIX 1 — Factor B del dipolo (física)
#   Problema:  B = 1 - (1+z)/(E·Dc)   ← +1 espurio
#   Correcto:  B = -(1+z)/(E·Dc)
#   Derivación: GPT confirmó que el +1 del boost Doppler directo se cancela
#               exactamente con el término de perturbación de redshift cuando
#               se expresa todo a z_obs fijo (Bonvin 2006, Hui & Greene,
#               Peterson et al. 2022 Pantheon+).
#   Impacto:   vp sobreestimado ~5% a z=0.05, ~10% a z=0.10, ~16% a z=0.15
#              La señal NO muere — la amplitud se ajusta ~5-15%.
#
# FIX 2 — run_diagnostics usa forward() real del PINN (bug crítico)
#   Problema:  Los diagnósticos reconstruían E(z) analíticamente (THDE puro),
#              ignorando los pesos del MLP entrenado. Los flags ✅ FLAT/CLEAN
#              podían ser falsos positivos.
#   Solución:  run_on_catalog ahora devuelve también el model state_dict.
#              Cell 7 guarda el state_dict en cross_results.
#              run_diagnostics carga el state_dict y llama model.forward()
#              real para calcular mu_th.
#
# APLICACIÓN:
#   Busca los comentarios "# ← FIX 1" y "# ← FIX 2" en este archivo
#   y reemplaza las secciones correspondientes en v3.8.
# =============================================================================


# -----------------------------------------------------------------------
# FIX 1 — CELL 5: forward() del PINN  (reemplaza línea 564 de v3.8)
# -----------------------------------------------------------------------
# ANTES (v3.8, línea 564):
#   B = -1.0/z_s + (1.0+z_s)*(EDc - z_s)/(z_s * EDc)
#
# DESPUÉS (v3.9):
#   B = -(1.0 + z_s) / EDc
#
# El resto del forward() no cambia.

class HTANA_v39(nn.Module):
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
        z1    = z_sorted.squeeze(-1)
        z_wz  = torch.cat([torch.zeros(1, dtype=torch.float64, device=device), z1])
        E_wz  = self.E_of_z(z_wz.unsqueeze(-1))
        Dc_wz = cumulative_trapz(1.0/(E_wz + 1e-10), z_wz)
        Dc    = torch.clamp(Dc_wz[1:], min=1e-10)
        E     = E_wz[1:]
        z_s   = torch.clamp(z1, min=1e-8)
        EDc   = torch.clamp(E * Dc, min=1e-12)

        # ← FIX 1: B correcto — derivación relativista estándar
        # δ ln dL = -(1+z)/(E·Dc) · (vp/c) · cosθ
        # El +1 del boost Doppler directo se cancela con la perturbación de
        # redshift al expresar todo a z_obs fijo (Bonvin 2006 / Peterson 2022)
        B = -(1.0 + z_s) / EDc                        # ← FIX 1 (era: -1/z + (1+z)(EDc-z)/(z·EDc))

        _, vp = self.get_params()
        DL    = torch.clamp((1.0+z1)*Dc*DH*(1.0 + B*(vp/C_LIGHT)*cos_theta),
                            min=1e-6)
        return 5.0*torch.log10(DL) + 25.0

    def E_only(self, z):
        return self.E_of_z(z)

print("✅ PINN v3.9 definido (Fix 1: B correcto).")


# -----------------------------------------------------------------------
# FIX 2a — CELL 6: run_on_catalog devuelve state_dict
# -----------------------------------------------------------------------
# CAMBIO: añade model.state_dict() al return y al print final.

def run_on_catalog(z_arr, mu_arr, mu_err_arr, cos_theta_arr,
                   catalog_name="?", verbose=True):
    """
    Run PINN on real catalog data with fixed recipe.
    Returns: (delta_rec, vp_rec, final_loss, state_dict)   ← FIX 2a: +state_dict
    """
    torch.manual_seed(SEED)

    z_t   = torch.tensor(z_arr,         dtype=torch.float64, device=device).view(-1,1)
    mu_t  = torch.tensor(mu_arr,        dtype=torch.float64, device=device)
    me_t  = torch.tensor(mu_err_arr,    dtype=torch.float64, device=device)
    ct_t  = torch.tensor(cos_theta_arr, dtype=torch.float64, device=device)

    model = HTANA_v39().to(device)                        # ← usa v3.9
    opt   = torch.optim.Adam([
        {'params': model.mlp.parameters(), 'lr': 1e-3},
        {'params': [model.delta_raw],       'lr': 5e-3},
        {'params': [model.vp_raw],          'lr': 1e-2},
    ])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=EPOCHS, eta_min=1e-5)

    g = torch.Generator(device=device).manual_seed(SEED + 10)

    t0 = time.time()
    for epoch in range(EPOCHS):
        opt.zero_grad(set_to_none=True)

        mu_th    = model(z_t, ct_t)
        inv_sig2 = 1.0 / (me_t ** 2)
        delta_M  = (torch.sum((mu_t - mu_th) * inv_sig2) /
                    torch.sum(inv_sig2)).detach()
        mu_th_corr = mu_th + delta_M
        L_data = torch.mean(((mu_th_corr - mu_t) / me_t)**2)

        z_col = torch.rand(3000, generator=g, device=device, dtype=torch.float64) \
                * (Z_COL_MAX - 0.01) + 0.01
        z_col = z_col.view(-1, 1)
        E     = model.E_only(z_col)
        delta, _ = model.get_params()
        z1    = z_col.squeeze(-1)
        R     = E**2 - Om_m*(1+z1)**3 - Om_de*torch.pow(E, 4.0 - 2.0*delta)
        den   = Om_m*(1+z1)**3 + Om_de*torch.pow(E.detach(),
                4.0 - 2.0*delta) + 1e-8
        L_fried = torch.mean((R/den)**2)

        loss = L_data + LAMBDA_FRIED * L_fried
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        opt.step()
        sched.step()

        if verbose and (epoch+1) % 2000 == 0:
            d, v = model.get_params()
            print(f"    [{catalog_name}] epoch {epoch+1} | "
                  f"Δ={d.item():.3f} | vp={v.item():.0f} km/s | "
                  f"loss={loss.item():.4f}")

    delta, vp = model.get_params()
    elapsed   = time.time() - t0
    if verbose:
        print(f"  ✅ {catalog_name}: Δ={delta.item():.3f} | "
              f"vp={vp.item():.0f} km/s | ⏱️ {elapsed:.0f}s")

    # ← FIX 2a: devuelve state_dict para que diagnostics use el modelo real
    return delta.item(), vp.item(), loss.item(), model.state_dict()

print("✅ run_on_catalog v3.9 (Fix 2a: devuelve state_dict).")


# -----------------------------------------------------------------------
# FIX 2b — CELL 7: guardar state_dict en cross_results
# -----------------------------------------------------------------------
# CAMBIO: desempaqueta el 4to valor y lo guarda en cross_results.

print(f"\n🚀 Cross-catalog run")
print(f"   Recipe: λ_fried={LAMBDA_FRIED} | z_col_max={Z_COL_MAX} | epochs={EPOCHS}")
print(f"   Catalogs: {list(catalogs.keys())}")
print()

cross_results = {}

t_total = time.time()
for cat_name, (z, mu, mu_err, cos_th) in catalogs.items():
    print(f"  ▶ Running {cat_name} ({len(z)} SNe)...")
    dr, vr, lf, sd = run_on_catalog(z, mu, mu_err, cos_th,   # ← FIX 2b: +sd
                                     catalog_name=cat_name, verbose=True)
    cross_results[cat_name] = {
        'delta':      dr,
        'vp':         vr,
        'loss':       lf,
        'n_sne':      len(z),
        'state_dict': sd,   # ← FIX 2b: guarda pesos del MLP entrenado
    }
    print()

print(f"⏱️  Total: {time.time()-t_total:.0f}s")


# -----------------------------------------------------------------------
# FIX 2c — CELL 8 (run_diagnostics): usa model.forward() real
# -----------------------------------------------------------------------
# CAMBIO: reemplaza el bloque proxy analítico (líneas 919-962 de v3.8)
# por carga del state_dict y llamada directa a model.forward().

def run_diagnostics(catalogs, cross_results, d_hat):
    """
    Run 3 diagnostic tests per catalog after PINN convergence.
    v3.9: usa model.forward() real con los pesos entrenados.    ← FIX 2c
    """
    if len(cross_results) == 0:
        print("⚠️  No results to diagnose.")
        return

    print(f"\n{'='*70}")
    print(f"🔬 v3.9 DIAGNOSTIC AUDIT  (Fix 2: forward() real)")
    print(f"{'='*70}")
    print(f"  Purpose: verify that δM* uniform assumption holds.")
    print(f"  Uses: converged PINN weights (MLP + Δ + vp) — NOT analytic proxy.\n")

    diag_summary = {}

    for cat_name, (z_arr, mu_arr, mu_err_arr, cos_th_arr) in catalogs.items():
        if cat_name not in cross_results:
            continue

        print(f"\n  ━━━ {cat_name} ━━━")

        z_t  = torch.tensor(z_arr,       dtype=torch.float64, device=device).view(-1,1)
        ct_t = torch.tensor(cos_th_arr,  dtype=torch.float64, device=device)

        # ← FIX 2c: carga el modelo real entrenado — NO re-entrena, NO proxy
        if 'state_dict' not in cross_results[cat_name]:
            print(f"  ⚠️  state_dict no disponible para {cat_name}.")
            print(f"     Re-corre Cell 7 con run_on_catalog v3.9.")
            continue

        model_diag = HTANA_v39().to(device)
        model_diag.load_state_dict(cross_results[cat_name]['state_dict'])
        model_diag.eval()

        with torch.no_grad():
            mu_th_t = model_diag(z_t, ct_t)             # ← forward() real del PINN
            mu_th   = mu_th_t.cpu().numpy()

        mu_obs = mu_arr
        mu_err = mu_err_arr
        cos_th = cos_th_arr
        z_np   = z_arr

        # Global δM*
        inv_s2  = 1.0 / mu_err**2
        dM_glob = np.sum((mu_obs - mu_th) * inv_s2) / np.sum(inv_s2)
        resid   = mu_obs - (mu_th + dM_glob)

        diag_summary[cat_name] = {}

        # ── TEST 1: Tomography — δM* by z-bin ──────────────────────────────
        print(f"\n  TEST 1 — Zero-point tomography (δM* by z-bin)")
        print(f"  If flat → offset is uniform → safe. If drifts → z-structure.")
        z_edges = [0.005, 0.10, 0.30, 0.60]
        dM_bins = []
        for i in range(len(z_edges)-1):
            mask = (z_np >= z_edges[i]) & (z_np < z_edges[i+1])
            if mask.sum() < 5:
                print(f"    z=[{z_edges[i]:.2f},{z_edges[i+1]:.2f}]: too few SNe ({mask.sum()})")
                continue
            inv_s2_b = 1.0 / mu_err[mask]**2
            dM_b     = np.sum((mu_obs[mask] - mu_th[mask]) * inv_s2_b) / np.sum(inv_s2_b)
            dM_bins.append(dM_b)
            print(f"    z=[{z_edges[i]:.2f},{z_edges[i+1]:.2f}]"
                  f"  N={mask.sum():4d}  δM*={dM_b:+.4f}")

        if len(dM_bins) >= 2:
            dM_range = max(dM_bins) - min(dM_bins)
            flag1 = "✅ FLAT" if dM_range < 0.03 else "⚠️  DRIFTS"
            print(f"    Range: {dM_range:.4f} mag  →  {flag1}")
            if dM_range >= 0.03:
                print(f"    ⚠️  z-dependent offset detected. Δ may be contaminated.")
                print(f"       Consider adding δM(z)=a0+a1*z nuisance term.")
            diag_summary[cat_name]['test1_range'] = dM_range
            diag_summary[cat_name]['test1_flag']  = flag1

        # ── TEST 2: Angular residuals — residual dipole ─────────────────────
        print(f"\n  TEST 2 — Residual dipole (angular bias check)")
        print(f"  If slope ≈ 0 → no angular photometric bias → vp is clean.")
        from scipy.stats import linregress
        slope, intercept, r_val, p_val, std_err = linregress(cos_th, resid)
        flag2 = "✅ CLEAN" if p_val > 0.05 else "⚠️  DIPOLE RESIDUAL"
        print(f"    Residual dipole slope: {slope:+.4f} mag/unit_cos")
        print(f"    p-value: {p_val:.4f}  →  {flag2}")
        if p_val <= 0.05:
            print(f"    ⚠️  Significant angular structure in residuals.")
            print(f"       vp measurement may absorb photometric angular bias.")
        diag_summary[cat_name]['test2_slope'] = slope
        diag_summary[cat_name]['test2_pval']  = p_val
        diag_summary[cat_name]['test2_flag']  = flag2

        # ── TEST 3: Nuisance richness — does Δ survive δM(z)=a0+a1*z? ──────
        print(f"\n  TEST 3 — Nuisance richness (linear δM(z) = a0 + a1*z)")
        print(f"  If Δ barely moves → signal is robust to calibration assumptions.")
        W   = 1.0 / mu_err**2
        Y   = (mu_obs - mu_th) * W
        Z   = z_np
        A00 = np.sum(W);      A01 = np.sum(W * Z);    A11 = np.sum(W * Z**2)
        b0  = np.sum(Y);      b1  = np.sum(Y * Z)
        det = A00*A11 - A01**2
        if abs(det) < 1e-15:
            print(f"    ⚠️  Degenerate system — cannot fit linear offset.")
        else:
            a0_fit = (A11*b0 - A01*b1) / det
            a1_fit = (A00*b1 - A01*b0) / det
            resid_linear = mu_obs - mu_th - (a0_fit + a1_fit * z_np)
            chi2_global  = np.sum(resid**2          / mu_err**2) / len(resid)
            chi2_linear  = np.sum(resid_linear**2   / mu_err**2) / len(resid_linear)
            print(f"    a0={a0_fit:+.4f}  a1={a1_fit:+.4f} mag/unit_z")
            print(f"    χ²_red (global offset): {chi2_global:.4f}")
            print(f"    χ²_red (linear offset): {chi2_linear:.4f}")
            print(f"    Δχ²_red improvement:    {chi2_global-chi2_linear:+.4f}")
            flag3_a1 = "✅ FLAT" if abs(a1_fit) < 0.05 else "⚠️  z-SLOPE"
            print(f"    a1 significance: {flag3_a1}")
            if abs(a1_fit) >= 0.05:
                print(f"    ⚠️  Linear z-trend in offset. Run PINN with δM(z) nuisance")
                print(f"       to check if Δ is stable.")
            diag_summary[cat_name]['test3_a1']   = a1_fit
            diag_summary[cat_name]['test3_flag'] = flag3_a1

    # ── Final audit summary ─────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"📋 DIAGNOSTIC AUDIT SUMMARY  (v3.9 — forward() real)")
    print(f"{'='*70}")
    print(f"  {'Catalog':>12} | {'Test1 (z-tomo)':>16} | "
          f"{'Test2 (angular)':>17} | {'Test3 (linear δM)':>18}")
    print(f"  {'-'*68}")
    for cat_name, diag in diag_summary.items():
        t1 = diag.get('test1_flag', 'N/A')
        t2 = diag.get('test2_flag', 'N/A')
        t3 = diag.get('test3_flag', 'N/A')
        print(f"  {cat_name:>12} | {t1:>16} | {t2:>17} | {t3:>18}")

    print(f"\n  Interpretation:")
    print(f"  ✅ ALL FLAT/CLEAN → δM* scalar is sufficient. Results trustworthy.")
    print(f"  ⚠️  ANY WARNING   → declare in paper. Consider extended nuisance model.")
    print(f"\n  Key quote for paper methods section:")
    print(f"  'A diagnostic analysis of Hubble residuals using the PINN forward pass")
    print(f"   confirms that inter-catalog discrepancies are dominated by monopolar")
    print(f"   magnitude offsets, with no statistically significant redshift drift or")
    print(f"   angular structure that could bias the recovery of the bulk flow amplitude")
    print(f"   or Tsallis index.'")
    print(f"  (Only include this sentence if Tests 1-3 all pass.)")

    rows = [{'catalog': k, **v} for k, v in diag_summary.items()]
    pd.DataFrame(rows).to_csv(
        DATA_PATH + "HTANA_v39_diagnostics.csv", index=False)
    print(f"\n✅ Diagnostics saved: HTANA_v39_diagnostics.csv")
    return diag_summary


# -----------------------------------------------------------------------
# FIX 3 — CELL NUEVA: Mocks ΛCDM puros — test de bias en Δ
# -----------------------------------------------------------------------
# Pregunta: si los datos vienen de ΛCDM puro (Δ=2, vp=0),
# ¿el PINN recupera Δ=2.0 o se sesga hacia 1.52?
#
# Si histograma de Δ_rec se centra en 2.0 → limpio, el resultado es física.
# Si se corre hacia ~1.52                 → bias del optimizer → problema serio.
#
# GPT y Grok coinciden: esto es lo que falta para cerrar el argumento de Δ.
# Mientras no se corra, Δ≈1.52 es "plausible" pero no "demostrado".

def run_lcdm_bias_test(z_arr, mu_err_arr, cos_th_arr,
                       n_mocks=20, seed_base=77777):
    """
    Genera n_mocks realizaciones sintéticas desde ΛCDM puro:
      Δ_fid = 2.0  (ΛCDM exacto)
      vp_fid = 0   (sin bulk flow)
      ruido ~ N(0, mu_err)

    Corre PINN v3.9 en cada mock y reporta Δ_rec y vp_rec.

    Interpretación:
      mean(Δ_rec) ≈ 2.0  → sin bias → Δ≈1.52 en datos reales es física real
      mean(Δ_rec) ≪ 2.0  → bias del optimizer → Δ resultado no es confiable
    """
    print(f"\n{'='*70}")
    print(f"🧪 FIX 3 — ΛCDM BIAS TEST")
    print(f"{'='*70}")
    print(f"  Fiducial: Δ=2.0 (ΛCDM), vp=0 km/s")
    print(f"  N mocks: {n_mocks} | Noise: observational mu_err")
    print(f"  Pregunta: ¿el PINN recupera Δ=2.0 o se sesga?")
    print()

    z_t  = torch.tensor(z_arr,     dtype=torch.float64, device=device).view(-1,1)
    ct_t = torch.tensor(cos_th_arr, dtype=torch.float64, device=device)
    z1   = z_t.squeeze(-1)

    # Generar μ fiducial ΛCDM (Δ=2 exacto, vp=0)
    # Con Δ=2: E(z)² = Ωm(1+z)³ + ΩΛ  → ΛCDM estándar
    Om_m_v  = 0.315
    Om_de_v = 1.0 - Om_m_v
    with torch.no_grad():
        z_full   = torch.cat([torch.zeros(1, dtype=torch.float64, device=device), z1])
        E_lcdm   = torch.sqrt(Om_m_v*(1+z_full)**3 + Om_de_v)
        Dc_full  = cumulative_trapz(1.0/(E_lcdm + 1e-10), z_full)
        Dc_lcdm  = torch.clamp(Dc_full[1:], min=1e-10)
        DL_lcdm  = (1.0 + z1) * Dc_lcdm * DH
        mu_fid   = (5.0*torch.log10(DL_lcdm) + 25.0).cpu().numpy()

    print(f"  μ_fid range: [{mu_fid.min():.2f}, {mu_fid.max():.2f}]")
    print()

    results = []
    rng = np.random.default_rng(seed_base)

    for i in range(n_mocks):
        seed_i = seed_base + i * 10000
        noise  = rng.normal(0.0, mu_err_arr)
        mu_mock = mu_fid + noise

        delta_rec, vp_rec, loss_rec, _ = run_on_catalog(
            z_arr, mu_mock, mu_err_arr, cos_th_arr,
            catalog_name=f"ΛCDM-mock-{i+1:02d}", verbose=False)

        results.append({'mock': i+1, 'delta_rec': delta_rec, 'vp_rec': vp_rec,
                        'loss': loss_rec})

        bias_d = delta_rec - 2.0
        print(f"  Mock {i+1:02d}: Δ_rec={delta_rec:.3f} (bias={bias_d:+.3f}) | "
              f"vp_rec={vp_rec:+.0f} km/s")

    # Resumen estadístico
    deltas = np.array([r['delta_rec'] for r in results])
    vps    = np.array([r['vp_rec']    for r in results])

    print(f"\n{'='*70}")
    print(f"📋 ΛCDM BIAS TEST — RESUMEN")
    print(f"{'='*70}")
    print(f"  Δ_rec:  mean={deltas.mean():.3f}  std={deltas.std():.3f}  "
          f"[{deltas.min():.3f}, {deltas.max():.3f}]")
    print(f"  vp_rec: mean={vps.mean():.1f}  std={vps.std():.1f} km/s")
    print()

    bias_mean = deltas.mean() - 2.0
    if abs(bias_mean) < 0.05:
        verdict = "✅ SIN BIAS — PINN recupera Δ=2.0 correctamente"
        detail  = "   Δ≈1.52 en datos reales refleja física, no artifact del optimizer."
    elif abs(bias_mean) < 0.15:
        verdict = "⚠️  BIAS MODERADO — revisar lambda_fried y z_col_max"
        detail  = "   Δ≈1.52 puede estar parcialmente inflado. Declarar en paper."
    else:
        verdict = "❌ BIAS SEVERO — Δ resultado no confiable"
        detail  = "   El optimizer tiene preferencia por Δ<2 independiente de los datos."

    print(f"  Bias en Δ: {bias_mean:+.3f}")
    print(f"  Veredicto: {verdict}")
    print(f"  {detail}")

    # Guardar resultados
    import pandas as pd
    df_bias = pd.DataFrame(results)
    df_bias['delta_fid'] = 2.0
    df_bias['vp_fid']    = 0.0
    df_bias['bias_delta'] = df_bias['delta_rec'] - 2.0
    df_bias.to_csv(DATA_PATH + "HTANA_v39_lcdm_bias_test.csv", index=False)
    print(f"\n✅ Resultados guardados: HTANA_v39_lcdm_bias_test.csv")

    return results, deltas.mean(), deltas.std()


# Ejecutar el bias test sobre Pantheon+ (o el catálogo que tengas cargado)
if 'Pantheon+' in catalogs:
    z_arr, mu_arr, mu_err_arr, cos_th_arr = catalogs['Pantheon+']
    lcdm_results, delta_mean, delta_std = run_lcdm_bias_test(
        z_arr, mu_err_arr, cos_th_arr, n_mocks=20)
else:
    # Usa el primer catálogo disponible
    cat_name = list(catalogs.keys())[0]
    z_arr, mu_arr, mu_err_arr, cos_th_arr = catalogs[cat_name]
    print(f"  Usando catálogo: {cat_name}")
    lcdm_results, delta_mean, delta_std = run_lcdm_bias_test(
        z_arr, mu_err_arr, cos_th_arr, n_mocks=20)


# -----------------------------------------------------------------------
# RESUMEN DE CAMBIOS v3.8 → v3.9
# -----------------------------------------------------------------------
# FIX 1 (física):
#   Archivo:   HTANA_v39.forward()  línea ~35 de este archivo
#   Antes:     B = -1.0/z_s + (1.0+z_s)*(EDc - z_s)/(z_s * EDc)
#   Después:   B = -(1.0 + z_s) / EDc
#   Impacto:   vp baja ~5-16% dependiendo de z efectivo de la señal.
#              La detección sobrevive. La amplitud se recalibra.
#
# FIX 2 (diagnósticos):
#   2a: run_on_catalog devuelve state_dict como 4to valor
#   2b: Cell 7 guarda state_dict en cross_results[cat]['state_dict']
#   2c: run_diagnostics carga state_dict y usa model.forward() real
#   Impacto:   Los flags ✅/⚠️ ahora reflejan el PINN real entrenado,
#              no un modelo THDE analítico perfecto.
#
# FIX 3 (bias test):
#   Cell nueva: run_lcdm_bias_test()
#   Genera 20 mocks desde ΛCDM puro (Δ=2, vp=0) con ruido observacional
#   Corre PINN v3.9 en cada uno y reporta distribución de Δ_rec
#   Veredicto:
#     |bias| < 0.05  → ✅ Δ≈1.52 en datos reales es física
#     |bias| < 0.15  → ⚠️  bias moderado — declarar en paper
#     |bias| ≥ 0.15  → ❌ bias severo — Δ no confiable
#   Salida: HTANA_v39_lcdm_bias_test.csv
#
# NULL MOCKS:
#   Los 20 mocks actuales (corriendo con v3.8) son válidos como:
#   → Capa A: el pipeline no genera falsos positivos triviales
#   → p≲0.048 si 20/20 limpios
#   Después de v3.9: rehacer null mocks con B corregido para:
#   → Capa B: validación física publicable
# -----------------------------------------------------------------------

print("\n✅ v3.9 lista para audit.")
print("   Cambios vs v3.8:")
print("   1. B = -(1+z)/EDc  (física relativista estándar)")
print("   2. run_diagnostics usa model.forward() real (no proxy analítico)")
