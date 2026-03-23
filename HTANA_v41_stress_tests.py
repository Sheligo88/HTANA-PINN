# =============================================================================
# H-TANA PINN — v4.1 STRESS TESTS
# Autores: Iván Sheligo + Claude (Anthropic) + GPT (OpenAI) + Grok (xAI)
# Fecha: 2026-03-18
#
# INSTRUCCIONES:
#   Pegar DEBAJO de v3.9 + v4_audit_scaffold.py
#   Orden de ejecución:
#     1. run_v4_audit_pipeline(...)          ← preflight + injection + null ΛCDM
#     2. stress_test_zcol_and_errors_v41(...)← robustez z_col y errores
#     3. summarize_stress_gate_v41(...)       ← veredicto automático
#
# QUÉ ATACA ESTE PARCHE:
#   Tres debilidades reales sin covmat completa:
#   - Sensibilidad a z_col_max (colocation range)
#   - Sensibilidad a inflación de errores diagonales
#   - Gate comparativa automática PASS/MIXED/FAIL
#
# INTERPRETACIÓN:
#   ✅ PASS   → señal no depende del setup — claim robusto
#   ⚠️ MIXED  → señal vive pero con sensibilidad no trivial — narrativa conservadora
#   ❌ FAIL   → señal depende demasiado del setup — no publicar aún
# =============================================================================

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# v4.1 — inflación diagonal de errores
# ---------------------------------------------------------------------

def inflate_mu_errors_v41(mu_err_arr, mode="quadrature", frac=0.10, floor=None):
    """
    Infla errores diagonales como aproximación conservadora a systematics/covmat.

    Parámetros
    ----------
    mode : str
        "quadrature"    → sigma_new = sqrt(sigma² + (frac·sigma_med)²)
        "multiplicative"→ sigma_new = sigma · (1 + frac)
        "floor"         → sigma_new = max(sigma, floor)
    frac : float
        fracción para quadrature o multiplicative
    floor : float or None
        piso absoluto en magnitudes si mode="floor"
    """
    mu_err_arr = np.asarray(mu_err_arr, dtype=float)

    if mode == "quadrature":
        sigma_med  = np.median(mu_err_arr)
        extra      = frac * sigma_med
        return np.sqrt(mu_err_arr**2 + extra**2)

    elif mode == "multiplicative":
        return mu_err_arr * (1.0 + frac)

    elif mode == "floor":
        if floor is None:
            raise ValueError("mode='floor' requiere floor")
        return np.maximum(mu_err_arr, floor)

    else:
        raise ValueError("mode debe ser 'quadrature', 'multiplicative' o 'floor'")


# ---------------------------------------------------------------------
# v4.1 — wrapper de run_on_catalog_v4 con overrides limpios
# ---------------------------------------------------------------------

def run_on_catalog_v41(z_arr, mu_arr, mu_err_arr, cos_theta_arr,
                       catalog_name="?",
                       use_linear_nuisance=False,
                       seed=None, epochs=None,
                       lambda_fried=None, z_col_max=None,
                       error_inflation=None,
                       verbose=False):
    """
    Wrapper ligero sobre run_on_catalog_v4:
    - permite inflar errores antes del fit
    - permite override de z_col_max
    - todo lo demás delega a v4
    """
    mu_err_use = np.asarray(mu_err_arr, dtype=float).copy()

    if error_inflation is not None:
        mu_err_use = inflate_mu_errors_v41(
            mu_err_use,
            mode=error_inflation.get("mode", "quadrature"),
            frac=error_inflation.get("frac", 0.10),
            floor=error_inflation.get("floor", None)
        )

    fit = run_on_catalog_v4(
        z_arr=z_arr,
        mu_arr=mu_arr,
        mu_err_arr=mu_err_use,
        cos_theta_arr=cos_theta_arr,
        catalog_name=catalog_name,
        use_linear_nuisance=use_linear_nuisance,
        seed=seed,
        epochs=epochs,
        lambda_fried=lambda_fried,
        z_col_max=z_col_max,
        verbose=verbose
    )

    fit["mu_err_used_mean"]   = float(np.mean(mu_err_use))
    fit["mu_err_used_median"] = float(np.median(mu_err_use))
    fit["error_inflation"]    = error_inflation

    return fit


# ---------------------------------------------------------------------
# v4.1 — stress test principal
# ---------------------------------------------------------------------

def stress_test_zcol_and_errors_v41(
    z_arr, mu_arr, mu_err_arr, cos_theta_arr,
    label="Pantheon+",
    base_seed=None, epochs=None, lambda_fried=None,
    z_col_grid=(0.6, 0.8, 1.0, 1.4),
    error_scenarios=None,
    run_linear_nuisance=True,
    verbose=True
):
    """
    Corre grilla mínima de robustez:
    - varios z_col_max
    - varios escenarios de inflación diagonal de errores
    - opcionalmente compara base vs nuisance lineal

    Pregunta que responde:
    '¿Mi señal vive solo porque elegí colocation corta y errores optimistas?'
    """
    if base_seed    is None: base_seed    = SEED
    if epochs       is None: epochs       = EPOCHS
    if lambda_fried is None: lambda_fried = LAMBDA_FRIED

    if error_scenarios is None:
        error_scenarios = [
            {"name": "orig",    "inflation": None},
            {"name": "quad10",  "inflation": {"mode": "quadrature",     "frac": 0.10}},
            {"name": "quad20",  "inflation": {"mode": "quadrature",     "frac": 0.20}},
            {"name": "mult10",  "inflation": {"mode": "multiplicative", "frac": 0.10}},
        ]

    rows     = []
    case_idx = 0

    if verbose:
        print(f"\n{'='*78}")
        print(f"🧪 v4.1 STRESS TEST — {label}")
        print(f"{'='*78}")
        print(f"  z_col_grid      = {z_col_grid}")
        print(f"  error_scenarios = {[x['name'] for x in error_scenarios]}")
        print(f"  linear nuisance = {run_linear_nuisance}")

    for zc in z_col_grid:
        for es in error_scenarios:
            case_idx += 1
            case_name = f"{label}__zcol{zc}__{es['name']}"

            if verbose:
                print(f"\n  [{case_idx:02d}] {case_name}")

            # — fit base —
            fit_base = run_on_catalog_v41(
                z_arr=z_arr, mu_arr=mu_arr,
                mu_err_arr=mu_err_arr, cos_theta_arr=cos_theta_arr,
                catalog_name=case_name + "__base",
                use_linear_nuisance=False,
                seed=base_seed, epochs=epochs,
                lambda_fried=lambda_fried, z_col_max=zc,
                error_inflation=es["inflation"], verbose=False
            )

            rows.append({
                "label":           label,
                "case":            case_name,
                "mode":            "base",
                "z_col_max":       float(zc),
                "error_scenario":  es["name"],
                "delta":           fit_base["delta"],
                "vp":              fit_base["vp"],
                "chi2_red":        fit_base["chi2_red_data"],
                "resid_rms":       fit_base["resid_rms"],
                "a0":              np.nan,
                "a1":              np.nan,
                "seed":            base_seed,
            })

            if verbose:
                print(f"     base   | Δ={fit_base['delta']:.3f} | "
                      f"vp={fit_base['vp']:+7.1f} | "
                      f"χ²={fit_base['chi2_red_data']:.4f}")

            # — fit con nuisance lineal —
            if run_linear_nuisance:
                fit_lin = run_on_catalog_v41(
                    z_arr=z_arr, mu_arr=mu_arr,
                    mu_err_arr=mu_err_arr, cos_theta_arr=cos_theta_arr,
                    catalog_name=case_name + "__lin",
                    use_linear_nuisance=True,
                    seed=base_seed, epochs=epochs,
                    lambda_fried=lambda_fried, z_col_max=zc,
                    error_inflation=es["inflation"], verbose=False
                )

                rows.append({
                    "label":           label,
                    "case":            case_name,
                    "mode":            "linear",
                    "z_col_max":       float(zc),
                    "error_scenario":  es["name"],
                    "delta":           fit_lin["delta"],
                    "vp":              fit_lin["vp"],
                    "chi2_red":        fit_lin["chi2_red_data"],
                    "resid_rms":       fit_lin["resid_rms"],
                    "a0":              fit_lin.get("a0", np.nan),
                    "a1":              fit_lin.get("a1", np.nan),
                    "seed":            base_seed,
                })

                if verbose:
                    print(f"     linear | Δ={fit_lin['delta']:.3f} | "
                          f"vp={fit_lin['vp']:+7.1f} | "
                          f"χ²={fit_lin['chi2_red_data']:.4f} | "
                          f"a1={fit_lin.get('a1', np.nan):+.4f}")

    df      = pd.DataFrame(rows)
    df_base = df[df["mode"] == "base"].copy()

    summary = {
        "delta_mean": float(df_base["delta"].mean()),
        "delta_std":  float(df_base["delta"].std(ddof=0)),
        "delta_min":  float(df_base["delta"].min()),
        "delta_max":  float(df_base["delta"].max()),
        "vp_mean":    float(df_base["vp"].mean()),
        "vp_std":     float(df_base["vp"].std(ddof=0)),
        "vp_min":     float(df_base["vp"].min()),
        "vp_max":     float(df_base["vp"].max()),
        "chi2_mean":  float(df_base["chi2_red"].mean()),
        "n_cases":    int(len(df_base)),
    }

    if verbose:
        print(f"\n{'-'*78}")
        print("  Resumen base-only")
        print(f"  Δ:  mean={summary['delta_mean']:.3f} | std={summary['delta_std']:.3f} | "
              f"range=[{summary['delta_min']:.3f}, {summary['delta_max']:.3f}]")
        print(f"  vp: mean={summary['vp_mean']:.1f} | std={summary['vp_std']:.1f} | "
              f"range=[{summary['vp_min']:.1f}, {summary['vp_max']:.1f}] km/s")
        print(f"  χ²_red mean={summary['chi2_mean']:.4f}")

    # guardar resultados
    df.to_csv(DATA_PATH + "HTANA_v41_stress_rows.csv", index=False)
    print(f"\n✅ Stress test guardado: HTANA_v41_stress_rows.csv")

    return {"rows": df, "summary": summary}


# ---------------------------------------------------------------------
# v4.1 — gate automática de robustez
# ---------------------------------------------------------------------

def summarize_stress_gate_v41(
    stress_result,
    ref_z_col_max=0.6,
    ref_error_scenario="orig",
    vp_tol_soft=80.0,
    vp_tol_hard=140.0,
    delta_tol_soft=0.08,
    delta_tol_hard=0.15,
    verbose=True
):
    """
    Compara todos los casos base contra el caso de referencia.

    Tolerancias:
      soft: umbral para PASS
      hard: umbral para MIXED (encima → FAIL)

    Retorna veredicto: ✅ PASS / ⚠️ MIXED / ❌ FAIL
    """
    df  = stress_result["rows"].copy()
    dfb = df[df["mode"] == "base"].copy()

    ref_mask = (
        (dfb["z_col_max"]       == ref_z_col_max) &
        (dfb["error_scenario"]  == ref_error_scenario)
    )

    if ref_mask.sum() != 1:
        raise ValueError(
            f"No se encontró caso de referencia único: "
            f"z_col_max={ref_z_col_max}, error={ref_error_scenario}"
        )

    ref       = dfb.loc[ref_mask].iloc[0]
    delta_ref = float(ref["delta"])
    vp_ref    = float(ref["vp"])

    dfb = dfb.copy()
    dfb["delta_shift"]     = dfb["delta"] - delta_ref
    dfb["vp_shift"]        = dfb["vp"]    - vp_ref
    dfb["abs_delta_shift"] = np.abs(dfb["delta_shift"])
    dfb["abs_vp_shift"]    = np.abs(dfb["vp_shift"])

    max_delta_shift = float(dfb["abs_delta_shift"].max())
    max_vp_shift    = float(dfb["abs_vp_shift"].max())

    if   max_delta_shift < delta_tol_soft and max_vp_shift < vp_tol_soft:
        verdict = "✅ PASS"
    elif max_delta_shift < delta_tol_hard and max_vp_shift < vp_tol_hard:
        verdict = "⚠️ MIXED"
    else:
        verdict = "❌ FAIL"

    out = {
        "reference": {
            "z_col_max":      ref_z_col_max,
            "error_scenario": ref_error_scenario,
            "delta_ref":      delta_ref,
            "vp_ref":         vp_ref,
        },
        "max_abs_delta_shift": max_delta_shift,
        "max_abs_vp_shift":    max_vp_shift,
        "verdict":             verdict,
        "table":               dfb.sort_values(["z_col_max", "error_scenario"]).reset_index(drop=True),
    }

    if verbose:
        print(f"\n{'='*78}")
        print("🏁 v4.1 STRESS GATE")
        print(f"{'='*78}")
        print(f"  Referencia: z_col_max={ref_z_col_max} | error_scenario={ref_error_scenario}")
        print(f"  Δ_ref  = {delta_ref:.3f}")
        print(f"  vp_ref = {vp_ref:+.1f} km/s")
        print(f"  max |Δ shift|  = {max_delta_shift:.3f}  (soft<{delta_tol_soft}, hard<{delta_tol_hard})")
        print(f"  max |vp shift| = {max_vp_shift:.1f} km/s  (soft<{vp_tol_soft}, hard<{vp_tol_hard})")
        print(f"\n  Veredicto: {verdict}")

        print(f"\n  {'z_col':>6} | {'error':>8} | {'Δ':>7} | {'vp':>8} | {'Δ_shift':>8} | {'vp_shift':>9}")
        print(f"  {'-'*60}")
        for _, row in out["table"].iterrows():
            print(f"  {row['z_col_max']:>6.1f} | {row['error_scenario']:>8} | "
                  f"{row['delta']:>7.3f} | {row['vp']:>8.1f} | "
                  f"{row['delta_shift']:>+8.3f} | {row['vp_shift']:>+9.1f}")

    # guardar gate
    out["table"].to_csv(DATA_PATH + "HTANA_v41_stress_gate.csv", index=False)
    print(f"\n✅ Gate guardado: HTANA_v41_stress_gate.csv")

    return out


print("✅ HTANA v4.1 stress tools definidos.")

# =============================================================================
# USO RECOMENDADO — orden correcto:
#
# PASO 1 — audit pipeline (v4)
# report_v4 = run_v4_audit_pipeline(
#     z_arr, mu_arr, mu_err_arr, cos_th_arr,
#     label="Pantheon+", null_mocks=5, inj_seeds=2
# )
#
# PASO 2 — stress test corto (v4.1)
# stress_v41 = stress_test_zcol_and_errors_v41(
#     z_arr, mu_arr, mu_err_arr, cos_th_arr,
#     label="Pantheon+",
#     z_col_grid=(0.6, 1.0),
#     error_scenarios=[
#         {"name": "orig",    "inflation": None},
#         {"name": "quad10",  "inflation": {"mode": "quadrature",     "frac": 0.10}},
#         {"name": "mult10",  "inflation": {"mode": "multiplicative", "frac": 0.10}},
#     ],
#     run_linear_nuisance=True,
#     verbose=True
# )
#
# PASO 3 — gate automática
# gate_v41 = summarize_stress_gate_v41(
#     stress_v41,
#     ref_z_col_max=0.6,
#     ref_error_scenario="orig",
#     verbose=True
# )
# print(gate_v41["verdict"])
#
# Si PASS → siguiente paso es sky scan libre de dirección con v3.9.
# Si MIXED → narrativa conservadora, declarar en paper.
# Si FAIL → revisar setup antes de continuar.
# =============================================================================
