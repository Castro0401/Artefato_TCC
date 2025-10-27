# pages/03_Analise_Detalhada.py
from __future__ import annotations
import io
import numpy as np
import pandas as pd
import streamlit as st

st.title("🧪 Análise Detalhada — Diagnósticos essenciais")

# ---------- Guardas ----------
if "ts_df_norm" not in st.session_state:
    st.warning("Preciso da série do Passo 1 (Upload) antes de continuar.")
    st.page_link("pages/01_Upload.py", label="Ir para o Passo 1 — Upload", icon="📤")
    st.stop()

# ---------- Helpers de data / preenchimento ----------
_PT = {"Jan":1,"Fev":2,"Mar":3,"Abr":4,"Mai":5,"Jun":6,"Jul":7,"Ago":8,"Set":9,"Out":10,"Nov":11,"Dez":12}
_REV_PT = {v:k for k, v in _PT.items()}

def to_period(lbl: str) -> pd.Period:
    mon = lbl[:3].title()
    yy = 2000 + int(lbl[-2:])
    return pd.Period(freq="M", year=yy, month=_PT[mon])

def fill_missing_neighbors_with_linear_fallback(y_series: pd.Series) -> pd.Series:
    y0 = y_series.copy()
    prev = y0.shift(1); next_ = y0.shift(-1)
    neigh = (prev + next_) / 2.0
    mask = y0.isna() & prev.notna() & next_.notna()
    y_out = y0.copy(); y_out.loc[mask] = neigh.loc[mask]
    if y_out.isna().any():
        y_out = y_out.interpolate(limit_direction="both")
    return y_out

# ---------- Série base (mensal contínua) ----------
df = st.session_state["ts_df_norm"].copy()  # ['ds','y'] com labels 'Set/25'
df["p"] = df["ds"].apply(to_period)
df = df.sort_values("p").reset_index(drop=True)
full_idx = pd.period_range(df["p"].min(), df["p"].max(), freq="M")
df_full = pd.DataFrame({"p": full_idx}).merge(df[["p","y"]], on="p", how="left")
df_full["ts"] = df_full["p"].dt.to_timestamp()
y_raw = df_full["y"].astype(float)
y = fill_missing_neighbors_with_linear_fallback(y_raw)
y_pos = y.copy()

# ---------- Painel de configuração leve ----------
st.caption("Esta página aplica **diagnósticos clássicos** para orientar transformação da série, escolha de modelos e uso de Croston/SBA/TSB em demandas intermitentes.")
c1, c2 = st.columns(2)
with c1:
    period = st.number_input("Periodicidade (meses por ano) para STL", min_value=2, max_value=24, value=12, step=1)
with c2:
    apply_log_preview = st.checkbox("Pré-visualizar resultados também no log(y>0)", value=False)

# =========================================================================
# 1) ADI e CV² — Classificação do tipo de demanda (Croston/SBA/TSB)
# =========================================================================
# ADI = intervalo médio entre demandas positivas
nz = y[y > 0]
if len(nz) > 0:
    pos_idx = y.index[y > 0].to_list()
    gaps = np.diff(pos_idx) if len(pos_idx) > 1 else np.array([np.nan])
    ADI = float(np.nanmean(gaps)) if len(pos_idx) > 1 else float(np.inf)
    CV2 = float((nz.std(ddof=1) / nz.mean())**2) if nz.mean() else np.inf
else:
    ADI, CV2 = float(np.inf), float(np.inf)

# Regras Syntetos-Boylan (literatura): thresholds usuais
# Smooth: ADI < 1.32 & CV2 < 0.49
# Intermittent: ADI >= 1.32 & CV2 < 0.49
# Erratic: ADI < 1.32 & CV2 >= 0.49
# Lumpy: ADI >= 1.32 & CV2 >= 0.49
def classify_demand(ADI, CV2):
    if np.isinf(ADI) or np.isinf(CV2) or np.isnan(ADI) or np.isnan(CV2):
        return "Sem base (série sem vendas>0)"
    if ADI < 1.32 and CV2 < 0.49:
        return "Regular (smooth)"
    if ADI >= 1.32 and CV2 < 0.49:
        return "Intermitente"
    if ADI < 1.32 and CV2 >= 0.49:
        return "Errática"
    return "Lumpy (intermitente + errática)"

tipo_demanda = classify_demand(ADI, CV2)

st.subheader("1) ADI e CV² — Tipo de demanda")
cA, cB, cC = st.columns(3)
cA.metric("ADI (intervalo médio)", "∞" if np.isinf(ADI) else f"{ADI:.2f}")
cB.metric("CV² (tamanho da demanda)", "∞" if np.isinf(CV2) else f"{CV2:.2f}")
cC.metric("Classificação", tipo_demanda)
st.caption("→ **Croston/SBA/TSB** são recomendados para **Intermitente/Lumpy**; modelos clássicos podem servir melhor para **Regular**; **Errática** sugere cuidado (alta variância com pouca intermitência).")

# =========================================================================
# 2) Heterocedasticidade (variância crescente) — log/Box-Cox?
# =========================================================================
# Heurística: correlação entre nível e variação absoluta; e teste de tendência da variância móvel
dy = y.diff().abs()
lvl = y.shift(1)
corr_level_change = float(pd.concat([lvl, dy], axis=1).dropna().corr().iloc[0,1]) if len(y) > 3 else np.nan
roll_var = y.rolling(window=max(6, int(len(y)*0.2))).var()
trend_var = np.polyfit(np.arange(len(roll_var.dropna())), roll_var.dropna().values, 1)[0] if roll_var.dropna().shape[0] > 5 else np.nan
hetero_flag = (np.isfinite(corr_level_change) and corr_level_change > 0.3) or (np.isfinite(trend_var) and trend_var > 0)

st.subheader("2) Heterocedasticidade e variância crescente")
cH1, cH2, cH3 = st.columns(3)
cH1.metric("|Δy| ~ nível (corr)", f"{corr_level_change:.2f}" if corr_level_change==corr_level_change else "—")
cH2.metric("Tendência da var. móvel", f"{trend_var:.2e}" if trend_var==trend_var else "—")
cH3.metric("Sinal de heterocedasticidade?", "Sim" if hetero_flag else "Não")
st.caption("→ **Sinal positivo** sugere **log** ou **Box-Cox** para estabilizar variância.")

# =========================================================================
# 3) Assimetria e positividade
# =========================================================================
try:
    from scipy.stats import skew
    sk = float(skew(y.dropna().values)) if y.dropna().size > 2 else np.nan
except Exception:
    sk = float(pd.Series(y.dropna()).skew()) if y.dropna().size > 2 else np.nan

has_nonpositive = bool((y <= 0).any())

st.subheader("3) Assimetria e positividade")
cS1, cS2 = st.columns(2)
cS1.metric("Assimetria (skew)", f"{sk:.2f}" if sk==sk else "—")
cS2.metric("Há valores ≤ 0?", "Sim" if has_nonpositive else "Não")
st.caption("→ **Skew > 0** e dados **> 0** reforçam uso de **log**; se houver ≤0, prefira **Box-Cox** (com deslocamento interno).")

# =========================================================================
# 4) Decomposição STL — Força da tendência e da sazonalidade (Hyndman, 2018)
# =========================================================================
st.subheader("4) Decomposição STL e forças")

try:
    from statsmodels.tsa.seasonal import STL
    y_for_stl = y.dropna()
    stl = STL(y_for_stl, period=period, robust=True).fit()
    trend = pd.Series(stl.trend, index=y_for_stl.index)
    seas  = pd.Series(stl.seasonal, index=y_for_stl.index)
    rem   = pd.Series(stl.resid, index=y_for_stl.index)

    # forças (Hyndman): 1 - Var(resid)/Var(resid + componente)
    def variance(x): 
        return float(np.nanvar(x)) if len(pd.Series(x).dropna())>1 else np.nan
    F_trend = max(0.0, 1.0 - variance(rem)/variance(rem + trend)) if variance(rem + trend) else np.nan
    F_seas  = max(0.0, 1.0 - variance(rem)/variance(rem + seas))  if variance(rem + seas)  else np.nan

    cF1, cF2 = st.columns(2)
    cF1.metric("Força da tendência", f"{F_trend:.2f}" if F_trend==F_trend else "—")
    cF2.metric("Força da sazonalidade", f"{F_seas:.2f}" if F_seas==F_seas else "—")
    st.caption("→ **Valores próximos de 1** indicam componente forte; próximo de 0, componente fraco/ausente.")

except Exception as e:
    st.info("STL indisponível (statsmodels). Instale/atualize `statsmodels` para calcular forças de tendência/sazonalidade.")
    st.caption(f"Detalhe técnico: {e}")

# =========================================================================
# 5) ADF e KPSS — Estacionariedade
# =========================================================================
st.subheader("5) Testes ADF e KPSS (estacionariedade)")

adf_p = kpss_p = np.nan
try:
    from statsmodels.tsa.stattools import adfuller, kpss
    y_for_tests = y.dropna()
    if y_for_tests.size >= 8:
        adf_stat, adf_p, *_ = adfuller(y_for_tests.values, autolag="AIC")
        kpss_stat, kpss_p, *_ = kpss(y_for_tests.values, regression="c", nlags="auto")
except Exception as e:
    st.info("Para ADF/KPSS é necessário `statsmodels`."); st.caption(f"Detalhe técnico: {e}")

cT1, cT2 = st.columns(2)
cT1.metric("ADF p-valor (H0: não estacionária)", f"{adf_p:.4f}" if adf_p==adf_p else "—")
cT2.metric("KPSS p-valor (H0: estacionária)", f"{kpss_p:.4f}" if kpss_p==kpss_p else "—")
st.caption("→ **ADF p<0.05** sugere estacionariedade; **KPSS p<0.05** sugere não estacionariedade. Valores conflitantes apontam para **diferenciação (Δ)**.")

# =========================================================================
# 6) Box-Cox: λ (MLE) — validação da transformação
# =========================================================================
st.subheader("6) Box-Cox: λ (MLE)")

lambda_mle = np.nan
shift_applied = 0.0
try:
    from scipy.stats import boxcox_normmax, boxcox
    y_bc = y.dropna().astype(float)
    # Box-Cox requer valores > 0; aplica deslocamento mínimo, se necessário
    if (y_bc <= 0).any():
        shift_applied = float(1 - y_bc.min() + 1e-6)
        y_bc = y_bc + shift_applied
    lambda_mle = float(boxcox_normmax(y_bc.values, method="mle"))
except Exception as e:
    st.info("Para estimar λ via MLE é necessário `scipy`.")
    st.caption(f"Detalhe técnico: {e}")

cL1, cL2 = st.columns(2)
cL1.metric("λ (MLE)", f"{lambda_mle:.2f}" if lambda_mle==lambda_mle else "—")
cL2.metric("Deslocamento aplicado", f"{shift_applied:.2g}")
st.caption("→ **λ≈0** reforça uso de **log**; λ≈1 sugere não transformar; outros λ indicam **Box-Cox**.")

# =========================================================================
# Recomendações resumo (bullet points)
# =========================================================================
st.subheader("Recomendações")
recs = []

# 1) Intermitência
if "Intermitente" in tipo_demanda or "Lumpy" in tipo_demanda:
    recs.append("Aplicar **Croston/SBA/TSB** (demanda intermitente).")
elif "Errática" in tipo_demanda:
    recs.append("Demanda **errática**: considerar suavização robusta, outlier handling e modelos sem sazonalidade forte.")
else:
    recs.append("Demanda **regular**: modelos clássicos com ou sem sazonalidade podem funcionar bem.")

# 2) Heterocedasticidade
if hetero_flag:
    recs.append("Sinal de **heterocedasticidade** → usar **log** ou **Box-Cox** para estabilizar variância.")

# 3) Assimetria / positividade
if not has_nonpositive and (sk==sk and sk>0.5):
    recs.append("Distribuição **assimétrica à direita** e **positiva** → **log(y)** apropriado.")
elif has_nonpositive:
    recs.append("Há valores **≤ 0** → prefira **Box-Cox** (com deslocamento interno).")

# 4) Forças STL
try:
    if 'F_trend' in locals() and F_trend==F_trend and F_trend < 0.2:
        recs.append("**Tendência fraca** (STL) → evitar modelos com tendência rígida.")
    if 'F_seas' in locals() and F_seas==F_seas and F_seas < 0.2:
        recs.append("**Sazonalidade fraca** (STL) → considerar modelos **sem sazonalidade**.")
except Exception:
    pass

# 5) ADF/KPSS
if adf_p==adf_p and kpss_p==kpss_p:
    if adf_p >= 0.05 or kpss_p < 0.05:
        recs.append("Testes indicam **não estacionariedade** → usar **diferenciação (Δ)** antes de modelos ARIMA, se aplicável.")

# 6) Box-Cox
if lambda_mle==lambda_mle:
    if abs(lambda_mle) < 0.15:
        recs.append("**λ≈0** → **log(y)** é adequado.")
    elif abs(lambda_mle-1) < 0.15:
        recs.append("**λ≈1** → transformação pode não ser necessária.")
    else:
        recs.append(f"**λ≈{lambda_mle:.2f}** → usar **Box-Cox** com esse λ.")

if recs:
    st.markdown("\n".join([f"- {r}" for r in recs]))
else:
    st.markdown("- Sem recomendações automáticas neste momento (série possivelmente muito curta).")

# ---------- Navegação ----------
st.divider()
cL, cR = st.columns(2)
with cL:
    st.page_link("pages/02_Serie_Temporal.py", label="⬅️ Voltar para Análise Exploratória", icon="📈")
with cR:
    st.page_link("pages/04_Previsao.py", label="➡️ Seguir para Previsão", icon="🔮")
