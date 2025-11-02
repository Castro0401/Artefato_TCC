# pages/03_Analise_Detalhada.py
from __future__ import annotations
import io
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import streamlit as st

st.title("🧪 Análise Detalhada — Diagnósticos essenciais")

# =========================
# Guardas
# =========================
if "ts_df_norm" not in st.session_state:
    st.warning("Preciso da série do Passo 1 (Upload) antes de continuar.")
    st.page_link("pages/01_Upload.py", label="Ir para o Passo 1 — Upload", icon="📤")
    st.stop()

# =========================
# Importa util de estatísticas (opcional)
# =========================
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from core import Estatisticas as EST
except Exception:
    EST = None

# =========================
# CSS – compacta tudo e remove negrito
# =========================
st.markdown("""
<style>
/* títulos mais compactos */
h2, h3 { margin-top:.35rem !important; margin-bottom:.35rem !important; }

/* wrapper de elementos que contêm kpis: tira folga vertical grande do Streamlit */
div.element-container:has(.metric-kpi){ margin-bottom:.25rem !important; }

/* captions com menos respiro */
div[data-testid="stCaptionContainer"]{ margin-top:.20rem !important; margin-bottom:.35rem !important; }

/* parágrafos markdown */
.stMarkdown p { margin:.15rem 0 !important; }

/* cards KPI compactos e SEM negrito */
.metric-kpi{ display:flex; flex-direction:column; gap:.08rem; margin:.10rem 0 .20rem 0; }
.metric-kpi .lbl{ font-size:.95rem; color:var(--secondary-text,#6b7280); line-height:1.05; font-weight:400; margin:0; }
.metric-kpi .val{ font-size:1.9rem; font-weight:400; line-height:1.02; margin:0; }
</style>
""", unsafe_allow_html=True)

# =========================
# Helper KPI
# =========================
def kpi(col, label: str, value: str) -> None:
    col.markdown(
        f"""<div class="metric-kpi">
                <div class="lbl">{label}</div>
                <div class="val">{value}</div>
            </div>""",
        unsafe_allow_html=True
    )

# =========================
# Helpers de datas e imputação
# =========================
_PT = {"Jan":1,"Fev":2,"Mar":3,"Abr":4,"Mai":5,"Jun":6,"Jul":7,"Ago":8,"Set":9,"Out":10,"Nov":11,"Dez":12}
def to_period(lbl: str) -> pd.Period:
    mon = lbl[:3].title()
    yy = 2000 + int(lbl[-2:])
    return pd.Period(freq="M", year=yy, month=_PT[mon])

def fill_missing_neighbors_with_linear_fallback(y_series: pd.Series) -> pd.Series:
    y0 = y_series.copy()
    prev, nxt = y0.shift(1), y0.shift(-1)
    neigh = (prev + nxt) / 2.0
    mask = y0.isna() & prev.notna() & nxt.notna()
    y_out = y0.copy()
    y_out.loc[mask] = neigh.loc[mask]
    if y_out.isna().any():
        y_out = y_out.interpolate(limit_direction="both")
    return y_out

# =========================
# Série base (mensal contínua)
# =========================
df_up = st.session_state["ts_df_norm"].copy()  # ['ds','y'] com labels 'Set/25'
df_up["p"] = df_up["ds"].apply(to_period)
df_up = df_up.sort_values("p").reset_index(drop=True)
full_idx = pd.period_range(df_up["p"].min(), df_up["p"].max(), freq="M")
df_full = pd.DataFrame({"p": full_idx}).merge(df_up[["p","y"]], on="p", how="left")
df_full["ts"] = df_full["p"].dt.to_timestamp()

y_raw = df_full["y"].astype(float)
y = fill_missing_neighbors_with_linear_fallback(y_raw)

st.caption(
    "Esta página aplica **diagnósticos clássicos** (e versões do seu módulo auxiliar) "
    "para orientar transformações, modelo e políticas de estoque/demanda intermitente."
)

# =========================
# Painel de configuração
# =========================
c1, c2 = st.columns(2, gap="small")
with c1:
    stl_period = st.number_input("Periodicidade para STL (ex.: 12 para mensal)", 2, 24, 12, 1)
with c2:
    nlags = st.number_input("Lags para ACF/PACF", 8, 48, 24, 1)

# =============================================================================
# 1) ADI & CV²
# =============================================================================
def _fallback_calcular_adi_cv2(y_series: pd.Series) -> dict:
    v = y_series.fillna(0.0).astype(float)
    N = int(v.size)
    positivos = v[v > 0]
    Nz = int(positivos.size)
    frac_zeros = (N - Nz) / N if N else np.nan
    if Nz == 0:
        adi, cv2 = float("inf"), float("inf")
    elif Nz == 1:
        adi, cv2 = N / Nz, 0.0
    else:
        adi = N / Nz
        m = float(positivos.mean()); s = float(positivos.std(ddof=1))
        cv2 = (s / m) ** 2 if m > 0 else np.nan
    return dict(N=N, Nz=Nz, FracZeros=frac_zeros, ADI=adi, CV2=cv2, CV=np.sqrt(cv2) if pd.notna(cv2) else np.nan)

def _fallback_classificar(adi: float, cv2: float) -> str:
    LIM_ADI, LIM_CV2 = 1.32, 0.49
    if np.isinf(adi): return "Sem Demanda"
    if (adi < LIM_ADI) and (cv2 < LIM_CV2): return "Regular"
    if (adi >= LIM_ADI) and (cv2 < LIM_CV2): return "Intermittent"
    if (adi < LIM_ADI) and (cv2 >= LIM_CV2): return "Erratic"
    return "Lumpy"

met = EST.calcular_adi_cv2(y) if EST and hasattr(EST, "calcular_adi_cv2") else _fallback_calcular_adi_cv2(y)
tipo_demanda = EST.classificar_demanda(met["ADI"], met["CV2"]) if EST and hasattr(EST, "classificar_demanda") else _fallback_classificar(met["ADI"], met["CV2"])

st.subheader("1) ADI e CV² — Tipo de demanda")
cA, cB, cC = st.columns(3, gap="small")
kpi(cA, "ADI (intervalo médio)", "∞" if np.isinf(met["ADI"]) else f"{met['ADI']:.2f}")
kpi(cB, "CV² (positivos)", "n/d" if not pd.notna(met["CV2"]) else f"{met['CV2']:.2f}")
kpi(cC, "Classificação", tipo_demanda)
st.caption("→ **Croston/SBA/TSB** para **Intermittent/Lumpy**; **Regular** tende a funcionar com modelos clássicos; **Erratic** requer cautela.")

# =============================================================================
# 2) Heterocedasticidade
# =============================================================================
def _fallback_hetero(y_series: pd.Series):
    dy = y_series.diff().abs()
    lvl = y_series.shift(1)
    corr_level_change = float(pd.concat([lvl, dy], axis=1).dropna().corr().iloc[0,1]) if len(y_series) > 3 else np.nan
    roll_var = y_series.rolling(window=max(6, int(len(y_series)*0.2))).var()
    tmp = roll_var.dropna()
    trend_var = np.polyfit(np.arange(len(tmp)), tmp.values, 1)[0] if tmp.shape[0] > 5 else np.nan
    hetero_flag = (np.isfinite(corr_level_change) and corr_level_change > 0.3) or (np.isfinite(trend_var) and trend_var > 0)
    return dict(corr_level_change=corr_level_change, trend_var=trend_var, hetero_flag=hetero_flag)

het = EST.heterocedasticidade_indicadores(y) if EST and hasattr(EST, "heterocedasticidade_indicadores") else _fallback_hetero(y)

st.subheader("2) Heterocedasticidade e variância crescente")
cH1, cH2, cH3 = st.columns(3, gap="small")
kpi(cH1, "|Δy| ~ nível (corr)", "—" if not pd.notna(het["corr_level_change"]) else f"{het['corr_level_change']:.2f}")
kpi(cH2, "Tendência da var. móvel", "—" if not pd.notna(het["trend_var"]) else f"{het['trend_var']:.2e}")
kpi(cH3, "Sinal de heterocedasticidade?", "Sim" if het["hetero_flag"] else "Não")
st.caption("→ **Sinal positivo** sugere **log** ou **Box-Cox** para estabilizar variância.")

# =============================================================================
# 3) Assimetria / positividade
# =============================================================================
try:
    from scipy.stats import skew
    sk = float(skew(y.dropna().values)) if y.dropna().size > 2 else np.nan
except Exception:
    sk = float(pd.Series(y.dropna()).skew()) if y.dropna().size > 2 else np.nan

has_nonpositive = bool((y <= 0).any())
st.subheader("3) Assimetria e positividade")
cS1, cS2 = st.columns(2, gap="small")
kpi(cS1, "Assimetria (skew)", "—" if not pd.notna(sk) else f"{sk:.2f}")
kpi(cS2, "Há valores ≤ 0?", "Sim" if has_nonpositive else "Não")
st.caption("→ **Skew > 0** e dados **> 0** reforçam uso de **log**; com ≤0 prefira **Box-Cox** (com deslocamento).")

# =============================================================================
# 4) Decomposição STL — Forças
# =============================================================================
def _fallback_stl_strengths(y_series: pd.Series, period: int):
    try:
        from statsmodels.tsa.seasonal import STL
        y_for_stl = y_series.dropna()
        stl = STL(y_for_stl, period=period, robust=True).fit()
        trend, seas, rem = map(pd.Series, (stl.trend, stl.seasonal, stl.resid))
        def var(x):
            x = pd.Series(x).dropna().values
            return float(np.nanvar(x)) if x.size > 1 else np.nan
        F_trend = max(0.0, 1.0 - var(rem)/var(rem + trend)) if var(rem + trend) else np.nan
        F_seas  = max(0.0, 1.0 - var(rem)/var(rem + seas))  if var(rem + seas)  else np.nan
        return dict(F_trend=F_trend, F_seas=F_seas)
    except Exception as e:
        return dict(F_trend=np.nan, F_seas=np.nan, _err=str(e))

stl_res = EST.stl_strengths(y, stl_period) if EST and hasattr(EST, "stl_strengths") else _fallback_stl_strengths(y, stl_period)

st.subheader("4) Decomposição STL — forças (Hyndman)")
cF1, cF2 = st.columns(2, gap="small")
kpi(cF1, "Força da tendência", "—" if not pd.notna(stl_res.get("F_trend", np.nan)) else f"{stl_res['F_trend']:.2f}")
kpi(cF2, "Força da sazonalidade", "—" if not pd.notna(stl_res.get("F_seas", np.nan)) else f"{stl_res['F_seas']:.2f}")
st.caption("→ **1** = componente forte; **0** = fraca/ausente.")

# =============================================================================
# 5) ADF / KPSS
# =============================================================================
def _fallback_stationarity(y_series: pd.Series):
    out = dict(adf_p=np.nan, kpss_p=np.nan, _err=None)
    try:
        from statsmodels.tsa.stattools import adfuller, kpss
        yy = y_series.dropna()
        if yy.size >= 8:
            out["adf_p"] = float(adfuller(yy.values, autolag="AIC")[1])
            out["kpss_p"] = float(kpss(yy.values, regression="c", nlags="auto")[1])
    except Exception as e:
        out["_err"] = str(e)
    return out

stn = EST.stationarity_tests(y) if EST and hasattr(EST, "stationarity_tests") else _fallback_stationarity(y)

st.subheader("5) Testes ADF e KPSS (estacionariedade)")
cT1, cT2 = st.columns(2, gap="small")
kpi(cT1, "ADF p-valor (H0: não estacionária)", "—" if not pd.notna(stn["adf_p"]) else f"{stn['adf_p']:.4f}")
kpi(cT2, "KPSS p-valor (H1: estacionária)", "—" if not pd.notna(stn["kpss_p"]) else f"{stn['kpss_p']:.4f}")
st.caption("→ **ADF p<0.05** sugere estacionariedade; **KPSS p<0.05** sugere não estacionariedade.")

# =============================================================================
# 6) Box-Cox λ
# =============================================================================
def _fallback_boxcox_lambda(y_series: pd.Series):
    try:
        from scipy.stats import boxcox_normmax
        v = y_series.dropna().astype(float)
        shift = 0.0
        if (v <= 0).any():
            shift = float(1 - v.min() + 1e-6)
            v = v + shift
        lam = float(boxcox_normmax(v.values, method="mle"))
        return dict(lmbda=lam, shift=shift)
    except Exception as e:
        return dict(lmbda=np.nan, shift=0.0, _err=str(e))

bc = EST.boxcox_lambda_mle(y) if EST and hasattr(EST, "boxcox_lambda_mle") else _fallback_boxcox_lambda(y)

st.subheader("6) Box-Cox — λ (MLE)")
cL1, cL2 = st.columns(2, gap="small")
kpi(cL1, "λ (MLE)", "—" if not pd.notna(bc.get("lmbda", np.nan)) else f"{bc['lmbda']:.2f}")
kpi(cL2, "Deslocamento aplicado", f"{bc.get('shift', 0.0):.2g}")
st.caption("→ **λ≈0** reforça **log(y)**; **λ≈1** sugere manter escala; outros λ indicam **Box-Cox**.")

# =============================================================================
# 7) ACF / PACF
# =============================================================================
st.subheader("7) Dependência serial — FAC (ACF) e FACP (PACF)")
try:
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    import matplotlib.pyplot as plt

    yy = y.dropna()
    if yy.size >= 10:
        cG1, cG2 = st.columns(2, gap="small")
        with cG1:
            fig1, ax1 = plt.subplots(figsize=(5.2, 3.2))
            plot_acf(yy.values, lags=min(int(nlags), len(yy)-1), ax=ax1)
            buf1 = io.BytesIO(); fig1.savefig(buf1, format="png", bbox_inches="tight", dpi=150)
            st.image(buf1.getvalue(), caption="FAC (ACF)", use_container_width=True)
            plt.close(fig1)
        with cG2:
            fig2, ax2 = plt.subplots(figsize=(5.2, 3.2))
            plot_pacf(yy.values, lags=min(int(nlags), len(yy)//2), ax=ax2, method="ywm")
            buf2 = io.BytesIO(); fig2.savefig(buf2, format="png", bbox_inches="tight", dpi=150)
            st.image(buf2.getvalue(), caption="FACP (PACF)", use_container_width=True)
            plt.close(fig2)
    else:
        st.info("Série muito curta para FAC/FACP (precisa de ≳10 observações).")
except ModuleNotFoundError:
    st.error("Para plotar **FAC/FACP** é necessário `statsmodels` e `matplotlib`.")
except Exception as e:
    st.info(f"Não foi possível gerar ACF/PACF: {e}")

# =========================
# Recomendações
# =========================
st.subheader("Recomendações (automáticas)")
recs = []

# tipo de demanda
if tipo_demanda in {"Intermittent", "Lumpy"}:
    recs.append("Aplicar **Croston/SBA/TSB** (demanda intermitente).")
elif tipo_demanda == "Erratic":
    recs.append("Demanda **errática**: suavização robusta / outlier handling e modelos sem sazonalidade rígida.")
else:
    recs.append("Demanda **regular**: modelos clássicos (com/sem sazonalidade) tendem a funcionar.")

# hetero
if het["hetero_flag"]:
    recs.append("Sinais de **heterocedasticidade** → considerar **log** ou **Box-Cox**.")

# assimetria / positividade
if not has_nonpositive and (pd.notna(sk) and sk > 0.5):
    recs.append("Distribuição **positiva** e **assimétrica** → **log(y)** é apropriado.")
elif has_nonpositive:
    recs.append("Há valores **≤ 0** → usar **Box-Cox** com deslocamento.")

# forças STL
Ft, Fs = stl_res.get("F_trend", np.nan), stl_res.get("F_seas", np.nan)
if pd.notna(Ft) and Ft < 0.2: recs.append("**Tendência fraca** (STL) → evitar modelos com tendência rígida.")
if pd.notna(Fs) and Fs < 0.2: recs.append("**Sazonalidade fraca** (STL) → considerar modelos **sem sazonalidade**.")

# estacionariedade
adf_p, kpss_p = stn.get("adf_p", np.nan), stn.get("kpss_p", np.nan)
if pd.notna(adf_p) and pd.notna(kpss_p):
    if adf_p >= 0.05 or kpss_p < 0.05:
        recs.append("Evidências de **não estacionariedade** → considerar **diferenciação (Δ)** antes de ARIMA.")

# Box-Cox
lam = bc.get("lmbda", np.nan)
if pd.notna(lam):
    if abs(lam) < 0.15:
        recs.append("**λ≈0** → **log(y)** recomendado.")
    elif abs(lam-1) < 0.15:
        recs.append("**λ≈1** → transformação pode ser dispensada.")
    else:
        recs.append(f"**λ≈{lam:.2f}** → usar **Box-Cox** com esse λ.")

st.markdown("\n".join(f"- {r}" for r in recs) if recs else "- Sem recomendações automáticas (série possivelmente muito curta).")

# =========================
# Navegação
# =========================
st.divider()
cL, cR = st.columns(2, gap="large")
with cL:
    st.page_link("pages/02_Serie_Temporal.py", label="⬅️ Voltar — Série Temporal", icon="📈")
with cR:
    st.page_link("pages/04_Previsao.py", label="➡️ Seguir — Previsão", icon="🔮")
