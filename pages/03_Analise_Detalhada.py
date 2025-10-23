# pages/03_Analise_Detalhada.py
from __future__ import annotations
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Análise Detalhada — Testes de Série Temporal", page_icon="🧪", layout="wide")
st.title("🧪 Análise Detalhada — Testes e Diagnósticos")

# --- guarda: precisa do upload ---
if "ts_df_norm" not in st.session_state:
    st.warning("Preciso da série do Passo 1 (Upload) antes de continuar.")
    st.page_link("pages/01_Upload.py", label="Ir para o Passo 1 — Upload", icon="📤")
    st.stop()

# ---------------- helpers de data / preenchimento ----------------
_PT = {"Jan":1,"Fev":2,"Mar":3,"Abr":4,"Mai":5,"Jun":6,"Jul":7,"Ago":8,"Set":9,"Out":10,"Nov":11,"Dez":12}
_REV_PT = {v:k for k, v in _PT.items()}
def to_period(lbl: str) -> pd.Period:
    mon = lbl[:3].title()
    yy = 2000 + int(lbl[-2:])
    return pd.Period(freq="M", year=yy, month=_PT[mon])

def fill_missing_neighbors_with_linear_fallback(y_series: pd.Series) -> pd.Series:
    y0 = y_series.copy()
    prev = y0.shift(1)
    next_ = y0.shift(-1)
    neigh_mean = (prev + next_) / 2.0
    mask = y0.isna() & prev.notna() & next_.notna()
    y_out = y0.copy()
    y_out.loc[mask] = neigh_mean.loc[mask]
    if y_out.isna().any():
        y_out = y_out.interpolate(limit_direction="both")
    return y_out

# ---------------- prepara série ----------------
df = st.session_state["ts_df_norm"].copy()  # ['ds','y'] com labels 'Set/25'
df["p"] = df["ds"].apply(to_period)
df = df.sort_values("p").reset_index(drop=True)
full_idx = pd.period_range(df["p"].min(), df["p"].max(), freq="M")
df_full = pd.DataFrame({"p": full_idx}).merge(df[["p","y"]], on="p", how="left")
df_full["ts"] = df_full["p"].dt.to_timestamp()
y_raw = df_full["y"].astype(float)
y = fill_missing_neighbors_with_linear_fallback(y_raw)

st.caption("Esta página é **técnica**: aplica testes de estacionariedade e autocorrelação para diagnóstico da série.")

# ---------------- opções rápidas ----------------
st.subheader("Configuração da análise")
c1, c2 = st.columns(2)
with c1:
    do_diff = st.checkbox("Aplicar 1a diferença (Δy)", value=False,
                          help="Útil quando há tendência. Aplica y[t]-y[t-1] antes dos testes/ACF/PACF.")
with c2:
    nlags = st.number_input("Lags para ACF/PACF e Ljung–Box", min_value=8, max_value=48, value=24, step=1)

if do_diff:
    y_work = y.diff().dropna()
else:
    y_work = y.dropna()

# ---------------- testes (statsmodels) ----------------
try:
    from statsmodels.tsa.stattools import adfuller, kpss
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    import matplotlib.pyplot as plt
    import io

    st.subheader("Testes de estacionariedade")
    colA, colB = st.columns(2)

    # ADF
    with colA:
        try:
            adf_res = adfuller(y_work.values, autolag="AIC")
            adf_stat, adf_p = adf_res[0], adf_res[1]
            st.metric("ADF p-valor", f"{adf_p:.4f}")
            st.write(f"Estatística: **{adf_stat:.3f}** | n={len(y_work)}")
            st.caption("H0 (ADF): a série possui raiz unitária (não estacionária). p<0.05 sugere estacionariedade.")
        except Exception as e:
            st.error(f"Falha no ADF: {e}")

    # KPSS
    with colB:
        try:
            kpss_stat, kpss_p, _, _ = kpss(y_work.values, regression="c", nlags="auto")
            st.metric("KPSS p-valor", f"{kpss_p:.4f}")
            st.write(f"Estatística: **{kpss_stat:.3f}** | n={len(y_work)}")
            st.caption("H0 (KPSS): a série é estacionária em torno de uma constante. p<0.05 sugere não estacionariedade.")
        except Exception as e:
            st.error(f"Falha no KPSS: {e}")

    # Ljung–Box
    st.subheader("Autocorrelação de resíduos (Ljung–Box) — aplicado à própria série")
    try:
        lb = acorr_ljungbox(y_work.values, lags=[min(int(nlags), len(y_work)//2)], return_df=True)
        lb_p = float(lb["lb_pvalue"].iloc[-1])
        st.metric(f"Ljung–Box p-valor (lag={min(int(nlags), len(y_work)//2)})", f"{lb_p:.4f}")
        st.caption("H0: ausência de autocorrelação até o lag testado. p<0.05 indica autocorrelação significativa.")
    except Exception as e:
        st.error(f"Falha no Ljung–Box: {e}")

    # ACF/PACF (matplotlib -> imagem)
    st.subheader("ACF e PACF")
    c3, c4 = st.columns(2)
    with c3:
        fig1, ax1 = plt.subplots()
        plot_acf(y_work.values, lags=min(int(nlags), len(y_work)-1), ax=ax1)
        buf1 = io.BytesIO()
        fig1.savefig(buf1, format="png", bbox_inches="tight")
        st.image(buf1.getvalue(), caption="ACF", use_container_width=True)
        plt.close(fig1)

    with c4:
        fig2, ax2 = plt.subplots()
        plot_pacf(y_work.values, lags=min(int(nlags), len(y_work)//2), ax=ax2, method="ywm")
        buf2 = io.BytesIO()
        fig2.savefig(buf2, format="png", bbox_inches="tight")
        st.image(buf2.getvalue(), caption="PACF", use_container_width=True)
        plt.close(fig2)

except ModuleNotFoundError:
    st.error(
        "Pacote **statsmodels** não encontrado. "
        "Instale com `pip install statsmodels` para habilitar ADF, KPSS, Ljung–Box, ACF/PACF."
    )

# ---------------- interpretação rápida ----------------
with st.expander("Como interpretar rapidamente os p-values (resumo)", expanded=False):
    st.markdown("""
- **ADF**: H0 = *não estacionária*. p < 0,05 → **estacionária**.
- **KPSS**: H0 = *estacionária em torno de constante*. p < 0,05 → **não estacionária**.
- **Ljung–Box**: H0 = *sem autocorrelação*. p < 0,05 → **há autocorrelação** até o lag testado.
- Use **Δy** (1ª diferença) se ADF alto / KPSS baixo sugerirem não estacionariedade por tendência.
""")

# ---------------- navegação ----------------
st.divider()
cL, cR = st.columns(2)
with cL:
    st.page_link("pages/02_Serie_Temporal.py", label="⬅️ Voltar para Análise Exploratória", icon="📈")
with cR:
    st.page_link("pages/04_Previsao.py", label="➡️ Seguir para Previsão", icon="🔮")
