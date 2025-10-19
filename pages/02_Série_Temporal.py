# pages/02_Serie_Temporal.py
from __future__ import annotations
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Série Temporal — Análise Exploratória", page_icon="📈", layout="wide")
st.title("📈 Série Temporal — Análise Exploratória")

# ---------------------------------------------------------------------
# 0) Carrega a série mensal do Passo 1 (ds,y) ou interrompe
# ---------------------------------------------------------------------
if "ts_df_norm" not in st.session_state:
    st.warning("Preciso da série do Passo 1 (Upload).")
    st.page_link("pages/01_Upload.py", label="Ir para o Passo 1 — Upload", icon="📤")
    st.stop()

df = st.session_state["ts_df_norm"].copy()  # colunas: ds (ex.: 'Set/25'), y (quantidade)

# ---------------------------------------------------------------------
# helpers de data
# ---------------------------------------------------------------------
_PT = {"Jan":1,"Fev":2,"Mar":3,"Abr":4,"Mai":5,"Jun":6,"Jul":7,"Ago":8,"Set":9,"Out":10,"Nov":11,"Dez":12}
_REV_PT = {v:k for k, v in _PT.items()}

def label_to_period(lbl: str) -> pd.Period:
    mon = lbl[:3].title()
    yy = 2000 + int(lbl[-2:])
    return pd.Period(freq="M", year=yy, month=_PT[mon])

# ordenar e criar eixo temporal contínuo
df["p"] = df["ds"].apply(label_to_period)
df = df.sort_values("p").reset_index(drop=True)
full_idx = pd.period_range(df["p"].min(), df["p"].max(), freq="M")
df_full = pd.DataFrame({"p": full_idx}).merge(df[["p","y"]], on="p", how="left")
df_full["ts"] = df_full["p"].dt.to_timestamp()
y = df_full["y"].astype(float)

# ---------------------------------------------------------------------
# 1) KPIs básicos
# ---------------------------------------------------------------------
n = len(df_full)
n_missing = int(y.isna().sum())
pct_missing = 100 * n_missing / n
n_zeros = int((y.fillna(0) == 0).sum())
mean = float(y.mean())
std = float(y.std())
cv = 100 * std / mean if mean else np.nan

def cagr_simple(y_series: pd.Series) -> float | None:
    v = y_series.dropna().values
    if len(v) < 6:
        return np.nan
    a = float(np.nanmean(v[:max(1, len(v)//4)]))
    b = float(np.nanmean(v[-max(1, len(v)//4):]))
    return (b/a - 1) * 100 if a else np.nan

cagr = cagr_simple(y)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Observações (meses)", f"{n}")
c2.metric("Faltas", f"{n_missing} ({pct_missing:.1f}%)")
c3.metric("Zeros", f"{n_zeros}")
c4.metric("CV (%)", f"{cv:.1f}" if np.isfinite(cv) else "—")
c5.metric("Crescimento (~%)", f"{cagr:.1f}" if cagr==cagr else "—")
st.caption("Obs.: CV = desvio padrão / média; crescimento aproximado usando médias do início e do fim da série.")

# ---------------------------------------------------------------------
# 2) Gráfico da série
# ---------------------------------------------------------------------
st.subheader("Série mensal")
st.line_chart(df_full.set_index("ts")["y"], height=300, use_container_width=True)

# ---------------------------------------------------------------------
# 3) Distribuição e boxplot por mês
# ---------------------------------------------------------------------
df_full["mes_num"] = df_full["p"].dt.month
df_full["mes_lab"]  = df_full["mes_num"].map(_REV_PT)

left, right = st.columns(2)
with left:
    st.markdown("**Histograma da demanda**")
    st.bar_chart(df_full["y"].dropna(), height=260, use_container_width=True)

with right:
    st.markdown("**Boxplot por mês**")
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        data_months = [df_full.loc[df_full["mes_num"]==m, "y"].dropna().values for m in range(1,13)]
        ax.boxplot(data_months, showfliers=True)
        ax.set_xticklabels([_REV_PT[m] for m in range(1,13)])
        ax.set_ylabel("y")
        st.pyplot(fig, clear_figure=True, use_container_width=True)
    except ModuleNotFoundError:
        st.warning("Para ver o boxplot por mês, instale `matplotlib`:\n\n`python -m pip install matplotlib`")

# ---------------------------------------------------------------------
# 4) Qualidade dos dados: faltas e outliers (IQR)
# ---------------------------------------------------------------------
q1, q3 = y.quantile(0.25), y.quantile(0.75)
iqr = q3 - q1
low_thr, high_thr = q1 - 1.5*iqr, q3 + 1.5*iqr
outliers = df_full.loc[(y < low_thr) | (y > high_thr), ["ts","y"]]

with st.expander("Qualidade dos dados"):
    st.write(f"**Limiares IQR**: baixo < {low_thr:.1f} | alto > {high_thr:.1f}")
    colA, colB = st.columns(2)
    with colA:
        st.write("**Meses faltantes**")
        st.dataframe(df_full.loc[y.isna(), ["ts"]].rename(columns={"ts":"mês"}), use_container_width=True, height=180)
    with colB:
        st.write("**Outliers (IQR)**")
        st.dataframe(outliers.rename(columns={"ts":"mês"}), use_container_width=True, height=180)

# ---------------------------------------------------------------------
# 5) Estacionariedade (ADF) e Correlogramas (ACF/PACF)
# ---------------------------------------------------------------------
st.subheader("Propriedades temporais")
try:
    from statsmodels.tsa.stattools import adfuller, acf, pacf
    y_clean = y.interpolate(limit_direction="both")  # evita NaN
    # ADF
    try:
        adf_stat, pval, *_ = adfuller(y_clean.values, autolag="AIC")
        st.metric("ADF p-valor", f"{pval:.4f}", help="p < 0.05 sugere estacionariedade (nível).")
    except Exception as e:
        st.warning(f"Não foi possível rodar o ADF: {e}")

    # Correlogramas
    max_lag = int(min(24, max(1, len(y_clean)-2)))
    acf_vals = acf(y_clean.values, nlags=max_lag, fft=True)
    try:
        pacf_vals = pacf(y_clean.values, nlags=max_lag, method="yw")
    except ValueError:
        pacf_vals = pacf(y_clean.values, nlags=max_lag, method="ywmle")

    try:
        import matplotlib.pyplot as plt
        # ACF
        fig1, ax1 = plt.subplots()
        ax1.stem(range(len(acf_vals)), acf_vals)  # sem use_line_collection (compat)
        ax1.set_title("ACF")
        ax1.set_xlabel("defasagem (meses)")
        ax1.set_ylabel("correlação")
        st.pyplot(fig1, clear_figure=True, use_container_width=True)

        # PACF
        fig2, ax2 = plt.subplots()
        ax2.stem(range(len(pacf_vals)), pacf_vals)
        ax2.set_title("PACF")
        ax2.set_xlabel("defasagem (meses)")
        ax2.set_ylabel("correlação parcial")
        st.pyplot(fig2, clear_figure=True, use_container_width=True)
    except ModuleNotFoundError:
        st.warning("Para visualizar ACF/PACF, instale `matplotlib`:\n\n`python -m pip install matplotlib`")

except ModuleNotFoundError:
    st.info("Para testes ADF/ACF/PACF instale `statsmodels` (e opcionalmente `matplotlib`):\n\n"
            "`python -m pip install statsmodels matplotlib`")

# ---------------------------------------------------------------------
# 6) Navegação
# ---------------------------------------------------------------------
st.divider()
st.page_link("pages/03_Previsão.py", label="➡️ Seguir para **Previsão (Passo 3)**", icon="🔮")
