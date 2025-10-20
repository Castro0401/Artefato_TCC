# pages/02_Serie_Temporal.py
from __future__ import annotations
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Série Temporal — Análise Exploratória", page_icon="📈", layout="wide")
st.title("📈 Série Temporal — Análise Exploratória")

# ---------------------------------------------------------------------
# 0) Entrada (ds, y) do Passo 1
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

def period_to_label(p: pd.Period) -> str:
    return f"{_REV_PT[p.month]}/{str(p.year)[-2:]}"

# ordenar e criar eixo temporal contínuo
df["p"] = df["ds"].apply(label_to_period)
df = df.sort_values("p").reset_index(drop=True)
full_idx = pd.period_range(df["p"].min(), df["p"].max(), freq="M")
df_full = pd.DataFrame({"p": full_idx}).merge(df[["p","y"]], on="p", how="left")
df_full["ts"] = df_full["p"].dt.to_timestamp()
y = df_full["y"].astype(float)

# ---------------------------------------------------------------------
# 1) Indicadores descritivos
# ---------------------------------------------------------------------
n = len(df_full)
n_missing = int(y.isna().sum())
pct_missing = 100 * n_missing / n
n_zeros = int((y.fillna(0) == 0).sum())

mean = float(y.mean())
median = float(y.median())
std = float(y.std())
min_ = float(y.min()) if y.size else np.nan
max_ = float(y.max()) if y.size else np.nan
cv = 100 * std / mean if mean else np.nan

q1, q3 = y.quantile(0.25), y.quantile(0.75)
iqr = float(q3 - q1)
# crescimento aproximado (média do início vs fim da série)
def cagr_simple(y_series: pd.Series) -> float | None:
    v = y_series.dropna().values
    if len(v) < 6:
        return np.nan
    a = float(np.nanmean(v[:max(1, len(v)//4)]))
    b = float(np.nanmean(v[-max(1, len(v)//4):]))
    return (b/a - 1) * 100 if a else np.nan
cagr = cagr_simple(y)

st.subheader("Análise descritiva")
k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Média", f"{mean:.1f}" if mean==mean else "—")
k2.metric("Mediana", f"{median:.1f}" if median==median else "—")
k3.metric("Desv. Padrão", f"{std:.1f}" if std==std else "—")
k4.metric("Mín / Máx", f"{min_:.0f} / {max_:.0f}" if min_==min_ and max_==max_ else "—")
k5.metric("CV (%)", f"{cv:.1f}" if np.isfinite(cv) else "—")
k6.metric("Crescimento (~%)", f"{cagr:.1f}" if cagr==cagr else "—")

k7, k8, k9 = st.columns(3)
k7.metric("Observações (meses)", f"{n}")
k8.metric("Faltas", f"{n_missing} ({pct_missing:.1f}%)")
k9.metric("Zeros", f"{n_zeros}")

st.caption("Notas: CV = desvio padrão / média. Crescimento aproximado compara início vs. fim da série.")

# ---------------------------------------------------------------------
# 2) Gráfico da série
# ---------------------------------------------------------------------
st.subheader("Série mensal")
st.line_chart(df_full.set_index("ts")["y"], height=300, use_container_width=True)

# ---------------------------------------------------------------------
# 3) Distribuição e boxplot por mês (mesmo tamanho)
# ---------------------------------------------------------------------
df_full["mes_num"] = df_full["p"].dt.month
df_full["mes_lab"]  = df_full["mes_num"].map(_REV_PT)

colL, colR = st.columns(2)
with colL:
    st.markdown("**Histograma da demanda**")
    import matplotlib.pyplot as plt
    fig_h, ax_h = plt.subplots(figsize=(6, 3))  # mesma “altura” do boxplot
    ax_h.hist(df_full["y"].dropna().values, bins="auto")
    ax_h.set_xlabel("y"); ax_h.set_ylabel("freq.")
    st.pyplot(fig_h, clear_figure=True, use_container_width=True)

with colR:
    st.markdown("**Boxplot por mês**")
    try:
        fig_b, ax_b = plt.subplots(figsize=(6, 3))  # mesmo tamanho do histograma
        data_months = [df_full.loc[df_full["mes_num"]==m, "y"].dropna().values for m in range(1,13)]
        ax_b.boxplot(data_months, showfliers=True)
        ax_b.set_xticklabels([_REV_PT[m] for m in range(1,13)])
        ax_b.set_ylabel("y")
        st.pyplot(fig_b, clear_figure=True, use_container_width=True)
    except ModuleNotFoundError:
        st.warning("Para ver o boxplot por mês, instale `matplotlib`:\n\n`python -m pip install matplotlib`")

# ---------------------------------------------------------------------
# 4) Qualidade dos dados (aberta, em linhas) + outliers (IQR)
# ---------------------------------------------------------------------
st.subheader("Qualidade dos dados")
low_thr, high_thr = q1 - 1.5*(q3-q1), q3 + 1.5*(q3-q1)

# Meses faltantes (NaN)
missing_df = df_full.loc[y.isna(), ["p"]].copy()
missing_df["Mês"] = missing_df["p"].apply(period_to_label)
missing_df = missing_df[["Mês"]]

# Outliers (IQR)
outliers_df = df_full.loc[(y < low_thr) | (y > high_thr), ["p","y"]].copy()
outliers_df["Mês"] = outliers_df["p"].apply(period_to_label)
outliers_df.rename(columns={"y":"Quantidade"}, inplace=True)
outliers_df = outliers_df[["Mês","Quantidade"]]

row1, row2 = st.columns(2)
with row1:
    st.write(f"**Faltas:** {len(missing_df)} mês(es).")
    if len(missing_df):
        st.dataframe(missing_df, use_container_width=True, height=150)
    else:
        st.success("Sem meses faltantes.")

with row2:
    st.write(f"**Outliers (IQR):** {len(outliers_df)} ocorrência(s).")
    if len(outliers_df):
        st.dataframe(outliers_df, use_container_width=True, height=150)
    else:
        st.success("Sem outliers pelo critério IQR.")

st.caption(f"Limiares IQR: baixo < {low_thr:.1f} | alto > {high_thr:.1f} (Q1={q1:.1f}, Q3={q3:.1f}, IQR={iqr:.1f}).")

# ---------------------------------------------------------------------
# 5) Estacionariedade (ADF) — opcional (sem ACF/PACF)
# ---------------------------------------------------------------------
try:
    from statsmodels.tsa.stattools import adfuller
    y_clean = y.interpolate(limit_direction="both")
    try:
        _, pval, *_ = adfuller(y_clean.values, autolag="AIC")
        st.metric("ADF (p-valor)", f"{pval:.4f}", help="p < 0.05 sugere estacionariedade (nível).")
    except Exception as e:
        st.warning(f"Não foi possível rodar o ADF: {e}")
except ModuleNotFoundError:
    pass  # não exibe nada se não houver statsmodels

# ---------------------------------------------------------------------
# 6) Navegação
# ---------------------------------------------------------------------
st.divider()
st.page_link("pages/03_Previsao.py", label="➡️ Seguir para **Previsão (Passo 3)**", icon="🔮")
