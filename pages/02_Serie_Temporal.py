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

# ordenar e criar eixo contínuo (preenche meses faltantes com NaN)
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

# Crescimento aproximado: média do início vs. fim (janelas iguais)
def cagr_simple(y_series: pd.Series) -> float | None:
    v = y_series.dropna().values
    if len(v) < 6:
        return np.nan
    w = max(1, len(v)//4)
    a = float(np.nanmean(v[:w]))
    b = float(np.nanmean(v[-w:]))
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

st.caption(
    "CV = desvio padrão / média. Crescimento (~%) compara médias do início e do fim da série para suavizar ruído."
)

# ---------------------------------------------------------------------
# 2) Gráfico da série
# ---------------------------------------------------------------------
st.subheader("Série mensal")
st.line_chart(df_full.set_index("ts")["y"], height=300, use_container_width=True)

# ---------------------------------------------------------------------
# 3) Distribuição (histograma interativo) e boxplot por mês (mesmo tamanho)
# ---------------------------------------------------------------------
import altair as alt
df_plot = df_full.copy()
df_plot["mes_lab"] = df_plot["p"].apply(period_to_label)

colL, colR = st.columns(2)

with colL:
    st.markdown("**Histograma da demanda (interativo)**")
    # histograma com binning automático do Altair + tooltip
    hist = (
        alt.Chart(df_plot.dropna(subset=["y"]))
        .mark_bar()
        .encode(
            x=alt.X("y:Q", bin=alt.Bin(maxbins=20), title="y"),
            y=alt.Y("count()", title="freq."),
            tooltip=[alt.Tooltip("count():Q", title="freq."), alt.Tooltip("y:Q", bin=True)],
        )
        .properties(height=260)
    )
    st.altair_chart(hist, use_container_width=True)

with colR:
    st.markdown("**Boxplot por mês (Jan–Dez)**")

    # extrair o número e o nome do mês
    df_plot["mes_num"] = df_plot["p"].dt.month
    df_plot["mes_lab"] = df_plot["mes_num"].map(_REV_PT)

    # gerar boxplot consolidado por mês (todos os anos)
    box = (
        alt.Chart(df_plot.dropna(subset=["y"]))
        .mark_boxplot(size=30)
        .encode(
            x=alt.X("mes_lab:N", title="Mês", sort=list(_REV_PT.values())),
            y=alt.Y("y:Q", title="Demanda"),
            tooltip=["mes_lab:N", "y:Q"]
        )
        .properties(height=300)
    )
    st.altair_chart(box, use_container_width=True)

# ---------------------------------------------------------------------
# 4) Tendência e Sazonalidade (periodicidade mensal = 12)
# ---------------------------------------------------------------------
st.subheader("Tendência e Sazonalidade")
try:
    from statsmodels.tsa.seasonal import seasonal_decompose
    y_clean = y.interpolate(limit_direction="both")
    # exige série regular; period=12 para mensal
    result = seasonal_decompose(y_clean, model="additive", period=12, extrapolate_trend="freq")
    # monta dataframes para plotar
    ts_index = df_full["ts"]
    trend_df = pd.DataFrame({"ts": ts_index, "Tendência": result.trend})
    seas_df  = pd.DataFrame({"ts": ts_index, "Sazonalidade": result.seasonal})

    cA, cB = st.columns(2)
    with cA:
        st.line_chart(trend_df.set_index("ts"), height=260, use_container_width=True)
    with cB:
        st.line_chart(seas_df.set_index("ts"), height=260, use_container_width=True)

except Exception as e:
    st.info("Para exibir tendência e sazonalidade, é necessário `statsmodels`. "
            "Se já estiver instalado, talvez a série esteja curta ou com poucos pontos válidos.")
    st.caption(f"Detalhe técnico: {e}")

# ---------------------------------------------------------------------
# 5) Qualidade dos dados (resumo textual + tabela de outliers)
# ---------------------------------------------------------------------
st.subheader("Qualidade dos Dados")

# Faltas
missing_df = df_full.loc[y.isna(), ["p"]].copy()
missing_df["Mês"] = missing_df["p"].apply(period_to_label)
missing_list = ", ".join(missing_df["Mês"].tolist())

# Outliers (IQR)
low_thr, high_thr = q1 - 1.5*(q3-q1), q3 + 1.5*(q3-q1)
outliers_df = df_full.loc[(y < low_thr) | (y > high_thr), ["p","y"]].copy()
outliers_df["Mês"] = outliers_df["p"].apply(period_to_label)
outliers_df.rename(columns={"y":"Quantidade"}, inplace=True)
outliers_df = outliers_df[["Mês","Quantidade"]]

st.markdown(f"**Dados faltando:** {len(missing_df)}"
            + (f" — meses: {missing_list}" if len(missing_df) else ""))

st.markdown(f"**Outliers (IQR):** **{len(outliers_df)}** ocorrência(s) "
            f"(limiares: baixo < {low_thr:.1f} | alto > {high_thr:.1f} — Q1={q1:.1f}, Q3={q3:.1f}, IQR={iqr:.1f}).")

# Tabela apenas para os outliers (se houver)
if len(outliers_df):
    st.dataframe(outliers_df, use_container_width=True, height=160)

# ---------------------------------------------------------------------
# 6) Navegação
# ---------------------------------------------------------------------
st.divider()
st.page_link("pages/03_Previsao.py", label="➡️ Seguir para **Previsão (Passo 3)**", icon="🔮")

