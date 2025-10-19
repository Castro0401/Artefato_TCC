# pages/02_🔮_Previsão.py
from __future__ import annotations
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Previsão & PCP — Previsão (h=6)", page_icon="🔮", layout="wide")
st.title("🔮 Passo 2 — Previsão de Demanda (6 meses)")

# ---------- helpers ----------
_PT = {1:"Jan",2:"Fev",3:"Mar",4:"Abr",5:"Mai",6:"Jun",7:"Jul",8:"Ago",9:"Set",10:"Out",11:"Nov",12:"Dez"}
def label_pt(ts: pd.Timestamp) -> str:
    return f"{_PT[ts.month]}/{str(ts.year)[-2:]}"

def to_period(s: str) -> pd.Period:
    # converte "Set/25" -> Period('2025-09','M'); "2024-01-01" também funciona
    try:
        return pd.to_datetime(s, dayfirst=True).to_period("M")
    except Exception:
        # tenta formato "Set/25"
        mon = s[:3].capitalize()
        yy = int(s[-2:]) + 2000
        rev = {v:k for k, v in _PT.items()}
        month_num = rev.get(mon, None)
        if month_num is None:
            raise ValueError(f"Formato de mês inválido: {s}")
        return pd.Period(freq="M", year=yy, month=month_num)

def next_6_months(last_period: pd.Period) -> list[str]:
    out = []
    p = last_period + 1
    for _ in range(6):
        ts = p.to_timestamp()
        out.append(label_pt(ts))
        p += 1
    return out

# ---------- entrada: série mensal do Passo 1 ----------
if "ts_df_norm" not in st.session_state:
    st.warning("Preciso da série mensal do Passo 1 (Upload).")
    st.page_link("pages/01_📤_Upload.py", label="Ir para o Passo 1 — Upload", icon="📤")
    st.stop()

hist = st.session_state["ts_df_norm"].copy()      # ['ds','y'] com labels tipo 'Set/25'

# ---------- “modelo” (simulado) ----------
with st.expander("Configuração do modelo (simulado por enquanto)", expanded=True):
    model_choice = st.selectbox("Modelo candidato", ["AutoARIMA", "ETS (Holt-Winters)", "Prophet", "XGBoost"], index=0)
    horizon = st.number_input("Horizonte (meses)", min_value=1, max_value=12, value=6, step=1, help="Usaremos 6 meses por padrão.")
    seed = st.number_input("Semente aleatória (reprodutibilidade)", min_value=0, value=42, step=1)

# ---------- previsão (SIMULAÇÃO enquanto o modelo real não chega) ----------
np.random.seed(int(seed))

# Para simular um “ajuste”, fazemos uma média móvel + ruído
hist_work = hist.copy()
# converte labels para Period ordenado
hist_work["p"] = hist_work["ds"].apply(to_period)
hist_work = hist_work.sort_values("p").reset_index(drop=True)

y = hist_work["y"].astype(float).values
y_ma = pd.Series(y).rolling(3, min_periods=1).mean().values
sigma = max(np.std(y - y_ma), 1.0)

last_p = hist_work["p"].iloc[-1]
future_labels = next_6_months(last_p)[:horizon]

# simula tendência leve + ruído proporcional
trend = (y_ma[-1] - y_ma[max(len(y_ma)-4,0)]) / max(3, len(y_ma)-1)
sim_vals = []
base = y_ma[-1]
for i in range(horizon):
    base = base + trend  # tendência linear simples
    sim_vals.append(max(0, base + np.random.normal(0, 0.6*sigma)))

forecast_df_6m = pd.DataFrame({"ds": future_labels, "y": np.round(sim_vals).astype(int)})

# ---------- métricas fake (meramente ilustrativas) ----------
mape = np.clip(np.random.normal(8, 2), 4, 15)    # %
rmse = max(1.0, np.random.normal(25, 8))

# guarda no estado (para o MPS usar depois)
st.session_state["forecast_df_6m"] = forecast_df_6m

# ---------- visualizações ----------
left, right = st.columns([2,1])

with left:
    st.subheader("Histórico + Previsão (6 meses)")
    # prepara série contínua com datas reais para o gráfico
    hist_plot = hist_work.assign(ts=hist_work["p"].dt.to_timestamp())[["ts","y"]]
    last_ts = hist_plot["ts"].iloc[-1]
    fut_ts = pd.date_range(last_ts + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
    fut_plot = pd.DataFrame({"ts": fut_ts, "y": forecast_df_6m["y"]})

    chart_df = (pd.concat([
                    hist_plot.assign(tipo="Histórico"),
                    fut_plot.assign(tipo="Previsão")
                ])
                .set_index("ts"))
    st.line_chart(chart_df, x=None, y="y", color="tipo", height=320, use_container_width=True)

with right:
    st.subheader("Resumo do modelo")
    st.metric("Modelo escolhido", model_choice)
    st.metric("Horizonte", f"{horizon} meses")
    st.metric("MAPE (simulado)", f"{mape:.1f} %")
    st.metric("RMSE (simulado)", f"{rmse:.1f}")

st.subheader("Previsão (tabela)")
st.dataframe(forecast_df_6m, use_container_width=True, height=220)

st.info("Quando o módulo real de previsão estiver pronto, basta substituir esta simulação por `run_best_model_monthly(...)` e manter o `st.session_state['forecast_df_6m']`.")

# ---------- navegação ----------
st.divider()
go_mps = st.button("➡️ Usar esta previsão e seguir para o MPS", type="primary")
if go_mps:
    try:
        st.switch_page("pages/03_🗓️_MPS.py")
    except Exception:
        st.success("Previsão salva! Abra o MPS no menu lateral.")
