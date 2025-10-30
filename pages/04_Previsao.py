# -*- coding: utf-8 -*-
from __future__ import annotations
"""
04_Previsao.py — versão com menu lateral, "modo rápido" por grade reduzida e gráfico real+previsão.
- Consome a série validada no 01_Upload.py via st.session_state["ts_df_norm"] (['ds','y'])
- Converte rótulos "Mon/YY" para datas (MS) e cria série mensal contínua (asfreq + interpolate)
- Importa core/pipeline.py com caminho robusto
- Modo rápido: reduz combinações de hiperparâmetros (grids) e diminui réplicas do bootstrap
"""

import sys
from pathlib import Path
import traceback
import numpy as np
import pandas as pd
import streamlit as st

# =============================
# Inserir caminhos para importar core/pipeline
# =============================
ROOT = Path(__file__).resolve().parent.parent      # .../artefato_tcc (raiz)
CORE = ROOT / "core"
for p in (ROOT, CORE):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

try:
    from core import pipeline as pipe
except ModuleNotFoundError:
    import pipeline as pipe  # fallback

st.set_page_config(page_title="Previsão", page_icon="🔮", layout="wide")
st.title("🔮 Passo 2 — Previsão")

# =============================
# Recupera a série do Upload
# =============================
if not st.session_state.get("upload_ok"):
    st.error("Nenhuma série encontrada. Volte ao Passo 1 (Upload) para carregar os dados.")
    st.stop()

_ts = st.session_state.get("ts_df_norm")
if not isinstance(_ts, pd.DataFrame) or not {"ds", "y"}.issubset(_ts.columns):
    st.error("Formato inesperado da série: esperado DataFrame com colunas ['ds','y'].")
    st.stop()

product_name = st.session_state.get("product_name", "Produto")
st.caption(f"Série atual: **{product_name}**")
st.dataframe(_ts, use_container_width=True, height=400)

# =============================
# Converte rótulos para datas (primeiro dia do mês) e cria Series mensal contínua
# =============================
_PT_MON2NUM = {
    "Jan": 1, "Fev": 2, "Mar": 3, "Abr": 4, "Mai": 5, "Jun": 6,
    "Jul": 7, "Ago": 8, "Set": 9, "Out": 10, "Nov": 11, "Dez": 12,
}

def _label_to_month_start(val) -> pd.Timestamp:
    # Se já vier datetime, só converte
    if isinstance(val, (pd.Timestamp,)):
        return pd.to_datetime(val)
    s = str(val)
    try:
        if "/" in s:
            mon, yy = s.split("/")
            y = 2000 + int(yy)  # regra simples; ajuste se necessário
            m = _PT_MON2NUM.get(mon)
            if m is None:
                return pd.to_datetime(s, errors="coerce")
            return pd.Timestamp(year=y, month=m, day=1)
        return pd.to_datetime(s, errors="coerce")
    except Exception:
        return pd.NaT

_idx = _ts["ds"].map(_label_to_month_start)
if _idx.isna().any():
    st.warning("Alguns rótulos de data não foram convertidos e serão descartados.")

s_monthly = (
    pd.Series(_ts.loc[_idx.notna(), "y"].astype(float).to_numpy(), index=_idx[_idx.notna()])
      .sort_index()
      .asfreq("MS")           # grade mensal contínua
      .interpolate("linear")  # igual ao preparo usado ao ler Excel no terminal
      .bfill()
      .ffill()
)

st.caption(
    f"Registros após normalização: {len(s_monthly)} — "
    f"janela: {s_monthly.index.min():%b/%Y} → {s_monthly.index.max():%b/%Y}"
)

# =============================
# Configurações (menu lateral)
# =============================
st.sidebar.header("⚙️ Configurações")
HORIZON = st.sidebar.selectbox("Horizonte (meses)", options=[6, 8, 12], index=0)
FAST_MODE = st.sidebar.toggle("Modo rápido (grade reduzida)", value=True,
                              help="Usa menos combinações de hiperparâmetros e menos réplicas de bootstrap.")
SEASONAL_PERIOD = 12
DO_ORIGINAL = True
DO_LOG = True

# =============================
# Redução defensiva de grids quando FAST_MODE
# (apenas aplica se os atributos existirem no pipeline; caso contrário, ignora)
# =============================

def apply_fast_grids(module):
    """Tenta reduzir combinatórias de busca do pipeline de forma segura.
    Não quebra se algum atributo não existir.
    """
    # Random Forest
    if hasattr(module, "RF_N_ESTIMATORS_GRID"): module.RF_N_ESTIMATORS_GRID = [200]
    if hasattr(module, "RF_MAX_DEPTH_GRID"):    module.RF_MAX_DEPTH_GRID    = [None, 10]
    if hasattr(module, "RF_MIN_SAMPLES_SPLIT_GRID"): module.RF_MIN_SAMPLES_SPLIT_GRID = [2]
    if hasattr(module, "RF_MIN_SAMPLES_LEAF_GRID"):  module.RF_MIN_SAMPLES_LEAF_GRID  = [1]

    # XGBoost/GBM (se houver)
    for name, val in [
        ("XGB_N_ESTIMATORS_GRID", [200]),
        ("XGB_MAX_DEPTH_GRID", [3, 5]),
        ("XGB_LEARNING_RATE_GRID", [0.1]),
        ("XGB_SUBSAMPLE_GRID", [0.8]),
    ]:
        if hasattr(module, name): setattr(module, name, val)

    # ARIMA/SARIMA
    for name, val in [
        ("ARIMA_P_GRID", [0,1,2]),
        ("ARIMA_D_GRID", [0,1]),
        ("ARIMA_Q_GRID", [0,1,2]),
        ("SEASONAL_P_GRID", [0,1]),
        ("SEASONAL_D_GRID", [0,1]),
        ("SEASONAL_Q_GRID", [0,1]),
    ]:
        if hasattr(module, name): setattr(module, name, val)

    if hasattr(module, "SARIMA_SEASONAL_PERIODS"):
        module.SARIMA_SEASONAL_PERIODS = [12]

    # ETS/Exponential Smoothing
    for name, val in [
        ("ETS_TREND_OPTS", [None, "add"]),
        ("ETS_SEASONAL_OPTS", [None, "add"]),
        ("ETS_DAMPED_TREND_OPTS", [False, True]),
    ]:
        if hasattr(module, name): setattr(module, name, val)

    # Prophet (se houver)
    for name, val in [
        ("PROPHET_SEASONALITY_MODE", ["additive"]),
        ("PROPHET_CHGPOINT_RANGE", [0.8]),
        ("PROPHET_N_CHGPOINTS", [10]),
    ]:
        if hasattr(module, name): setattr(module, name, val)

    # LSTM/Deep (se houver)
    for name, val in [
        ("LSTM_EPOCHS", 10),
        ("LSTM_BATCH_SIZE", 32),
        ("LSTM_HIDDEN_UNITS", [32]),
    ]:
        if hasattr(module, name): setattr(module, name, val)


# Bootstrap: reduz réplicas quando rápido; mantém ligado para comparabilidade
if FAST_MODE:
    DO_BOOTSTRAP = True
    N_BOOTSTRAP = 5
    apply_fast_grids(pipe)
else:
    DO_BOOTSTRAP = True
    N_BOOTSTRAP = 20

BOOTSTRAP_BLOCK = 24

run = st.button("▶️ Rodar previsão", type="primary")

if run:
    try:
        with st.spinner("Executando pipeline… isso pode levar alguns minutos…"):
            resultados = pipe.run_full_pipeline(
                data_input=s_monthly,
                sheet_name=None, date_col=None, value_col=None,
                horizon=HORIZON, seasonal_period=SEASONAL_PERIOD,
                do_original=DO_ORIGINAL, do_log=DO_LOG, do_bootstrap=DO_BOOTSTRAP,
                n_bootstrap=N_BOOTSTRAP, bootstrap_block=BOOTSTRAP_BLOCK,
                save_dir=None,
            )

        champ = resultados.attrs.get("champion", {})

        st.subheader("🏆 Modelo Campeão (métricas)")
        def _fmt(x):
            try:
                return f"{float(x):.4g}"
            except Exception:
                return str(x)

        cols = st.columns(4)
        cols[0].metric("MAE", _fmt(champ.get("MAE")))
        cols[1].metric("sMAPE (%)", _fmt(champ.get("sMAPE")))
        cols[2].metric("RMSE", _fmt(champ.get("RMSE")))
        cols[3].metric("MAPE (%)", _fmt(champ.get("MAPE")))

        st.write("Parâmetros do campeão:")
        st.json({
            "preprocess": champ.get("preprocess"),
            "preprocess_params": champ.get("preprocess_params"),
            "model": champ.get("model"),
            "model_params": champ.get("model_params"),
        })

        # =============================
        # Real + Previsão (gráfico)
        # =============================
        forecast = None
        for key in ("forecast", "forecast_df", "yhat", "pred", "prediction"):
            if key in resultados.attrs:
                forecast = resultados.attrs[key]
                break

        # Normaliza forecast para Series; se não vier do pipeline, usa sazonal ingênuo
        if isinstance(forecast, pd.DataFrame) and {"ds", "yhat"}.issubset(forecast.columns):
            f_idx = (pd.to_datetime(forecast["ds"]) if not isinstance(forecast.index, pd.DatetimeIndex)
                     else forecast.index)
            forecast_s = pd.Series(forecast["yhat"].astype(float).to_numpy(), index=f_idx)
        elif isinstance(forecast, pd.Series):
            forecast_s = forecast.astype(float)
        else:
            last = s_monthly[-SEASONAL_PERIOD:]
            reps = int((HORIZON + SEASONAL_PERIOD - 1) // SEASONAL_PERIOD)
            vals = np.tile(last.to_numpy(), reps)[:HORIZON]
            f_idx = pd.date_range(s_monthly.index[-1] + pd.offsets.MonthBegin(1),
                                  periods=HORIZON, freq="MS")
            forecast_s = pd.Series(vals, index=f_idx)

        plot_df = pd.DataFrame({"Real": s_monthly, "Previsão": forecast_s})
        cut = max(36, HORIZON + 6)
        plot_tail = plot_df.iloc[-cut:]

        st.subheader("📈 Histórico + Previsão")
        st.line_chart(plot_tail, height=280)

        st.subheader("📋 Experimentos (resumo)")
        st.dataframe(resultados.reset_index(drop=True), use_container_width=True)

    except Exception:
        st.error("Falha ao executar a previsão. Veja o traceback abaixo:")
        st.code("
".join(traceback.format_exc().splitlines()), language="text")
