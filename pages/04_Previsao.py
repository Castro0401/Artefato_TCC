# -*- coding: utf-8 -*-
from __future__ import annotations
"""
04_Previsao.py — com:
- menu lateral (horizonte 6/8/12 + modo rápido = grade reduzida),
- LSTM desativado explicitamente (compatível com seu pipeline),
- barra de progresso lendo os logs do pipeline (… progresso A/B),
- gráfico Real + Previsão ao final.
"""

import sys, re, inspect
from pathlib import Path
import traceback
import numpy as np
import pandas as pd
import streamlit as st

# ===== importar core/pipeline =====
ROOT = Path(__file__).resolve().parent.parent
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

# ===== recuperar série do Upload =====
if not st.session_state.get("upload_ok"):
    st.error("Nenhuma série encontrada. Volte ao Passo 1 (Upload).")
    st.stop()

_ts = st.session_state.get("ts_df_norm")
if not isinstance(_ts, pd.DataFrame) or not {"ds", "y"}.issubset(_ts.columns):
    st.error("Formato inesperado: esperado DataFrame com colunas ['ds','y'].")
    st.stop()

# rótulo "Mon/YY" -> Timestamp MS e série mensal contínua
_PT_MON2NUM = {"Jan":1,"Fev":2,"Mar":3,"Abr":4,"Mai":5,"Jun":6,"Jul":7,"Ago":8,"Set":9,"Out":10,"Nov":11,"Dez":12}
def _label_to_month_start(v) -> pd.Timestamp:
    if isinstance(v, pd.Timestamp): return v
    s = str(v)
    try:
        if "/" in s:
            mon, yy = s.split("/")
            return pd.Timestamp(year=2000+int(yy), month=_PT_MON2NUM.get(mon, 1), day=1)
        return pd.to_datetime(s, errors="coerce")
    except Exception:
        return pd.NaT

_idx = _ts["ds"].map(_label_to_month_start)
s_monthly = (
    pd.Series(_ts.loc[_idx.notna(), "y"].astype(float).to_numpy(), index=_idx[_idx.notna()])
      .sort_index().asfreq("MS").interpolate("linear").bfill().ffill()
)

# ===== barra lateral =====
st.sidebar.header("⚙️ Configurações")
HORIZON = st.sidebar.selectbox("Horizonte (meses)", [6, 8, 12], index=0)
FAST_MODE = st.sidebar.toggle("Modo rápido (grade reduzida)", value=False,
                              help="Menos combinações + bootstrap reduzido.")
SEASONAL_PERIOD = 12
DO_ORIGINAL = True
DO_LOG = True
DO_BOOTSTRAP = True
BOOTSTRAP_BLOCK = 24

# ===== desativar LSTM do seu pipeline =====
# seu pipeline só liga LSTM se KERAS_AVAILABLE=True; forço para False aqui:
if hasattr(pipe, "KERAS_AVAILABLE"):
    pipe.KERAS_AVAILABLE = False

# ===== reduzir grades exatamente com os nomes do seu pipeline =====
def apply_fast_grids(module):
    # Intermitentes
    if hasattr(module, "CROSTON_ALPHAS"):     module.CROSTON_ALPHAS     = [0.1, 0.3]
    if hasattr(module, "SBA_ALPHAS"):         module.SBA_ALPHAS         = [0.1, 0.3]
    if hasattr(module, "TSB_ALPHA_GRID"):     module.TSB_ALPHA_GRID     = [0.1, 0.3]
    if hasattr(module, "TSB_BETA_GRID"):      module.TSB_BETA_GRID      = [0.1, 0.3]
    # RandomForest
    if hasattr(module, "RF_LAGS_GRID"):        module.RF_LAGS_GRID        = [6]           # antes: [3, 6, 12]
    if hasattr(module, "RF_N_ESTIMATORS_GRID"):module.RF_N_ESTIMATORS_GRID= [200]         # antes: [200, 500]
    if hasattr(module, "RF_MAX_DEPTH_GRID"):   module.RF_MAX_DEPTH_GRID   = [None, 10]    # antes: [None, 5, 10]
    # SARIMA
    if hasattr(module, "SARIMA_GRID"):
        module.SARIMA_GRID = {"p":[0,1], "d":[0,1], "q":[0,1], "P":[0], "D":[0,1], "Q":[0]}  # reduzida

if FAST_MODE:
    apply_fast_grids(pipe)
    DO_BOOTSTRAP = True
    N_BOOTSTRAP = 5
else:
    DO_BOOTSTRAP = True
    N_BOOTSTRAP = 20

# ===== estimador de progresso a partir dos logs do pipeline =====
# Calcula total de combinações com as grades atuais (já reduzidas se FAST_MODE)
def _total_steps(mod) -> int:
    tot = 0
    if hasattr(mod, "CROSTON_ALPHAS"):          tot += len(mod.CROSTON_ALPHAS)
    if hasattr(mod, "SBA_ALPHAS"):              tot += len(mod.SBA_ALPHAS)
    if hasattr(mod, "TSB_ALPHA_GRID") and hasattr(mod, "TSB_BETA_GRID"):
        tot += len(mod.TSB_ALPHA_GRID) * len(mod.TSB_BETA_GRID)
    if (hasattr(mod, "RF_LAGS_GRID") and hasattr(mod, "RF_N_ESTIMATORS_GRID")
        and hasattr(mod, "RF_MAX_DEPTH_GRID")):
        tot += len(mod.RF_LAGS_GRID) * len(mod.RF_N_ESTIMATORS_GRID) * len(mod.RF_MAX_DEPTH_GRID)
    if hasattr(mod, "SARIMA_GRID"):
        g = mod.SARIMA_GRID
        tot += len(g["p"])*len(g["d"])*len(g["q"])*len(g["P"])*len(g["D"])*len(g["Q"])
    # LSTM não conta (desligada)
    return tot

TOTAL = _total_steps(pipe)
prog = st.progress(0)
prog_text = st.empty()

# monkey patch no logger do pipeline para extrair “A/B” e atualizar barra
_original_log = pipe.log
_step = {"done": 0}

def _progress_from_msg(msg: str):
    # padrões que o seu pipeline imprime:
    # "… Croston progresso j/N", "… SBA progresso j/N", "… TSB progresso step/N"
    # "… RF progresso cnt/tot", "… SARIMAX progresso i/N"
    m = re.search(r"progresso\s+(\d+)\s*/\s*(\d+)", msg)
    if m:
        cur, tot = int(m.group(1)), int(m.group(2))
        # atualiza em relação à fração daquele bloco -> aproximamos como passos absolutos
        # incremento mínimo 1 para evitar voltar
        inc = max(1, cur - (_step.get("__last_block_cur", 0)))
        _step["done"] = min(TOTAL, _step["done"] + inc)
        _step["__last_block_cur"] = cur

def _patched_log(msg: str):
    try:
        _progress_from_msg(str(msg))
        pct = 0 if TOTAL == 0 else min(100, int(round(_step["done"] * 100 / TOTAL)))
        prog.progress(pct)
        if pct < 100:
            prog_text.write(f"{pct}% — {msg}")
        else:
            prog_text.write("100% — concluído")
    except Exception:
        pass
    # ainda imprime no console
    _original_log(msg)

# ===== botão =====
run = st.button("▶️ Rodar previsão", type="primary")

if run:
    try:
        # ativa logger com progresso
        pipe.log = _patched_log

        # tenta também suportar progress_cb= se futuramente existir
        extra = {}
        try:
            sig = inspect.signature(pipe.run_full_pipeline)
            if "progress_cb" in sig.parameters:
                # compat futuro: se você adicionar isso no pipeline
                def _cb(curr:int, total:int, desc:str=""):
                    _step["done"] = curr; prog.progress(int(curr*100/total) if total else 0)
                    prog_text.write(f"{int(curr*100/total)}% — {desc}" if total else desc)
                extra["progress_cb"] = _cb
        except Exception:
            pass

        with st.spinner("Executando pipeline…"):
            resultados = pipe.run_full_pipeline(
                data_input=s_monthly,
                sheet_name=None, date_col=None, value_col=None,
                horizon=HORIZON, seasonal_period=SEASONAL_PERIOD,
                do_original=DO_ORIGINAL, do_log=DO_LOG, do_bootstrap=DO_BOOTSTRAP,
                n_bootstrap=N_BOOTSTRAP, bootstrap_block=BOOTSTRAP_BLOCK,
                save_dir=None,
                **extra
            )
        prog.progress(100); prog_text.write("100% — concluído")

        champ = resultados.attrs.get("champion", {})

        st.subheader("🏆 Modelo Campeão (métricas)")
        def _fmt(x):
            try: return f"{float(x):.4g}"
            except Exception: return str(x)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("MAE", _fmt(champ.get("MAE")))
        c2.metric("sMAPE (%)", _fmt(champ.get("sMAPE")))
        c3.metric("RMSE", _fmt(champ.get("RMSE")))
        c4.metric("MAPE (%)", _fmt(champ.get("MAPE")))

        st.write("Parâmetros do campeão:")
        st.json({
            "preprocess": champ.get("preprocess"),
            "preprocess_params": champ.get("preprocess_params"),
            "model": champ.get("model"),
            "model_params": champ.get("model_params"),
        })

        # ===== gráfico Real + Previsão =====
        forecast = None
        for key in ("forecast","forecast_df","yhat","pred","prediction"):  # pode não existir
            if key in resultados.attrs:
                forecast = resultados.attrs[key]; break

        if isinstance(forecast, pd.DataFrame) and {"ds","yhat"}.issubset(forecast.columns):
            f_idx = pd.to_datetime(forecast["ds"]) if not isinstance(forecast.index, pd.DatetimeIndex) else forecast.index
            forecast_s = pd.Series(forecast["yhat"].astype(float).to_numpy(), index=f_idx)
        elif isinstance(forecast, pd.Series):
            forecast_s = forecast.astype(float)
        else:
            last = s_monthly[-SEASONAL_PERIOD:]
            reps = int((HORIZON + SEASONAL_PERIOD - 1) // SEASONAL_PERIOD)
            vals = np.tile(last.to_numpy(), reps)[:HORIZON]
            f_idx = pd.date_range(s_monthly.index[-1] + pd.offsets.MonthBegin(1), periods=HORIZON, freq="MS")
            forecast_s = pd.Series(vals, index=f_idx)

        plot_df = pd.DataFrame({"Real": s_monthly, "Previsão": forecast_s})
        cut = max(36, HORIZON + 6)
        st.subheader("📈 Histórico + Previsão")
        st.line_chart(plot_df.iloc[-cut:], height=280)

        st.subheader("📋 Experimentos (resumo)")
        st.dataframe(resultados.reset_index(drop=True), use_container_width=True)

    except Exception:
        st.error("Falha ao executar a previsão. Veja o traceback abaixo:")
        st.code("\n".join(traceback.format_exc().splitlines()), language="text")
    finally:
        # restaura o logger original
        pipe.log = _original_log
