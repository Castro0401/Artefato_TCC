# -*- coding: utf-8 -*-
from __future__ import annotations
"""
04_Previsao.py — Página de previsão com:
- snapshot/restauração das grades (evita “grudar” modo rápido)
- LSTM/Prophet desativados (para equivaler ao terminal)
- barra de progresso incluindo bootstrap
- console de logs ao vivo (exibe mensagens do pipeline)
- guard de execução (evita reexecutar após concluir)
- modelo campeão + métricas + gráfico Real + Previsão
"""

import sys, re, inspect, copy, contextlib, io, traceback, time
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

# =============================
# Importar core/pipeline
# =============================
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

# ---- guards para não reexecutar sem clique novo
if "is_running" not in st.session_state:
    st.session_state.is_running = False
if "run_id" not in st.session_state:
    st.session_state.run_id = None

# =============================
# Recuperar série do Upload
# =============================
if not st.session_state.get("upload_ok"):
    st.error("Nenhuma série encontrada. Volte ao Passo 1 (Upload).")
    st.stop()

_ts = st.session_state.get("ts_df_norm")
if not isinstance(_ts, pd.DataFrame) or not {"ds", "y"}.issubset(_ts.columns):
    st.error("Formato inesperado: esperado DataFrame com colunas ['ds','y'].")
    st.stop()

_PT_MON2NUM = {"Jan":1,"Fev":2,"Mar":3,"Abr":4,"Mai":5,"Jun":6,
               "Jul":7,"Ago":8,"Set":9,"Out":10,"Nov":11,"Dez":12}

def _label_to_month_start(v) -> pd.Timestamp:
    if isinstance(v, pd.Timestamp): return v
    s = str(v)
    try:
        if "/" in s:
            mon, yy = s.split("/")
            return pd.Timestamp(year=2000 + int(yy), month=_PT_MON2NUM.get(mon, 1), day=1)
        return pd.to_datetime(s, errors="coerce")
    except Exception:
        return pd.NaT

_idx = _ts["ds"].map(_label_to_month_start)
s_monthly = (
    pd.Series(_ts.loc[_idx.notna(), "y"].astype(float).to_numpy(), index=_idx[_idx.notna()])
      .sort_index()
      .asfreq("MS")
      .interpolate("linear")
      .bfill()
      .ffill()
)

# =============================
# Snapshot das grades originais
# =============================
_ORIG = {}
def _snap_if_exists(name: str):
    if hasattr(pipe, name) and name not in _ORIG:
        _ORIG[name] = copy.deepcopy(getattr(pipe, name))

for nm in [
    "CROSTON_ALPHAS", "SBA_ALPHAS", "TSB_ALPHA_GRID", "TSB_BETA_GRID",
    "RF_LAGS_GRID", "RF_N_ESTIMATORS_GRID", "RF_MAX_DEPTH_GRID",
    "SARIMA_GRID"
]:
    _snap_if_exists(nm)

def restore_full_grids(module):
    for name, val in _ORIG.items():
        setattr(module, name, copy.deepcopy(val))

def apply_fast_grids(module):
    # Intermitentes
    if hasattr(module, "CROSTON_ALPHAS"):       module.CROSTON_ALPHAS       = [0.1, 0.3]
    if hasattr(module, "SBA_ALPHAS"):           module.SBA_ALPHAS           = [0.1, 0.3]
    if hasattr(module, "TSB_ALPHA_GRID"):       module.TSB_ALPHA_GRID       = [0.1, 0.3]
    if hasattr(module, "TSB_BETA_GRID"):        module.TSB_BETA_GRID        = [0.1, 0.3]
    # Random Forest
    if hasattr(module, "RF_LAGS_GRID"):         module.RF_LAGS_GRID         = [6]
    if hasattr(module, "RF_N_ESTIMATORS_GRID"): module.RF_N_ESTIMATORS_GRID = [200]
    if hasattr(module, "RF_MAX_DEPTH_GRID"):    module.RF_MAX_DEPTH_GRID    = [None, 10]
    # SARIMA
    if hasattr(module, "SARIMA_GRID"):
        module.SARIMA_GRID = {"p":[0,1], "d":[0,1], "q":[0,1], "P":[0], "D":[0,1], "Q":[0]}

# =============================
# Menu lateral
# =============================
st.sidebar.header("⚙️ Configurações")
HORIZON = st.sidebar.selectbox("Horizonte (meses)", [6, 8, 12], index=0)
FAST_MODE = st.sidebar.toggle("Modo rápido (grade reduzida)", value=False,
                              help="Menos combinações + bootstrap reduzido.")
SEASONAL_PERIOD = 12
DO_ORIGINAL = True
DO_LOG = True

# Desliga LSTM/Deep e Prophet
if hasattr(pipe, "KERAS_AVAILABLE"):
    pipe.KERAS_AVAILABLE = False
for flag in ("ENABLE_PROPHET", "USE_PROPHET", "HAS_PROPHET", "FBPROPHET_ENABLED", "ENABLE_FBPROPHET"):
    if hasattr(pipe, flag):
        setattr(pipe, flag, False)

if FAST_MODE:
    apply_fast_grids(pipe)
    DO_BOOTSTRAP = True
    N_BOOTSTRAP = 5
else:
    restore_full_grids(pipe)
    DO_BOOTSTRAP = True
    N_BOOTSTRAP = 20

BOOTSTRAP_BLOCK = 24

# =============================
# Mostrar configuração efetiva
# =============================
def _total_steps(mod) -> int:
    base = 0
    if hasattr(mod, "CROSTON_ALPHAS"): base += len(mod.CROSTON_ALPHAS)
    if hasattr(mod, "SBA_ALPHAS"):     base += len(mod.SBA_ALPHAS)
    if hasattr(mod, "TSB_ALPHA_GRID") and hasattr(mod, "TSB_BETA_GRID"):
        base += len(mod.TSB_ALPHA_GRID) * len(mod.TSB_BETA_GRID)
    if (hasattr(mod, "RF_LAGS_GRID") and hasattr(mod, "RF_N_ESTIMATORS_GRID")
        and hasattr(mod, "RF_MAX_DEPTH_GRID")):
        base += len(mod.RF_LAGS_GRID) * len(mod.RF_N_ESTIMATORS_GRID) * len(mod.RF_MAX_DEPTH_GRID)
    if hasattr(mod, "SARIMA_GRID"):
        g = mod.SARIMA_GRID
        base += len(g["p"]) * len(g["d"]) * len(g["q"]) * len(g["P"]) * len(g["D"]) * len(g["Q"])
    total = base * (N_BOOTSTRAP + 1)
    return max(1, total)

TOTAL = _total_steps(pipe)
st.caption(f"Configuração: rápido={'ON' if FAST_MODE else 'OFF'} | combinações≈{TOTAL // (N_BOOTSTRAP+1)} | bootstrap={N_BOOTSTRAP} | total_passos≈{TOTAL}")

# =============================
# Botão + barra + console de logs
# =============================
clicked = st.button("▶️ Rodar previsão", type="primary", disabled=st.session_state.is_running)
if clicked and not st.session_state.is_running:
    st.session_state.is_running = True
    st.session_state.run_id = time.time()

prog = st.progress(0)
prog_text = st.empty()

log_box = st.expander("📜 Console de logs (ao vivo)", expanded=True)
log_area = log_box.empty()
_log_lines: list[str] = []

def _push_log(line: str):
    _log_lines.append(str(line))
    if len(_log_lines) > 400:
        del _log_lines[:len(_log_lines) - 400]
    log_area.text("\n".join(_log_lines))

_original_log = getattr(pipe, "log", print)
_step = {"done": 0}

_bootstrap_pat = re.compile(r"bootstrap\s*(\d+)\s*/\s*(\d+)", re.IGNORECASE)
_generic_pat   = re.compile(r"progresso\s+(\d+)\s*/\s*(\d+)", re.IGNORECASE)

def _progress_from_msg(msg: str):
    s = str(msg)
    m = _bootstrap_pat.search(s)
    if m:
        cur, tot = int(m.group(1)), int(m.group(2))
        base = max(1, TOTAL // (N_BOOTSTRAP + 1))
        target = min(TOTAL, base * (1 + cur))
        if target > _step["done"]:
            _step["done"] = target
        return
    m2 = _generic_pat.search(s)
    if m2:
        cur, tot = int(m2.group(1)), int(m2.group(2))
        base = max(1, TOTAL // (N_BOOTSTRAP + 1))
        target = min(TOTAL, int(round(base * cur / max(1, tot))))
        if target > _step["done"]:
            _step["done"] = target

def _patched_log(msg: str):
    s = str(msg)
    try:
        _push_log(s)
        _progress_from_msg(s)
        pct = int(round(_step["done"] * 100 / TOTAL))
        prog.progress(min(100, max(0, pct)))
        prog_text.write(f"{pct}% — {s}" if pct < 100 else "100% — concluído")
    except Exception:
        pass
    _original_log(msg)

def _wire_progress():
    wired = False
    if hasattr(pipe, "log"):
        pipe.log = _patched_log
        wired = True
    extra = {}
    try:
        sig = inspect.signature(pipe.run_full_pipeline)
        if "progress_cb" in sig.parameters:
            def _cb(curr:int, total:int, desc:str=""):
                pct = 0 if total == 0 else int(round(curr * 100 / total))
                prog.progress(min(100, max(0, pct)))
                prog_text.write(f"{pct}% — {desc}" if desc else f"{pct}%")
            extra["progress_cb"] = _cb
    except Exception:
        pass
    return wired, extra

# =============================
# Execução (guard)
# =============================
if st.session_state.is_running:
    try:
        wired, extra = _wire_progress()
        with st.spinner("Executando pipeline…"):
            _stdout = io.StringIO()
            _stderr = io.StringIO()
            with contextlib.redirect_stdout(_stdout), contextlib.redirect_stderr(_stderr):
                resultados = pipe.run_full_pipeline(
                    data_input=s_monthly,
                    sheet_name=None, date_col=None, value_col=None,
                    horizon=HORIZON, seasonal_period=SEASONAL_PERIOD,
                    do_original=DO_ORIGINAL, do_log=DO_LOG, do_bootstrap=True,
                    n_bootstrap=N_BOOTSTRAP, bootstrap_block=BOOTSTRAP_BLOCK,
                    save_dir=None,
                    **extra
                )
            if _stdout.getvalue(): _push_log(_stdout.getvalue())
            if _stderr.getvalue(): _push_log(_stderr.getvalue())

        # guarda o resultado para reexibir sem reprocessar
        st.session_state["last_result"] = resultados

        prog.progress(100)
        prog_text.write("100% — concluído")

    except Exception:
        st.error("Falha ao executar a previsão. Veja o traceback abaixo:")
        st.code("\n".join(traceback.format_exc().splitlines()), language="text")
    finally:
        if hasattr(pipe, "log"):
            pipe.log = _original_log
        st.session_state.is_running = False

# =============================
# Render da saída se já houver resultado salvo
# =============================
resultados = st.session_state.get("last_result")
if resultados is not None:
    champ = resultados.attrs.get("champion", {})
    modelo_nome = champ.get("model", "Desconhecido")
    titulo_modelo = f"🏆 Modelo Campeão: {modelo_nome}" + (" (Modo rápido)" if FAST_MODE else "")
    st.subheader(titulo_modelo)

    def _fmt(x):
        try: return f"{float(x):.4g}"
        except Exception: return str(x)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MAE", _fmt(champ.get("MAE")))
    c2.metric("sMAPE (%)", _fmt(champ.get("sMAPE")))
    c3.metric("RMSE", _fmt(champ.get("RMSE")))
    c4.metric("MAPE (%)", _fmt(champ.get("MAPE")))

    st.caption("Parâmetros do modelo campeão:")
    st.json({
        "preprocess": champ.get("preprocess"),
        "preprocess_params": champ.get("preprocess_params"),
        "model_params": champ.get("model_params"),
    })

    # Gráfico Real + Previsão
    forecast = None
    for key in ("forecast","forecast_df","yhat","pred","prediction"):
        if key in resultados.attrs:
            forecast = resultados.attrs[key]; break

    if isinstance(forecast, pd.DataFrame) and {"ds","yhat"}.issubset(forecast.columns):
        f_idx = pd.to_datetime(forecast["ds"])
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
