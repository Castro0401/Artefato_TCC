# -*- coding: utf-8 -*-
from __future__ import annotations
"""
04_Previsao.py — robusto e integrado ao MPS:
- submit via st.form (1 clique = 1 execução) + fingerprint de configuração
- snapshot/restauração de grades (modo rápido não “gruda”)
- LSTM/Prophet desativados (para espelhar terminal)
- console de logs FILTRADO (eventos essenciais)
- barra de progresso proporcional e ACUMULATIVA (base × (bootstrap+1))
- resultado persiste em session_state
- botões: salvar previsão → MPS, ir para Inputs (05) e ir para MPS (06)
"""

import sys, re, inspect, copy, contextlib, io, traceback, time, hashlib, json
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

# ---- guards / estado
ss = st.session_state
ss.setdefault("is_running", False)
ss.setdefault("last_result", None)
ss.setdefault("last_cfg_key", None)

# =============================
# Recuperar série do Upload
# =============================
if not ss.get("upload_ok"):
    st.error("Nenhuma série encontrada. Volte ao Passo 1 (Upload).")
    st.page_link("pages/01_Upload.py", label="Ir para o Passo 1 — Upload")
    st.stop()

_ts = ss.get("ts_df_norm")
if not isinstance(_ts, pd.DataFrame) or not {"ds", "y"}.issubset(_ts.columns):
    st.error("Formato inesperado: esperado DataFrame com colunas ['ds','y'].")
    st.stop()

_PT_MON2NUM = {"Jan":1,"Fev":2,"Mar":3,"Abr":4,"Mai":5,"Jun":6,"Jul":7,"Ago":8,"Set":9,"Out":10,"Nov":11,"Dez":12}
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
      .sort_index().asfreq("MS").interpolate("linear").bfill().ffill()
)

# =============================
# Snapshot das grades originais
# =============================
_ORIG = {}
def _snap_if_exists(name: str):
    if hasattr(pipe, name) and name not in _ORIG:
        _ORIG[name] = copy.deepcopy(getattr(pipe, name))

for nm in ["CROSTON_ALPHAS","SBA_ALPHAS","TSB_ALPHA_GRID","TSB_BETA_GRID",
           "RF_LAGS_GRID","RF_N_ESTIMATORS_GRID","RF_MAX_DEPTH_GRID","SARIMA_GRID"]:
    _snap_if_exists(nm)

def restore_full_grids(module):
    for name, val in _ORIG.items():
        setattr(module, name, copy.deepcopy(val))

def apply_fast_grids(module):
    if hasattr(module, "CROSTON_ALPHAS"):        module.CROSTON_ALPHAS        = [0.1, 0.3]
    if hasattr(module, "SBA_ALPHAS"):            module.SBA_ALPHAS            = [0.1, 0.3]
    if hasattr(module, "TSB_ALPHA_GRID"):        module.TSB_ALPHA_GRID        = [0.1, 0.3]
    if hasattr(module, "TSB_BETA_GRID"):         module.TSB_BETA_GRID         = [0.1, 0.3]
    if hasattr(module, "RF_LAGS_GRID"):          module.RF_LAGS_GRID          = [6]
    if hasattr(module, "RF_N_ESTIMATORS_GRID"):  module.RF_N_ESTIMATORS_GRID  = [200]
    if hasattr(module, "RF_MAX_DEPTH_GRID"):     module.RF_MAX_DEPTH_GRID     = [None, 10]
    if hasattr(module, "SARIMA_GRID"):
        module.SARIMA_GRID = {"p":[0,1], "d":[0,1], "q":[0,1], "P":[0], "D":[0,1], "Q":[0]}

# =============================
# Menu lateral (dentro de form)
# =============================
with st.form(key="previsao_form"):
    st.sidebar.header("⚙️ Configurações")
    HORIZON = st.sidebar.selectbox("Horizonte (meses)", [6,8,12], index=0)
    FAST_MODE = st.sidebar.toggle("Modo rápido (grade reduzida)", value=False,
                                  help="Menos combinações + bootstrap reduzido.")
    SEASONAL_PERIOD = 12
    DO_ORIGINAL = True
    DO_LOG = True

    # Desliga LSTM/Deep e Prophet
    if hasattr(pipe, "KERAS_AVAILABLE"):
        pipe.KERAS_AVAILABLE = False
    for flag in ("ENABLE_PROPHET","USE_PROPHET","HAS_PROPHET","FBPROPHET_ENABLED","ENABLE_FBPROPHET"):
        if hasattr(pipe, flag):
            setattr(pipe, flag, False)

    if FAST_MODE:
        apply_fast_grids(pipe); DO_BOOTSTRAP = True; N_BOOTSTRAP = 5
    else:
        restore_full_grids(pipe); DO_BOOTSTRAP = True; N_BOOTSTRAP = 20
    BOOTSTRAP_BLOCK = 24

    # passos totais (incluindo bootstrap)
    def _base_steps(mod) -> int:
        base = 0
        if hasattr(mod,"CROSTON_ALPHAS"): base += len(mod.CROSTON_ALPHAS)
        if hasattr(mod,"SBA_ALPHAS"):     base += len(mod.SBA_ALPHAS)
        if hasattr(mod,"TSB_ALPHA_GRID") and hasattr(mod,"TSB_BETA_GRID"):
            base += len(mod.TSB_ALPHA_GRID) * len(mod.TSB_BETA_GRID)
        if (hasattr(mod,"RF_LAGS_GRID") and hasattr(mod,"RF_N_ESTIMATORS_GRID")
            and hasattr(mod,"RF_MAX_DEPTH_GRID")):
            base += len(mod.RF_LAGS_GRID) * len(mod.RF_N_ESTIMATORS_GRID) * len(mod.RF_MAX_DEPTH_GRID)
        if hasattr(mod,"SARIMA_GRID"):
            g = mod.SARIMA_GRID
            base += len(g["p"])*len(g["d"])*len(g["q"])*len(g["P"])*len(g["D"])*len(g["Q"])
        return max(1, base)

    BASE = _base_steps(pipe)
    TOTAL = BASE * (N_BOOTSTRAP + 1)
    st.caption(f"Configuração: rápido={'ON' if FAST_MODE else 'OFF'} | combinações≈{BASE} | bootstrap={N_BOOTSTRAP} | total_passos≈{TOTAL}")

    submitted = st.form_submit_button("▶️ Rodar previsão", type="primary", disabled=ss.is_running)

# =============================
# Console de logs (FILTRADO) e progresso acumulativo
# =============================
prog = st.progress(0)
prog_text = st.empty()

log_box = st.expander("📜 Console (passos essenciais)", expanded=False)
log_area = log_box.empty()
_raw_lines: list[str] = []

_WHITELIST = [
    r"==== PIPELINE INICIADO ====",
    r"^Params:",
    r"Realizando testes da série ORIGINAL",
    r"^→\s*Croston|^→\s*SBA|^→\s*TSB|^→\s*RF|^→\s*SARIMAX",
    r"Concluídos testes:\s*original",
    r"Realizando testes da série .*log",
    r"Concluídos testes:\s*log",
    r"^•\s*Testes — bootstrap",
    r"bootstrap\s*\(.*\)\s*—\s*fim",
    r"===== CAMPEÃO",
    r"==== PIPELINE FINALIZADO ====",
    r"Linhas totais de experimentos:",
    r"Resumo rápido por preprocess:",
    r"(ERROR|EXCEPTION|Traceback)",
]
_WHITELIST_RE = [re.compile(p, re.IGNORECASE) for p in _WHITELIST]

def _show_log_filtered():
    filtered = []
    for ln in _raw_lines[-500:]:
        s = str(ln).strip()
        if any(rx.search(s) for rx in _WHITELIST_RE):
            filtered.append(s)
    log_area.text("\n".join(filtered[-200:]))

def _push_log(line: str):
    _raw_lines.append(str(line))
    _show_log_filtered()

_original_log = getattr(pipe, "log", print)

# padrões de progresso do log
_RE_BOOT   = re.compile(r"bootstrap\s*(\d+)\s*/\s*(\d+)", re.IGNORECASE)   # i/N
_RE_GEN    = re.compile(r"progresso\s+(\d+)\s*/\s*(\d+)", re.IGNORECASE)    # j/N (rodada base)

# --- Progresso acumulativo corrigido ---
_state = {
    "base_done": 0,      # 0..BASE
    "boot_cur": 0,       # 0..N_BOOTSTRAP
    "progress_main": 0,  # fração 0..1 da parte principal
    "progress_boot": 0,  # fração 0..1 do bootstrap
    "BASE": BASE,
    "TOTAL": TOTAL,
    "N_BOOTSTRAP": N_BOOTSTRAP,
}

def _emit_progress():
    """Calcula o progresso acumulativo total com base em base_done + bootstrap."""
    BASE = _state["BASE"]
    N_BOOTSTRAP = _state["N_BOOTSTRAP"]
    # parte principal: 0..1
    base_ratio = _state["progress_main"]
    # parte bootstrap: 0..1
    boot_ratio = _state["progress_boot"]
    # peso da parte base é 1/(1+N), e o do bootstrap é N/(1+N)
    # equivalendo a TOTAL = BASE * (1 + N_BOOTSTRAP)
    total_ratio = min(1.0, (base_ratio + (boot_ratio * N_BOOTSTRAP)) / (1.0 + N_BOOTSTRAP))
    pct = int(round(100 * total_ratio))
    prog.progress(min(100, max(0, pct)))
    return pct

def _patched_log(msg: str):
    s = str(msg)
    try:
        _push_log(s)

        # Detecta progresso base: "progresso j/N"
        m = _RE_GEN.search(s)
        if m:
            j, N = int(m.group(1)), max(1, int(m.group(2)))
            progress_val = j / float(N)
            if progress_val > _state["progress_main"]:
                _state["progress_main"] = progress_val
                _state["base_done"] = int(round(progress_val * _state["BASE"]))

        # Detecta progresso do bootstrap: "bootstrap i/N"
        m2 = _RE_BOOT.search(s)
        if m2:
            i, N = int(m2.group(1)), max(1, int(m2.group(2)))
            progress_val = i / float(N)
            if progress_val > _state["progress_boot"]:
                _state["progress_boot"] = progress_val
                _state["boot_cur"] = i

        pct = _emit_progress()
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
# Fingerprint de configuração (evita run duplicado em reruns)
# =============================
def _cfg_key() -> str:
    def _len_or_zero(x): 
        try: return len(x)
        except Exception: return 0
    g = {
        "HORIZON": HORIZON, "FAST_MODE": FAST_MODE,
        "N_BOOTSTRAP": N_BOOTSTRAP, "BOOTSTRAP_BLOCK": BOOTSTRAP_BLOCK,
        "CROSTON": _len_or_zero(getattr(pipe,"CROSTON_ALPHAS",[])),
        "SBA": _len_or_zero(getattr(pipe,"SBA_ALPHAS",[])),
        "TSB_A": _len_or_zero(getattr(pipe,"TSB_ALPHA_GRID",[])),
        "TSB_B": _len_or_zero(getattr(pipe,"TSB_BETA_GRID",[])),
        "RF_L": _len_or_zero(getattr(pipe,"RF_LAGS_GRID",[])),
        "RF_N": _len_or_zero(getattr(pipe,"RF_N_ESTIMATORS_GRID",[])),
        "RF_D": _len_or_zero(getattr(pipe,"RF_MAX_DEPTH_GRID",[])),
        "SARIMA": {k:_len_or_zero(getattr(pipe,"SARIMA_GRID",{}).get(k,[])) for k in ["p","d","q","P","D","Q"]},
    }
    return hashlib.sha1(json.dumps(g, sort_keys=True).encode()).hexdigest()

cfg_key = _cfg_key()

# =============================
# Execução (só com submit + cfg diferente)
# =============================
if submitted and not ss.is_running and (ss.last_cfg_key != cfg_key or ss.last_result is None):
    ss.is_running = True
    try:
        _wire_progress()
        with st.spinner("Executando pipeline…"):
            _stdout, _stderr = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(_stdout), contextlib.redirect_stderr(_stderr):
                resultados = pipe.run_full_pipeline(
                    data_input=s_monthly,
                    sheet_name=None, date_col=None, value_col=None,
                    horizon=HORIZON, seasonal_period=SEASONAL_PERIOD,
                    do_original=DO_ORIGINAL, do_log=DO_LOG, do_bootstrap=True,
                    n_bootstrap=N_BOOTSTRAP, bootstrap_block=BOOTSTRAP_BLOCK,
                    save_dir=None,
                )
            if _stdout.getvalue(): _push_log(_stdout.getvalue())
            if _stderr.getvalue(): _push_log(_stderr.getvalue())
        prog.progress(100); prog_text.write("100% — concluído")
        ss.last_result = resultados
        ss.last_cfg_key = cfg_key
    except Exception:
        st.error("Falha ao executar a previsão. Traceback abaixo:")
        st.code("\n".join(traceback.format_exc().splitlines()), language="text")
    finally:
        if hasattr(pipe,"log"): pipe.log = _original_log
        ss.is_running = False

# =============================
# Render da saída se já houver resultado salvo
# =============================
res = ss.get("last_result")
if res is not None:
    champ = res.attrs.get("champion", {})
    modelo_nome = champ.get("model", "Desconhecido")
    st.subheader(f"🏆 Modelo Campeão: {modelo_nome}" + (" (Modo rápido)" if FAST_MODE else ""))

    def _fmt(x):
        try: return f"{float(x):.4g}"
        except Exception: return str(x)

    c1,c2,c3,c4 = st.columns(4)
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
        if key in res.attrs:
            forecast = res.attrs[key]; break
    if isinstance(forecast, pd.DataFrame) and {"ds","yhat"}.issubset(forecast.columns):
        f_idx = pd.to_datetime(forecast["ds"])
        forecast_s = pd.Series(forecast["yhat"].astype(float).to_numpy(), index=f_idx)
        forecast_df_std = forecast.rename(columns={"yhat":"y"})[["ds","y"]].copy()
    elif isinstance(forecast, pd.Series):
        forecast_s = forecast.astype(float)
        forecast_df_std = pd.DataFrame({"ds": forecast.index, "y": forecast.values})
    else:
        last = s_monthly[-SEASONAL_PERIOD:]; reps = int((HORIZON+SEASONAL_PERIOD-1)//SEASONAL_PERIOD)
        vals = np.tile(last.to_numpy(), reps)[:HORIZON]
        f_idx = pd.date_range(s_monthly.index[-1] + pd.offsets.MonthBegin(1), periods=HORIZON, freq="MS")
        forecast_s = pd.Series(vals, index=f_idx)
        forecast_df_std = pd.DataFrame({"ds": f_idx, "y": vals})

    plot_df = pd.DataFrame({"Real": s_monthly, "Previsão": forecast_s})
    st.subheader("📈 Histórico + Previsão")
    st.line_chart(plot_df.iloc[-max(36, HORIZON+6):], height=280)

    st.subheader("📋 Experimentos (resumo)")
    st.dataframe(res.reset_index(drop=True), use_container_width=True)

    # =============================
    # 🔗 Integração com MPS: salvar e navegar
    # =============================
    st.divider()
    st.subheader("➡️ Próximos passos")

    c_save, c_inputs, c_mps = st.columns([1.3, 1.3, 1.0])

    with c_save:
        can_save = forecast_df_std is not None and len(forecast_df_std) > 0
        if st.button("💾 Salvar previsão para o MPS", disabled=not can_save, use_container_width=True):
            st.session_state["forecast_df"] = forecast_df_std.copy()
            st.session_state["forecast_h"] = int(HORIZON)
            st.session_state["forecast_committed"] = True
            st.success("Previsão salva para o MPS.")

    with c_inputs:
        st.page_link("pages/05_Inputs_MPS.py", label="⚙️ Ir para Inputs do MPS", icon="⚙️", use_container_width=True)

    with c_mps:
        st.page_link("pages/06_MPS.py", label="🗓️ Ir para o MPS", icon="🗓️", use_container_width=True)

    if not st.session_state.get("forecast_committed", False):
        st.info("Clique em **Salvar previsão para o MPS** antes de avançar.", icon="ℹ️")
