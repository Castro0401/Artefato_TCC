# -*- coding: utf-8 -*-
from __future__ import annotations

# ==== checagem de sintaxe do pipeline, antes de importar
from pathlib import Path
import py_compile, streamlit as st

ROOT = Path(__file__).resolve().parent.parent
CORE = ROOT / "core"
PIPE_PATH = CORE / "pipeline.py"

try:
    py_compile.compile(str(PIPE_PATH), doraise=True)
except py_compile.PyCompileError as e:
    st.error("Erro de sintaxe em core/pipeline.py")
    st.code(e.msg, language="text")
    st.stop()


import sys, re, copy, contextlib, io, traceback, time
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# ===== import pipeline
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
st.title("🔮 Previsão")

ss = st.session_state
ss.setdefault("is_running", False)
ss.setdefault("last_result", None)

# ===== série do upload
if not ss.get("upload_ok"):
    st.warning("Nenhuma série encontrada. Volte ao Passo 1 (Upload).")
    st.page_link("pages/01_Upload.py", label="Ir para o Passo 1 — Upload")
    st.stop()

_ts = ss.get("ts_df_norm")
if not isinstance(_ts, pd.DataFrame) or not {"ds", "y"}.issubset(_ts.columns):
    st.error("Formato inesperado: esperado DataFrame com colunas ['ds','y'].")
    st.stop()

# ===== helpers de datas
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

# série mensal contínua (MS)
_idx = _ts["ds"].map(_label_to_month_start)
s_monthly = (
    pd.Series(_ts.loc[_idx.notna(), "y"].astype(float).to_numpy(), index=_idx[_idx.notna()])
      .sort_index().asfreq("MS").interpolate("linear").bfill().ffill()
)

# ===== snapshot das grades (para ligar/desligar modo rápido)
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
    # só é aplicado quando FAST_MODE=True
    if hasattr(module, "CROSTON_ALPHAS"):        module.CROSTON_ALPHAS        = [0.1, 0.3]
    if hasattr(module, "SBA_ALPHAS"):            module.SBA_ALPHAS            = [0.1, 0.3]
    if hasattr(module, "TSB_ALPHA_GRID"):        module.TSB_ALPHA_GRID        = [0.1, 0.3]
    if hasattr(module, "TSB_BETA_GRID"):         module.TSB_BETA_GRID         = [0.1, 0.3]
    if hasattr(module, "RF_LAGS_GRID"):          module.RF_LAGS_GRID          = [6]
    if hasattr(module, "RF_N_ESTIMATORS_GRID"):  module.RF_N_ESTIMATORS_GRID  = [200]
    if hasattr(module, "RF_MAX_DEPTH_GRID"):     module.RF_MAX_DEPTH_GRID     = [None, 10]
    if hasattr(module, "SARIMA_GRID"):
        module.SARIMA_GRID = {"p":[0,1], "d":[0,1], "q":[0,1], "P":[0], "D":[0,1], "Q":[0]}

# ===== form (config)
with st.form(key="previsao_form"):
    st.sidebar.header("⚙️ Configurações")
    HORIZON = st.sidebar.selectbox("Horizonte (meses)", [6,8,12], index=0)

    FAST_MODE = st.sidebar.toggle(
        "Modo rápido (grade reduzida)",
        value=False,
        help="Quando desligado (padrão), usa as grades integrais do pipeline."
    )

    # Controles de LOG/BOOTSTRAP
    DO_LOG = st.sidebar.checkbox("Aplicar log", value=True)
    DO_BOOTSTRAP = st.sidebar.checkbox("Ativar bootstrap", value=True)

    # Grava no session_state para outros pontos da página
    ss["FAST_MODE"] = bool(FAST_MODE)
    ss["HORIZON"] = int(HORIZON)

    # *** Ordem garantida: restaura grades completas e só depois aplica rápido, se for o caso
    restore_full_grids(pipe)
    if FAST_MODE:
        apply_fast_grids(pipe)
        max_boot, default_boot = 10, 10
    else:
        max_boot, default_boot = 50, 20

    if DO_BOOTSTRAP:
        N_BOOTSTRAP = st.sidebar.slider(
            "Réplicas do bootstrap",
            min_value=1, max_value=max_boot, value=default_boot, step=1,
            help="Reduzir as réplicas acelera a execução, mas pode não encontrar o melhor modelo."
        )
        st.sidebar.caption(
            "⚠️ **Menos réplicas** → roda mais rápido, mas **pode não encontrar** o melhor modelo."
        )
    else:
        N_BOOTSTRAP = 0

    # Exibição (informativa) do tamanho da grade
    def _len(x):
        try: return len(x)
        except Exception: return 0
    base = 0
    base += _len(getattr(pipe,"CROSTON_ALPHAS",[]))
    base += _len(getattr(pipe,"SBA_ALPHAS",[]))
    base += _len(getattr(pipe,"TSB_ALPHA_GRID",[])) * _len(getattr(pipe,"TSB_BETA_GRID",[]))
    base += _len(getattr(pipe,"RF_LAGS_GRID",[])) * _len(getattr(pipe,"RF_N_ESTIMATORS_GRID",[])) * _len(getattr(pipe,"RF_MAX_DEPTH_GRID",[]))
    if hasattr(pipe,"SARIMA_GRID"):
        g = pipe.SARIMA_GRID
        base += _len(g.get("p",[]))*_len(g.get("d",[]))*_len(g.get("q",[]))*_len(g.get("P",[]))*_len(g.get("D",[]))*_len(g.get("Q",[]))

    rounds_desc = ["original"]
    if DO_LOG: rounds_desc.append("log")
    if DO_BOOTSTRAP: rounds_desc.append(f"bootstrap×{N_BOOTSTRAP}")
    st.caption(f"Configuração: rápido={'ON' if FAST_MODE else 'OFF'} | combinações≈{max(1,base)} | rodadas: {', '.join(rounds_desc)}")

    submitted = st.form_submit_button("▶️ Rodar previsão", type="primary", disabled=ss.is_running)

# ===== console filtrado + progresso acumulativo (rodadas/famílias)
prog = st.progress(0)
prog_text = st.empty()
log_box = st.expander("📜 Console (passos essenciais)", expanded=False)
log_area = log_box.empty()
_raw_lines: list[str] = []

_WHITELIST = [
    r"==== PIPELINE INICIADO ====",
    r"^Params:",
    r"Realizando testes da série ORIGINAL",
    r"Realizando testes da série .*log",
    r"^→\s*Croston", r"^→\s*SBA", r"^→\s*TSB", r"^→\s*RF", r"^→\s*SARIMAX",
    r"^•\s*Testes — bootstrap", r"bootstrap",
    r"===== CAMPEÃO",
    r"==== PIPELINE FINALIZADO ====",
    r"(ERROR|EXCEPTION|Traceback)",
]
_WHITELIST_RE = [re.compile(p, re.IGNORECASE) for p in _WHITELIST]
def _show_log_filtered():
    filtered = []
    for ln in _raw_lines[-600:]:
        s = str(ln).strip()
        if any(rx.search(s) for rx in _WHITELIST_RE):
            filtered.append(s)
    log_area.text("\n".join(filtered[-250:]))
def _push_log(line: str):
    _raw_lines.append(str(line)); _show_log_filtered()
_original_log = getattr(pipe, "log", print)

def _len_safe(x):
    try: return len(x)
    except Exception: return 0
FAMILY_WEIGHT = {
    "CROSTON": _len_safe(getattr(pipe, "CROSTON_ALPHAS", [])),
    "SBA":     _len_safe(getattr(pipe, "SBA_ALPHAS", [])),
    "TSB":     (_len_safe(getattr(pipe, "TSB_ALPHA_GRID", [])) * _len_safe(getattr(pipe, "TSB_BETA_GRID", []))),
    "RF":      (_len_safe(getattr(pipe, "RF_LAGS_GRID", [])) * _len_safe(getattr(pipe, "RF_N_ESTIMATORS_GRID", [])) * _len_safe(getattr(pipe, "RF_MAX_DEPTH_GRID", []))),
    "SARIMAX": 0,
}
if hasattr(pipe, "SARIMA_GRID"):
    g = pipe.SARIMA_GRID
    FAMILY_WEIGHT["SARIMAX"] = _len_safe(g.get("p",[]))*_len_safe(g.get("d",[]))*_len_safe(g.get("q",[]))*_len_safe(g.get("P",[]))*_len_safe(g.get("D",[]))*_len_safe(g.get("Q",[]))
FAMILY_ORDER = [f for f in ["CROSTON","SBA","TSB","RF","SARIMAX"] if FAMILY_WEIGHT[f] > 0]
ROUND_WEIGHT = sum(FAMILY_WEIGHT[f] for f in FAMILY_ORDER)

TOTAL_ROUNDS = 1 + (1 if DO_LOG else 0) + (N_BOOTSTRAP if DO_BOOTSTRAP else 0)
TOTAL_WEIGHT = max(1, ROUND_WEIGHT * TOTAL_ROUNDS)
_state = {"round_idx": 0, "family_pos": 0, "weight_done": 0}
_FAM_PAT = {
    "CROSTON": re.compile(r"^→\s*Croston", re.IGNORECASE),
    "SBA":     re.compile(r"^→\s*SBA", re.IGNORECASE),
    "TSB":     re.compile(r"^→\s*TSB", re.IGNORECASE),
    "RF":      re.compile(r"^→\s*RF", re.IGNORECASE),
    "SARIMAX": re.compile(r"^→\s*SARIMAX", re.IGNORECASE),
}
_START_ORIG = re.compile(r"Realizando testes da série ORIGINAL", re.IGNORECASE)
_START_LOG  = re.compile(r"Realizando testes da série .*log", re.IGNORECASE)
_BOOT_ANY   = re.compile(r"bootstrap", re.IGNORECASE)
def _emit_progress():
    pct = int(round(100 * _state["weight_done"] / float(TOTAL_WEIGHT)))
    prog.progress(min(100, max(0, pct))); return pct
def _advance_family(fam: str):
    w = FAMILY_WEIGHT.get(fam, 0)
    if w > 0: _state["weight_done"] = min(TOTAL_WEIGHT, _state["weight_done"] + w)
def _maybe_start_round(line: str):
    if _START_ORIG.search(line): _state["round_idx"] = 0; _state["family_pos"] = 0; return
    if DO_LOG and _START_LOG.search(line):  _state["round_idx"] = 1; _state["family_pos"] = 0; return
    if DO_BOOTSTRAP and _BOOT_ANY.search(line) and _state["round_idx"] < (1 + (1 if DO_LOG else 0) + 1):
        _state["round_idx"] = max(_state["round_idx"], 1 + (1 if DO_LOG else 0))
        _state["family_pos"] = 0
        return
def _patched_log(msg: str):
    s = str(msg).strip()
    try:
        _push_log(s); _maybe_start_round(s)
        for fam in FAMILY_ORDER:
            if _FAM_PAT[fam].search(s):
                _advance_family(fam)
                _state["family_pos"] = (_state["family_pos"] + 1) % max(1, len(FAMILY_ORDER))
                if _state["family_pos"] == 0 and DO_BOOTSTRAP and _state["round_idx"] >= (1 + (1 if DO_LOG else 0)):
                    _state["round_idx"] = min((1 + (1 if DO_LOG else 0)) + max(0, N_BOOTSTRAP), _state["round_idx"] + 1)
                pct = _emit_progress()
                prog_text.write(f"{pct}% — {s}" if pct < 100 else "100% — concluído")
                break
    except Exception:
        pass
    _original_log(msg)
def _wire_progress():
    if hasattr(pipe, "log"): pipe.log = _patched_log

# ===== execução (sempre reroda ao clicar)
if submitted and not ss.is_running:
    ss.is_running = True
    ss.last_result = None
    try:
        _wire_progress()

        # Painel de depuração — parâmetros que SERÃO passados
        debug_info = {
            "horizon": int(HORIZON),
            "seasonal_period": 12,
            "do_original": True,
            "do_log": bool(DO_LOG),
            "do_bootstrap": bool(DO_BOOTSTRAP),
            "n_bootstrap": int(N_BOOTSTRAP),
            "FAST_MODE": bool(FAST_MODE),
            "grid_sizes": {
                "Croston": len(getattr(pipe, "CROSTON_ALPHAS", [])),
                "SBA": len(getattr(pipe, "SBA_ALPHAS", [])),
                "TSB": (
                    len(getattr(pipe, "TSB_ALPHA_GRID", [])) *
                    len(getattr(pipe, "TSB_BETA_GRID", []))
                ),
                "RF": (
                    len(getattr(pipe, "RF_LAGS_GRID", [])) *
                    len(getattr(pipe, "RF_N_ESTIMATORS_GRID", [])) *
                    len(getattr(pipe, "RF_MAX_DEPTH_GRID", []))
                ),
                "SARIMAX": (
                    len(getattr(pipe, "SARIMA_GRID", {}).get("p", [])) *
                    len(getattr(pipe, "SARIMA_GRID", {}).get("d", [])) *
                    len(getattr(pipe, "SARIMA_GRID", {}).get("q", [])) *
                    len(getattr(pipe, "SARIMA_GRID", {}).get("P", [])) *
                    len(getattr(pipe, "SARIMA_GRID", {}).get("D", [])) *
                    len(getattr(pipe, "SARIMA_GRID", {}).get("Q", []))
                )
            }
        }
        with st.expander("🔎 Depuração — parâmetros efetivos", expanded=False):
            st.json(debug_info)

        t0 = time.perf_counter()
        with st.spinner("Executando pipeline…"):
            _stdout, _stderr = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(_stdout), contextlib.redirect_stderr(_stderr):
                resultados = pipe.run_full_pipeline(
                    data_input=s_monthly,
                    sheet_name=None, date_col=None, value_col=None,
                    horizon=int(HORIZON), seasonal_period=12,
                    do_original=True,
                    do_log=bool(DO_LOG),
                    do_bootstrap=bool(DO_BOOTSTRAP),
                    n_bootstrap=int(N_BOOTSTRAP),
                    bootstrap_block=24,
                    save_dir=None,
                )
            if _stdout.getvalue(): _push_log(_stdout.getvalue())
            if _stderr.getvalue(): _push_log(_stderr.getvalue())
        elapsed = time.perf_counter() - t0
        prog.progress(100); prog_text.write("100% — concluído")
        st.caption(f"⏱️ Tempo de execução: {elapsed:.1f}s")

        # =========================================================
        # Salva resultado “bruto” (experimentos) e artefatos úteis
        # =========================================================
        ss.last_result = resultados

        # Experimentos (opcional para a página 07 e download)
        exp_df = None
        if isinstance(resultados, pd.DataFrame) and not resultados.empty:
            exp_df = resultados.copy()
        elif hasattr(resultados, "attrs"):
            for key in ["experiments", "experiments_df", "resultados", "table", "exp"]:
                val = resultados.attrs.get(key)
                if isinstance(val, pd.DataFrame) and not val.empty:
                    exp_df = val.copy(); break
        if exp_df is None and isinstance(resultados, dict):
            for key in ["experiments", "experiments_df", "resultados", "table", "exp"]:
                val = resultados.get(key)
                if isinstance(val, pd.DataFrame) and not val.empty:
                    exp_df = val.copy(); break

        if isinstance(exp_df, pd.DataFrame) and not exp_df.empty:
            ss["exp_table"] = exp_df.copy()
            ss["experiments_df"] = exp_df.copy()
            ss["experiments_csv"] = exp_df.to_csv(index=False).encode("utf-8")
            ss["experiments_ready"] = True
        else:
            ss["experiments_ready"] = False

    except Exception:
        st.error("Falha ao executar a previsão. Traceback abaixo:")
        st.code("\n".join(traceback.format_exc().splitlines()), language="text")
    finally:
        if hasattr(pipe,"log"): pipe.log = _original_log
        ss.is_running = False

# ===== render
res = ss.get("last_result")
if res is not None:
    # ----- Campeão
    champ = {}
    if hasattr(res, "attrs"):
        champ = res.attrs.get("champion", {}) or {}
    elif isinstance(res, dict):
        champ = res.get("champion", {}) or {}

    modelo_nome = champ.get("model", "Desconhecido")
    st.subheader(f"🏆 Modelo Campeão: {modelo_nome}" + (" (Modo rápido)" if st.session_state.get('FAST_MODE', False) else ""))

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

    # ===== PREVISÃO do pipeline (sem fallback)
    forecast = None
    if hasattr(res, "attrs"):
        forecast = res.attrs.get("forecast_df", None)
        if forecast is None:
            # se o pipeline reportou erro, exibir claramente
            f_err = res.attrs.get("forecast_error")
            if f_err:
                st.error(f"Pipeline não conseguiu gerar previsão do campeão: {f_err}")
            else:
                st.error("Pipeline não retornou 'forecast_df'.")
    elif isinstance(res, dict):
        forecast = res.get("forecast_df", None)

    if forecast is None:
        st.stop()

    # Normaliza para ['ds','y']
    if isinstance(forecast, pd.DataFrame):
        if {"ds","yhat"}.issubset(forecast.columns):
            forecast_df_std = forecast.rename(columns={"yhat":"y"})[["ds","y"]].copy()
        elif {"ds","y"}.issubset(forecast.columns):
            forecast_df_std = forecast[["ds","y"]].copy()
        else:
            st.error("Formato inesperado de 'forecast_df'. Esperado colunas ['ds','y'] ou ['ds','yhat'].")
            st.stop()
    elif isinstance(forecast, pd.Series):
        forecast_df_std = pd.DataFrame({"ds": forecast.index, "y": forecast.values})
    else:
        st.error("Tipo inesperado de 'forecast_df' retornado pelo pipeline.")
        st.stop()

    # ===== Gráfico Real + Previsão (Altair)
    st.subheader("📈 Histórico + Previsão")
    hist_df = pd.DataFrame({"ds": s_monthly.index, "valor": s_monthly.values, "série": "Real"})
    prev_df = pd.DataFrame({"ds": pd.to_datetime(forecast_df_std["ds"]), "valor": forecast_df_std["y"].values, "série": "Previsão"})
    plot_long = pd.concat([hist_df, prev_df], ignore_index=True)

    chart = (
        alt.Chart(plot_long.reset_index(drop=True))
        .mark_line()
        .encode(
            x=alt.X("ds:T", title="Mês"),
            y=alt.Y("valor:Q", title="Quantidade"),
            color=alt.Color(
                "série:N",
                scale=alt.Scale(domain=["Real","Previsão"], range=["#1e3a8a", "#60a5fa"]),
                legend=alt.Legend(title=None, orient="top")
            ),
            tooltip=[alt.Tooltip("ds:T", title="Período"),
                     alt.Tooltip("série:N", title="Série"),
                     alt.Tooltip("valor:Q", title="Valor", format=",.0f")]
        )
        .properties(height=280, width="container")
        .interactive()
    )
    st.altair_chart(chart, use_container_width=True)

    # ===== Experimentos — botão de download (sem mostrar tabela)
    st.subheader("📦 Experimentos")
    try:
        exp_df = res.reset_index(drop=True)
        st.session_state["experiments_df"] = exp_df.copy()
        csv_bytes = exp_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Baixar todos os experimentos (CSV)",
            data=csv_bytes,
            file_name="experimentos_previsao.csv",
            mime="text/csv",
            help="Contém todas as combinações testadas com métricas e parâmetros."
        )
    except Exception:
        st.info("Resultados dos experimentos indisponíveis para exportação.")

    # =============================
    # 🔗 Próximos passos
    # =============================
    st.divider()
    st.subheader("➡️ Próximos passos")

    # CSS: afinar botão primário
    st.markdown("""
    <style>
    div.stButton > button[kind="primary"]{
        padding: 0.45rem 0.9rem !important;
        font-size: 0.95rem !important;
        border-radius: 10px !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # Botão Salvar para MPS (usa exatamente o horizonte selecionado)
    can_save = forecast_df_std is not None and len(forecast_df_std) > 0
    if st.button("Salvar previsão para o MPS", type="primary", disabled=not can_save):
        st.session_state["forecast_df"] = forecast_df_std.copy()
        st.session_state["forecast_h"] = int(ss.get("HORIZON", HORIZON))
        st.session_state["forecast_committed"] = True
        st.success("Previsão salva para o MPS. Agora você pode ir aos Inputs do MPS.")

    st.markdown("---")
    st.page_link("pages/05_Inputs_MPS.py", label="➡️ Ir para Inputs do MPS")

    if not st.session_state.get("forecast_committed", False):
        st.info("Clique em **Salvar previsão para o MPS** antes de avançar aos Inputs.", icon="ℹ️")
