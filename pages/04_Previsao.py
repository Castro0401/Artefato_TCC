# pages/04_Previsao.py
from __future__ import annotations
import time
import numpy as np
import pandas as pd
import streamlit as st

# ------------------------------
# Import do pipeline central
# ------------------------------
# Estrutura esperada: core/__init__.py  e  core/pipeline.py
from core import pipeline as pl

# ------------------------------
# Título
# ------------------------------
st.title("🔮 Passo 2 — Previsão de Demanda")

# ------------------------------
# Guarda de etapa (precisa do Upload)
# ------------------------------
if "ts_df_norm" not in st.session_state:
    st.warning("Preciso da série do Passo 1 (Upload) antes de continuar.")
    st.page_link("pages/01_Upload.py", label="Ir para o Passo 1 — Upload", icon="📤")
    st.stop()

# ------------------------------
# Helpers para rótulos
# ------------------------------
_PT = {1:"Jan",2:"Fev",3:"Mar",4:"Abr",5:"Mai",6:"Jun",7:"Jul",8:"Ago",9:"Set",10:"Out",11:"Nov",12:"Dez"}
_REV_PT = {v:k for k, v in _PT.items()}

def label_pt(ts: pd.Timestamp) -> str:
    return f"{_PT[ts.month]}/{str(ts.year)[-2:]}"

def to_period(s: str) -> pd.Period:
    # "Set/25" -> Period('2025-09','M'); YYYY-MM-DD também funciona
    try:
        return pd.to_datetime(s, dayfirst=True).to_period("M")
    except Exception:
        mon = s[:3].capitalize()
        yy = int(s[-2:]) + 2000
        month_num = _REV_PT.get(mon, None)
        if month_num is None:
            raise ValueError(f"Formato de mês inválido: {s}")
        return pd.Period(freq="M", year=yy, month=month_num)

# ------------------------------
# Série histórica do Passo 1
# (não altere o session_state aqui)
# ------------------------------
hist = st.session_state["ts_df_norm"].copy()  # ['ds','y'] com labels tipo 'Set/25'
hist_work = hist.copy()
hist_work["p"] = hist_work["ds"].apply(to_period)
hist_work = hist_work.sort_values("p").reset_index(drop=True)

# ------------------------------
# Sidebar — Parâmetros
# ------------------------------
with st.sidebar:
    st.header("⚙️ Configurações da previsão")

    # Horizonte e período sazonal (não resetam upload)
    last_h = int(st.session_state.get("forecast_h", 6))
    horizon = st.selectbox(
        "Horizonte (meses)",
        options=[6, 8, 12],
        index={6:0, 8:1, 12:2}.get(last_h, 0),
        help="O MPS usará este mesmo horizonte."
    )
    seasonal_period = st.number_input(
        "Período sazonal (m)",
        min_value=1, max_value=24, value=int(st.session_state.get("seasonal_period", 12)), step=1,
        help="Para série mensal, geralmente 12."
    )

    st.divider()
    st.subheader("Transformações")
    do_log = st.checkbox("Aplicar Log + ε (auto)", value=True,
                         help="Aplica log com ε e shift automáticos para estabilizar a variância.")
    do_bootstrap = st.checkbox("Ativar Bootstrap FPP", value=False,
                               help="Gera réplicas sintéticas via Box-Cox + STL + bootstrap em blocos.")

    # Parâmetros do bootstrap (mostrados apenas se ativo)
    if do_bootstrap:
        st.caption("**Bootstrap FPP** — Réplicas reconstroem séries sintéticas preservando tendência e sazonalidade.")
        st.markdown(
            "- **Réplicas**: quantas séries sintéticas gerar (mais réplicas = avaliação mais robusta, mais tempo).\n"
            "- **Tamanho do bloco**: tamanho dos blocos contíguos ao reamostrar resíduos (blocos maiores preservam mais autocorrelação)."
        )
        n_bootstrap = st.slider("Réplicas (n)", min_value=5, max_value=100, value=20, step=5)
        bootstrap_block = st.slider("Tamanho do bloco", min_value=3, max_value=48, value=24, step=1)
    else:
        n_bootstrap = 0
        bootstrap_block = 0

    st.divider()
    fast_mode = st.toggle("🏁 Modo Rápido", value=False,
                          help="Reduz grades (SARIMAX/RF/TSB) e desliga bootstrap para acelerar (resultado menos robusto).")

# Guarda sazonal no estado para manter padrão na próxima visita
st.session_state["seasonal_period"] = int(seasonal_period)

# ------------------------------
# Ajuste dinâmico das grades (Modo Rápido)
# ------------------------------
# Observação: fazemos monkey-patch nas constantes do pipeline para acelerar em tempo de execução.
# Isso não altera o arquivo no disco; só vale para esta sessão.
original_grids_snapshot = None
def _apply_fast_mode():
    global original_grids_snapshot
    if original_grids_snapshot is not None:
        return  # já aplicado

    # snapshot para possível restauração futura
    original_grids_snapshot = {
        "CROSTON_ALPHAS": pl.CROSTON_ALPHAS[:],
        "SBA_ALPHAS": pl.SBA_ALPHAS[:],
        "TSB_ALPHA_GRID": pl.TSB_ALPHA_GRID[:],
        "TSB_BETA_GRID": pl.TSB_BETA_GRID[:],
        "RF_LAGS_GRID": pl.RF_LAGS_GRID[:],
        "RF_N_ESTIMATORS_GRID": pl.RF_N_ESTIMATORS_GRID[:],
        "RF_MAX_DEPTH_GRID": pl.RF_MAX_DEPTH_GRID[:],
        "SARIMA_GRID": {k: v[:] for k, v in pl.SARIMA_GRID.items()},
    }

    # Grades compactas
    pl.CROSTON_ALPHAS = [0.1, 0.3]
    pl.SBA_ALPHAS = [0.1, 0.3]
    pl.TSB_ALPHA_GRID = [0.3]
    pl.TSB_BETA_GRID = [0.3]
    pl.RF_LAGS_GRID = [6]              # só lags 1..6
    pl.RF_N_ESTIMATORS_GRID = [150]    # menos árvores
    pl.RF_MAX_DEPTH_GRID = [None, 5]
    pl.SARIMA_GRID = {"p":[0,1], "d":[0,1], "q":[0,1], "P":[0,1], "D":[0,1], "Q":[0]}
    # Modo rápido também desliga bootstrap para acelerar
    st.session_state["_force_disable_bootstrap"] = True

def _restore_grids_if_needed():
    global original_grids_snapshot
    if original_grids_snapshot is None:
        return
    pl.CROSTON_ALPHAS = original_grids_snapshot["CROSTON_ALPHAS"]
    pl.SBA_ALPHAS = original_grids_snapshot["SBA_ALPHAS"]
    pl.TSB_ALPHA_GRID = original_grids_snapshot["TSB_ALPHA_GRID"]
    pl.TSB_BETA_GRID = original_grids_snapshot["TSB_BETA_GRID"]
    pl.RF_LAGS_GRID = original_grids_snapshot["RF_LAGS_GRID"]
    pl.RF_N_ESTIMATORS_GRID = original_grids_snapshot["RF_N_ESTIMATORS_GRID"]
    pl.RF_MAX_DEPTH_GRID = original_grids_snapshot["RF_MAX_DEPTH_GRID"]
    pl.SARIMA_GRID = original_grids_snapshot["SARIMA_GRID"]
    original_grids_snapshot = None
    st.session_state.pop("_force_disable_bootstrap", None)

if fast_mode:
    _apply_fast_mode()
else:
    _restore_grids_if_needed()

# Se o modo rápido estiver ativo, força desativar bootstrap na chamada
if st.session_state.get("_force_disable_bootstrap", False):
    do_bootstrap = False
    n_bootstrap = 0
    bootstrap_block = 0

# ------------------------------
# UI: botão principal
# ------------------------------
run = st.button("▶️ Rodar previsão", type="primary")

# ------------------------------
# Execução do pipeline com barra de progresso
# ------------------------------
forecast_df = None
champion = None

if run:
    # Container visual da barra de progresso e rótulo curto
    progress_box = st.container()
    pbar = progress_box.progress(0, text="Iniciando…")

    # Monkey-patch do logger para alimentar a barra (sem mostrar textos de experimento)
    # A barra vai avançando por "fases" aproximadas com base nos logs do pipeline.
    phases = {
        "PIPELINE INICIADO": 0.05,
        "Realizando testes da série ORIGINAL": 0.15,
        "Preparando transformação LOG": 0.20,
        "Realizando testes da série LOG": 0.35,
        "Geração das réplicas sintéticas": 0.40,
        "réplicas geradas": 0.45,
        "SARIMAX": 0.70,        # durante SARIMAX subiremos mais
        "RandomForest": 0.55,
        "TSB": 0.50,
        "Croston": 0.30,
        "SBA": 0.40,
        "CAMPEÃO": 0.92,
        "PIPELINE FINALIZADO": 1.00,
    }
    # Como o pipeline emite logs com vários textos, vamos detectar trechos-chave:
    def _progress_from_msg(msg: str) -> float:
        m = msg.upper()
        for key, val in phases.items():
            if key in m:
                return val
        # ajustes finos: durante loops doravante, incrementos pequenos
        if "PROGRESSO" in m or "RÉPLICA" in m:
            return min(pbar_value + 0.02, 0.9)
        return None

    # Estado local do progresso
    pbar_value = 0.0
    def _bar_update(x: float | None, fallback: float = 0.01):
        nonlocal pbar_value
        if x is None:
            # pequeno incremento para não ficar parado
            x = min(pbar_value + fallback, 0.98)
        if x > pbar_value:
            pbar_value = x
            pbar.progress(min(1.0, pbar_value))

    # Captura o logger original
    pl_log_original = pl.log

    def _hooked_log(msg: str):
        # Atualiza barra com base nos eventos, mas não mostra texto dos experimentos
        try:
            x = _progress_from_msg(str(msg))
            _bar_update(x)
        except Exception:
            pass
        # Mantém o logger original (útil para debug no terminal)
        pl_log_original(msg)

    # Aponta o logger do pipeline para nosso hook
    pl.log = _hooked_log

    try:
        # Monta DataFrame para o loader do pipeline (ele aceita df com ds/y)
        df_in = hist_work[["p", "y"]].copy()
        df_in = df_in.rename(columns={"p": "ds"})  # o pipeline aceita 'ds'/'y'

        t0 = time.time()
        resultados = pl.run_full_pipeline(
            data_input=df_in,
            sheet_name=None, date_col="ds", value_col="y",
            horizon=int(horizon),
            seasonal_period=int(seasonal_period),
            do_original=True,
            do_log=bool(do_log),
            do_bootstrap=bool(do_bootstrap),
            n_bootstrap=int(n_bootstrap) if do_bootstrap else 0,
            bootstrap_block=int(bootstrap_block) if do_bootstrap else 0,
            save_dir=None  # sem persistência em disco pela UI
        )
        # Força a barra para ~100% ao concluir
        _bar_update(0.99)
        time.sleep(0.1)
        _bar_update(1.0)

        # Recupera forecast real e campeão
        forecast_df = resultados.attrs.get("forecast_df", pd.DataFrame())
        champion = resultados.attrs.get("champion", {})

    except Exception as e:
        pbar.progress(0.0, text="Erro ao rodar previsão.")
        st.exception(e)
    finally:
        # Restaura o logger original
        pl.log = pl_log_original
        # Remove a barra (limpa o container) para dar lugar às saídas
        progress_box.empty()

# ------------------------------
# Exibição da previsão + métricas
# ------------------------------
if isinstance(forecast_df, pd.DataFrame) and len(forecast_df):
    # Persistência para MPS
    st.session_state["forecast_df"] = forecast_df.copy()
    st.session_state["forecast_h"] = int(horizon)
    st.session_state["forecast_committed"] = True

    # Gráfico histórico + previsão
    st.subheader(f"Histórico + Previsão ({horizon} meses)")
    hist_plot = hist_work.assign(ts=hist_work["p"].dt.to_timestamp())[["ts","y"]]
    last_ts = hist_plot["ts"].iloc[-1]
    fut_ts = pd.date_range(last_ts + pd.offsets.MonthBegin(1), periods=int(horizon), freq="MS")
    fut_plot = pd.DataFrame({"ts": fut_ts, "y": forecast_df["y"]})

    chart_df = pd.concat([
        hist_plot.assign(tipo="Histórico"),
        fut_plot.assign(tipo="Previsão")
    ]).set_index("ts")

    st.line_chart(chart_df, x=None, y="y", color="tipo", height=340, use_container_width=True)

    # Métricas do campeão (exibimos sMAPE como principal em %)
    left, right = st.columns([2,1])
    with right:
        st.subheader("Resumo do campeão")
        if isinstance(champion, dict) and champion:
            st.metric("Modelo", f"{champion.get('model','—')}")
            st.metric("Preprocess", f"{champion.get('preprocess','—')}")
            smape_val = champion.get("sMAPE", None)
            if smape_val is not None:
                # já vem em %, garantimos formatação
                try:
                    st.metric("sMAPE (holdout)", f"{float(smape_val):.2f} %")
                except Exception:
                    st.metric("sMAPE (holdout)", str(smape_val))
        else:
            st.caption("Campeão indisponível (ver logs).")

    st.subheader("Previsão (tabela)")
    st.dataframe(forecast_df, use_container_width=True, height=260)

    st.divider()
    st.page_link("pages/05_Inputs_MPS.py", label="➡️ Usar esta previsão e ir para Inputs do MPS", icon="🗓️", disabled=False)

else:
    st.info("Clique em **Rodar previsão** para gerar o horizonte selecionado.")
    st.page_link("pages/05_Inputs_MPS.py", label="➡️ Ir para Inputs do MPS", icon="🗓️", disabled=True)
