# pages/04_Previsao.py
from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd
import streamlit as st

# -----------------------------------------------------------------------------
# Título
# -----------------------------------------------------------------------------
st.title("🔮 Passo 2 — Previsão de Demanda")

# -----------------------------------------------------------------------------
# Guardas de etapa
# -----------------------------------------------------------------------------
if "ts_df_norm" not in st.session_state:
    st.warning("Preciso da série do Passo 1 (Upload) antes de continuar.")
    st.page_link("pages/01_Upload.py", label="Ir para o Passo 1 — Upload", icon="📤")
    st.stop()

# -----------------------------------------------------------------------------
# Importa core.pipeline
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import core.pipeline as pl
except Exception as e:
    st.error(f"Falha ao importar core.pipeline: {e}")
    st.stop()

# -----------------------------------------------------------------------------
# Helpers: converte labels 'Set/25' -> Timestamp MS
# -----------------------------------------------------------------------------
_PT = {1:"Jan",2:"Fev",3:"Mar",4:"Abr",5:"Mai",6:"Jun",7:"Jul",8:"Ago",9:"Set",10:"Out",11:"Nov",12:"Dez"}
_REV_PT = {v:k for k, v in _PT.items()}

def to_period(label: str) -> pd.Period:
    try:
        return pd.to_datetime(label, dayfirst=True).to_period("M")
    except Exception:
        mon = label[:3].capitalize()
        yy = int(label[-2:]) + 2000
        m = _REV_PT.get(mon)
        if m is None:
            raise ValueError(f"Formato de mês inválido: {label}")
        return pd.Period(year=yy, month=m, freq="M")

def df_upload_to_pipeline(df_upload: pd.DataFrame) -> pd.DataFrame:
    tmp = df_upload.copy()
    tmp["p"] = tmp["ds"].apply(to_period)
    tmp = tmp.sort_values("p")
    tmp["ds"] = tmp["p"].dt.to_timestamp(how="start")
    return tmp[["ds", "y"]].dropna(subset=["ds"])

# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configurações")

    last_h = int(st.session_state.get("forecast_h", 6))
    horizon = st.selectbox("Horizonte (meses)", [6, 8, 12],
                           index=[6, 8, 12].index(last_h) if last_h in (6,8,12) else 0)

    seasonal_period = st.number_input("Período sazonal (m)", 1, 24, 12, step=1)

    st.markdown("**Pré-processamentos**")
    use_log = st.checkbox("Aplicar log + ε (auto)", value=True)
    use_boot = st.checkbox("Gerar séries sintéticas (bootstrap FPP)", value=True)

    if use_boot:
        st.caption("**Dicas**  \n• Réplicas ↑ → experimentos mais robustos (mais lento)  \n• Bloco ↑ → preserva mais autocorrelação (um pouco mais lento)")
        n_boot = st.slider("Réplicas (bootstrap)", 1, 100, 20, step=1)
        block = st.slider("Tamanho do bloco", 3, 48, 24, step=1)
    else:
        n_boot, block = 0, 0

    fast_mode = st.toggle("🏎️ Modo rápido", value=False,
                          help="Reduz o grid (SARIMAX/RF) e limita Bootstrap.")
    force_full = st.checkbox("Forçar modo completo (pode demorar muito)", value=False)

# -----------------------------------------------------------------------------
# Estimador de carga e auto-limite
# -----------------------------------------------------------------------------
def estimate_workload():
    # tamanhos dos grids atuais do módulo
    cro = len(getattr(pl, "CROSTON_ALPHAS", [0.1]))
    sba = len(getattr(pl, "SBA_ALPHAS", [0.1]))
    tsb = len(getattr(pl, "TSB_ALPHA_GRID", [0.3])) * len(getattr(pl, "TSB_BETA_GRID", [0.3]))
    rf  = (len(getattr(pl, "RF_LAGS_GRID", [6])) *
           len(getattr(pl, "RF_N_ESTIMATORS_GRID", [200])) *
           len(getattr(pl, "RF_MAX_DEPTH_GRID", [None])))
    sar = (len(getattr(pl, "SARIMA_GRID", {"p":[0],"d":[0],"q":[0],"P":[0],"D":[0],"Q":[0]})["p"]) *
           len(getattr(pl, "SARIMA_GRID")["d"]) *
           len(getattr(pl, "SARIMA_GRID")["q"]) *
           len(getattr(pl, "SARIMA_GRID")["P"]) *
           len(getattr(pl, "SARIMA_GRID")["D"]) *
           len(getattr(pl, "SARIMA_GRID")["Q"]))
    models_per_series = cro + sba + tsb + rf + sar
    n_series = 1 + (1 if use_log else 0) + (n_boot if use_boot else 0)
    return models_per_series * n_series

work = estimate_workload()
HARD_LIMIT = 1200  # limiar prático (~minutos em CPU média)

# aplica “rápido seguro” automaticamente se necessário
auto_fast = False
if (not fast_mode) and (not force_full) and work > HARD_LIMIT:
    auto_fast = True

def apply_fast_caps():
    # encolhe grids pesados (sem alterar o arquivo core)
    pl.CROSTON_ALPHAS[:] = [0.1, 0.3]
    pl.SBA_ALPHAS[:]      = [0.1, 0.3]
    pl.TSB_ALPHA_GRID[:]  = [0.3]
    pl.TSB_BETA_GRID[:]   = [0.3]
    pl.RF_LAGS_GRID[:]    = [6]
    pl.RF_N_ESTIMATORS_GRID[:] = [200]
    pl.RF_MAX_DEPTH_GRID[:]    = [None]
    pl.SARIMA_GRID["p"] = [0,1]
    pl.SARIMA_GRID["d"] = [0,1]
    pl.SARIMA_GRID["q"] = [0,1]
    pl.SARIMA_GRID["P"] = [0]
    pl.SARIMA_GRID["D"] = [0,1]
    pl.SARIMA_GRID["Q"] = [0]
    # limitar bootstrap
    return min(n_boot, 5), min(block, 12)

if fast_mode or auto_fast:
    n_boot, block = apply_fast_caps()

# -----------------------------------------------------------------------------
# Barra (sem textos) e Execução
# -----------------------------------------------------------------------------
bar_slot = st.empty()
run = st.button("▶️ Rodar previsão", type="primary")

def run_with_progress(df_in: pd.DataFrame):
    bar = bar_slot.progress(0)
    p = {"v": 0}

    def bump(d=1):
        p["v"] = min(96, p["v"] + d)
        bar.progress(p["v"])

    original_log = pl.log
    def silent_log(msg: str):
        # só avança a barra. Não escreve nada.
        low = msg.lower()
        if "pipeline iniciado" in low: bump(4)
        elif "realizando testes da série original" in low: bump(6)
        elif "transformação log" in low or "log-transformada" in low: bump(6)
        elif "bootstrap" in low and ("gerando" in low or "réplicas" in low): bump(10)
        elif "croston" in low or "sba" in low or "tsb" in low: bump(2)
        elif "randomforest" in low: bump(2)
        elif "sarimax" in low: bump(2)
        elif "pipeline finalizado" in low: bump(3)
        else:
            bump(1)
        # mantém o print no stdout para depuração, sem poluir UI
        try:
            original_log(msg)
        except Exception:
            pass

    pl.log = silent_log
    try:
        df_out = pl.run_full_pipeline(
            data_input=df_in,
            sheet_name=None,
            date_col="ds",
            value_col="y",
            horizon=int(horizon),
            seasonal_period=int(seasonal_period),
            do_original=True,
            do_log=bool(use_log),
            do_bootstrap=bool(use_boot),
            n_bootstrap=int(n_boot),
            bootstrap_block=int(block),
            save_dir=None,
        )
        bar.progress(100)
        return df_out
    finally:
        pl.log = original_log
        bar_slot.empty()

# -----------------------------------------------------------------------------
# Rodar
# -----------------------------------------------------------------------------
if run:
    try:
        df_in = df_upload_to_pipeline(st.session_state["ts_df_norm"])
        with st.spinner("Executando…"):
            resultados = run_with_progress(df_in)

        champ = resultados.attrs.get("champion", {})
        st.success("✅ Experimentos concluídos!")

        # KPIs e tabela só após finalizar
        if champ:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Modelo campeão", str(champ.get("model", "—")))
            try:
                c2.metric("sMAPE (%)", f"{float(champ.get('sMAPE', float('nan'))):.2f}")
            except Exception:
                c2.metric("sMAPE (%)", "—")
            try:
                c3.metric("MAE", f"{float(champ.get('MAE', float('nan'))):.2f}")
                c4.metric("RMSE", f"{float(champ.get('RMSE', float('nan'))):.2f}")
            except Exception:
                pass

        with st.expander("Ver tabela completa de experimentos"):
            st.dataframe(resultados, use_container_width=True, height=420)

        # estado mínimo p/ próximas páginas
        st.session_state["forecast_h"] = int(horizon)
        st.session_state["forecast_committed"] = False

        # aviso se o app precisou “segurar” a carga
        if auto_fast and not force_full:
            st.info("Rodei em **modo rápido seguro** porque a carga estimada estava muito alta. "
                    "Se quiser o conjunto completo, marque **Forçar modo completo** e rode novamente.")

    except Exception as e:
        bar_slot.empty()
        st.error(f"Ocorreu um erro durante a execução: {e}")
