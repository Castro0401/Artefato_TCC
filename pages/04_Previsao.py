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
# Import do pipeline
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import core.pipeline as pl
except Exception as e:
    st.error(f"Não consegui importar core.pipeline: {e}")
    st.stop()

# -----------------------------------------------------------------------------
# Helpers de data (conversão 'Set/25' → Period → Timestamp)
# -----------------------------------------------------------------------------
_PT = {1:"Jan",2:"Fev",3:"Mar",4:"Abr",5:"Mai",6:"Jun",7:"Jul",8:"Ago",9:"Set",10:"Out",11:"Nov",12:"Dez"}
_REV_PT = {v:k for k, v in _PT.items()}

def to_period(label: str) -> pd.Period:
    # aceita "Set/25" ou datas YYYY-MM-DD
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
    """
    Converte df ['ds','y'] (onde 'ds' é label tipo 'Set/25') em
    ['ds','y'] com 'ds' Timestamp mensal (MS), adequado ao pipeline.
    """
    tmp = df_upload.copy()
    tmp["p"] = tmp["ds"].apply(to_period)
    tmp = tmp.sort_values("p")
    tmp["ds"] = tmp["p"].dt.to_timestamp(how="start")
    return tmp[["ds", "y"]].dropna(subset=["ds"])

# -----------------------------------------------------------------------------
# Sidebar — parâmetros
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
        st.caption("Réplicas ↑ → mais robustez\n\nTamanho do bloco ↑ → preserva mais autocorrelação.")
        n_boot = st.slider("Réplicas (bootstrap)", 1, 100, 20, step=1)
        block = st.slider("Tamanho do bloco", 3, 48, 24, step=1)
    else:
        n_boot, block = 0, 0

    fast_mode = st.toggle("🏎️ Modo rápido", value=False,
                          help="Reduz custo experimental (principalmente no bootstrap).")

# aplica modo rápido apenas dosando bootstrap (não mexe na lógica interna)
if fast_mode and use_boot:
    n_boot = min(n_boot, 5)
    block = min(block, 12)

# -----------------------------------------------------------------------------
# UI: botão e barra (sem textos auxiliares)
# -----------------------------------------------------------------------------
bar_slot = st.empty()
run = st.button("▶️ Rodar previsão", type="primary")

def run_with_progress(df_in: pd.DataFrame):
    """
    Executa o pipeline com uma barra de progresso silenciosa.
    A barra avança usando os logs do pipeline, mas sem exibir textos.
    """
    bar = bar_slot.progress(0)
    pct = {"v": 0}

    def tick(step: int = 1):
        pct["v"] = min(95, pct["v"] + step)
        bar.progress(pct["v"])

    # intercepta logs para apenas progredir (sem renderizar nada na tela)
    original_log = pl.log
    def ui_log(msg: str):
        low = msg.lower()
        if "pipeline iniciado" in low: tick(3)
        elif "original" in low and "realizando testes" in low: tick(6)
        elif "transformação log" in low or "log-transformada" in low: tick(6)
        elif "bootstrap" in low and ("gerando" in low or "réplicas" in low): tick(10)
        elif "croston" in low or "sba" in low or "tsb" in low: tick(2)
        elif "randomforest" in low: tick(2)
        elif "sarimax" in low: tick(2)
        elif "pipeline finalizado" in low or "concluídos testes" in low: tick(5)
        else:
            tick(1)  # fallback leve
        # não escreve nada na interface; só mantém original_log por segurança
        try:
            original_log(msg)
        except Exception:
            pass

    pl.log = ui_log
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

if run:
    try:
        df_in = df_upload_to_pipeline(st.session_state["ts_df_norm"])
        with st.spinner("Executando…"):
            resultados = run_with_progress(df_in)

        champ = resultados.attrs.get("champion", {})
        st.success("✅ Experimentos concluídos!")

        if champ:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Modelo campeão", str(champ.get("model", "—")))
            # sMAPE já é % no pipeline; aqui mostramos com 2 casas
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

        # Atualiza estado mínimo (sem gravar forecast ainda)
        st.session_state["forecast_h"] = int(horizon)
        st.session_state["forecast_committed"] = False

    except Exception as e:
        st.error(f"Ocorreu um erro durante a execução: {e}")
