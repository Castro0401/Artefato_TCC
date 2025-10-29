# pages/04_Previsao.py
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

# =========================================================
# Título
# =========================================================
st.title("🔮 Passo 2 — Previsão de Demanda")

# =========================================================
# Guarda de etapa: precisa do Upload (Passo 1)
# =========================================================
if "ts_df_norm" not in st.session_state:
    st.warning("Preciso da série do Passo 1 (Upload) antes de continuar.")
    st.page_link("pages/01_Upload.py", label="Ir para o Passo 1 — Upload", icon="📤")
    st.stop()

# =========================================================
# Import do pipeline (core/pipeline.py) como módulo
# =========================================================
ROOT = Path(__file__).resolve().parents[1]  # raiz do projeto (onde fica a pasta core/)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import core.pipeline as pl  # noqa: E402

# =========================================================
# Helpers para converter "Jan/25" -> Timestamp("2025-01-01")
# =========================================================
_PT = {1:"Jan",2:"Fev",3:"Mar",4:"Abr",5:"Mai",6:"Jun",7:"Jul",8:"Ago",9:"Set",10:"Out",11:"Nov",12:"Dez"}
_REV_PT = {v: k for k, v in _PT.items()}

def parse_label_pt(label: str) -> pd.Timestamp:
    # aceita "Set/25" etc.
    mon = label[:3].capitalize()
    yy = int(label[-2:]) + 2000
    mm = _REV_PT.get(mon)
    if mm is None:
        # fallback: tentar parser direto
        return pd.to_datetime(label, dayfirst=True)
    return pd.Timestamp(year=yy, month=mm, day=1)

# =========================================================
# Sidebar — parâmetros essenciais (simples)
# =========================================================
with st.sidebar:
    st.header("Configurar experimento")
    horizon = st.selectbox("Horizonte (meses)", [6, 8, 12], index=0)
    seasonal_period = st.number_input("Período sazonal (m)", min_value=1, value=12, step=1)
    use_log = st.checkbox("Usar transformação log + ε", value=True)
    use_boot = st.checkbox("Usar bootstrap FPP", value=False)

    if use_boot:
        st.caption("Bootstrap FPP gera réplicas sintéticas para robustez.")
        n_boot = st.slider("Réplicas (n)", 5, 40, 20, step=1)
        block = st.slider("Tamanho do bloco", 6, 48, 24, step=1)
    else:
        n_boot, block = 0, 0

    quick = st.checkbox("Modo rápido (grades menores)", value=True)
    st.caption("Ative para acelerar (restringe combinações de hiperparâmetros).")

# =========================================================
# Preparar DataFrame de entrada no formato esperado
# (converter rótulos 'ds' como 'Jan/25' para datetimes de 1º dia do mês)
# =========================================================
hist = st.session_state["ts_df_norm"].copy()  # colunas: ['ds','y'] com 'ds' tipo 'Set/25'
df_in = pd.DataFrame({
    "ds": hist["ds"].apply(parse_label_pt),
    "y": pd.to_numeric(hist["y"], errors="coerce")
}).dropna().sort_values("ds")

# =========================================================
# Ajustes de "modo rápido" no próprio módulo do pipeline
# (reduz grades para acelerar; não altera a lógica)
# =========================================================
if quick:
    pl.CROSTON_ALPHAS = [0.1, 0.3]
    pl.SBA_ALPHAS = [0.1, 0.3]
    pl.TSB_ALPHA_GRID = [0.1, 0.5]
    pl.TSB_BETA_GRID = [0.1, 0.5]
    pl.RF_LAGS_GRID = [3, 6]
    pl.RF_N_ESTIMATORS_GRID = [200]
    pl.RF_MAX_DEPTH_GRID = [None, 5]
    pl.SARIMA_GRID = {"p":[0,1], "d":[0,1], "q":[0,1], "P":[0,1], "D":[0,1], "Q":[0,1]}

# =========================================================
# Botão: Rodar previsão (experimentos do pipeline)
# =========================================================
run = st.button("▶️ Rodar previsão (experimentar modelos)", type="primary")

if run:
    # --------- UI: barra de progresso sem textos extras ---------
    prog = st.progress(0.0)
    holder = st.empty()  # espaço reservado para resultados após terminar

    # Pequeno “pré-aquecimento” visual da barra
    prog.progress(0.05)

    try:
        # Chamada do pipeline (sem salvar arquivos em disco)
        resultados = pl.run_full_pipeline(
            data_input=df_in,
            sheet_name=None,
            date_col="ds",
            value_col="y",
            horizon=int(horizon),
            seasonal_period=int(seasonal_period),
            do_original=True,
            do_log=bool(use_log),
            do_bootstrap=bool(use_boot),
            n_bootstrap=int(n_boot) if use_boot else 0,
            bootstrap_block=int(block) if use_boot else 0,
            save_dir=None
        )

        # Avança a barra perto de 100% e conclui
        prog.progress(0.95)

        # Guardar apenas no session_state (NADA em query params!)
        st.session_state["experiments_df"] = resultados
        st.session_state["champion"] = resultados.attrs.get("champion", {})

        # Remover a barra e renderizar resultados
        prog.progress(1.0)
        prog.empty()

        champ = st.session_state.get("champion", {})

        with holder.container():
            st.subheader("🏆 Modelo campeão (critério do pipeline)")
            if champ:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Preprocess", str(champ.get("preprocess", "-")))
                c2.metric("Modelo", str(champ.get("model", "-")))
                # Mostrar sMAPE (percentual) porque é mais intuitivo
                smape_val = champ.get("sMAPE", np.nan)
                c3.metric("sMAPE (%)", f"{float(smape_val):.2f}" if pd.notna(smape_val) else "-")
                c4.metric("RMSE", f"{float(champ.get('RMSE', np.nan)):.2f}" if pd.notna(champ.get("RMSE", np.nan)) else "-")
            else:
                st.info("Nenhum campeão retornado.")

            st.subheader("Resultados dos experimentos")
            st.dataframe(
                st.session_state["experiments_df"],
                use_container_width=True,
                hide_index=True
            )

            st.success("Experimento concluído com sucesso!")

    except Exception as e:
        prog.empty()
        holder.empty()
        st.error(f"Ocorreu um erro durante a execução: {e}")

else:
    # Não mostra tabela/resultado antes de rodar
    st.info("Defina os parâmetros na barra lateral e clique em **Rodar previsão**.")
# pages/04_Previsao.py
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

# =========================================================
# Título
# =========================================================
st.title("🔮 Passo 2 — Previsão de Demanda")

# =========================================================
# Guarda de etapa: precisa do Upload (Passo 1)
# =========================================================
if "ts_df_norm" not in st.session_state:
    st.warning("Preciso da série do Passo 1 (Upload) antes de continuar.")
    st.page_link("pages/01_Upload.py", label="Ir para o Passo 1 — Upload", icon="📤")
    st.stop()

# =========================================================
# Import do pipeline (core/pipeline.py) como módulo
# =========================================================
ROOT = Path(__file__).resolve().parents[1]  # raiz do projeto (onde fica a pasta core/)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import core.pipeline as pl  # noqa: E402

# =========================================================
# Helpers para converter "Jan/25" -> Timestamp("2025-01-01")
# =========================================================
_PT = {1:"Jan",2:"Fev",3:"Mar",4:"Abr",5:"Mai",6:"Jun",7:"Jul",8:"Ago",9:"Set",10:"Out",11:"Nov",12:"Dez"}
_REV_PT = {v: k for k, v in _PT.items()}

def parse_label_pt(label: str) -> pd.Timestamp:
    # aceita "Set/25" etc.
    mon = label[:3].capitalize()
    yy = int(label[-2:]) + 2000
    mm = _REV_PT.get(mon)
    if mm is None:
        # fallback: tentar parser direto
        return pd.to_datetime(label, dayfirst=True)
    return pd.Timestamp(year=yy, month=mm, day=1)

# =========================================================
# Sidebar — parâmetros essenciais (simples)
# =========================================================
with st.sidebar:
    st.header("Configurar experimento")
    horizon = st.selectbox("Horizonte (meses)", [6, 8, 12], index=0)
    seasonal_period = st.number_input("Período sazonal (m)", min_value=1, value=12, step=1)
    use_log = st.checkbox("Usar transformação log + ε", value=True)
    use_boot = st.checkbox("Usar bootstrap FPP", value=False)

    if use_boot:
        st.caption("Bootstrap FPP gera réplicas sintéticas para robustez.")
        n_boot = st.slider("Réplicas (n)", 5, 40, 20, step=1)
        block = st.slider("Tamanho do bloco", 6, 48, 24, step=1)
    else:
        n_boot, block = 0, 0

    quick = st.checkbox("Modo rápido (grades menores)", value=True)
    st.caption("Ative para acelerar (restringe combinações de hiperparâmetros).")

# =========================================================
# Preparar DataFrame de entrada no formato esperado
# (converter rótulos 'ds' como 'Jan/25' para datetimes de 1º dia do mês)
# =========================================================
hist = st.session_state["ts_df_norm"].copy()  # colunas: ['ds','y'] com 'ds' tipo 'Set/25'
df_in = pd.DataFrame({
    "ds": hist["ds"].apply(parse_label_pt),
    "y": pd.to_numeric(hist["y"], errors="coerce")
}).dropna().sort_values("ds")

# =========================================================
# Ajustes de "modo rápido" no próprio módulo do pipeline
# (reduz grades para acelerar; não altera a lógica)
# =========================================================
if quick:
    pl.CROSTON_ALPHAS = [0.1, 0.3]
    pl.SBA_ALPHAS = [0.1, 0.3]
    pl.TSB_ALPHA_GRID = [0.1, 0.5]
    pl.TSB_BETA_GRID = [0.1, 0.5]
    pl.RF_LAGS_GRID = [3, 6]
    pl.RF_N_ESTIMATORS_GRID = [200]
    pl.RF_MAX_DEPTH_GRID = [None, 5]
    pl.SARIMA_GRID = {"p":[0,1], "d":[0,1], "q":[0,1], "P":[0,1], "D":[0,1], "Q":[0,1]}

# =========================================================
# Botão: Rodar previsão (experimentos do pipeline)
# =========================================================
run = st.button("▶️ Rodar previsão (experimentar modelos)", type="primary")

if run:
    # --------- UI: barra de progresso sem textos extras ---------
    prog = st.progress(0.0)
    holder = st.empty()  # espaço reservado para resultados após terminar

    # Pequeno “pré-aquecimento” visual da barra
    prog.progress(0.05)

    try:
        # Chamada do pipeline (sem salvar arquivos em disco)
        resultados = pl.run_full_pipeline(
            data_input=df_in,
            sheet_name=None,
            date_col="ds",
            value_col="y",
            horizon=int(horizon),
            seasonal_period=int(seasonal_period),
            do_original=True,
            do_log=bool(use_log),
            do_bootstrap=bool(use_boot),
            n_bootstrap=int(n_boot) if use_boot else 0,
            bootstrap_block=int(block) if use_boot else 0,
            save_dir=None
        )

        # Avança a barra perto de 100% e conclui
        prog.progress(0.95)

        # Guardar apenas no session_state (NADA em query params!)
        st.session_state["experiments_df"] = resultados
        st.session_state["champion"] = resultados.attrs.get("champion", {})

        # Remover a barra e renderizar resultados
        prog.progress(1.0)
        prog.empty()

        champ = st.session_state.get("champion", {})

        with holder.container():
            st.subheader("🏆 Modelo campeão (critério do pipeline)")
            if champ:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Preprocess", str(champ.get("preprocess", "-")))
                c2.metric("Modelo", str(champ.get("model", "-")))
                # Mostrar sMAPE (percentual) porque é mais intuitivo
                smape_val = champ.get("sMAPE", np.nan)
                c3.metric("sMAPE (%)", f"{float(smape_val):.2f}" if pd.notna(smape_val) else "-")
                c4.metric("RMSE", f"{float(champ.get('RMSE', np.nan)):.2f}" if pd.notna(champ.get("RMSE", np.nan)) else "-")
            else:
                st.info("Nenhum campeão retornado.")

            st.subheader("Resultados dos experimentos")
            st.dataframe(
                st.session_state["experiments_df"],
                use_container_width=True,
                hide_index=True
            )

            st.success("Experimento concluído com sucesso!")

    except Exception as e:
        prog.empty()
        holder.empty()
        st.error(f"Ocorreu um erro durante a execução: {e}")

else:
    # Não mostra tabela/resultado antes de rodar
    st.info("Defina os parâmetros na barra lateral e clique em **Rodar previsão**.")
