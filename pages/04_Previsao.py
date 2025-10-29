# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------
# Configuração da página
# ---------------------------------------------------------------------
st.title("🔮 Passo 2 — Previsão de Demanda")

# ---------------------------------------------------------------------
# Guarda de etapa: precisa do Upload (Passo 1)
# ---------------------------------------------------------------------
if "ts_df_norm" not in st.session_state:
    st.warning("Preciso da série do Passo 1 (Upload) antes de continuar.")
    st.page_link("pages/01_Upload.py", label="Ir para o Passo 1 — Upload", icon="📤")
    st.stop()

# ---------------------------------------------------------------------
# Import do pipeline (core/pipeline.py)
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]  # raiz do projeto
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import core.pipeline as pipe  # type: ignore
except Exception as e:
    st.error(f"Não consegui importar core/pipeline.py: {e}")
    st.stop()

# ---------------------------------------------------------------------
# Helpers de data (rótulos 'Jan/25' -> Timestamp)
# ---------------------------------------------------------------------
_PT2NUM = {"Jan": 1, "Fev": 2, "Mar": 3, "Abr": 4, "Mai": 5, "Jun": 6,
           "Jul": 7, "Ago": 8, "Set": 9, "Out": 10, "Nov": 11, "Dez": 12}

def parse_label_to_timestamp(s: str) -> pd.Timestamp:
    """
    Converte 'Set/25' para primeiro dia do mês correspondente.
    Também aceita strings padrão reconhecíveis por to_datetime().
    """
    s = str(s).strip()
    # tentativa direta
    ts = pd.to_datetime(s, errors="coerce", dayfirst=True)
    if pd.notna(ts):
        return pd.Timestamp(ts.normalize()).to_period("M").to_timestamp(how="start")
    # formato 'Mmm/YY' em PT-BR
    try:
        mon = s[:3].capitalize()
        yy = int(s[-2:]) + 2000
        mm = _PT2NUM[mon]
        return pd.Timestamp(year=yy, month=mm, day=1)
    except Exception:
        return pd.NaT

def df_labels_to_datetime(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Recebe df ['ds','y'] com 'ds' tipo 'Set/25' e devolve ['ds','y'] com 'ds' datetime (1º dia do mês).
    Remove linhas não conversíveis.
    """
    out = df_in.copy()
    out["ds"] = out["ds"].apply(parse_label_to_timestamp)
    out["y"] = pd.to_numeric(out["y"], errors="coerce")
    out = out.dropna(subset=["ds", "y"])
    out = out.sort_values("ds").reset_index(drop=True)
    return out

# ---------------------------------------------------------------------
# Sidebar – parâmetros do experimento
# ---------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Parâmetros do experimento")

    horizon = st.selectbox("Horizonte (meses)", [6, 8, 12],
                           index={6: 0, 8: 1, 12: 2}.get(int(st.session_state.get("forecast_h", 6)), 0))
    seasonal_period = st.number_input("Período sazonal (m)", min_value=1, max_value=24, value=12, step=1)

    st.subheader("Pré-processamentos")
    use_log = st.checkbox("Aplicar log + ε", value=True,
                          help="Transforma y → log(y+shift+ε) com ε escolhido automaticamente para estabilizar a variância.")
    use_boot = st.checkbox("Usar Bootstrap FPP", value=True,
                           help="Gera séries sintéticas com Box–Cox (λ MLE) + STL robusta + bootstrap em blocos dos resíduos.")

    if use_boot:
        st.caption("**Bootstrap — o que significam os parâmetros**\n"
                   "- **Réplicas**: quantas séries sintéticas serão geradas.\n"
                   "- **Tamanho do bloco**: quantos pontos consecutivos do resíduo são reamostrados de uma vez (preserva autocorrelação).")
        n_boot = st.slider("Réplicas (n_bootstrap)", 5, 50, 20, step=1)
        block = st.slider("Tamanho do bloco", 6, 48, 24, step=1)
    else:
        n_boot = 0
        block = 24

    st.subheader("🏎️ Desempenho")
    fast_mode = st.toggle("Modo rápido (menos combinações)", value=False)
    st.caption("Deixe **desligado** para avaliar mais combinações (padrão).")

# ---------------------------------------------------------------------
# Área principal – instrução
# ---------------------------------------------------------------------
st.info("Clique em **Rodar previsão** para executar os experimentos. "
        "Nenhuma tabela é exibida até a conclusão.")

run_btn = st.button("▶️ Rodar previsão", type="primary")

# zona para resultados (preenchido só depois)
res_container = st.container()

if run_btn:
    # 1) Prepara DataFrame de entrada com datas reais
    hist = st.session_state["ts_df_norm"]
    try:
        df_in = df_labels_to_datetime(hist[["ds", "y"]])
    except Exception as e:
        st.error(f"Erro ao normalizar datas do histórico: {e}")
        st.stop()

    if df_in.empty or df_in["ds"].isna().any():
        st.error("Após conversão de rótulos, a série ficou vazia ou com datas inválidas. "
                 "Verifique a coluna de datas do Upload.")
        st.stop()

    # 2) Ajuste de 'modo rápido' (monkeypatch leve nas grades do pipeline)
    #    Obs.: apenas valores simples; não mexe na lógica interna.
    if fast_mode:
        try:
            pipe.CROSTON_ALPHAS = [0.1, 0.3]
            pipe.SBA_ALPHAS = [0.1, 0.3]
            pipe.TSB_ALPHA_GRID = [0.3]
            pipe.TSB_BETA_GRID = [0.3]
            pipe.RF_LAGS_GRID = [3]
            pipe.RF_N_ESTIMATORS_GRID = [200]
            pipe.RF_MAX_DEPTH_GRID = [None]
            pipe.SARIMA_GRID = {"p": [0, 1], "d": [0, 1], "q": [0, 1], "P": [0], "D": [0, 1], "Q": [0]}
        except Exception:
            # se o módulo não expuser alguma grade, simplesmente ignore
            pass

    # 3) Executa pipeline (sem logs de texto na UI)
    with st.spinner("Processando sua previsão… isso pode levar alguns minutos."):
        try:
            resultados = pipe.run_full_pipeline(
                data_input=df_in,                 # DataFrame já com 'ds' datetime e 'y' numérico
                sheet_name=None,
                date_col="ds",
                value_col="y",
                horizon=int(horizon),
                seasonal_period=int(seasonal_period),
                do_original=True,
                do_log=bool(use_log),
                do_bootstrap=bool(use_boot),
                n_bootstrap=int(n_boot) if use_boot else 0,
                bootstrap_block=int(block) if use_boot else 24,
                save_dir=None,                   # sem salvar em disco pela UI
            )
        except Exception as e:
            st.error(f"Ocorreu um erro durante a execução: {e}")
            st.stop()

    # 4) Exibe resultados (somente agora)
    with res_container:
        st.success("Experimentos concluídos com sucesso! ✅")

        # Campeão
        champ = resultados.attrs.get("champion", {})
        st.subheader("🏆 Modelo campeão")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Pré-processamento", str(champ.get("preprocess", "-")))
        c2.metric("Modelo", str(champ.get("model", "-")))
        # sMAPE do pipeline já é %; exibimos com 1 casa
        try:
            smape_val = float(champ.get("sMAPE", np.nan))
            c3.metric("sMAPE", f"{smape_val:.1f} %")
        except Exception:
            c3.metric("sMAPE", "-")
        try:
            mae_val = float(champ.get("MAE", np.nan))
            c4.metric("MAE", f"{mae_val:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
        except Exception:
            c4.metric("MAE", "-")

        # Tabela completa (ordenada)
        st.subheader("Resultados dos experimentos")
        st.dataframe(resultados, use_container_width=True, height=380)

        # Guarda informações úteis para as próximas etapas
        st.session_state["forecast_h"] = int(horizon)        # horizonte escolhido
        st.session_state["exp_results"] = resultados         # DataFrame com as linhas (ok no session_state)
        st.session_state["champion"] = champ                 # dict do campeão
        # Como o pipeline ainda não traz a previsão futura, não marcamos como 'committed'
        st.session_state["forecast_committed"] = False

    st.divider()
    st.page_link("pages/05_Inputs_MPS.py",
                 label="➡️ Ir para 05_Inputs_MPS (configurar Inputs do MPS)",
                 icon="⚙️")
