# pages/04_Previsao.py
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------
# Config da página
# ---------------------------------------------------------------------
st.title("🔮 Passo 2 — Previsão de Demanda")

# ---------------------------------------------------------------------
# Guardas de etapa
# ---------------------------------------------------------------------
if "ts_df_norm" not in st.session_state:
    st.warning("Preciso da série do Passo 1 (Upload) antes de continuar.")
    st.page_link("pages/01_Upload.py", label="Ir para o Passo 1 — Upload", icon="📤")
    st.stop()

# ---------------------------------------------------------------------
# Import seguro do core.pipeline (sem quebrar paths quando rodar via streamlit)
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]  # raiz do projeto (onde fica a pasta core/)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import core.pipeline as pl
except Exception as e:
    st.error(f"Não consegui importar `core.pipeline`. Erro: {e}")
    st.stop()

# ---------------------------------------------------------------------
# Helpers de data (labels 'Set/25' → Period mensal; Period → Timestamp)
# ---------------------------------------------------------------------
_PT = {1: "Jan", 2: "Fev", 3: "Mar", 4: "Abr", 5: "Mai", 6: "Jun",
       7: "Jul", 8: "Ago", 9: "Set", 10: "Out", 11: "Nov", 12: "Dez"}
_REV_PT = {v: k for k, v in _PT.items()}

def to_period(s: str) -> pd.Period:
    # aceita "Set/25" ou datas YYYY-MM-DD
    try:
        return pd.to_datetime(s, dayfirst=True).to_period("M")
    except Exception:
        mon = s[:3].capitalize()
        yy = int(s[-2:]) + 2000
        month_num = _REV_PT.get(mon)
        if month_num is None:
            raise ValueError(f"Formato de mês inválido: {s}")
        return pd.Period(freq="M", year=yy, month=month_num)

# ---------------------------------------------------------------------
# Série histórica do Passo 1 (não alterar o objeto original no session_state)
# ---------------------------------------------------------------------
_hist = st.session_state["ts_df_norm"].copy()     # colunas ['ds','y'] com label tipo 'Set/25'
_hist["p"] = _hist["ds"].apply(to_period)         # Period(M)
_hist = _hist.sort_values("p").reset_index(drop=True)

# ---------------------------------------------------------------------
# MENU LATERAL — parâmetros do pipeline
# ---------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configurações da previsão")

    # Lembrar último horizonte
    last_h = int(st.session_state.get("forecast_h", 6))
    horizon = st.selectbox("Horizonte (meses)",
                           [6, 8, 12],
                           index=[6, 8, 12].index(last_h) if last_h in (6, 8, 12) else 0,
                           help="Esse horizonte será usado para o holdout do pipeline e para a exibição.")

    seasonal_period = st.number_input("Período sazonal (m)", min_value=1, max_value=24, value=12, step=1)

    st.markdown("**Pré-processamentos**")
    use_log = st.checkbox("Aplicar log + ε (auto)", value=True,
                          help="Aplica log(y + shift + ε) escolhendo ε para estabilizar a variância; métricas avaliadas na escala original.")
    use_boot = st.checkbox("Gerar séries sintéticas (bootstrap FPP)", value=True)

    if use_boot:
        st.caption("**Réplicas**: quantas séries sintéticas serão geradas.\n\n"
                   "**Tamanho do bloco**: controla a preservação da autocorrelação ao reamostrar resíduos.")
        n_boot = st.slider("Réplicas (bootstrap)", min_value=1, max_value=100, value=20, step=1)
        block = st.slider("Tamanho do bloco", min_value=3, max_value=48, value=24, step=1)
    else:
        n_boot, block = 0, 0

    fast_mode = st.toggle("🏎️ Modo rápido", value=False,
                          help="Reduz o custo experimental (menos combinações e/ou menos réplicas). "
                               "Use quando quiser um resultado mais veloz para protótipos.")

# Aplicação do modo rápido (sem alterar a lógica do pipeline; só dosamos parâmetros)
if fast_mode:
    # Mantemos log+ε conforme seleção; reduzimos bootstrap e não mexemos na grade interna de SARIMAX/RF.
    if use_boot:
        n_boot = min(n_boot, 5)
        block = min(block, 12)

# ---------------------------------------------------------------------
# Área de progresso + mensagens (a tabela de previsão fica em branco enquanto roda)
# ---------------------------------------------------------------------
progress_ph = st.empty()
status_box = st.container()
results_area = st.container()   # onde aparecerão os resultados quando terminar

# Tabela em branco até terminar
st.subheader("Previsão (tabela)")
st.dataframe(pd.DataFrame(columns=["ds", "y"]), use_container_width=True, height=220)

# ---------------------------------------------------------------------
# Botão para rodar o pipeline
# ---------------------------------------------------------------------
run = st.button("▶️ Rodar previsão", type="primary")

def _make_df_input_from_hist(df_hist: pd.DataFrame) -> pd.DataFrame:
    """Converte df com Period para Timestamp (MS) para o pipeline."""
    df = df_hist[["p", "y"]].copy()
    df["ds"] = df["p"].dt.to_timestamp(how="start")  # Period -> Timestamp
    return df[["ds", "y"]].dropna(subset=["ds"])

# Captura de logs do pipeline para dirigir a barra
def run_with_progress(df_in: pd.DataFrame):
    # barra e contador
    bar = progress_ph.progress(0, text="Inicializando…")
    prog = {"value": 0}

    def tick(step: int = 1, label: str | None = None):
        prog["value"] = min(95, prog["value"] + step)  # segura até 95%
        bar.progress(prog["value"], text=label or "Executando…")

    # intercepta logs do pipeline
    original_log = pl.log

    def ui_log(msg: str):
        # atualiza mensagens e barra com pequenos avanços
        status_box.write(msg)
        # heurística leve por etapas:
        lower = msg.lower()
        if "pipeline iniciado" in lower or "realizando testes da série original" in lower:
            tick(5, "Carregando e preparando dados…")
        elif "transformação log" in lower or "log-transformada" in lower:
            tick(5, "Aplicando log + ε…")
        elif "bootstrap" in lower and ("gerando" in lower or "réplicas" in lower):
            tick(10, "Gerando réplicas (bootstrap)…")
        elif "croston" in lower or "sba" in lower or "tsb" in lower:
            tick(3, "Modelos para demanda intermitente…")
        elif "randomforest" in lower:
            tick(3, "Random Forest…")
        elif "sarimax" in lower:
            tick(3, "SARIMAX…")
        elif "concluídos testes" in lower or "pipeline finalizado" in lower:
            tick(5, "Finalizando…")
        else:
            tick(1, "Executando…")
        try:
            original_log(msg)  # ainda imprime no stdout, se desejar
        except Exception:
            pass

    # injeta o logger de UI
    pl.log = ui_log
    try:
        # chamada do pipeline
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
            save_dir=None,   # você pode plugar um diretório se quiser exportar CSV/XLSX direto
        )
        # fecha barra em 100%
        bar.progress(100, text="Concluído!")
        return df_out
    finally:
        # restaura logger original e limpa barra
        pl.log = original_log
        progress_ph.empty()

if run:
    try:
        df_in = _make_df_input_from_hist(_hist)
        with st.spinner("Executando o pipeline…"):
            resultados = run_with_progress(df_in)

        # Exibe resumo e campeão (inclui sMAPE)
        champ = resultados.attrs.get("champion", {})
        with results_area:
            st.success("✅ Experimentos concluídos!")
            if champ:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Modelo campeão", str(champ.get("model", "—")))
                c2.metric("sMAPE (%)", f"{float(champ.get('sMAPE', float('nan'))):.2f}")
                c3.metric("MAE", f"{float(champ.get('MAE', float('nan'))):.2f}")
                c4.metric("RMSE", f"{float(champ.get('RMSE', float('nan'))):.2f}")
                st.caption(f"Pré-processamento: **{champ.get('preprocess','-')}**  |  "
                           f"Params: {champ.get('preprocess_params','-')}  |  "
                           f"Hiperparâmetros: {champ.get('model_params','-')}")

            with st.expander("Ver tabela completa de experimentos"):
                st.dataframe(resultados, use_container_width=True, height=420)

            st.info("Por ora mantemos a tabela de previsão **em branco durante a execução**. "
                    "Se quiser, posso plugar a geração da série prevista do campeão em seguida.")

        # Atualiza estado mínimo (somente horizonte definido; não salvamos forecast_df aqui)
        st.session_state["forecast_h"] = int(horizon)
        st.session_state["forecast_committed"] = False

    except Exception as e:
        st.error(f"Ocorreu um erro durante a execução: {e}")
