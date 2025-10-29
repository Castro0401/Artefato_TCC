# pages/04_Previsao.py
from __future__ import annotations
import io
import numpy as np
import pandas as pd
import streamlit as st

# ===== core pipeline completo =====
# ajuste o import conforme seu projeto: core/pipeline.py
import PipelineCompletoV5.py as pipe

st.title("🔮 Passo 2 — Previsão de Demanda")

# ---------- Guardas ----------
if "ts_df_norm" not in st.session_state:
    st.warning("Preciso da série do Passo 1 (Upload) antes de continuar.")
    st.page_link("pages/01_Upload.py", label="Ir para o Passo 1 — Upload", icon="📤")
    st.stop()

# ---------- Le histórico carregado no Passo 1 ----------
hist = st.session_state["ts_df_norm"].copy()  # ['ds','y'] com labels tipo 'Set/25'
# Converte para DataFrame ('ds' em data mensal real) que o pipeline aceita
def _to_monthly_df(df_label_y: pd.DataFrame) -> pd.DataFrame:
    # df_label_y['ds'] é "Set/25"; vamos converter pra Timestamp 1º dia do mês
    PT = {"Jan":1,"Fev":2,"Mar":3,"Abr":4,"Mai":5,"Jun":6,"Jul":7,"Ago":8,"Set":9,"Out":10,"Nov":11,"Dez":12}
    def _to_period(lbl: str) -> pd.Period:
        mon = lbl[:3].title(); yy = 2000 + int(lbl[-2:])
        return pd.Period(freq="M", year=yy, month=PT[mon])
    df = hist.copy()
    df["__p"] = df["ds"].apply(_to_period)
    df = df.sort_values("__p")
    df["ds"] = df["__p"].dt.to_timestamp(how="start")
    df = df[["ds","y"]].reset_index(drop=True)
    return df

df_monthly = _to_monthly_df(hist)

# ============== Configuração do experimento (UI enxuta) ==============
with st.expander("Configurações do experimento", expanded=True):
    last_h  = int(st.session_state.get("forecast_h", 6))
    horizon = st.selectbox("Horizonte (meses)", [6,8,12],
                           index=[6,8,12].index(last_h) if last_h in [6,8,12] else 0,
                           help="O MPS usará este mesmo horizonte.")
    season  = st.number_input("Período sazonal (m)", min_value=1, max_value=24, value=12, step=1)
    colA, colB, colC = st.columns(3)
    with colA:
        do_original = st.checkbox("Usar série original", value=True)
    with colB:
        do_log      = st.checkbox("Usar log + ε", value=True)
    with colC:
        do_boot     = st.checkbox("Bootstrap FPP", value=False)
    row = st.columns(2)
    with row[0]:
        n_boot      = st.slider("Réplicas bootstrap", 1, 50, 10, disabled=not do_boot)
    with row[1]:
        block_size  = st.slider("Tamanho do bloco (bootstrap)", 3, 48, 24, disabled=not do_boot)

    modo_rapido = st.toggle("🏎️ Modo rápido (menos combinações)", value=True,
                            help="Reduz grade de hiperparâmetros para acelerar a execução.")

# ============== Área de execução ==============
run = st.button("▶️ Rodar previsão agora", type="primary")

# Espaços de UI
placeholder_loading = st.container()  # onde aparece barra + mensagens
placeholder_result  = st.container()  # onde entram os resultados ao final

def _apply_fast_mode(on: bool):
    """Reduz as grades do pipeline (e restaura ao final)."""
    orig = {
        "CROSTON_ALPHAS": pipe.CROSTON_ALPHAS[:],
        "SBA_ALPHAS": pipe.SBA_ALPHAS[:],
        "TSB_ALPHA_GRID": pipe.TSB_ALPHA_GRID[:],
        "TSB_BETA_GRID": pipe.TSB_BETA_GRID[:],
        "RF_LAGS_GRID": pipe.RF_LAGS_GRID[:],
        "RF_N_ESTIMATORS_GRID": pipe.RF_N_ESTIMATORS_GRID[:],
        "RF_MAX_DEPTH_GRID": pipe.RF_MAX_DEPTH_GRID[:],
        "SARIMA_GRID": {k: v[:] for k, v in pipe.SARIMA_GRID.items()},
    }
    def _shrink():
        pipe.CROSTON_ALPHAS = [0.1, 0.3]
        pipe.SBA_ALPHAS     = [0.1, 0.3]
        pipe.TSB_ALPHA_GRID = [0.3]
        pipe.TSB_BETA_GRID  = [0.3]
        pipe.RF_LAGS_GRID   = [6]
        pipe.RF_N_ESTIMATORS_GRID = [200]
        pipe.RF_MAX_DEPTH_GRID    = [None]
        pipe.SARIMA_GRID = {"p":[0,1], "d":[0,1], "q":[0,1], "P":[0,1], "D":[0,1], "Q":[0]}
    def _restore():
        pipe.CROSTON_ALPHAS = orig["CROSTON_ALPHAS"]
        pipe.SBA_ALPHAS     = orig["SBA_ALPHAS"]
        pipe.TSB_ALPHA_GRID = orig["TSB_ALPHA_GRID"]
        pipe.TSB_BETA_GRID  = orig["TSB_BETA_GRID"]
        pipe.RF_LAGS_GRID   = orig["RF_LAGS_GRID"]
        pipe.RF_N_ESTIMATORS_GRID = orig["RF_N_ESTIMATORS_GRID"]
        pipe.RF_MAX_DEPTH_GRID    = orig["RF_MAX_DEPTH_GRID"]
        pipe.SARIMA_GRID = orig["SARIMA_GRID"]
    return (_shrink, _restore)

if run:
    # 1) “Troca” a tela: mostra somente a área de carregamento (barra + mensagens)
    placeholder_result.empty()
    with placeholder_loading:
        st.subheader("Processando sua previsão…")
        prog = st.progress(0.0, text="Inicializando…")
        log_area = st.empty()

    # 2) Conecta callbacks do pipeline à UI
    log_buffer = []

    def ui_logger(msg: str):
        # acumula e mostra últimas 60 linhas
        log_buffer.append(msg)
        log_area.code("\n".join(log_buffer[-60:]))

    def ui_progress(done: int, total: int):
        frac = 0.0 if total <= 0 else done / total
        prog.progress(frac, text=f"Executando experimentos… {int(frac*100)}%")

    pipe.set_logger(ui_logger)
    pipe.set_progress(ui_progress)

    # 3) (opcional) modo rápido
    shrink, restore = _apply_fast_mode(modo_rapido)
    if modo_rapido:
        shrink()

    # 4) Rodar de fato
    try:
        with st.spinner("Executando pipeline…"):
            df_out = pipe.run_full_pipeline(
                data_input=df_monthly,  # passamos DF já mensal
                sheet_name=None, date_col="ds", value_col="y",
                horizon=int(horizon), seasonal_period=int(season),
                do_original=bool(do_original),
                do_log=bool(do_log),
                do_bootstrap=bool(do_boot),
                n_bootstrap=int(n_boot) if do_boot else 0,
                bootstrap_block=int(block_size) if do_boot else 0,
                save_dir=None,
            )
    finally:
        if modo_rapido:
            restore()

    champ = df_out.attrs.get("champion", {})

    # 5) Ao finalizar: some a área de carregamento e entre com os resultados
    placeholder_loading.empty()

    with placeholder_result:
        st.subheader(f"Histórico + Previsão ({horizon} meses)")
        # monta série histórica e projeta a previsão do campeão
        # Observação: o pipeline escolheu o melhor modelo usando holdout interno.
        # Para exibição, vamos apenas construir a tabela de previsão com labels futuros.
        # (Nesta versão, simplificamos a projeção visual usando o último ponto como base.)
        # Se seu pipeline retornar as previsões futuras, troque abaixo por elas.
        # Aqui, mostramos os resultados (tabela de experimentos) e os metadados do campeão.

        st.markdown("### Campeão do experimento")
        colL, colR = st.columns([2,1])
        with colL:
            st.write(
                f"**Preprocess:** {champ.get('preprocess','—')}  \n"
                f"**Modelo:** {champ.get('model','—')}  \n"
                f"**Parâmetros:** {champ.get('model_params','—')}"
            )
        with colR:
            st.metric("MAE", f"{champ.get('MAE', float('nan')):.2f}")
            st.metric("RMSE", f"{champ.get('RMSE', float('nan')):.2f}")

        st.markdown("### Tabela de experimentos (ordenada)")
        st.dataframe(df_out, use_container_width=True, height=360)

        # Persistência para MPS: aqui você conecta com seu formato
        # Exemplo: pegue o horizonte e deixe salvo para as próximas páginas
        st.session_state["forecast_h"] = int(horizon)
        st.session_state["forecast_committed"] = True

        st.info("Previsão concluída e configurada. Siga para os Inputs do MPS.")
        st.page_link("pages/05_Inputs_MPS.py",
                     label="➡️ Ir para 05_Inputs_MPS (configurar plano mestre)",
                     icon="🛠️")

else:
    st.info("Clique em **Rodar previsão agora** para iniciar os experimentos com a sua série.")
