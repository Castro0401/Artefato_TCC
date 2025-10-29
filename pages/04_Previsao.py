# pages/04_Previsao.py
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------
# Título
# ---------------------------------------------------------------------
st.title("🔮 Passo 2 — Previsão de Demanda")

# ---------------------------------------------------------------------
# Guarda: precisa do Passo 1
# ---------------------------------------------------------------------
if "ts_df_norm" not in st.session_state:
    st.warning("Preciso da série do Passo 1 (Upload) antes de continuar.")
    st.page_link("pages/01_Upload.py", label="Ir para o Passo 1 — Upload", icon="📤")
    st.stop()

# ---------------------------------------------------------------------
# Importa o pipeline como MÓDULO (Opção A)
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]  # raiz do projeto (onde fica a pasta core/)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import core.PipelineCompletoV5 as pipe  # <<< agora 'pipe' existe no escopo

# ---------------------------------------------------------------------
# Helpers de data (para converter 'Set/25' -> timestamp mensal)
# ---------------------------------------------------------------------
_PT = {"Jan":1,"Fev":2,"Mar":3,"Abr":4,"Mai":5,"Jun":6,"Jul":7,"Ago":8,"Set":9,"Out":10,"Nov":11,"Dez":12}
_REV_PT = {v:k for k,v in _PT.items()}

def to_period(lbl: str) -> pd.Period:
    try:
        # tenta parsear como data normal também
        return pd.to_datetime(lbl, dayfirst=True).to_period("M")
    except Exception:
        mon = lbl[:3].title()
        yy = 2000 + int(lbl[-2:])
        return pd.Period(freq="M", year=yy, month=_PT[mon])

def label_pt(ts: pd.Timestamp) -> str:
    return f"{_REV_PT[ts.month]}/{str(ts.year)[-2:]}"

# ---------------------------------------------------------------------
# Entrada base vinda do Passo 1 (normalizada mensalmente)
#  - Convertemos para DataFrame com 'ds' = Timestamp mensal e 'y'
#    porque o pipeline espera datas de verdade.
# ---------------------------------------------------------------------
hist_norm = st.session_state["ts_df_norm"].copy()  # colunas: ['ds','y'] mas 'ds' é um label tipo 'Set/25'
hist_norm["p"] = hist_norm["ds"].apply(to_period)
hist_norm = hist_norm.sort_values("p").reset_index(drop=True)
df_pipe = pd.DataFrame({
    "ds": hist_norm["p"].dt.to_timestamp(),   # 1º dia do mês
    "y":  hist_norm["y"].astype(float),
})

# ---------------------------------------------------------------------
# UI – parâmetros do experimento
# ---------------------------------------------------------------------
with st.expander("Configurações do experimento", expanded=True):
    c1, c2, c3 = st.columns(3)
    with c1:
        horizon = st.selectbox("Horizonte (meses)", [6, 8, 12], index=0)
    with c2:
        seasonal_period = st.number_input("Período sazonal (m)", 1, 24, 12, step=1)
    with c3:
        modo = st.radio("Modo", ["Rápido", "Completo"], index=0, horizontal=True)

    c4, c5, c6 = st.columns(3)
    with c4:
        do_original  = st.checkbox("Usar Original", True)
    with c5:
        do_log       = st.checkbox("Usar Log + ε", True)
    with c6:
        do_bootstrap = st.checkbox("Usar Bootstrap FPP", True)

    nb = 10 if modo == "Rápido" else 20
    blk = 24
    if do_bootstrap:
        nb = st.slider("Réplicas Bootstrap", 1, 50, nb)
        blk = st.slider("Tamanho do bloco (bootstrap)", 3, 60, blk)

# ---------------------------------------------------------------------
# Rodar
# ---------------------------------------------------------------------
run_btn = st.button("▶️ Rodar previsão (executar experimentos)")

if run_btn:
    # Barra de status + logger conectado ao pipeline
    status = st.status("Inicializando…", expanded=True)
    progress = st.progress(0)

    # mapeia palavras-chave de log para “pontos” do progresso (heurístico)
    steps = {
        "Série ORIGINAL": 0.20,
        "Transformação LOG": 0.35,
        "SÉRIE LOG-transformada": 0.55,
        "réplicas bootstrap": 0.65,
        "SÉRIE SINTÉTICA": 0.80,
        "FINALIZADO": 1.00,
    }
    pct = 0.02

    def ui_logger(msg: str):
        nonlocal pct
        status.write(msg)
        for key, target in steps.items():
            if key in msg:
                pct = max(pct, target)
        progress.progress(min(pct, 0.98))
        pct = min(pct + 0.01, 0.98)

    # injeta logger da UI no pipeline
    pipe.set_logger(ui_logger)

    try:
        with st.spinner("Executando pipeline… isso pode levar alguns minutos."):
            resultados = pipe.run_full_pipeline(
                data_input=df_pipe,           # passamos o DF com 'ds' Timestamp e 'y'
                sheet_name=None,
                date_col="ds",
                value_col="y",
                horizon=int(horizon),
                seasonal_period=int(seasonal_period),
                do_original=bool(do_original),
                do_log=bool(do_log),
                do_bootstrap=bool(do_bootstrap),
                n_bootstrap=int(nb),
                bootstrap_block=int(blk),
                save_dir=None,                # ajuste se quiser salvar CSV/XLSX
            )
        progress.progress(1.0)
        status.update(label="Concluído!", state="complete")

        # mostra tabela de experimentos
        st.subheader("Resultados dos experimentos")
        st.dataframe(resultados, use_container_width=True, height=360)

        champ = resultados.attrs.get("champion", {})
        if champ:
            st.success("🏆 Campeão selecionado (critério: menor MAE; desempates por RMSE/soma/simplicidade)")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Pré-processamento", champ.get("preprocess", "-"))
            c2.metric("Modelo", champ.get("model", "-"))
            c3.metric("MAE", f"{champ.get('MAE', float('nan')):.3f}")
            c4.metric("RMSE", f"{champ.get('RMSE', float('nan')):.3f}")
            st.caption(f"Parâmetros: {champ.get('model_params','-')}")
        else:
            st.info("Não foi possível identificar o campeão a partir da tabela.")

        # ------------------------------------------------------------
        # PLACEHOLDER de previsão futura (até expormos a função do campeão):
        # gera uma extrapolação simples com tendência local para alimentar o MPS.
        # ------------------------------------------------------------
        y = df_pipe["y"].values.astype(float)
        ma = pd.Series(y).rolling(3, min_periods=1).mean().values
        trend = (ma[-1] - ma[max(len(ma)-4, 0)]) / max(3, len(ma)-1)
        base = ma[-1]
        rng = np.random.default_rng(42)
        fut_vals = []
        for _ in range(int(horizon)):
            base = base + trend
            fut_vals.append(max(0.0, base + rng.normal(0, 0.1*max(1.0, np.std(y)))))

        last_ts = df_pipe["ds"].iloc[-1]
        fut_idx = pd.date_range(last_ts + pd.offsets.MonthBegin(1), periods=int(horizon), freq="MS")
        forecast_df = pd.DataFrame({"ds": [label_pt(ts) for ts in fut_idx], "y": np.round(fut_vals).astype(int)})

        # Persistência para o MPS
        st.session_state["forecast_df"] = forecast_df
        st.session_state["forecast_h"] = int(horizon)
        st.session_state["forecast_committed"] = True

        # Visual
        st.subheader(f"Histórico + Previsão ({horizon} meses)")
        hist_plot = hist_norm.assign(ts=hist_norm["p"].dt.to_timestamp())[["ts","y"]]
        fut_plot = pd.DataFrame({"ts": fut_idx, "y": forecast_df["y"]})
        chart_df = pd.concat([
            hist_plot.assign(tipo="Histórico"),
            fut_plot.assign(tipo="Previsão"),
        ]).set_index("ts")
        st.line_chart(chart_df, y="y", color="tipo", height=320, use_container_width=True)

        st.subheader("Previsão (tabela)")
        st.dataframe(forecast_df, use_container_width=True, height=220)

    except Exception as e:
        progress.progress(0)
        status.update(label="Falhou", state="error")
        st.error(f"Ocorreu um erro ao executar o pipeline: {e}")
        st.exception(e)  # opcional, para ver stacktrace no dev

# Navegação
st.divider()
st.page_link("pages/05_Inputs_MPS.py", label="➡️ Usar esta previsão nos Inputs do MPS", icon="⚙️")
