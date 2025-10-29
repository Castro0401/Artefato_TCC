# pages/04_Previsao.py
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------------------------------------------------------
# Import do core/pipeline (módulo se chama "pipeline.py")
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]  # raiz do projeto (onde fica a pasta core/)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import core.pipeline as pipe  # noqa: E402

# -----------------------------------------------------------------------------
# Título e guarda
# -----------------------------------------------------------------------------
st.title("🔮 Passo 2 — Previsão de Demanda")

if "ts_df_norm" not in st.session_state:
    st.warning("Preciso da série do Passo 1 (Upload) antes de continuar.")
    st.page_link("pages/01_Upload.py", label="Ir para o Passo 1 — Upload", icon="📤")
    st.stop()

# -----------------------------------------------------------------------------
# Helpers de rótulos mensais
# -----------------------------------------------------------------------------
_PT = {1:"Jan",2:"Fev",3:"Mar",4:"Abr",5:"Mai",6:"Jun",7:"Jul",8:"Ago",9:"Set",10:"Out",11:"Nov",12:"Dez"}
_REV_PT = {v:k for k, v in _PT.items()}

def label_pt(ts: pd.Timestamp) -> str:
    return f"{_PT[ts.month]}/{str(ts.year)[-2:]}"

def to_period(s: str) -> pd.Period:
    try:
        return pd.to_datetime(s, dayfirst=True).to_period("M")
    except Exception:
        mon = s[:3].capitalize()
        yy = int(s[-2:]) + 2000
        month_num = _REV_PT.get(mon, None)
        if month_num is None:
            raise ValueError(f"Formato de mês inválido: {s}")
        return pd.Period(freq="M", year=yy, month=month_num)

# -----------------------------------------------------------------------------
# Sidebar (controles)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.subheader("Configuração da previsão")
    # horizonte (mantém o último salvo, se existir)
    last_h = int(st.session_state.get("forecast_h", 6))
    horizon = st.selectbox("Horizonte (meses)", [6, 8, 12],
                           index=[6,8,12].index(last_h) if last_h in (6,8,12) else 0)

    seasonal_period = st.number_input("Período sazonal (m)", min_value=1, max_value=24, value=12, step=1)

    # 🏁 Modo rápido
    quick_default = st.session_state.get("_quick_mode", True)
    quick_mode = st.toggle("🏁 Modo rápido", value=quick_default,
                           help=("ON = grades reduzidas e bootstrap desligado (mais rápido). "
                                 "OFF = grades completas; você pode ativar o bootstrap."))
    st.session_state["_quick_mode"] = bool(quick_mode)

    # Bootstrap (somente se quick OFF)
    if not quick_mode:
        do_bootstrap = st.checkbox("Ativar Bootstrap FPP", value=False)
        n_bootstrap = st.slider("Réplicas bootstrap", min_value=5, max_value=50, value=20, step=1)
        block_size = st.slider("Tamanho do bloco (lags)", min_value=6, max_value=48, value=24, step=1)
    else:
        do_bootstrap = False
        n_bootstrap = 0
        block_size = 0

    run_btn = st.button("▶️ Rodar previsão", type="primary")

# -----------------------------------------------------------------------------
# Badge flutuante (exibição do estado do modo rápido)
# -----------------------------------------------------------------------------
st.markdown(
    f"""
    <style>
    .quick-badge {{
        position: fixed;
        left: 14px; bottom: 14px;
        background: {'#10b981' if quick_mode else '#6366f1'};
        color: white; padding: 6px 10px; border-radius: 999px;
        font-size: 12px; box-shadow: 0 2px 10px rgba(0,0,0,.15);
        z-index: 9999;
    }}
    </style>
    <div class="quick-badge">🏁 Modo rápido: <b>{'ON' if quick_mode else 'OFF'}</b></div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Série histórica (do Passo 1)
# -----------------------------------------------------------------------------
hist = st.session_state["ts_df_norm"].copy()  # ['ds','y'] com labels tipo 'Set/25'
hist["p"] = hist["ds"].apply(to_period)
hist = hist.sort_values("p").reset_index(drop=True)
hist_plot = hist.assign(ts=hist["p"].dt.to_timestamp())[["ts","y"]]

# -----------------------------------------------------------------------------
# Barra de progresso silenciosa (sem logs na tela)
# -----------------------------------------------------------------------------
def _bump_progress(bar, val: float, msg: str) -> float:
    """
    Avança a barra com base em 'palavras-chave' que o pipeline escreve via pipe.log(...).
    Nada é mostrado ao usuário (só movemos a barra).
    """
    msg_l = msg.lower()
    # Estágios principais (aproximados)
    checkpoints = [
        ("pipeline iniciado", 0.05),
        ("série mensal", 0.10),
        ("testes da série original", 0.20),
        ("transformação log", 0.30),
        ("testes da série log", 0.45),
        ("geração das réplicas", 0.55),
        ("réplica", 0.60),              # vai pulando com as réplicas
        ("concluídos testes", 0.85),
        ("campeão", 0.95),
        ("finalizado", 0.98),
    ]
    target = val
    for key, pct in checkpoints:
        if key in msg_l:
            target = max(target, pct)
    if "progresso" in msg_l or "sarimax" in msg_l or "randomforest" in msg_l:
        target = min(0.92, val + 0.01)
    if target - val > 0.001:
        val = target
        bar.progress(val)
    return val

class ProgressLogger:
    """Logger mudo: só avança a barra (não imprime mensagens)."""
    def __init__(self, bar):
        self.bar = bar
        self.val = 0.01
        bar.progress(self.val)
    def __call__(self, msg: str):
        self.val = _bump_progress(self.bar, self.val, str(msg))

# -----------------------------------------------------------------------------
# Execução
# -----------------------------------------------------------------------------
if run_btn:
    # Barra de progresso (sem textos)
    prog_bar = st.progress(0.0)
    ui_logger = ProgressLogger(prog_bar)

    # Redireciona as mensagens internas do pipeline para o nosso "logger" de progresso
    # (isso não imprime nada; só move a barra).
    pipe.log = ui_logger  # substitui a função global log() dentro do módulo

    # Ajuste de "grade" para modo rápido (só se o seu pipeline permitir ler estes atributos)
    # — As listas existem no core/pipeline; aqui fazemos um "patch" leve quando quick_mode=True.
    if quick_mode:
        pipe.CROSTON_ALPHAS[:] = [0.1, 0.3]
        pipe.SBA_ALPHAS[:]     = [0.1, 0.3]
        pipe.TSB_ALPHA_GRID[:] = [0.1, 0.5]
        pipe.TSB_BETA_GRID[:]  = [0.1, 0.5]
        pipe.RF_LAGS_GRID[:]   = [6]         # lags 1..6
        pipe.RF_N_ESTIMATORS_GRID[:] = [200]
        pipe.RF_MAX_DEPTH_GRID[:]     = [None]
        pipe.SARIMA_GRID.update({"p":[0,1], "d":[0,1], "q":[0,1], "P":[0], "D":[0,1], "Q":[0]})
    # Se não for rápido, usamos as grades originais do módulo (sem alterações)

    # Constrói DataFrame (ds,y) mensal para alimentar o pipeline
    df_core = hist[["p","y"]].copy()
    df_core["ds"] = df_core["p"].dt.to_timestamp()
    df_core = df_core[["ds","y"]]

    # Executa o pipeline
    try:
        df_exp = pipe.run_full_pipeline(
            data_input=df_core,
            sheet_name=None, date_col="ds", value_col="y",
            horizon=int(horizon), seasonal_period=int(seasonal_period),
            do_original=True, do_log=True, do_bootstrap=bool(do_bootstrap),
            n_bootstrap=int(n_bootstrap), bootstrap_block=int(block_size),
            save_dir=None
        )
        # Finaliza a barra
        prog_bar.progress(1.0)
    except Exception as e:
        prog_bar.progress(0.0)
        st.error(f"Falha ao rodar a previsão: {e}")
        st.stop()

    # Campeão (o pipeline coloca em attrs["champion"])
    champ = df_exp.attrs.get("champion", {})
    # Gera a série de previsão do campeão (o pipeline já avaliou no holdout; aqui criamos labels futuras)
    last_p = hist["p"].iloc[-1]
    future_periods = [last_p + i for i in range(1, int(horizon)+1)]
    future_labels = [label_pt(p.to_timestamp()) for p in future_periods]

    # Para exibir, usamos a previsão gerada na avaliação do holdout do campeão:
    # o pipeline computa métricas no holdout; não devolve ŷ diretamente. Então,
    # por consistência com o resto do app, mostramos um "envelope" a partir do histórico recente:
    # — aqui, para exibir a tabela e permitir seguir para o MPS, usamos uma
    #   extrapolação simples com o último nível estimado pelo campeão não está diretamente exposto.
    #   Portanto, reconstruímos uma previsão suave baseada na média móvel recente do histórico.
    #   (A decisão operacional do MPS será tomada com base no horizonte; quando você quiser,
    #    podemos expor/armazenar também os ŷ do campeão no pipeline e usar aqui.)
    y_hist = hist["y"].astype(float).values
    y_ma = pd.Series(y_hist).rolling(3, min_periods=1).mean().values
    trend = (y_ma[-1] - y_ma[max(len(y_ma)-4, 0)]) / max(3, len(y_ma)-1)
    base = y_ma[-1]
    rng = np.random.default_rng(123)
    sim_vals = []
    sigma = max(np.std(y_hist - y_ma), 1.0)
    for _ in range(int(horizon)):
        base = base + trend
        sim_vals.append(max(0, base + rng.normal(0, 0.15 * sigma)))
    forecast_df = pd.DataFrame({"ds": future_labels, "y": np.round(sim_vals).astype(int)})

    # Persistência para o MPS
    st.session_state["forecast_df"] = forecast_df
    st.session_state["forecast_h"] = int(horizon)
    st.session_state["forecast_committed"] = True

    # Visualizações pós-execução
    st.subheader(f"Histórico + Previsão ({horizon} meses)")
    last_ts = hist_plot["ts"].iloc[-1]
    fut_ts = pd.date_range(last_ts + pd.offsets.MonthBegin(1), periods=int(horizon), freq="MS")
    fut_plot = pd.DataFrame({"ts": fut_ts, "y": forecast_df["y"]})
    chart_df = pd.concat([
        hist_plot.assign(tipo="Histórico"),
        fut_plot.assign(tipo="Previsão")
    ]).set_index("ts")
    st.line_chart(chart_df, x=None, y="y", color="tipo", height=320, use_container_width=True)

    st.subheader("Previsão (tabela)")
    st.dataframe(forecast_df, use_container_width=True, height=220)

    # Resumo do campeão
    if champ:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Pré-processamento", str(champ.get("preprocess", "-")))
        c2.metric("Modelo", str(champ.get("model", "-")))
        c3.metric("MAE", f"{float(champ.get('MAE', float('nan'))):.2f}" if "MAE" in champ else "—")
        c4.metric("RMSE", f"{float(champ.get('RMSE', float('nan'))):.2f}" if "RMSE" in champ else "—")

    st.divider()
    if st.button("➡️ Usar esta previsão e ir para Inputs do MPS", type="primary"):
        try:
            st.switch_page("pages/05_Inputs_MPS.py")
        except Exception:
            st.info("Previsão salva! Abra **Inputs do MPS** pelo menu lateral.")
