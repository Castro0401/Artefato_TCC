# pages/04_Previsao.py
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------------------------------------------------------
# Configuração da página
# -----------------------------------------------------------------------------
st.title("🔮 Passo 2 — Previsão de Demanda")

# -----------------------------------------------------------------------------
# Guarda de etapa: precisa do Upload (Passo 1)
# -----------------------------------------------------------------------------
if "ts_df_norm" not in st.session_state:
    st.warning("Preciso da série do Passo 1 (Upload) antes de continuar.")
    st.page_link("pages/01_Upload.py", label="Ir para o Passo 1 — Upload", icon="📤")
    st.stop()

# -----------------------------------------------------------------------------
# Import do pipeline como módulo (core/pipeline.py)
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]  # raiz do projeto (onde fica a pasta core/)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import core.pipeline as pl  # noqa: E402
except Exception as e:
    st.error(
        "Não consegui importar `core/pipeline.py`. "
        "Confira se o arquivo existe e se a pasta `core/` tem um `__init__.py`."
    )
    st.exception(e)
    st.stop()

# -----------------------------------------------------------------------------
# Helpers de data
# -----------------------------------------------------------------------------
_PT = {1: "Jan", 2: "Fev", 3: "Mar", 4: "Abr", 5: "Mai", 6: "Jun", 7: "Jul",
       8: "Ago", 9: "Set", 10: "Out", 11: "Nov", 12: "Dez"}
_REV_PT = {v: k for k, v in _PT.items()}


def label_pt(ts: pd.Timestamp) -> str:
    return f"{_PT[ts.month]}/{str(ts.year)[-2:]}"


def to_period(s: str) -> pd.Period:
    # converte "Set/25" -> Period('2025-09','M'); datas YYYY-MM-DD também funcionam
    try:
        return pd.to_datetime(s, dayfirst=True).to_period("M")
    except Exception:
        mon = s[:3].capitalize()
        yy = int(s[-2:]) + 2000
        month_num = _REV_PT.get(mon, None)
        if month_num is None:
            raise ValueError(f"Formato de mês inválido: {s}")
        return pd.Period(freq="M", year=yy, month=month_num)


def next_n_months(last_period: pd.Period, n: int) -> list[str]:
    out, p = [], last_period + 1
    for _ in range(n):
        out.append(label_pt(p.to_timestamp()))
        p += 1
    return out


# -----------------------------------------------------------------------------
# Entrada: série mensal do Passo 1
# Formato guardado no upload: df com ['ds','y'] onde ds é "Set/25", "Out/25", ...
# -----------------------------------------------------------------------------
hist = st.session_state["ts_df_norm"].copy()
hist["p"] = hist["ds"].apply(to_period)
hist = hist.sort_values("p").reset_index(drop=True)

# -----------------------------------------------------------------------------
# Sidebar — parâmetros do experimento
# -----------------------------------------------------------------------------
with st.sidebar:
    st.subheader("⚙️ Configurações da previsão")

    # lembrar último horizonte salvo no session_state
    last_h = int(st.session_state.get("forecast_h", 6))
    horizon = st.selectbox(
        "Horizonte (meses)",
        [6, 8, 12],
        index={6: 0, 8: 1, 12: 2}.get(last_h, 0),
        help="O MPS usará esse mesmo horizonte.",
    )

    seasonal_period = st.number_input(
        "Período sazonal (m)",
        min_value=1,
        max_value=24,
        value=int(st.session_state.get("seasonal_period", 12)),
        step=1,
        help="Para série mensal tipicamente use 12.",
    )
    # persistir sem resetar upload
    st.session_state["seasonal_period"] = int(seasonal_period)

    st.markdown("---")
    do_log = st.checkbox("Usar transformação Log + ε (auto)", value=True)
    do_bootstrap = st.checkbox("Usar Bootstrap FPP", value=True)

    if do_bootstrap:
        st.caption("**Bootstrap FPP** = Box–Cox (λ MLE) + STL robusta + reamostragem em blocos dos resíduos + inversão.")
        st.markdown(
            "_Réplicas:_ quantas séries sintéticas serão geradas para testar robustez.  "
            "_Tamanho do bloco:_ controla a preservação da autocorrelação nos resíduos; "
            "valores maiores preservam dependências mais longas."
        )
        n_bootstrap = st.slider("Réplicas (n)", min_value=5, max_value=50, value=20, step=1)
        bootstrap_block = st.slider("Tamanho do bloco", min_value=3, max_value=48, value=24, step=1)
    else:
        n_bootstrap = 0
        bootstrap_block = 0

    st.markdown("---")
    fast_mode = st.toggle("🏎️ Modo rápido (menos testes)", value=False, help="Desliga o Bootstrap automaticamente e usa grades compactas.")
    if fast_mode:
        # acelera: priorize original + log, sem bootstrap
        do_bootstrap = False
        n_bootstrap = 0
        bootstrap_block = 0

    st.markdown("---")
    run = st.button("▶️ Rodar previsão", type="primary", use_container_width=True)

# -----------------------------------------------------------------------------
# Execução do pipeline com barra de progresso silenciosa
# -----------------------------------------------------------------------------
forecast_df = None
champion = None

if run:
    progress_box = st.container()
    pbar = progress_box.progress(0, text="Iniciando…")

    # mapa de fases -> progresso aproximado
    phases = {
        "PIPELINE INICIADO": 0.05,
        "REALIZANDO TESTES DA SÉRIE ORIGINAL": 0.15,
        "PREPARANDO TRANSFORMAÇÃO LOG": 0.20,
        "REALIZANDO TESTES DA SÉRIE LOG": 0.35,
        "GERAÇÃO DAS RÉPLICAS SINTÉTICAS": 0.40,
        "RÉPLICAS GERADAS": 0.45,
        "CROSTON": 0.30,
        "SBA": 0.40,
        "TSB": 0.50,
        "RANDOMFOREST": 0.60,
        "SARIMAX": 0.75,
        "CAMPEÃO": 0.92,
        "PIPELINE FINALIZADO": 1.00,
    }

    def _progress_from_msg(msg: str) -> float | None:
        m = str(msg).upper()
        for key, val in phases.items():
            if key in m:
                return val
        # heurística para pequenos avanços entre logs conhecidos
        if "PROGRESSO" in m or "RÉPLICA" in m:
            return min(state["v"] + 0.02, 0.9)
        return None

    # estado mutável p/ usar em closures (evita 'nonlocal' no nível de módulo)
    state = {"v": 0.0}

    def _bar_update(x: float | None, fallback: float = 0.01):
        if x is None:
            x = min(state["v"] + fallback, 0.98)
        if x > state["v"]:
            state["v"] = x
            pbar.progress(min(1.0, state["v"]))

    # hook no logger do pipeline
    pl_log_original = pl.log

    def _hooked_log(msg: str):
        try:
            _bar_update(_progress_from_msg(msg))
        except Exception:
            pass
        pl_log_original(msg)

    pl.log = _hooked_log

    try:
        # prepara df de entrada no formato aceito pelo pipeline (ds,y mensais)
        df_in = hist[["p", "y"]].rename(columns={"p": "ds"})
        # roda o pipeline
        resultados = pl.run_full_pipeline(
            data_input=df_in,
            sheet_name=None,
            date_col="ds",
            value_col="y",
            horizon=int(horizon),
            seasonal_period=int(seasonal_period),
            do_original=True,
            do_log=bool(do_log),
            do_bootstrap=bool(do_bootstrap),
            n_bootstrap=int(n_bootstrap) if do_bootstrap else 0,
            bootstrap_block=int(bootstrap_block) if do_bootstrap else 0,
            save_dir=None,
        )
        _bar_update(0.99)
        _bar_update(1.0)

        # Campeão e sMAPE reais vindos do pipeline
        champion = resultados.attrs.get("champion", {})
        # Observação: o pipeline não entrega a série prevista; abaixo montamos
        # uma projeção simples (apenas para visual) até você estender o pipeline.
    except Exception as e:
        pbar.progress(0.0, text="Erro ao rodar previsão.")
        st.exception(e)
    finally:
        # restaura logger e remove barra
        pl.log = pl_log_original
        progress_box.empty()

# -----------------------------------------------------------------------------
# Geração de uma previsão visual (placeholder) e UI de resultados
# -----------------------------------------------------------------------------
# Mesmo que não clique em "Rodar", deixamos a visualização pronta com horizonte atual
hist_work = hist.copy()
y = hist_work["y"].astype(float).values
y_ma = pd.Series(y).rolling(3, min_periods=1).mean().values
sigma = max(np.std(y - y_ma), 1.0)
last_p = hist_work["p"].iloc[-1]
future_labels = next_n_months(last_p, int(horizon))

# projeção simples (placeholder visual)
trend = (y_ma[-1] - y_ma[max(len(y_ma) - 4, 0)]) / max(3, len(y_ma) - 1)
sim_vals, base = [], y_ma[-1]
rng = np.random.default_rng(123)
for _ in range(int(horizon)):
    base = base + trend
    sim_vals.append(max(0, base + rng.normal(0, 0.15 * sigma)))
forecast_df = pd.DataFrame({"ds": future_labels, "y": np.round(sim_vals, 0).astype(int)})

# Persistência para MPS
st.session_state["forecast_df"] = forecast_df
st.session_state["forecast_h"] = int(horizon)
st.session_state["forecast_committed"] = True
if horizon == 6:
    st.session_state["forecast_df_6m"] = forecast_df.copy()

# -----------------------------------------------------------------------------
# Visualizações
# -----------------------------------------------------------------------------
left, right = st.columns([2, 1])

with left:
    st.subheader(f"Histórico + Previsão ({horizon} meses)")
    hist_plot = hist_work.assign(ts=hist_work["p"].dt.to_timestamp())[["ts", "y"]]
    last_ts = hist_plot["ts"].iloc[-1]
    fut_ts = pd.date_range(last_ts + pd.offsets.MonthBegin(1), periods=int(horizon), freq="MS")
    fut_plot = pd.DataFrame({"ts": fut_ts, "y": forecast_df["y"]})

    chart_df = pd.concat(
        [hist_plot.assign(tipo="Histórico"), fut_plot.assign(tipo="Previsão")]
    ).set_index("ts")

    st.line_chart(chart_df, x=None, y="y", color="tipo", height=340, use_container_width=True)

with right:
    st.subheader("Resumo do experimento")
    if champion:
        smape = champion.get("sMAPE", None)
        mae = champion.get("MAE", None)
        rmse = champion.get("RMSE", None)
        st.metric("sMAPE (campeão)", f"{smape:.2f} %" if smape is not None else "—")
        st.metric("Modelo", f"{champion.get('model','—')}")
        st.caption(
            f"Pré-processamento: **{champion.get('preprocess','—')}**  \n"
            f"Parâmetros: {champion.get('model_params','—')}"
        )
        if mae is not None and rmse is not None:
            st.caption(f"MAE={mae:.2f} | RMSE={rmse:.2f}")
    else:
        st.info("Rode o experimento para ver o campeão e a sMAPE.")

st.subheader("Previsão (tabela)")
st.dataframe(forecast_df, use_container_width=True, height=240)

st.caption(
    "Obs.: a série prevista exibida é um **placeholder visual**. "
    "O pipeline atual escolhe o melhor modelo e reporta métricas (incluindo sMAPE), "
    "mas não retorna a série prevista. Quando você estender `pipeline.py` para "
    "expor as previsões, basta substituir este bloco."
)

# -----------------------------------------------------------------------------
# Navegação
# -----------------------------------------------------------------------------
st.divider()
go_mps = st.button(
    "➡️ Usar esta previsão e ir para configuração dos Inputs do MPS",
    type="primary",
)
if go_mps:
    try:
        st.switch_page("pages/05_Inputs_MPS.py")
    except Exception:
        st.info("Previsão salva! Abra **Inputs do MPS** pelo menu lateral.")
