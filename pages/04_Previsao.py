# pages/04_Previsao.py
from __future__ import annotations
import re
import time
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------------------------------------------------------
# Título
# -----------------------------------------------------------------------------
st.title("🔮 Passo 2 — Previsão de Demanda")

# -----------------------------------------------------------------------------
# Guardas: precisa do Upload (Passo 1)
# -----------------------------------------------------------------------------
if "ts_df_norm" not in st.session_state:
    st.warning("Preciso da série do Passo 1 (Upload) antes de continuar.")
    st.page_link("pages/01_Upload.py", label="Ir para o Passo 1 — Upload", icon="📤")
    st.stop()

# -----------------------------------------------------------------------------
# Importa o pipeline como módulo (core/pipeline.py)
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import core.pipeline as pipe  # <<< sem ".py"

# -----------------------------------------------------------------------------
# Helpers de datas e série mensal
# -----------------------------------------------------------------------------
_PT = {1:"Jan",2:"Fev",3:"Mar",4:"Abr",5:"Mai",6:"Jun",7:"Jul",8:"Ago",9:"Set",10:"Out",11:"Nov",12:"Dez"}
_REV_PT = {v:k for k, v in _PT.items()}

def _label_pt(ts: pd.Timestamp) -> str:
    return f"{_PT[ts.month]}/{str(ts.year)[-2:]}"

def _to_period_from_label(lbl: str) -> pd.Period:
    try:
        return pd.to_datetime(lbl, dayfirst=True).to_period("M")
    except Exception:
        mon = lbl[:3].title()
        yy = 2000 + int(lbl[-2:])
        return pd.Period(freq="M", year=yy, month=_REV_PT[mon])

def _monthly_series_from_session() -> pd.Series:
    df = st.session_state["ts_df_norm"].copy()
    df["p"] = df["ds"].apply(_to_period_from_label)
    df = df.sort_values("p")
    idx = df["p"].dt.to_timestamp(how="start")
    s = pd.Series(df["y"].astype(float).values, index=idx, name="y").asfreq("MS")
    s = s.interpolate("linear").bfill().ffill().astype(float)
    return s

def _next_n_month_labels(last_ts: pd.Timestamp, n: int) -> list[str]:
    fut = pd.date_range(last_ts + pd.offsets.MonthBegin(1), periods=n, freq="MS")
    return [_label_pt(ts) for ts in fut]

# -----------------------------------------------------------------------------
# Parâmetros de execução
# -----------------------------------------------------------------------------
with st.expander("Parâmetros do experimento", expanded=True):
    c1, c2, c3 = st.columns(3)
    with c1:
        horizon = st.selectbox("Horizonte (meses)", [6, 8, 12], index=0)
    with c2:
        seasonal_period = st.number_input("Período sazonal (m)", 1, 24, 12, step=1)
    with c3:
        mode_fast = st.toggle("🏁 Modo rápido", value=True, help="Reduz grades, desliga/limita bootstrap e acelera o SARIMAX.")

    d1, d2, d3, d4 = st.columns(4)
    with d1:
        do_original = st.checkbox("Usar série original", True)
    with d2:
        do_log = st.checkbox("Usar log + ε", True)
    with d3:
        do_bootstrap = st.checkbox("Usar bootstrap FPP", True)
    with d4:
        n_bootstrap = st.slider("Réplicas bootstrap", 5, 60, 20, step=5, disabled=not do_bootstrap)
    block_size = st.slider("Tamanho do bloco (bootstrap)", 6, 48, 24, step=2, disabled=not do_bootstrap)

# Badge fixo no canto inferior esquerdo
st.markdown(
    f"""
    <style>
      .fast-badge {{
        position: fixed; left: 14px; bottom: 14px; z-index: 9999;
        background: {'#16a34a' if mode_fast else '#6b7280'};
        color: white; padding: 6px 10px; border-radius: 9999px; 
        font-weight: 600; box-shadow: 0 4px 14px rgba(0,0,0,.15);
      }}
    </style>
    <div class="fast-badge">🏁 Modo rápido: {'ON' if mode_fast else 'OFF'}</div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Progresso + logs
# -----------------------------------------------------------------------------
progress_ph = st.empty()
logs_ph = st.container()
st.session_state["_previsao_logs"] = logs_ph

# mapeia mensagens de log para marcos de progresso
_STAGE_HINTS = [
    ("ORIGINAL", 0.05),
    ("Transformação LOG", 0.10),
    ("Geração das réplicas", 0.15),
    ("Croston", 0.30),
    ("SBA", 0.45),
    ("TSB", 0.60),
    ("RandomForest", 0.75),
    ("SARIMAX", 0.90),
    ("CAMPEÃO", 0.98),
]

def _bump_progress(bar, current, msg):
    target = current
    for hint, val in _STAGE_HINTS:
        if hint.lower() in msg.lower():
            target = max(target, val)
    target = min(0.99, max(target, current + 0.01))
    bar.progress(target)
    return target

class ProgressLogger:
    """Substitui pipe.log para também alimentar a barra de progresso."""
    def __init__(self, bar):
        self.bar = bar
        self.val = 0.01
        bar.progress(self.val)
    def __call__(self, msg: str):
        try:
            st.session_state["_previsao_logs"].write(msg)
        except Exception:
            pass
        self.val = _bump_progress(self.bar, self.val, str(msg))

# -----------------------------------------------------------------------------
# Executar
# -----------------------------------------------------------------------------
if st.button("▶️ Rodar previsão", type="primary"):
    base_series = _monthly_series_from_session()
    last_ts = base_series.index[-1]

    # aplica “modo rápido” reduzindo grades e bootstrap no módulo pipeline
    if mode_fast:
        pipe.CROSTON_ALPHAS = [0.1]
        pipe.SBA_ALPHAS = [0.1]
        pipe.TSB_ALPHA_GRID = [0.3]
        pipe.TSB_BETA_GRID = [0.3]
        pipe.RF_LAGS_GRID = [6]
        pipe.RF_N_ESTIMATORS_GRID = [200]
        pipe.RF_MAX_DEPTH_GRID = [None]
        pipe.SARIMA_GRID = {"p":[0,1], "d":[0,1], "q":[0,1], "P":[0], "D":[0,1], "Q":[0]}
        # desliga bootstrap ou limita (você escolhe uma das linhas abaixo)
        do_bootstrap = False
        # n_bootstrap = min(n_bootstrap, 5); block_size = min(block_size, 12)

    # barra + logger conectado
    bar = progress_ph.progress(0.01)
    pipe.log = ProgressLogger(bar)  # redireciona logs do pipeline para UI + progresso

    try:
        with st.spinner("Processando sua previsão…"):
            df_exp = pipe.run_full_pipeline(
                data_input=base_series,
                horizon=int(horizon),
                seasonal_period=int(seasonal_period),
                do_original=bool(do_original),
                do_log=bool(do_log),
                do_bootstrap=bool(do_bootstrap),
                n_bootstrap=int(n_bootstrap),
                bootstrap_block=int(block_size),
                save_dir=None,
            )
        bar.progress(1.0)
        time.sleep(0.2)
        progress_ph.empty()
    except Exception as e:
        progress_ph.empty()
        st.exception(e)
        st.stop()

    # campeão
    champ = pd.Series(df_exp.attrs.get("champion", {}))
    model_name = champ.get("model", "SARIMAX")
    params = champ.get("model_params", "")
    preprocess = str(champ.get("preprocess", "-")).lower()

    # refit/inversão se o campeão veio de LOG
    fwd, inv = (None, None)
    if preprocess.startswith("log"):
        fwd, inv, _ = pipe.make_log_transformers(base_series, window=6)
    s_model = fwd(base_series) if fwd else base_series

    # ----- funções de previsão de cada modelo -----
    def _parse_int(s, key):
        m = re.search(rf"{key}\s*=\s*(-?\d+)", s)
        return int(m.group(1)) if m else None

    def _parse_sarima(pstr: str):
        om = re.search(r"order=\((\d+),(\d+),(\d+)\)", pstr)
        sm = re.search(r"seasonal=\((\d+),(\d+),(\d+),(\d+)\)", pstr)
        if not om: return (0,1,0, 0,1,0, seasonal_period)
        p,d,q = map(int, om.groups())
        if sm:
            P,D,Q,m = map(int, sm.groups())
        else:
            P,D,Q,m = 0,0,0, seasonal_period
        return (p,d,q,P,D,Q,m)

    h = int(horizon)
    y_hist = s_model.values.astype(float)

    def _forecast_croston(alpha):
        _, f = pipe.croston_forecast(y_hist, alpha=alpha, h=h); return f
    def _forecast_sba(alpha):
        _, f = pipe.sba_forecast(y_hist, alpha=alpha, h=h); return f
    def _forecast_tsb(alpha, beta):
        _, f = pipe.tsb_forecast(y_hist, alpha=alpha, beta=beta, h=h); return f
    def _forecast_sarimax(p,d,q,P,D,Q,m):
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        res = SARIMAX(s_model, order=(p,d,q), seasonal_order=(P,D,Q,m),
                      enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        return res.get_forecast(steps=h).predicted_mean.values.astype(float)
    def _forecast_rf(lags_k, n_estimators, max_depth):
        df_sup = pipe.make_supervised_from_series(s_model, list(range(1, lags_k+1)))
        y = df_sup["y"].values
        X = df_sup.drop(columns=["y"])
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=(None if str(max_depth)=="None" else int(max_depth)),
            random_state=42
        )
        model.fit(X.values, y)
        ext = list(s_model.values.astype(float))
        last = s_model.index[-1]
        fut_idx = pd.date_range(last + pd.offsets.MonthBegin(1), periods=h, freq="MS")
        preds = []
        for ts in fut_idx:
            row = {f"lag_{L}": ext[-L] for L in range(1, lags_k+1)}
            # dummies de mês (2..12 pois drop_first=True)
            for m_ in range(2, 12+1):
                row[f"month_{m_}"] = 1 if ts.month == m_ else 0
            xv = np.array([row.get(c, 0.0) for c in X.columns], dtype=float).reshape(1, -1)
            yhat = float(model.predict(xv)[0])
            preds.append(yhat); ext.append(yhat)
        return np.array(preds, dtype=float)

    # gera previsão futura
    try:
        if model_name == "Croston":
            a = _parse_int(params, "alpha") or 0.1
            y_pred_m = _forecast_croston(a)
        elif model_name == "SBA":
            a = _parse_int(params, "alpha") or 0.1
            y_pred_m = _forecast_sba(a)
        elif model_name == "TSB":
            a = _parse_int(params, "alpha") or 0.3
            b = _parse_int(params, "beta") or 0.3
            y_pred_m = _forecast_tsb(a, b)
        elif model_name == "RandomForest":
            k = _parse_int(params, "lags") or int(re.search(r"lags=1\.\.(\d+)", params).group(1))
            n_est = _parse_int(params, "n_estimators") or 200
            mdep = re.search(r"max_depth=([None\d]+)", params)
            mdep = mdep.group(1) if mdep else "None"
            y_pred_m = _forecast_rf(k, n_est, mdep)
        elif model_name == "SARIMAX":
            p,d,q,P,D,Q,m = _parse_sarima(params)
            y_pred_m = _forecast_sarimax(p,d,q,P,D,Q,m)
        else:
            # fallback: sazonal-naive (ex.: LSTM campeão)
            s = base_series
            y_pred_m = (s.values[-12:][:h] if len(s) >= 12 else np.full(h, s.values[-1]))
        y_pred = (pipe.np.clip((pipe.np.asarray(inv(y_pred_m)) if inv else y_pred_m), 0.0, None))
    except Exception as e:
        st.warning(f"Não consegui refazer o ajuste do campeão ({model_name}). Usei um fallback sazonal-naive. Detalhe: {e}")
        s = base_series
        y_pred = s.values[-12:][:h] if len(s) >= 12 else np.full(h, s.values[-1])

    forecast_df = pd.DataFrame({
        "ds": _next_n_month_labels(last_ts, h),
        "y": np.round(y_pred, 0).astype(int)
    })

    # persistência para MPS
    st.session_state["forecast_df"] = forecast_df
    st.session_state["forecast_h"] = int(horizon)
    st.session_state["forecast_committed"] = True

    # -----------------------------------------------------------------------------
    # Visualizações
    # -----------------------------------------------------------------------------
    st.subheader("Resultado")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Campeão", model_name)
    c2.metric("Pré-processamento", str(champ.get("preprocess","-")).capitalize())
    c3.metric("MAE", f"{champ.get('MAE', np.nan):.2f}")
    c4.metric("RMSE", f"{champ.get('RMSE', np.nan):.2f}")

    hist_df = st.session_state["ts_df_norm"].copy()
    hist_df["_ts"] = hist_df["ds"].apply(lambda s: _to_period_from_label(s).to_timestamp())
    fut_ts = pd.date_range(hist_df["_ts"].iloc[-1] + pd.offsets.MonthBegin(1), periods=h, freq="MS")
    chart_df = pd.concat([
        pd.DataFrame({"ts": hist_df["_ts"], "y": hist_df["y"].astype(float), "tipo": "Histórico"}),
        pd.DataFrame({"ts": fut_ts, "y": forecast_df["y"].astype(float), "tipo": "Previsão"})
    ]).set_index("ts")
    st.line_chart(chart_df, y="y", color="tipo", height=330, use_container_width=True)

    st.subheader("Tabela — Previsão")
    st.dataframe(forecast_df, use_container_width=True, height=260)

    with st.expander("Experimentos (todas as linhas)"):
        st.dataframe(df_exp, use_container_width=True, height=380)

    st.success("Previsão salva. Você já pode avançar para o **MPS**.")

st.divider()
st.page_link("pages/05_Inputs_MPS.py", label="➡️ Ir para 05_Inputs_MPS (configurar Inputs)", icon="⚙️")
