# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

st.title("🔮 Passo 2 — Previsão de Demanda")

# ---------------- Guardas ----------------
if "ts_df_norm" not in st.session_state:
    st.warning("Preciso da série do Passo 1 (Upload) antes de continuar.")
    st.page_link("pages/01_Upload.py", label="Ir para o Passo 1 — Upload", icon="📤")
    st.stop()

# --------- Import do pipeline ----------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
try:
    import core.pipeline as pipe
except Exception as e:
    st.error(f"Falha ao importar core/pipeline.py: {e}")
    st.stop()

# ---------- Helpers de data ------------
_PT2NUM = {"Jan":1,"Fev":2,"Mar":3,"Abr":4,"Mai":5,"Jun":6,"Jul":7,"Ago":8,"Set":9,"Out":10,"Nov":11,"Dez":12}
_NUM2PT = {v:k for k,v in _PT2NUM.items()}

def parse_label_to_timestamp(s: str) -> pd.Timestamp:
    s = str(s).strip()
    ts = pd.to_datetime(s, errors="coerce", dayfirst=True)
    if pd.notna(ts):
        return pd.Timestamp(ts.normalize()).to_period("M").to_timestamp(how="start")
    try:
        mon = s[:3].capitalize()
        yy = int(s[-2:]) + 2000
        mm = _PT2NUM[mon]
        return pd.Timestamp(year=yy, month=mm, day=1)
    except Exception:
        return pd.NaT

def label_pt(ts: pd.Timestamp) -> str:
    return f"{_NUM2PT[ts.month]}/{str(ts.year)[-2:]}"

def df_labels_to_datetime(df_in: pd.DataFrame) -> pd.DataFrame:
    out = df_in.copy()
    out["ds"] = out["ds"].apply(parse_label_to_timestamp)
    out["y"] = pd.to_numeric(out["y"], errors="coerce")
    out = out.dropna(subset=["ds", "y"]).sort_values("ds").reset_index(drop=True)
    return out

def seasonal_naive_forecast(hist_df: pd.DataFrame, h: int, m: int = 12) -> pd.DataFrame:
    """
    Previsão provisória: repete o valor do mesmo mês do ano anterior (sazonal ingênuo).
    hist_df: ['ds'(Timestamp MS), 'y'(float)]
    Retorna DataFrame ['ds'(label tipo "Set/25"), 'y'] para h meses à frente.
    """
    hist_df = hist_df.sort_values("ds").reset_index(drop=True)
    if len(hist_df) < m:
        base = float(hist_df["y"].iloc[-1])
        fut_ts = pd.date_range(hist_df["ds"].iloc[-1] + pd.offsets.MonthBegin(1), periods=h, freq="MS")
        return pd.DataFrame({"ds":[label_pt(t) for t in fut_ts], "y":[base]*h})

    # mapa de (mês) -> último valor observado naquele mês (do último ano completo)
    hist_df["month"] = hist_df["ds"].dt.month
    last_year = hist_df["ds"].dt.year.max()
    # preferir ano passado se existir, senão usa os últimos valores por mês
    ref = hist_df[hist_df["ds"].dt.year == last_year - 1]
    if ref.empty:
        ref = hist_df.copy()
    month_map = ref.dropna(subset=["y"]).groupby("month")["y"].last().to_dict()

    fut_ts = pd.date_range(hist_df["ds"].iloc[-1] + pd.offsets.MonthBegin(1), periods=h, freq="MS")
    fut_vals = []
    for t in fut_ts:
        fut_vals.append(float(month_map.get(t.month, hist_df["y"].iloc[-1])))
    return pd.DataFrame({"ds":[label_pt(t) for t in fut_ts], "y":fut_vals})

# --------- Sidebar de parâmetros ---------
with st.sidebar:
    st.header("⚙️ Parâmetros")
    horizon = st.selectbox("Horizonte (meses)", [6, 8, 12],
                           index={6:0, 8:1, 12:2}.get(int(st.session_state.get("forecast_h", 6)), 0))
    seasonal_period = st.number_input("Período sazonal (m)", min_value=1, max_value=24, value=12, step=1)

    st.subheader("Pré-processamentos")
    use_log = st.checkbox("Aplicar log + ε", value=True)
    use_boot = st.checkbox("Usar Bootstrap FPP", value=True)
    if use_boot:
        st.caption("**Bootstrap** — Réplicas: quantas séries sintéticas gerar. "
                   "Quanto maior, mais robusto (e mais lento).")
        n_boot = st.slider("Réplicas (n_bootstrap)", 5, 50, 20, step=1)
        block = 24  # tamanho de bloco fixo para mensal
    else:
        n_boot, block = 0, 24

    st.subheader("🏎️ Desempenho")
    fast_mode = st.toggle("Modo rápido (menos combinações)", value=False)

    st.subheader("Visualização")
    show_experiments = st.toggle("Deseja ver os experimentos?", value=False)

st.info("Clique em **Rodar previsão** para executar os experimentos.")

run_btn = st.button("▶️ Rodar previsão", type="primary")

if run_btn:
    # 1) Normaliza série do Upload
    hist_labels = st.session_state["ts_df_norm"][["ds","y"]].copy()
    try:
        df_in = df_labels_to_datetime(hist_labels)
    except Exception as e:
        st.error(f"Erro ao normalizar datas do histórico: {e}")
        st.stop()
    if df_in.empty:
        st.error("A série ficou vazia após a conversão de datas. Verifique o Upload.")
        st.stop()

    # 2) Modo rápido: grades menores
    if fast_mode:
        try:
            pipe.CROSTON_ALPHAS = [0.1, 0.3]
            pipe.SBA_ALPHAS = [0.1, 0.3]
            pipe.TSB_ALPHA_GRID = [0.3]
            pipe.TSB_BETA_GRID = [0.3]
            pipe.RF_LAGS_GRID = [3]
            pipe.RF_N_ESTIMATORS_GRID = [200]
            pipe.RF_MAX_DEPTH_GRID = [None]
            pipe.SARIMA_GRID = {"p":[0,1], "d":[0,1], "q":[0,1], "P":[0], "D":[0,1], "Q":[0]}
        except Exception:
            pass

    # 3) Executa pipeline (sem logs visuais)
    with st.spinner("Processando sua previsão…"):
        try:
            resultados = pipe.run_full_pipeline(
                data_input=df_in,  # DataFrame com ds Timestamp + y
                sheet_name=None, date_col="ds", value_col="y",
                horizon=int(horizon), seasonal_period=int(seasonal_period),
                do_original=True, do_log=bool(use_log), do_bootstrap=bool(use_boot),
                n_bootstrap=int(n_boot) if use_boot else 0,
                bootstrap_block=int(block),
                save_dir=None,
            )
        except Exception as e:
            st.error(f"Ocorreu um erro durante a execução: {e}")
            st.stop()

    st.success("Experimentos concluídos com sucesso! ✅")

    # 4) Painel campeão
    champ = resultados.attrs.get("champion", {})
    st.subheader("🏆 Modelo campeão")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Pré-processamento", str(champ.get("preprocess", "-")))
    c2.metric("Modelo", str(champ.get("model", "-")))
    try:
        c3.metric("sMAPE", f"{float(champ.get('sMAPE', np.nan)):.1f} %")
    except Exception:
        c3.metric("sMAPE", "-")
    try:
        c4.metric("MAE", f"{float(champ.get('MAE', np.nan)):,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
    except Exception:
        c4.metric("MAE", "-")

    # 5) Histórico + Previsão (gráfico)
    #    Previsão provisória por sazonal ingênuo (até pipeline retornar forecast futuro)
    forecast_df = seasonal_naive_forecast(df_in[["ds","y"]], h=int(horizon), m=int(seasonal_period))
    st.session_state["forecast_df"] = forecast_df.copy()
    st.session_state["forecast_h"] = int(horizon)
    st.session_state["forecast_committed"] = True

    st.subheader(f"Histórico + Previsão ({int(horizon)} meses)")
    hist_plot = df_in[["ds","y"]].rename(columns={"ds":"ts"}).copy()
    hist_plot["tipo"] = "Histórico"
    # construir datas futuras a partir dos labels
    last_ts = df_in["ds"].max()
    fut_ts = pd.date_range(last_ts + pd.offsets.MonthBegin(1), periods=int(horizon), freq="MS")
    fut_plot = pd.DataFrame({"ts": fut_ts, "y": forecast_df["y"], "tipo": "Previsão"})
    chart_df = pd.concat([hist_plot, fut_plot], ignore_index=True).set_index("ts")
    st.line_chart(chart_df, x=None, y="y", color="tipo", height=340, use_container_width=True)

    # 6) Experimentos — exibir apenas se usuário quiser
    if show_experiments:
        st.subheader("Resultados dos experimentos")
        df = resultados.copy().reset_index(drop=True)

        # força tudo a string/num simples (sem objetos/arrays/Period/etc)
        def _cell_safe(v):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return ""
            if isinstance(v, (int, float, np.integer, np.floating)):
                # manter tipo numérico simples
                return float(v) if isinstance(v, np.floating) else int(v) if isinstance(v, np.integer) else v
            return str(v)

        rows = [{col: _cell_safe(df.iloc[i][col]) for col in df.columns} for i in range(len(df))]
        st.dataframe(rows, use_container_width=True, height=380)

    st.divider()
    st.page_link("pages/05_Inputs_MPS.py",
                 label="➡️ Ir para 05_Inputs_MPS (configurar Inputs do MPS)",
                 icon="⚙️")
