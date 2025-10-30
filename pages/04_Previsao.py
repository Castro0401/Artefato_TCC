# -*- coding: utf-8 -*-
from __future__ import annotations
"""
04_Previsao.py — versão final minimalista
- Consome a série validada no 01_Upload.py via st.session_state["ts_df_norm"] (colunas: ["ds","y"])
- Converte rótulos "Mon/YY" para datas (MS) e cria série mensal contínua (asfreq + interpolate)
- Importa core/pipeline.py com caminho robusto
"""

import sys
from pathlib import Path
import traceback
import pandas as pd
import streamlit as st

# =============================
# Inserir caminhos para importar core/pipeline
# =============================
ROOT = Path(__file__).resolve().parent.parent      # .../artefato_tcc (raiz)
CORE = ROOT / "core"
for p in (ROOT, CORE):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

try:
    from core import pipeline as pipe
except ModuleNotFoundError:
    import pipeline as pipe  # fallback

st.set_page_config(page_title="Previsão", page_icon="🔮", layout="wide")
st.title("🔮 Passo 2 — Previsão (1 clique)")

# =============================
# Recupera a série do Upload
# =============================
if not st.session_state.get("upload_ok"):
    st.error("Nenhuma série encontrada. Volte ao Passo 1 (Upload) para carregar os dados.")
    st.stop()

_ts = st.session_state.get("ts_df_norm")
if not isinstance(_ts, pd.DataFrame) or not {"ds", "y"}.issubset(_ts.columns):
    st.error("Formato inesperado da série: esperado DataFrame com colunas ['ds','y'].")
    st.stop()

product_name = st.session_state.get("product_name", "Produto")
st.caption(f"Série atual: **{product_name}**")
st.dataframe(_ts.head(12), use_container_width=True)

# =============================
# Converte rótulos para datas (primeiro dia do mês) e cria Series mensal contínua
# =============================
_PT_MON2NUM = {
    "Jan": 1, "Fev": 2, "Mar": 3, "Abr": 4, "Mai": 5, "Jun": 6,
    "Jul": 7, "Ago": 8, "Set": 9, "Out": 10, "Nov": 11, "Dez": 12,
}

def _label_to_month_start(val) -> pd.Timestamp:
    # Se já vier datetime, só converte
    if isinstance(val, (pd.Timestamp,)):
        return pd.to_datetime(val)
    s = str(val)
    try:
        if "/" in s:
            mon, yy = s.split("/")
            y = 2000 + int(yy)  # regra simples; ajuste se necessário
            m = _PT_MON2NUM.get(mon)
            if m is None:
                return pd.to_datetime(s, errors="coerce")
            return pd.Timestamp(year=y, month=m, day=1)
        return pd.to_datetime(s, errors="coerce")
    except Exception:
        return pd.NaT

_idx = _ts["ds"].map(_label_to_month_start)
if _idx.isna().any():
    st.warning("Alguns rótulos de data não foram convertidos e serão descartados.")

s_monthly = (
    pd.Series(_ts.loc[_idx.notna(), "y"].astype(float).to_numpy(), index=_idx[_idx.notna()])
      .sort_index()
      .asfreq("MS")           # grade mensal contínua
      .interpolate("linear")  # igual ao preparo usado ao ler Excel no terminal
      .bfill()
      .ffill()
)

# =============================
# Parâmetros (fixos para bater com o terminal agora)
# =============================
HORIZON = 6
SEASONAL_PERIOD = 12
DO_ORIGINAL = True
DO_LOG = True
DO_BOOTSTRAP = True
N_BOOTSTRAP = 20
BOOTSTRAP_BLOCK = 24

run = st.button("▶️ Rodar previsão", type="primary")

if run:
    try:
        with st.spinner("Executando pipeline… isso pode levar alguns minutos…"):
            resultados = pipe.run_full_pipeline(
                data_input=s_monthly,
                sheet_name=None, date_col=None, value_col=None,
                horizon=HORIZON, seasonal_period=SEASONAL_PERIOD,
                do_original=DO_ORIGINAL, do_log=DO_LOG, do_bootstrap=DO_BOOTSTRAP,
                n_bootstrap=N_BOOTSTRAP, bootstrap_block=BOOTSTRAP_BLOCK,
                save_dir=None,
            )

        champ = resultados.attrs.get("champion", {})

        st.subheader("🏆 Modelo Campeão (métricas)")
        def _fmt(x):
            try:
                return f"{float(x):.4g}"
            except Exception:
                return str(x)

        cols = st.columns(4)
        cols[0].metric("MAE", _fmt(champ.get("MAE")))
        cols[1].metric("sMAPE (%)", _fmt(champ.get("sMAPE")))
        cols[2].metric("RMSE", _fmt(champ.get("RMSE")))
        cols[3].metric("MAPE (%)", _fmt(champ.get("MAPE")))

        st.write("Parâmetros do campeão:")
        st.json({
            "preprocess": champ.get("preprocess"),
            "preprocess_params": champ.get("preprocess_params"),
            "model": champ.get("model"),
            "model_params": champ.get("model_params"),
        })

        st.subheader("📋 Experimentos (resumo)")
        st.dataframe(resultados.reset_index(drop=True), use_container_width=True)

    except Exception:
        st.error("Falha ao executar a previsão. Veja o traceback abaixo:")
        st.code("\n".join(traceback.format_exc().splitlines()), language="text")
