# -*- coding: utf-8 -*-
from __future__ import annotations

"""
04_Previsao.py — versão simples e acoplada ao fluxo
- Usa a série enviada/validada em pages/01_Upload.py via st.session_state
- Converte para série mensal contínua (freq='MS') com interpolação linear
- Chama pipe.run_full_pipeline passando uma *Series* para replicar o comportamento do terminal
"""

import traceback
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

import pipeline as pipe  # precisa estar acessível no PYTHONPATH/pasta do app

st.set_page_config(page_title="Previsão", page_icon="🔮", layout="wide")
st.title("🔮 Passo 2: Previsão (1 clique)")

# === helpers ===
_PT_MON2NUM = {
    "Jan": 1, "Fev": 2, "Mar": 3, "Abr": 4, "Mai": 5, "Jun": 6,
    "Jul": 7, "Ago": 8, "Set": 9, "Out": 10, "Nov": 11, "Dez": 12,
}

def _restore_month_start_from_label(label: str) -> pd.Timestamp:
    """Converte rótulos do tipo "Set/25" para Timestamp YYYY-MM-01.
    Regra do ano: 20 + YY (ex.: 25 -> 2025). Ajuste aqui se tiver sérios < 2000.
    """
    try:
        mon_pt, yy = label.split("/")
        y = 2000 + int(yy)  # heurística: anos 2000+
        m = _PT_MON2NUM[mon_pt]
        return pd.Timestamp(year=y, month=m, day=1)
    except Exception:
        # Tenta parsear diretamente (caso já venha em outro formato)
        return pd.to_datetime(label, errors="coerce")

# === checagens de sessão ===
if not st.session_state.get("upload_ok"):
    st.error("Nenhuma série carregada ainda. Volte ao Passo 1 para fazer o upload.")
    st.page_link("pages/01_Upload.py", label="⬅️ Ir para Upload")
    st.stop()

# Série mensal normalizada que vem do 01_Upload.py
# Esperado: DataFrame com colunas ["ds", "y"], onde "ds" está no formato "Set/25"
ts_df_norm = st.session_state.get("ts_df_norm")
product_name = st.session_state.get("product_name", "Produto")

if ts_df_norm is None or not isinstance(ts_df_norm, pd.DataFrame) or set(ts_df_norm.columns) != {"ds","y"}:
    st.error("Formato inesperado da série em memória. Refaça o upload no Passo 1.")
    st.stop()

st.write(f"Item atual: **{product_name}**")
st.dataframe(ts_df_norm.head(15), use_container_width=True)

# === converte rótulos para datas (MonthStart) e prepara Series mensal contínua ===
# Tenta restaurar timestamps a partir de rótulos "Mon/YY".
if pd.api.types.is_datetime64_any_dtype(ts_df_norm["ds"]):
    idx = pd.to_datetime(ts_df_norm["ds"])  # já são datas
else:
    idx = ts_df_norm["ds"].apply(_restore_month_start_from_label)

s_monthly = (
    pd.Series(ts_df_norm["y"].astype(float).to_numpy(), index=idx)
      .sort_index()
      .asfreq("MS")           # garante grade mensal contínua
      .interpolate("linear")  # igual ao terminal
      .bfill()
      .ffill()
)

# Parâmetros fixos para igualar ao terminal (ajuste se necessário)
HORIZON = 6
SEASONAL_PERIOD = 12
DO_ORIGINAL = True
DO_LOG = True
DO_BOOTSTRAP = True
N_BOOTSTRAP = 20
BOOTSTRAP_BLOCK = 24

st.divider()

if st.button("▶️ Rodar previsão", type="primary"):
    try:
        with st.status("Executando pipeline…", expanded=True) as status:
            st.write("Preparando série mensal contínua e chamando `run_full_pipeline`…")
            resultados = pipe.run_full_pipeline(
                data_input=s_monthly,          # passa Series para replicar caminho do terminal
                sheet_name=None, date_col=None, value_col=None,
                horizon=HORIZON, seasonal_period=SEASONAL_PERIOD,
                do_original=DO_ORIGINAL, do_log=DO_LOG, do_bootstrap=DO_BOOTSTRAP,
                n_bootstrap=N_BOOTSTRAP, bootstrap_block=BOOTSTRAP_BLOCK,
                save_dir=None,
            )
            status.update(label="Pipeline finalizado.", state="complete")

        champ = resultados.attrs.get("champion", {})

        st.subheader("🏆 Modelo Campeão")
        if champ:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("MAE", f"{champ.get('MAE', float('nan')):.4g}")
            c2.metric("sMAPE (%)", f"{champ.get('sMAPE', float('nan')):.4g}")
            c3.metric("RMSE", f"{champ.get('RMSE', float('nan')):.4g}")
            c4.metric("MAPE (%)", f"{champ.get('MAPE', float('nan')):.4g}")

            st.write({
                "preprocess": champ.get("preprocess"),
                "preprocess_params": champ.get("preprocess_params"),
                "model": champ.get("model"),
                "model_params": champ.get("model_params"),
            })
        else:
            st.warning("Campeão não encontrado nos atributos. Verifique logs do pipeline.")

        st.subheader("📋 Experimentos")
        st.dataframe(resultados.reset_index(drop=True), use_container_width=True)

    except Exception as e:
        st.error("Falha ao executar a previsão. Detalhes abaixo:")
        st.exception(e)
        import traceback as _tb
        st.code("\n".join(_tb.format_exc().splitlines()[-40:]), language="text")
