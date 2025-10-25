# pages/06_MPS.py
from __future__ import annotations
import io
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

st.title("🗓️ 06_MPS — Plano Mestre de Produção (mensal)")

# -------- Guardas --------
if "ts_df_norm" not in st.session_state:
    st.warning("Preciso da série do Passo 1 (Upload) antes de continuar.")
    st.page_link("pages/01_Upload.py", label="Ir para o Passo 1 — Upload")
    st.stop()

if not st.session_state.get("forecast_committed", False):
    st.warning("Preciso que você **salve a previsão** no Passo 2 (Previsão) antes de abrir o MPS.")
    st.page_link("pages/04_Previsao.py", label="Ir para o Passo 2 — Previsão")
    st.stop()

if "forecast_df" not in st.session_state or "forecast_h" not in st.session_state:
    st.error("Previsão não encontrada no estado. Volte ao Passo 2 e salve a previsão.")
    st.page_link("pages/04_Previsao.py", label="Ir para o Passo 2 — Previsão")
    st.stop()

if "mps_inputs" not in st.session_state:
    st.warning("Antes do MPS, configure os **Inputs do MPS**.")
    st.page_link("pages/05_Inputs_MPS.py", label="Ir para 05_Inputs_MPS")
    st.stop()

# -------- Core import --------
ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from core.mps import compute_mps_monthly

# -------- Dados / Inputs --------
fcst = st.session_state["forecast_df"].copy()[["ds", "y"]]
horizon = int(st.session_state["forecast_h"])
labels = fcst["ds"].tolist()
inp = st.session_state["mps_inputs"]

# Pedidos firmes (se não houver, cria zeros)
orders_df = st.session_state.get("mps_firm_orders", pd.DataFrame({"ds": labels, "y": [0]*len(labels)})).copy()
if list(orders_df["ds"]) != labels:
    orders_df = pd.DataFrame({"ds": labels, "y": [0]*len(labels)})

# Aviso: valores vêm da página anterior
st.info(
    "Os parâmetros abaixo **vêm da página 05_Inputs_MPS**. "
    "Para ajustar política de lote, tamanhos, estoque em mão, lead time, SS, pedidos em carteira e congelamento, "
    "volte à página de inputs.",
    icon="ℹ️",
)

# Snapshot dos parâmetros utilizados
lot_policy = inp.get("lot_policy_default", "FX")
lot_size   = int(inp.get("lot_size_default", 150))
initial_inventory = int(inp.get("initial_inventory_default", 55))
lead_time = int(inp.get("lead_time_default", 1))
frozen_range = inp.get("frozen_range", (labels[0], labels[0])) if labels else ("","")

# -------- SS variável (a partir dos inputs) --------
z_map = {"90%":1.282, "95%":1.645, "97.5%":1.960, "99%":2.326}
auto_ss = bool(inp.get("auto_ss", True))
ss_series = None
if auto_ss:
    method = inp.get("ss_method", "CV (%)")
    z = z_map.get(inp.get("z_choice","95%"), 1.645)
    if method == "CV (%)":
        cv = float(inp.get("cv_pct", 15.0)) / 100.0
        sigma_t = cv * fcst["y"].values
        ss_vals = np.ceil(z * sigma_t * np.sqrt(max(lead_time, 1)))
        ss_series = pd.Series(ss_vals.astype(int), index=labels, name="ss")
    else:
        sigma_abs = float(inp.get("sigma_abs", 20.0))
        ss_const = int(np.ceil(z * sigma_abs * np.sqrt(max(lead_time, 1))))
        ss_series = pd.Series([ss_const]*len(labels), index=labels, name="ss")

# -------- Chamada MPS (com fallback se série não suportada) --------
if auto_ss and ss_series is not None:
    safety_stock_for_core = int(np.ceil(ss_series.mean()))
else:
    safety_stock_for_core = 0

params = dict(
    lot_policy=lot_policy,
    lot_size=int(lot_size),
    safety_stock=int(safety_stock_for_core),
    lead_time=int(lead_time),
    initial_inventory=int(initial_inventory),
    scheduled_receipts={},
    firm_customer_orders=orders_df,
)

try:
    if auto_ss and ss_series is not None:
        params["safety_stock_series"] = ss_series.values
except Exception:
    pass

mps_df = compute_mps_monthly(fcst, **params)

# -------- Visual --------
previsto     = mps_df["gross_requirements"].astype(int).tolist()
em_carteira  = orders_df["y"].astype(int).tolist()
estoque_proj = mps_df["projected_on_hand_end"].astype(int).tolist()
qtd_mps      = mps_df["planned_order_receipts"].astype(int).tolist()
inicio_mps   = mps_df["planned_order_releases"].astype(int).tolist()
atp_cum      = (mps_df["atp"].astype(int).cumsum().tolist() if "atp" in mps_df.columns else [0]*len(labels))

display_tbl = pd.DataFrame(
    [previsto, em_carteira, estoque_proj, qtd_mps, inicio_mps, atp_cum],
    index=["Previsto", "Em carteira", "Estoque Proj.", "Qtde. MPS", "Início MPS", "ATP(cum)"],
    columns=labels,
)

st.subheader("📅 MPS — visualização mensal (somente leitura)")
st.dataframe(display_tbl, use_container_width=True, height=300)

# Mostrar parâmetros aplicados
st.subheader("Parâmetros aplicados")
p1, p2, p3, p4, p5 = st.columns(5)
p1.metric("Política", "Lote Fixo (FX)" if lot_policy == "FX" else "Lote-a-Lote (L4L)")
p2.metric("Tamanho do lote", f"{lot_size}")
p3.metric("Estoque inicial", f"{initial_inventory}")
p4.metric("Lead time (meses)", f"{lead_time}")
p5.metric("SS automático", "Sim" if auto_ss else "Não")
st.caption(f"Congelamento: **{frozen_range[0]} → {frozen_range[1]}**")

# -------- Download Excel --------
def to_excel_bytes(df_display: pd.DataFrame, fcst: pd.DataFrame, mps_df: pd.DataFrame,
                   orders_df: pd.DataFrame, ss_series: pd.Series | None) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df_display.to_excel(writer, sheet_name="MPS", index=True)
        fcst.to_excel(writer, sheet_name="Previsão", index=False)
        orders_df.to_excel(writer, sheet_name="Em_carteira", index=False)
        mps_df.to_excel(writer, sheet_name="Detalhe", index=False)
        if ss_series is not None:
            pd.DataFrame({"ds": ss_series.index, "ss": ss_series.values}).to_excel(
                writer, sheet_name="Estoque_Seguranca", index=False
            )
    buf.seek(0)
    return buf.getvalue()

st.download_button(
    "⬇️ Baixar MPS (Excel)",
    data=to_excel_bytes(display_tbl, fcst, mps_df, orders_df, ss_series),
    file_name=f"MPS_mensal_h{horizon}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

st.divider()

# Navegação final: voltar para Inputs e avançar para Conclusão
c_back, c_next = st.columns(2)
with c_back:
    st.page_link("pages/05_Inputs_MPS.py", label="⬅️ Voltar: Inputs do MPS", icon="⚙️")
with c_next:
    st.page_link("pages/07_Dashboard_Conclusao.py", label="➡️ Avançar: Conclusão", icon="✅")
