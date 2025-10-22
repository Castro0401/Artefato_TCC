# pages/03_mps.py  (ou 03_🗓️_MPS.py)
from __future__ import annotations
import io
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------
# Configuração da página
# ---------------------------------------------------------------------
st.set_page_config(page_title="MPS — Plano Mestre de Produção", page_icon="🗓️", layout="wide")
st.title("🗓️ MPS — Plano Mestre de Produção (mensal)")

# ---------------------------------------------------------------------
# Guardas de etapa (fluxo controlado)
# ---------------------------------------------------------------------
# 1) Precisa ter feito o Upload (Passo 1)
if "ts_df_norm" not in st.session_state:
    st.warning("Preciso da série do Passo 1 (Upload) antes de continuar.")
    # ajuste o path conforme o seu arquivo de upload
    st.page_link("pages/01_Upload.py", label="Ir para o Passo 1 — Upload")
    st.stop()

# 2) Precisa ter salvo/commitado a Previsão (Passo 2)
if not st.session_state.get("forecast_committed", False):
    st.warning("Preciso que você **salve a previsão** no Passo 2 (Previsão) antes de abrir o MPS.")
    # ajuste o path conforme o seu arquivo de previsão
    st.page_link("pages/02_Previsao.py", label="Ir para o Passo 2 — Previsão")
    st.stop()

# 3) Precisa ter forecast_df e forecast_h presentes
if "forecast_df" not in st.session_state or "forecast_h" not in st.session_state:
    st.error("Previsão não encontrada no estado. Volte ao Passo 2, salve a previsão e retorne.")
    st.page_link("pages/02_Previsao.py", label="Ir para o Passo 2 — Previsão")
    st.stop()

# ---------------------------------------------------------------------
# Imports do core (após as guardas)
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from core.mps import compute_mps_monthly

# ---------------------------------------------------------------------
# 0) Entrada vinda do Passo 2 — Previsão (sem fallback)
# ---------------------------------------------------------------------
fcst = st.session_state["forecast_df"].copy()[["ds", "y"]]
horizon = int(st.session_state["forecast_h"])
labels = fcst["ds"].tolist()

st.caption(f"🔗 Horizonte atual vindo da **Previsão**: **{horizon} mês(es)**.")

# ---------------------------------------------------------------------
# 1) Parâmetros
# ---------------------------------------------------------------------
st.subheader("Parâmetros do MPS")
c1, c2, c3 = st.columns(3)
with c1:
    pol_display = st.selectbox("Política de lote", ["Lote Fixo (FX)", "Lote-a-Lote (L4L)"], index=0)
    lot_policy = "FX" if pol_display.startswith("Lote Fixo") else "L4L"
with c2:
    safety_stock = st.number_input("Estoque de segurança (por mês)", min_value=0, value=0, step=10)
with c3:
    initial_inventory = st.number_input("Em mão (inicial)", min_value=0, value=55, step=5)

c4, c5, c6 = st.columns(3)
with c4:
    if lot_policy == "L4L":
        st.text_input("Tamanho do lote (FX)", value="—", disabled=True)
        lot_size = 1
    else:
        lot_size = st.number_input("Tamanho do lote (FX)", min_value=1, value=150, step=10)
with c5:
    lead_time = st.number_input("Lead time (meses)", min_value=0, value=1, step=1)
with c6:
    item_name = st.text_input("Item", value="Cadeira de ripas")

# ---------------------------------------------------------------------
# 2) Editor 1-linha: Em carteira (reativo ao horizonte)
# ---------------------------------------------------------------------
# Se o horizonte/labels mudarem, resetamos a linha editável para casar com as novas colunas
if "mps_orders_row" not in st.session_state or list(st.session_state["mps_orders_row"].columns) != labels:
    st.session_state["mps_orders_row"] = pd.DataFrame([[0]*len(labels)], index=["Em carteira"], columns=labels)

st.subheader("Pedidos firmes — **Em carteira** (edite a linha abaixo)")
orders_row = st.data_editor(
    st.session_state["mps_orders_row"],
    use_container_width=True,
    num_rows="fixed",
    column_config={lab: st.column_config.NumberColumn(lab, min_value=0, step=1) for lab in labels},
    key="orders_row_editor"
)
st.session_state["mps_orders_row"] = orders_row.copy()  # persiste

# converte a linha editada para df (ds,y)
orders_df = pd.DataFrame({"ds": labels, "y": orders_row.loc["Em carteira"].astype(int).values})

# ---------------------------------------------------------------------
# 3) Cálculo do MPS (reativo)
# ---------------------------------------------------------------------
params = dict(
    lot_policy=lot_policy,
    lot_size=int(lot_size),
    safety_stock=int(safety_stock),
    lead_time=int(lead_time),
    initial_inventory=int(initial_inventory),
    scheduled_receipts={},                  # pode virar editor depois
    firm_customer_orders=orders_df,         # <- pedidos firmes p/ ATP
)
mps_df = compute_mps_monthly(fcst, **params)

# ---------------------------------------------------------------------
# 4) Visual “PUC” (somente leitura)
# ---------------------------------------------------------------------
previsto      = mps_df["gross_requirements"].astype(int).tolist()
em_carteira   = orders_df["y"].astype(int).tolist()
estoque_proj  = mps_df["projected_on_hand_end"].astype(int).tolist()
qtd_mps       = mps_df["planned_order_receipts"].astype(int).tolist()
inicio_mps    = mps_df["planned_order_releases"].astype(int).tolist()
atp_cum       = (mps_df["atp"].astype(int).cumsum().tolist() if "atp" in mps_df.columns else [0]*len(labels))

display_tbl = pd.DataFrame(
    [previsto, em_carteira, estoque_proj, qtd_mps, inicio_mps, atp_cum],
    index=["Previsto", "Em carteira", "Estoque Proj.", "Qtde. MPS", "Início MPS", "ATP(cum)"],
    columns=labels,
)

st.subheader("MPS — visualização mensal")
st.dataframe(display_tbl, use_container_width=True, height=280)

# ---------------------------------------------------------------------
# 5) Download Excel (gera no clique)
# ---------------------------------------------------------------------
def to_excel_bytes(df_display: pd.DataFrame, fcst: pd.DataFrame, mps_df: pd.DataFrame, orders_df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df_display.to_excel(writer, sheet_name="MPS", index=True)
        fcst.to_excel(writer, sheet_name="Previsão", index=False)
        orders_df.to_excel(writer, sheet_name="Em_carteira", index=False)
        mps_df.to_excel(writer, sheet_name="Detalhe", index=False)
    buf.seek(0)
    return buf.getvalue()

st.caption("A linha **Em carteira** acima é a única editável. O MPS e o ATP(cum) recalculam automaticamente após cada ajuste.")
st.caption("ATP é calculado considerando a lógica clássica: **Estoque disponível + Qtde. MPS - máximo(Em carteira ; Previstos)**")

st.download_button(
    "⬇️ Baixar MPS (Excel)",
    data=to_excel_bytes(display_tbl, fcst, mps_df, orders_df),
    file_name=f"MPS_mensal_h{horizon}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)
