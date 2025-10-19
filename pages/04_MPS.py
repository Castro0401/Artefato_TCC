# pages/03_mps.py  (ou 02_..._MPS.py)
from __future__ import annotations
import io
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

# --- permitir importar core/ mesmo sendo página ---
ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from core.mps import compute_mps_monthly

st.set_page_config(page_title="MPS — Plano Mestre de Produção", page_icon="🗓️", layout="wide")
st.title("🗓️ MPS — Plano Mestre de Produção (mensal)")

# ---------------- 1) Previsão (ds,y) ----------------
_PT = {1:"Jan",2:"Fev",3:"Mar",4:"Abr",5:"Mai",6:"Jun",7:"Jul",8:"Ago",9:"Set",10:"Out",11:"Nov",12:"Dez"}
def next_month_labels(n=6, start=None):
    if start is None:
        start = pd.Timestamp.today().to_period("M").to_timestamp()
    cur = pd.Timestamp(start)
    labels = []
    for _ in range(n):
        labels.append(f"{_PT[cur.month]}/{str(cur.year)[-2:]}")
        cur = (cur.to_period("M")+1).to_timestamp()
    return labels

if "forecast_df_6m" in st.session_state:
    fcst = st.session_state["forecast_df_6m"][["ds","y"]].copy()
else:
    np.random.seed(42)
    fcst = pd.DataFrame({"ds": next_month_labels(6), "y": np.random.randint(250, 380, size=6)})

# ---------------- 2) Parâmetros ----------------
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

# ---------------- 3) Editor 1-linha: Em carteira ----------------
labels = fcst["ds"].tolist()
if "mps_orders_row" not in st.session_state or list(st.session_state["mps_orders_row"].columns) != labels:
    st.session_state["mps_orders_row"] = pd.DataFrame(
        [ [0]*len(labels) ], index=["Em carteira"], columns=labels
    )

st.subheader("Pedidos firmes — **Em carteira** (edite a linha abaixo)")
orders_row = st.data_editor(
    st.session_state["mps_orders_row"],
    use_container_width=True,
    num_rows="fixed",
    column_config={lab: st.column_config.NumberColumn(lab, min_value=0, step=1) for lab in labels},
    key="orders_row_editor"
)
# salva no estado (provoca rerun só quando houver mudança real)
st.session_state["mps_orders_row"] = orders_row.copy()

# converte a linha editada para df (ds,y)
orders_df = pd.DataFrame({"ds": labels, "y": orders_row.loc["Em carteira"].astype(int).values})

# ---------------- 4) Cálculo (reativo) ----------------
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

# ---------------- 5) Visual “PUC” (somente leitura) ----------------
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
st.dataframe(display_tbl, use_container_width=True, height=280)  # <- apenas exibição, não editável

# ---------------- 6) Download Excel (gera só quando clicar) ----------------
def to_excel_bytes(df_display: pd.DataFrame, fcst: pd.DataFrame, mps_df: pd.DataFrame, orders_df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df_display.to_excel(writer, sheet_name="MPS", index=True)
        fcst.to_excel(writer, sheet_name="Previsão", index=False)
        orders_df.to_excel(writer, sheet_name="Em_carteira", index=False)
        mps_df.to_excel(writer, sheet_name="Detalhe", index=False)
    buf.seek(0)
    return buf.getvalue()

st.caption("A linha **Em carteira** acima é a única editável.  O MPS e o ATP(cum) recalculam automaticamente após cada ajuste.")
st.caption("ATP é calculado considerando a lógica clássica: **Estoque disponível + Qtde. MPS - máximo (Em carteira ; Previstos)**")

st.download_button(
    "⬇️ Baixar MPS (Excel)",
    data=to_excel_bytes(display_tbl, fcst, mps_df, orders_df),
    file_name="MPS_mensal.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)