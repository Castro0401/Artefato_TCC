# pages/05_Inputs_MPS.py
from __future__ import annotations
import numpy as np
import pandas as pd
import streamlit as st

st.title("⚙️ Inputs do MPS")

# =========================
# GUARDAS DE ETAPA
# =========================
if "ts_df_norm" not in st.session_state:
    st.warning("Preciso da série do Passo 1 (Upload) antes de continuar.")
    st.page_link("pages/01_Upload.py", label="Ir para o Passo 1 — Upload")
    st.stop()

if not st.session_state.get("forecast_committed", False):
    st.warning("Preciso que você **salve a previsão** no Passo 2 (Previsão) antes de configurar os inputs.")
    st.page_link("pages/04_Previsao.py", label="Ir para o Passo 2 — Previsão")
    st.stop()

if "forecast_df" not in st.session_state or "forecast_h" not in st.session_state:
    st.error("Previsão não encontrada no estado. Volte ao Passo 2, salve a previsão e retorne.")
    st.page_link("pages/04_Previsao.py", label="Ir para o Passo 2 — Previsão")
    st.stop()

# =========================
# DADOS BASE
# =========================
fcst = st.session_state["forecast_df"][["ds", "y"]].copy()
horizon = int(st.session_state["forecast_h"])
labels = fcst["ds"].tolist()
st.caption(f"🔗 Horizonte atual da Previsão: **{horizon} mês(es)**.")

# defaults
defaults = st.session_state.get("mps_inputs", {})
def get(k, dv): return defaults.get(k, dv)

# =========================
# 1) ITEM E POLÍTICAS PADRÃO
# =========================
st.subheader("1) Item e políticas padrão")
c1, c2, c3, c4 = st.columns(4)
with c1:
    item_name = st.text_input("Item (nome)", value=get("item_name", "Cadeira de ripas"))
with c2:
    lot_policy_default = st.selectbox(
        "Política padrão", ["Lote Fixo (FX)", "Lote-a-Lote (L4L)"],
        index=0 if get("lot_policy_default", "FX") == "FX" else 1
    )
    lot_policy_default = "FX" if lot_policy_default.startswith("Lote Fixo") else "L4L"
with c3:
    lot_size_default = st.number_input(
        "Tamanho do lote padrão (se FX)", min_value=1, step=10, value=int(get("lot_size_default", 150))
    )
with c4:
    initial_inventory_default = st.number_input(
        "Estoque em mão inicial (padrão)", min_value=0, step=5, value=int(get("initial_inventory_default", 55))
    )

c5, c6 = st.columns(2)
with c5:
    lead_time_default = st.number_input("Lead time (meses)", min_value=0, value=int(get("lead_time_default", 1)), step=1)
with c6:
    st.write("")  # espaçamento

# =========================
# 2) ESTOQUE DE SEGURANÇA — PARÂMETROS
# =========================
st.subheader("2) Estoque de segurança — parâmetros")
row = st.columns([1.2, 1, 1])
with row[0]:
    auto_ss = st.checkbox("Ativar SS automático (variável por mês)", value=bool(get("auto_ss", True)))
    ss_method = st.radio("Método", ["CV (%)", "σ absoluto"],
                         index=0 if get("ss_method", "CV (%)") == "CV (%)" else 1, horizontal=True)
with row[1]:
    z_choice = st.selectbox(
        "Nível de serviço (z)", ["90%", "95%", "97.5%", "99%"],
        index={"90%":0, "95%":1, "97.5%":2, "99%":3}[get("z_choice", "95%")]
    )
with row[2]:
    if ss_method == "CV (%)":
        cv_pct = st.number_input("CV da demanda/erro (%)", min_value=0.0, value=float(get("cv_pct", 15.0)), step=1.0)
        sigma_abs = None
    else:
        sigma_abs = st.number_input("σ absoluto (unid/mês)", min_value=0.0, value=float(get("sigma_abs", 20.0)), step=1.0)
        cv_pct = None

# =========================
# 3) CONGELAMENTO DE HORIZONTE (intervalo)
# =========================
st.subheader("3) Congelamento de horizonte (intervalo)")
frozen_default = get("frozen_range", (labels[0], labels[0])) if labels else ("", "")
frozen_range = st.select_slider(
    "Selecione o intervalo a congelar (inclusive)",
    options=labels,
    value=frozen_default if isinstance(frozen_default, tuple) else (labels[0], labels[0]),
)
st.caption(f"Período congelado: **{frozen_range[0]} → {frozen_range[1]}**")

# =========================
# 4) PEDIDOS EM CARTEIRA
# =========================
st.subheader("4) Pedidos firmes — Em carteira")
# inicia/recupera
if "mps_firm_orders" in st.session_state:
    current_orders = st.session_state["mps_firm_orders"].copy()
    # sincroniza com labels atuais
    if list(current_orders["ds"]) != labels:
        current_orders = pd.DataFrame({"ds": labels, "y": [0]*len(labels)})
else:
    current_orders = pd.DataFrame({"ds": labels, "y": [0]*len(labels)})

# editor em linha única
orders_row_df = pd.DataFrame([current_orders["y"].tolist()], index=["Em carteira"], columns=labels)
orders_row_df = st.data_editor(
    orders_row_df,
    use_container_width=True,
    num_rows="fixed",
    column_config={lab: st.column_config.NumberColumn(lab, min_value=0, step=1) for lab in labels},
    key="orders_row_inputs_mps",
)
# reconstrói df (ds,y)
orders_df = pd.DataFrame({"ds": labels, "y": orders_row_df.loc["Em carteira"].astype(int).values})

# =========================
# 5) CUSTOS (opcional para páginas futuras)
# =========================
st.subheader("5) Custos (para consolidação — opcional)")
c9, c10, c11, c12 = st.columns(4)
with c9:
    unit_cost = st.number_input("Custo unitário (R$)", min_value=0.0, value=float(get("unit_cost", 0.0)), step=10.0)
with c10:
    holding_rate = st.number_input("Taxa de manutenção (%/mês)", min_value=0.0, value=float(get("holding_rate", 1.5)), step=0.1)
with c11:
    order_cost = st.number_input("Custo por pedido (R$)", min_value=0.0, value=float(get("order_cost", 50.0)), step=10.0)
with c12:
    shortage_cost = st.number_input("Custo de falta (R$/un)", min_value=0.0, value=float(get("shortage_cost", 0.0)), step=10.0)

# =========================
# SALVAR
# =========================
st.divider()
if st.button("💾 Salvar inputs do MPS", type="primary"):
    st.session_state["mps_inputs"] = {
        "item_name": item_name,
        "lot_policy_default": lot_policy_default,
        "lot_size_default": int(lot_size_default),
        "initial_inventory_default": int(initial_inventory_default),
        "lead_time_default": int(lead_time_default),
        "auto_ss": bool(auto_ss),
        "ss_method": ss_method,
        "z_choice": z_choice,
        "cv_pct": float(cv_pct) if cv_pct is not None else None,
        "sigma_abs": float(sigma_abs) if sigma_abs is not None else None,
        "frozen_range": tuple(frozen_range),
        # custos (para futura consolidação)
        "unit_cost": float(unit_cost),
        "holding_rate": float(holding_rate),
        "order_cost": float(order_cost),
        "shortage_cost": float(shortage_cost),
    }
    st.session_state["mps_firm_orders"] = orders_df.copy()
    st.success("Inputs do MPS salvos com sucesso! ✅")

# Preview
st.subheader("Prévia dos inputs atuais")
preview = pd.DataFrame([st.session_state.get("mps_inputs", {})]).T.rename(columns={0: "valor"})
st.dataframe(preview, use_container_width=True, height=280)

st.divider()
st.page_link("pages/06_MPS.py", label="➡️ Ir para 06_MPS (Plano Mestre de Produção)", icon="🗓️")
