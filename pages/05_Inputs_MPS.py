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
# DADOS BASE (apenas para rótulos e edição)
# =========================
fcst = st.session_state["forecast_df"][["ds", "y"]].copy()
horizon = int(st.session_state["forecast_h"])

# Garantimos que 'ds' está em Timestamp mensal (MS)
ds_ts = pd.to_datetime(fcst["ds"]).dt.to_period("M").dt.to_timestamp()

# rótulo bonito MÊS/ANO em PT-BR (ex.: Set/25)
PT_MON = ["Jan","Fev","Mar","Abr","Mai","Jun","Jul","Ago","Set","Out","Nov","Dez"]
def fmt_mmyy(ts: pd.Timestamp) -> str:
    m = ts.month
    yy = ts.year % 100
    return f"{PT_MON[m-1]}/{yy:02d}"

labels_raw: list[pd.Timestamp] = ds_ts.tolist()          # valores "reais"
labels_str: list[str] = [fmt_mmyy(ts) for ts in ds_ts]    # rótulos para UI

st.caption(f"🔗 Horizonte atual da Previsão: **{horizon} meses**.")

# defaults
defaults = st.session_state.get("mps_inputs", {})
def get(k, dv): return defaults.get(k, dv)

# =========================
# 1) ITEM E POLÍTICAS PADRÃO
# =========================
st.subheader("1) Item e políticas padrão")
c1, c2, c3, c4 = st.columns(4)
with c1:
    product_name = st.session_state.get("product_name", None)
    default_item = get("item_name", product_name if product_name else "Item (sem nome)")
    item_name = st.text_input("Item (nome)", value=default_item)
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

# Lead time (meses)
lt1, lt2, lt3, lt4 = st.columns(4, gap="small")
with lt1:
    lead_time_default = st.number_input(
        "Lead time (meses)", min_value=0, value=int(get("lead_time_default", 0)), step=1
    )
with lt2: st.write("")
with lt3: st.write("")
with lt4: st.write("")

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

start_label_str = labels_str[0]
freeze_on_default = bool(defaults.get("freeze_on", False))
freeze_on = st.checkbox("Ativar congelamento", value=freeze_on_default)

# tenta reaproveitar “fim” salvo
_saved = defaults.get("frozen_range", (start_label_str, start_label_str))
saved_end_str = _saved[1] if isinstance(_saved, (list, tuple)) and len(_saved) == 2 else start_label_str
if saved_end_str not in labels_str:
    try:
        ts_try = pd.to_datetime(saved_end_str, errors="coerce")
        if pd.notna(ts_try):
            saved_end_str = fmt_mmyy(ts_try.to_period("M").to_timestamp())
    except Exception:
        saved_end_str = start_label_str
if saved_end_str not in labels_str:
    saved_end_str = start_label_str

if freeze_on:
    end_label_str = st.select_slider(
        f"Selecione o **fim** do período a congelar (início fixo em {start_label_str})",
        options=labels_str,
        value=saved_end_str,
    )
    frozen_range = (start_label_str, end_label_str)
    st.caption(f"Período congelado: **{frozen_range[0]} → {frozen_range[1]}**")
else:
    frozen_range = None
    st.caption("Período congelado: **sem congelamento**")

# =========================
# 4) PEDIDOS EM CARTEIRA
# =========================
st.subheader("4) Pedidos firmes — Em carteira")

# inicia/recupera; mantemos ds “raw” como Timestamp; exibimos colunas com rótulos bonitos
if "mps_firm_orders" in st.session_state:
    current_orders = st.session_state["mps_firm_orders"].copy()
    # sincroniza com labels atuais (comparando pelo rótulo bonito)
    cur_labels_str = [fmt_mmyy(pd.to_datetime(x).to_period("M").to_timestamp())
                      for x in current_orders["ds"].tolist()]
    if cur_labels_str != labels_str:
        current_orders = pd.DataFrame({"ds": labels_raw, "y": [0]*len(labels_raw)})
else:
    current_orders = pd.DataFrame({"ds": labels_raw, "y": [0]*len(labels_raw)})

# editor em linha única: colunas são strings "MÊS/ANO"
orders_row_df = pd.DataFrame([current_orders["y"].tolist()], index=["Em carteira"], columns=labels_str)
orders_row_df = st.data_editor(
    orders_row_df,
    use_container_width=True,
    num_rows="fixed",
    column_config={lab: st.column_config.NumberColumn(lab, min_value=0, step=1) for lab in labels_str},
    key="orders_row_inputs_mps",
)
orders_row_df = orders_row_df.fillna(0)

# reconstrói df (ds original, y) preservando a ordem de labels_raw
values = orders_row_df.loc["Em carteira"].astype(int).reindex(labels_str).tolist()
orders_df = pd.DataFrame({"ds": labels_raw, "y": values})

# =========================
# 5) CUSTOS (apenas coleta; cálculos serão na página de conclusão)
# =========================
st.subheader("5) Custos (apenas coleta — cálculos na conclusão)")
c9, c10, c11, c12 = st.columns(4)
with c9:
    unit_cost = st.number_input("Custo de produzir (R$/un)", min_value=0.0, value=float(get("unit_cost", 1.0)), step=0.10)
with c10:
    order_cost = st.number_input("Custo por pedido (R$)", min_value=0.0, value=float(get("order_cost", 50.0)), step=10.0)
with c11:
    holding_rate = st.number_input("Custo de manter (% do valor/mês)", min_value=0.0, value=float(get("holding_rate", 1.5)), step=0.1)
with c12:
    shortage_cost = st.number_input("Custo de ruptura (R$/un não atendida)", min_value=0.0, value=float(get("shortage_cost", 0.0)), step=1.0)

st.caption("Aqui você apenas informa os valores. Os **cálculos** (Custo de produzir, encomendar, manter, ruptura e total) serão feitos na **página de conclusão**.")

# =========================
# 6) SALVAR OS INPUTS
# =========================
if st.button("Salvar inputs do MPS", type="primary"):
    st.session_state["mps_inputs"] = {
        "item_name": item_name,
        "lot_policy_default": "FX" if lot_policy_default == "FX" else "L4L",
        "lot_size_default": int(lot_size_default),
        "initial_inventory_default": int(initial_inventory_default),
        "lead_time_default": int(lead_time_default),
        "auto_ss": bool(auto_ss),
        "ss_method": ss_method,
        "z_choice": z_choice,
        "cv_pct": float(cv_pct) if ss_method == "CV (%)" else None,
        "sigma_abs": float(sigma_abs) if ss_method != "CV (%)" else None,
        "freeze_on": bool(freeze_on),
        "frozen_range": tuple(frozen_range) if freeze_on else None,
        # Custos (apenas armazenados)
        "unit_cost": float(unit_cost),
        "holding_rate": float(holding_rate),
        "order_cost": float(order_cost),
        "shortage_cost": float(shortage_cost),
        # Metadados úteis
        "labels_str": labels_str,
    }
    st.session_state["mps_firm_orders"] = orders_df.copy()
    st.success("Inputs do MPS salvos com sucesso! ✅")

# -------- Navegação --------
st.divider()
c_back, c_next = st.columns([1, 1], gap="large")
with c_back:
    st.page_link("pages/04_Previsao.py", label="⬅️ Retornar para Previsão")
with c_next:
    st.page_link("pages/06_MPS.py", label="➡️ Ir para Conclusão (usará estes inputs)")
