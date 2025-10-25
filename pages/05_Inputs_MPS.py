# pages/05_Inputs_MPS.py
from __future__ import annotations
import pandas as pd
import streamlit as st

st.title("⚙️ Inputs do MPS")

# =========================
# GUARDAS DE ETAPA (ACESSO BLOQUEADO)
# =========================
# 1) Exige Upload concluído
if "ts_df_norm" not in st.session_state:
    st.warning("Preciso da série do Passo 1 (Upload) antes de continuar.")
    st.page_link("pages/01_Upload.py", label="Ir para o Passo 1 — Upload")
    st.stop()

# 2) Exige previsão salva/commitada (Passo 2)
if not st.session_state.get("forecast_committed", False):
    st.warning("Preciso que você **salve a previsão** no Passo 2 (Previsão) antes de configurar os inputs.")
    st.page_link("pages/04_Previsao.py", label="Ir para o Passo 2 — Previsão")
    st.stop()

# 3) Exige objetos da previsão no estado
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

# =========================
# RECUPERA DEFAULTS
# =========================
defaults = st.session_state.get("mps_inputs", {})
def get(k, dv): return defaults.get(k, dv)

# =========================
# 1) ITEM E POLÍTICAS PADRÃO
# =========================
st.subheader("1) Item e políticas padrão")
c1, c2, c3 = st.columns(3)
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

c4, c5 = st.columns(2)
with c4:
    initial_inventory_default = st.number_input(
        "Estoque em mão inicial (padrão)", min_value=0, step=5, value=int(get("initial_inventory_default", 55))
    )
with c5:
    frozen_horizon = st.slider("Horizonte congelado (meses) — padrão", 0, horizon, int(get("frozen_horizon", 0)))

# =========================
# 2) ESTOQUE DE SEGURANÇA — PARÂMETROS
# =========================
st.subheader("2) Estoque de segurança — parâmetros")
c6, c7, c8 = st.columns(3)
with c6:
    auto_ss = st.checkbox("Ativar SS automático (variável por mês)", value=bool(get("auto_ss", True)))
with c7:
    ss_method = st.radio("Método", ["CV (%)", "σ absoluto"],
                         index=0 if get("ss_method", "CV (%)") == "CV (%)" else 1, horizontal=True)
with c8:
    z_choice = st.selectbox(
        "Nível de serviço (z)", ["90%", "95%", "97.5%", "99%"],
        index={"90%":0, "95%":1, "97.5%":2, "99%":3}[get("z_choice", "95%")]
    )

if ss_method == "CV (%)":
    cv_pct = st.number_input("CV da demanda/erro (%)", min_value=0.0, value=float(get("cv_pct", 15.0)), step=1.0)
    sigma_abs = None
else:
    sigma_abs = st.number_input("σ absoluto (unid/mês)", min_value=0.0, value=float(get("sigma_abs", 20.0)), step=1.0)
    cv_pct = None

# =========================
# 3) CUSTOS (para futuras páginas de consolidação — OPCIONAL)
# =========================
st.subheader("3) Custos (para páginas de consolidação — opcional)")
c9, c10, c11 = st.columns(3)
with c9:
    unit_cost = st.number_input("Custo unitário (R$)", min_value=0.0, value=float(get("unit_cost", 0.0)), step=10.0)
with c10:
    holding_rate = st.number_input("Taxa de manutenção (%/mês)", min_value=0.0, value=float(get("holding_rate", 1.5)), step=0.1)
with c11:
    order_cost = st.number_input("Custo por pedido (R$)", min_value=0.0, value=float(get("order_cost", 50.0)), step=10.0)

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
        "frozen_horizon": int(frozen_horizon),
        "auto_ss": bool(auto_ss),
        "ss_method": ss_method,
        "z_choice": z_choice,
        "cv_pct": float(cv_pct) if cv_pct is not None else None,
        "sigma_abs": float(sigma_abs) if sigma_abs is not None else None,
        # custos (para futura consolidação)
        "unit_cost": float(unit_cost),
        "holding_rate": float(holding_rate),
        "order_cost": float(order_cost),
        "shortage_cost": float(shortage_cost),
    }
    st.success("Inputs do MPS salvos com sucesso! ✅")

# =========================
# PRÉVIA + LINK
# =========================
st.subheader("Prévia dos inputs atuais")
preview = pd.DataFrame([st.session_state.get("mps_inputs", {})]).T.rename(columns={0: "valor"})
st.dataframe(preview, use_container_width=True, height=300)

st.divider()
st.page_link("pages/06_MPS.py", label="➡️ Ir para 06_MPS (Plano Mestre de Produção)", icon="🗓️")
