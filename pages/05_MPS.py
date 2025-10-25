# pages/05_MPS.py
from __future__ import annotations
import io
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------
# Configuração da página
# ---------------------------------------------------------------------
st.title("🗓️ MPS — Plano Mestre de Produção (mensal)")

# ---------------------------------------------------------------------
# Guardas de etapa (fluxo controlado)
# ---------------------------------------------------------------------
if "ts_df_norm" not in st.session_state:
    st.warning("Preciso da série do Passo 1 (Upload) antes de continuar.")
    st.page_link("pages/01_Upload.py", label="Ir para o Passo 1 — Upload")
    st.stop()

if not st.session_state.get("forecast_committed", False):
    st.warning("Preciso que você **salve a previsão** no Passo 2 (Previsão) antes de abrir o MPS.")
    st.page_link("pages/04_Previsao.py", label="Ir para o Passo 2 — Previsão")
    st.stop()

if "forecast_df" not in st.session_state or "forecast_h" not in st.session_state:
    st.error("Previsão não encontrada no estado. Volte ao Passo 2, salve a previsão e retorne.")
    st.page_link("pages/04_Previsao.py", label="Ir para o Passo 2 — Previsão")
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
# 0) Entrada vinda do Passo 2 — Previsão
# ---------------------------------------------------------------------
fcst = st.session_state["forecast_df"].copy()[["ds", "y"]]
horizon = int(st.session_state["forecast_h"])
labels = fcst["ds"].tolist()

st.caption(f"🔗 Horizonte atual vindo da **Previsão**: **{horizon} meses**.")

# ---------------------------------------------------------------------
# 1) Parâmetros
# ---------------------------------------------------------------------
st.subheader("Parâmetros do MPS")
c1, c2, c3 = st.columns(3)
with c1:
    pol_display = st.selectbox("Política de lote", ["Lote Fixo (FX)", "Lote-a-Lote (L4L)"], index=0)
    lot_policy = "FX" if pol_display.startswith("Lote Fixo") else "L4L"
with c2:
    # valor base (será sobrescrito se usarmos o automático)
    safety_stock_manual = st.number_input("Estoque de segurança (por mês)", min_value=0, value=0, step=10)
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
# 🔒 1.1) Congelamento de horizonte (baixo impacto)
# ---------------------------------------------------------------------
congelado = st.slider("Horizonte congelado (meses)", 0, horizon, 0)
editable_cols = labels[congelado:]  # só pode editar após o horizonte congelado

# ---------------------------------------------------------------------
# 1.2) Estoque de Segurança Automático (variável por mês)
# ---------------------------------------------------------------------
with st.expander("🛡️ Estoque de segurança automático (variável por mês)"):
    auto_ss = st.checkbox("Ativar estoque de segurança automático (recomendado)", value=False)
    metodo_ss = st.radio("Método", ["Coeficiente de variação (%)", "Desvio padrão absoluto (σ)"], index=0, horizontal=True)
    z_map = {"90%": 1.282, "95%": 1.645, "97.5%": 1.960, "99%": 2.326}
    z_choice = st.selectbox("Nível de serviço (z)", list(z_map.keys()), index=1)
    z = z_map[z_choice]

    ss_series = None  # por padrão
    if auto_ss:
        if metodo_ss == "Coeficiente de variação (%)":
            cv_pct = st.number_input("CV da demanda/erro (%)", min_value=0.0, value=15.0, step=1.0)
            cv = cv_pct / 100.0
            # σ_t ~ CV * previsão_t  (aproximação prática)
            sigma_t = cv * fcst["y"].values  # array por mês
            ss_vals = np.ceil(z * sigma_t * np.sqrt(max(lead_time, 1)))  # evita LT=0
            ss_series = pd.Series(ss_vals.astype(int), index=labels, name="ss")
        else:
            sigma_abs = st.number_input("σ absoluto (unidades/mês)", min_value=0.0, value=20.0, step=1.0)
            ss_const = int(np.ceil(z * sigma_abs * np.sqrt(max(lead_time, 1))))
            ss_series = pd.Series([ss_const]*len(labels), index=labels, name="ss")

        st.caption("Prévia do Estoque de Segurança por mês (usado no cálculo se o core suportar `safety_stock_series`):")
        st.dataframe(pd.DataFrame({"SS (unidades)": ss_series.values}, index=labels), use_container_width=True, height=180)

# ---------------------------------------------------------------------
# 2) Editor 1-linha: Em carteira (reativo ao horizonte)
# ---------------------------------------------------------------------
if "mps_orders_row" not in st.session_state or list(st.session_state["mps_orders_row"].columns) != labels:
    st.session_state["mps_orders_row"] = pd.DataFrame([[0]*len(labels)], index=["Em carteira"], columns=labels)

st.subheader("Pedidos firmes — **Em carteira** (edite a linha abaixo)")
column_cfg = {
    lab: st.column_config.NumberColumn(lab, min_value=0, step=1, disabled=(lab not in editable_cols))
    for lab in labels
}
orders_row = st.data_editor(
    st.session_state["mps_orders_row"],
    use_container_width=True,
    num_rows="fixed",
    column_config=column_cfg,
    key="orders_row_editor"
)
st.session_state["mps_orders_row"] = orders_row.copy()

orders_df = pd.DataFrame({"ds": labels, "y": orders_row.loc["Em carteira"].astype(int).values})

# ---------------------------------------------------------------------
# 3) Cálculo do MPS (com suporte a SS variável por mês via try/fallback)
# ---------------------------------------------------------------------
# Fallback: se o core não aceitar série, usa a média do ss variável ou o valor manual
if auto_ss and ss_series is not None:
    safety_stock_for_core = int(np.ceil(ss_series.mean()))
else:
    safety_stock_for_core = int(safety_stock_manual)

params = dict(
    lot_policy=lot_policy,
    lot_size=int(lot_size),
    safety_stock=int(safety_stock_for_core),   # compatível com core atual
    lead_time=int(lead_time),
    initial_inventory=int(initial_inventory),
    scheduled_receipts={},                  # pode virar editor depois
    firm_customer_orders=orders_df,         # <- pedidos firmes p/ ATP
)

# Tenta passar a série se o core aceitar; caso contrário, ignora silenciosamente
if auto_ss and ss_series is not None:
    try:
        params["safety_stock_series"] = ss_series.values  # ou list(ss_series.values)
    except Exception:
        pass

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
# 5) Download Excel (inclui SS por mês quando disponível)
# ---------------------------------------------------------------------
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

st.caption("A linha **Em carteira** é a única editável. O MPS e o ATP(cum) recalculam automaticamente.")
st.caption("Se 'SS automático' estiver ativo e o core suportar `safety_stock_series`, aplica SS por mês; caso contrário, usa a média como fallback.")

st.download_button(
    "⬇️ Baixar MPS (Excel)",
    data=to_excel_bytes(display_tbl, fcst, mps_df, orders_df, ss_series if (auto_ss and ss_series is not None) else None),
    file_name=f"MPS_mensal_h{horizon}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)
