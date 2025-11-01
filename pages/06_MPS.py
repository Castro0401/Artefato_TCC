# pages/06_MPS.py
from __future__ import annotations
import io
import sys
import inspect
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

# (ajuste o caminho da sua página final)
CONCLUSAO_PAGE = "pages/07_Dashboard_Conclusao.py"

st.title("🗓️ 06_MPS — Plano Mestre de Produção (mensal)")

# -------- Guardas de etapa --------
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from core.mps import compute_mps_monthly  # usa o core/mps.py atualizado

# -------- Dados / Inputs --------
fcst = st.session_state["forecast_df"].copy()[["ds", "y"]]
horizon = int(st.session_state["forecast_h"])
labels = fcst["ds"].tolist()
inp = st.session_state["mps_inputs"]

# Pedidos firmes (dos inputs). Se não houver ou se labels mudaram, zera.
orders_df = st.session_state.get(
    "mps_firm_orders", pd.DataFrame({"ds": labels, "y": [0] * len(labels)})
).copy()
if list(orders_df["ds"]) != labels:
    orders_df = pd.DataFrame({"ds": labels, "y": [0] * len(labels)})
orders_df = orders_df[["ds", "y"]].fillna(0)
orders_df["y"] = orders_df["y"].astype(int)

# -------- Aviso: parâmetros vêm da página anterior --------
st.info(
    "Todos os parâmetros abaixo **vêm da página 05_Inputs_MPS**. "
    "Para ajustar política, tamanhos de lote, estoque em mão, lead time, "
    "estoque de segurança, pedidos em carteira e congelamento, volte à página de inputs.",
    icon="ℹ️",
)

# -------- Snapshot dos parâmetros aplicados --------
lot_policy = inp.get("lot_policy_default", "FX")
lot_size = int(inp.get("lot_size_default", 150))
initial_inventory = int(inp.get("initial_inventory_default", 55))
lead_time = int(inp.get("lead_time_default", 1))

# chave liga/desliga de congelamento + range (pode ser None)
freeze_on = bool(inp.get("freeze_on", False))
frozen_range = inp.get("frozen_range", None)
no_freeze = (
    (not freeze_on)
    or (not frozen_range)
    or (not isinstance(frozen_range, (list, tuple)))
    or (len(frozen_range) != 2)
)

# -------- Estoque de segurança (série mensal) a partir dos inputs --------
z_map = {"90%": 1.282, "95%": 1.645, "97.5%": 1.960, "99%": 2.326}
auto_ss = bool(inp.get("auto_ss", True))
ss_series: pd.Series | None = None

if auto_ss and len(labels) > 0:
    method = inp.get("ss_method", "CV (%)")
    z = z_map.get(inp.get("z_choice", "95%"), 1.645)
    if method == "CV (%)":
        cv = float(inp.get("cv_pct", 15.0)) / 100.0
        sigma_t = cv * fcst["y"].values  # σ_t ≈ CV * forecast_t
        ss_vals = np.ceil(z * sigma_t * np.sqrt(max(lead_time, 1)))
        ss_series = pd.Series(ss_vals.astype(int), index=labels, name="ss")
    else:
        sigma_abs = float(inp.get("sigma_abs", 20.0))
        ss_const = int(np.ceil(z * sigma_abs * np.sqrt(max(lead_time, 1))))
        ss_series = pd.Series([ss_const] * len(labels), index=labels, name="ss")

# -------- Fallback: SS médio para o core --------
if auto_ss and ss_series is not None:
    safety_stock_for_core = int(np.ceil(ss_series.mean()))
else:
    safety_stock_for_core = 0

# -------- Monta base_params (agora que safety_stock_for_core existe) --------
base_params = dict(
    lot_policy=lot_policy,
    lot_size=int(lot_size),
    safety_stock=int(safety_stock_for_core),
    lead_time=int(lead_time),
    initial_inventory=int(initial_inventory),
    scheduled_receipts={},
    firm_customer_orders=orders_df,
)

# Só envia frozen_range se congelamento realmente ativado
if not no_freeze:
    base_params["frozen_range"] = tuple(frozen_range)

# Passa a série só se o core declarar esse parâmetro (por segurança)
accepts_series = "safety_stock_series" in inspect.signature(compute_mps_monthly).parameters
if auto_ss and ss_series is not None and accepts_series:
    mps_df = compute_mps_monthly(
        fcst, **base_params, safety_stock_series=ss_series.astype(int).values
    )
else:
    mps_df = compute_mps_monthly(fcst, **base_params)

# -------- Visualização --------
previsto = mps_df["gross_requirements"].astype(int).tolist()
em_carteira = orders_df["y"].astype(int).tolist()
estoque_proj = mps_df["projected_on_hand_end"].astype(int).tolist()
qtd_mps = mps_df["planned_order_receipts"].astype(int).tolist()
inicio_mps = mps_df["planned_order_releases"].astype(int).tolist()
atp_cum = (
    mps_df["atp"].astype(int).cumsum().tolist()
    if "atp" in mps_df.columns
    else [0] * len(labels)
)

display_tbl = pd.DataFrame(
    [previsto, em_carteira, estoque_proj, qtd_mps, inicio_mps, atp_cum],
    index=["Previsto", "Em carteira", "Estoque Proj.", "Qtde. MPS", "Início MPS", "ATP(cum)"],
    columns=labels,
)

st.subheader("📅 MPS — Visualização Mensal")
st.dataframe(display_tbl, use_container_width=True, height=300)

# Parâmetros aplicados (resumo) — versão compacta e sem truncar
st.subheader("Parâmetros aplicados")

st.markdown("""
<style>
.kpi {display:flex; flex-direction:column; gap:2px;}
.kpi small {color:#6b7280; font-size:0.85rem;}
.kpi .value {font-size:1.6rem; font-weight:600; line-height:1.1;}
.kpi .value.sm {font-size:1.2rem;}
</style>
""", unsafe_allow_html=True)

policy_label = "Lote Fixo (FX)" if lot_policy == "FX" else "Lote-a-Lote (L4L)"
lot_size_display = "Variável" if lot_policy == "L4L" else f"{lot_size}"

c1, c2, c3, c4, c5 = st.columns([1.6, 1, 1, 1, 1])
c1.markdown(f'<div class="kpi"><small>Política</small><div class="value sm">{policy_label}</div></div>', unsafe_allow_html=True)
c2.markdown(f'<div class="kpi"><small>Tamanho do lote</small><div class="value">{lot_size_display}</div></div>', unsafe_allow_html=True)
c3.markdown(f'<div class="kpi"><small>Estoque inicial</small><div class="value">{initial_inventory}</div></div>', unsafe_allow_html=True)
c4.markdown(f'<div class="kpi"><small>Lead time (meses)</small><div class="value">{lead_time}</div></div>', unsafe_allow_html=True)
c5_mark = "Sim" if auto_ss else "Não"
c5.markdown(f'<div class="kpi"><small>SS automático</small><div class="value">{c5_mark}</div></div>', unsafe_allow_html=True)

if no_freeze:
    st.caption("Período congelado: **sem congelamento**")
else:
    st.caption(f"Período congelado: **{frozen_range[0]} → {frozen_range[1]}**")

# -------- Exportação Excel --------
def to_excel_bytes(
    df_display: pd.DataFrame,
    fcst: pd.DataFrame,
    mps_df: pd.DataFrame,
    orders_df: pd.DataFrame,
    ss_series: pd.Series | None,
) -> bytes:
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

# -------- Navegação final --------
st.download_button(
    "⬇️ Baixar MPS (Excel)",
    data=to_excel_bytes(display_tbl, fcst, mps_df, orders_df, ss_series),
    file_name=f"MPS_mensal_h{horizon}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

st.divider()

c_back, c_next = st.columns(2)
with c_back:
    st.page_link("pages/05_Inputs_MPS.py", label="⬅️ Voltar: Inputs do MPS", icon="⚙️")
with c_next:
    st.page_link(CONCLUSAO_PAGE, label="➡️ Avançar: Conclusão", icon="✅")
