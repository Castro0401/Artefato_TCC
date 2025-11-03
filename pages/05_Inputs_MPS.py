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
idx_by_label_str = {s: i for i, s in enumerate(labels_str)}

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
        "Lead time (meses)", min_value=0, value=int(get("lead_time_default", 1)), step=1
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

if "mps_firm_orders" in st.session_state:
    current_orders = st.session_state["mps_firm_orders"].copy()
    cur_labels_str = [fmt_mmyy(pd.to_datetime(x).to_period("M").to_timestamp())
                      for x in current_orders["ds"].tolist()]
    if cur_labels_str != labels_str:
        current_orders = pd.DataFrame({"ds": labels_raw, "y": [0]*len(labels_raw)})
else:
    current_orders = pd.DataFrame({"ds": labels_raw, "y": [0]*len(labels_raw)})

orders_row_df = pd.DataFrame([current_orders["y"].tolist()], index=["Em carteira"], columns=labels_str)
orders_row_df = st.data_editor(
    orders_row_df,
    use_container_width=True,
    num_rows="fixed",
    column_config={lab: st.column_config.NumberColumn(lab, min_value=0, step=1) for lab in labels_str},
    key="orders_row_inputs_mps",
)
orders_row_df = orders_row_df.fillna(0)
values = orders_row_df.loc["Em carteira"].astype(int).reindex(labels_str).tolist()
orders_df = pd.DataFrame({"ds": labels_raw, "y": values})

# =========================
# 5) EPQ — Parâmetros de custos (nomenclatura da aula)
# =========================
st.subheader("5) EPQ — Parâmetros de custos (nomenclatura da aula)")

# Base de tempo para os parâmetros (apenas coleta; conversões ocorrerão na conclusão)
time_base = st.radio(
    "Base de tempo dos parâmetros (para D, H e r):",
    options=["por mês", "por ano"],
    index=0 if get("time_base", "por mês") == "por mês" else 1,
    horizontal=True,
    help="Escolha como você está informando as taxas. A conversão, se necessária, será feita na página de conclusão."
)

cA, cD, cP = st.columns(3)
with cA:
    A = st.number_input(
        "A — Custo fixo por setup/encomenda (R$)",
        min_value=0.0, step=10.0, value=float(get("A", 50.0)),
        help="Custo fixo cada vez que prepara a produção (setup) ou faz uma encomenda."
    )
with cD:
    D = st.number_input(
        "D — Taxa de demanda (unid/" + ("mês" if time_base=="por mês" else "ano") + ")",
        min_value=0.0, step=1.0, value=float(get("D", 0.0)),
        help="Demanda média estável na base de tempo escolhida."
    )
with cP:
    p = st.number_input(
        "p — Taxa de produção (unid/" + ("mês" if time_base=="por mês" else "ano") + ")",
        min_value=0.0, step=1.0, value=float(get("p", 0.0)),
        help="Capacidade de produção do item (deve ser p > D no EPQ)."
    )

st.markdown("**H — Custo de manter por unidade e por período**")
h_mode = st.radio(
    "Como deseja informar H?",
    options=["Informar H diretamente", "Calcular H a partir de r e v"],
    index=0 if get("h_mode", "Informar H diretamente") == "Informar H diretamente" else 1,
    horizontal=True
)

colH1, colH2, colH3 = st.columns(3)
if h_mode == "Informar H diretamente":
    with colH1:
        H = st.number_input(
            "H — Custo de manter (R$ por unid/" + ("mês" if time_base=="por mês" else "ano") + ")",
            min_value=0.0, step=1.0, value=float(get("H", 0.0)),
            help="Custo de manter uma unidade em estoque por período (na base escolhida)."
        )
    r = get("r", None)
    v = get("v", None)
else:
    with colH1:
        r = st.number_input(
            "r — Taxa de manutenção (R$/$ por " + ("mês" if time_base=="por mês" else "ano") + ")",
            min_value=0.0, step=0.01, value=float(get("r", 0.20 if time_base=="por ano" else 0.02)),
            help="Taxa de manutenção (por exemplo 0,20 R$/$/ano)."
        )
    with colH2:
        v = st.number_input(
            "v — Valor unitário do item (R$ por unid)",
            min_value=0.0, step=1.0, value=float(get("v", 0.0)),
            help="Usado com r para obter H = r·v."
        )
    H = None  # será calculado na página de conclusão

# (opcional) Custo de falta / ruptura por unidade
st.markdown("**π — Custo de falta (opcional)**")
colPi, colBlank, colBlank2 = st.columns(3)
with colPi:
    pi_shortage = st.number_input(
        "π — Custo de falta/ruptura (R$ por unidade não atendida)",
        min_value=0.0, step=1.0, value=float(get("pi_shortage", 0.0)),
        help="Opcional. Se informado, pode ser usado para analisar penalidade de ruptura."
    )

# =========================
# SALVAR
# =========================
if st.button("Salvar inputs do MPS", type="primary"):
    st.session_state["mps_inputs"] = {
        # Seções 1–3
        "item_name": item_name,
        "lot_policy_default": "FX" if lot_policy_default == "FX" else "L4L",
        "lot_size_default": int(lot_size_default),
        "initial_inventory_default": int(initial_inventory_default),
        "lead_time_default": int(lead_time_default),
        "auto_ss": bool(auto_ss),
        "ss_method": ss_method,
        "z_choice": z_choice,
        "cv_pct": float(cv_pct) if 'cv_pct' in locals() and cv_pct is not None else None,
        "sigma_abs": float(sigma_abs) if 'sigma_abs' in locals() and sigma_abs is not None else None,
        "freeze_on": bool(freeze_on),
        "frozen_range": tuple(frozen_range) if freeze_on else None,

        # Seção 4 — pedidos firmes
        "firm_orders": orders_df.copy(),

        # Seção 5 — EPQ (nomenclatura aula)
        "time_base": time_base,     # "por mês" ou "por ano"
        "A": float(A),
        "D": float(D),
        "p": float(p),
        "h_mode": h_mode,           # controla como H será derivado na conclusão
        "H": float(H) if H is not None else None,
        "r": float(r) if r is not None else None,
        "v": float(v) if v is not None else None,
        "pi_shortage": float(pi_shortage),
    }
    st.session_state["mps_firm_orders"] = orders_df.copy()
    st.success("Inputs do MPS salvos com sucesso! ✅")

# -------- Navegação --------
st.divider()
c_back, c_next = st.columns([1, 1], gap="large")
with c_back:
    st.page_link("pages/04_Previsao.py", label="⬅️ Retornar para Previsão")
with c_next:
    st.page_link("pages/06_MPS.py", label="➡️ Ir para MPS (Plano Mestre de Produção)")
