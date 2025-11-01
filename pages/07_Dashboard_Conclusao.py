# pages/07_Dashboard_Conclusao.py
from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# =========================
# Título
# =========================
st.title("✅ 07 — Conclusão (Painel de Decisão)")

# =========================
# Guardas de etapa (curtas e objetivas)
# =========================
if "ts_df_norm" not in st.session_state:
    st.warning("Preciso da série do Passo 1 (Upload) antes de continuar.")
    st.page_link("pages/01_Upload.py", label="Ir para 01_Upload", icon="📤")
    st.stop()

if not st.session_state.get("forecast_committed", False) or "forecast_df" not in st.session_state:
    st.warning("Preciso que você salve a previsão no Passo 2.")
    st.page_link("pages/04_Previsao.py", label="Ir para 04_Previsao", icon="🔮")
    st.stop()

if "mps_inputs" not in st.session_state:
    st.warning("Antes da conclusão, configure os Inputs do MPS.")
    st.page_link("pages/05_Inputs_MPS.py", label="Ir para 05_Inputs_MPS", icon="⚙️")
    st.stop()

# =========================
# Imports do core para recalcular MPS, se necessário
# =========================
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from core.mps import compute_mps_monthly  # mesmo core da página 06
except Exception as e:
    st.error(f"Não consegui importar o core do MPS (core/mps.py): {e}")
    st.stop()

# =========================
# Dados base
# =========================
ts_df_norm: pd.DataFrame = st.session_state["ts_df_norm"].copy()  # ['ds','y'] histórico normalizado
fcst_df: pd.DataFrame = st.session_state["forecast_df"].copy()[["ds", "y"]]  # previsão (futuro)
horizon: int = int(st.session_state.get("forecast_h", len(fcst_df)))

# =========================
# Utilidades locais
# =========================
def _excel_bytes(
    df_display: pd.DataFrame,
    fcst: pd.DataFrame,
    mps_df: pd.DataFrame,
    orders_df: pd.DataFrame,
    ss_series: Optional[pd.Series],
) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df_display.to_excel(writer, sheet_name="MPS", index=True)
        fcst.to_excel(writer, sheet_name="Previsao", index=False)
        orders_df.to_excel(writer, sheet_name="Em_carteira", index=False)
        mps_df.to_excel(writer, sheet_name="Detalhe", index=False)
        if ss_series is not None:
            pd.DataFrame({"ds": ss_series.index, "ss": ss_series.values}).to_excel(
                writer, sheet_name="SS_mensal", index=False
            )
    buf.seek(0)
    return buf.getvalue()

def _safe_col(df: pd.DataFrame, name: str, default=0) -> pd.Series:
    return df[name] if name in df.columns else pd.Series([default] * len(df), index=df.index)

# =========================
# (1) Recalcula MPS (mesma lógica da página 06)
# =========================
inp = st.session_state["mps_inputs"]
fcst = fcst_df.copy()

labels = fcst["ds"].tolist()

# Pedidos firmes (se existirem)
orders_df = st.session_state.get(
    "mps_firm_orders", pd.DataFrame({"ds": labels, "y": [0] * len(labels)})
).copy()
if list(orders_df["ds"]) != labels:
    orders_df = pd.DataFrame({"ds": labels, "y": [0] * len(labels)})
orders_df = orders_df[["ds", "y"]].fillna(0)
orders_df["y"] = orders_df["y"].astype(int)

# Snapshot/params principais
lot_policy = inp.get("lot_policy_default", "FX")
lot_size = int(inp.get("lot_size_default", 150))
initial_inventory = int(inp.get("initial_inventory_default", 55))
lead_time = int(inp.get("lead_time_default", 1))

# Congelamento
freeze_on = bool(inp.get("freeze_on", False))
frozen_range = inp.get("frozen_range", None)
no_freeze = (
    (not freeze_on)
    or (not frozen_range)
    or (not isinstance(frozen_range, (list, tuple)))
    or (len(frozen_range) != 2)
)

# Estoque de segurança (automático ou fixo médio, idêntico ao passo 06)
z_map = {"90%": 1.282, "95%": 1.645, "97.5%": 1.960, "99%": 2.326}
auto_ss = bool(inp.get("auto_ss", True))
ss_series: Optional[pd.Series] = None

if auto_ss and len(labels) > 0:
    method = inp.get("ss_method", "CV (%)")
    z = z_map.get(inp.get("z_choice", "95%"), 1.645)
    if method == "CV (%)":
        cv = float(inp.get("cv_pct", 15.0)) / 100.0
        sigma_t = cv * fcst["y"].values
        ss_vals = np.ceil(z * sigma_t * np.sqrt(max(lead_time, 1)))
        ss_series = pd.Series(ss_vals.astype(int), index=labels, name="ss")
    else:
        sigma_abs = float(inp.get("sigma_abs", 20.0))
        ss_const = int(np.ceil(z * sigma_abs * np.sqrt(max(lead_time, 1))))
        ss_series = pd.Series([ss_const] * len(labels), index=labels, name="ss")

safety_stock_for_core = int(np.ceil(ss_series.mean())) if (auto_ss and ss_series is not None) else 0

# Monta kwargs pro core
core_params = dict(
    lot_policy=lot_policy,
    lot_size=lot_size,
    safety_stock=int(safety_stock_for_core),
    lead_time=lead_time,
    initial_inventory=initial_inventory,
    scheduled_receipts={},          # não estamos usando aqui
    firm_customer_orders=orders_df, # pedidos firmes
)
if not no_freeze:
    core_params["frozen_range"] = tuple(frozen_range)

# Passa série de SS (se o core aceitar)
accepts_series = "safety_stock_series" in compute_mps_monthly.__code__.co_varnames
if auto_ss and ss_series is not None and accepts_series:
    mps_df = compute_mps_monthly(fcst, **core_params, safety_stock_series=ss_series.astype(int).values)
else:
    mps_df = compute_mps_monthly(fcst, **core_params)

# Tabela consolidada para visualização e export
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

# =========================
# (2) KPIs do MPS (simples e úteis)
# =========================
proj_stock_end = _safe_col(mps_df, "projected_on_hand_end")
avg_inventory = float(np.nanmean(proj_stock_end.values)) if len(proj_stock_end) else 0.0

atp = _safe_col(mps_df, "atp")
ruptures = int((atp < 0).sum())
rupture_rate = float(ruptures) / max(1, len(atp)) * 100.0

# custos (se informados nos inputs)
unit_cost = float(inp.get("unit_cost", 0.0))
holding_rate = float(inp.get("holding_rate", 0.0)) / 100.0
order_cost = float(inp.get("order_cost", 0.0))
shortage_cost = float(inp.get("shortage_cost", 0.0))

# custo de manutenção (aproxima): estoque médio * unit_cost * holding_rate
holding_cost = avg_inventory * unit_cost * holding_rate

# custo de pedido (aproxima): quantidade de ordens geradas (>0) * order_cost
orders_count = int((mps_df["planned_order_releases"] > 0).sum())
ordering_cost = orders_count * order_cost

# custo de falta (aproxima): soma dos ATP < 0 (em valor absoluto) * shortage_cost
stockout_units = float((-atp[atp < 0]).sum()) if len(atp) else 0.0
shortage_total_cost = stockout_units * shortage_cost

total_cost = holding_cost + ordering_cost + shortage_total_cost

# =========================
# (3) Experimentos da previsão (se existirem no estado)
# =========================
exp_df: Optional[pd.DataFrame] = st.session_state.get("experiments_df", None)

# Heurística para identificar as colunas usuais, se existirem
exp_cols = []
if isinstance(exp_df, pd.DataFrame):
    for c in ["model", "mae", "rmse", "mape", "smape", "criterion"]:
        if c in exp_df.columns:
            exp_cols.append(c)

# Melhor modelo (se disponível)
best_row = None
if isinstance(exp_df, pd.DataFrame) and "mae" in exp_df.columns:
    # escolha por MAE mínimo (ajuste se seu critério for diferente)
    best_row = exp_df.sort_values("mae", ascending=True).head(1)
elif isinstance(exp_df, pd.DataFrame) and "rmse" in exp_df.columns:
    best_row = exp_df.sort_values("rmse", ascending=True).head(1)

# =========================
# Layout principal (tabs)
# =========================
tab1, tab2, tab3 = st.tabs(["📊 Acurácia", "🏭 MPS & KPIs", "🧠 Recomendações"])

# --------- TAB 1: Acurácia ---------
with tab1:
    st.subheader("Desempenho dos modelos de previsão")

    if isinstance(exp_df, pd.DataFrame) and len(exp_df):
        # KPIs do melhor (se existir)
        if best_row is not None:
            b = best_row.iloc[0].to_dict()
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("MAE", f"{b.get('mae', np.nan):.2f}" if pd.notna(b.get("mae", np.nan)) else "—")
            c2.metric("sMAPE (%)", f"{b.get('smape', np.nan):.2f}" if pd.notna(b.get("smape", np.nan)) else "—")
            c3.metric("RMSE", f"{b.get('rmse', np.nan):.2f}" if pd.notna(b.get("rmse", np.nan)) else "—")
            c4.metric("MAPE (%)", f"{b.get('mape', np.nan):.2f}" if pd.notna(b.get("mape", np.nan)) else "—")
            st.caption(f"Modelo campeão: **{b.get('model','—')}**")

        # Ranking simples
        st.markdown("#### Ranking dos experimentos")
        show_cols = [c for c in ["model", "mae", "smape", "rmse", "mape"] if c in exp_df.columns]
        st.dataframe(
            exp_df[show_cols].sort_values(show_cols[1] if len(show_cols) > 1 else show_cols[0]),
            use_container_width=True,
            height=320,
        )

        # Download CSV dos experimentos
        st.download_button(
            "⬇️ Baixar experimentos (CSV)",
            data=exp_df.to_csv(index=False).encode("utf-8"),
            file_name="experimentos_previsao.csv",
            mime="text/csv",
        )
    else:
        st.info("Sem tabela de experimentos em memória. Gere na página de **Previsão** e volte aqui.")
        st.page_link("pages/04_Previsao.py", label="Ir para 04_Previsao", icon="🔮")

    # Gráfico real + previsão (histórico + futuro)
    try:
        hist = ts_df_norm.copy()[["ds", "y"]].rename(columns={"y": "Real"})
        fut = fcst_df.copy()[["ds", "y"]].rename(columns={"y": "Previsão"})
        both = hist.merge(fut, on="ds", how="outer").sort_values("ds")
        import plotly.express as px

        fig = px.line(both, x="ds", y=["Real", "Previsão"])
        # Cores: real = azul escuro; previsão = azul claro
        fig.update_traces(selector=dict(name="Real"), line=dict(color="#1e3a8a", width=2.5))
        fig.update_traces(selector=dict(name="Previsão"), line=dict(color="#60a5fa", width=2.5, dash="dash"))
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.info(f"Não foi possível exibir o gráfico Real x Previsão: {e}")

# --------- TAB 2: MPS ---------
with tab2:
    st.subheader("KPIs do Plano Mestre de Produção")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Estoque médio (un.)", f"{avg_inventory:.0f}")
    k2.metric("Ordens geradas", f"{orders_count}")
    k3.metric("Períodos com ruptura", f"{ruptures} ({rupture_rate:.1f}%)")
    k4.metric("Custo total (R$)", f"{total_cost:,.2f}")

    st.caption(
        f"**Custos (aprox.)** — Manutenção: R$ {holding_cost:,.2f} | Pedidos: R$ {ordering_cost:,.2f} | Falta: R$ {shortage_total_cost:,.2f}"
    )

    st.markdown("#### Tabela consolidada do MPS")
    st.dataframe(display_tbl, use_container_width=True, height=300)

    # Gráfico ATP mensal
    if "atp" in mps_df.columns:
        try:
            import plotly.express as px

            g = mps_df[["ds", "atp"]].copy()
            g["Sinal"] = np.where(g["atp"] < 0, "Ruptura (neg.)", "Disponível (pos.)")
            fig2 = px.bar(g, x="ds", y="atp", color="Sinal", barmode="relative")
            fig2.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=0))
            st.plotly_chart(fig2, use_container_width=True)
        except Exception as e:
            st.info(f"Não foi possível gerar o gráfico de ATP: {e}")

    # Download Excel MPS
    st.download_button(
        "⬇️ Baixar MPS (Excel)",
        data=_excel_bytes(display_tbl, fcst, mps_df, orders_df, ss_series),
        file_name=f"MPS_mensal_h{horizon}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

# --------- TAB 3: Recomendações ---------
with tab3:
    st.subheader("Sugestões automáticas")

    recs = []

    # recomendações pela acurácia (se disponível)
    if best_row is not None:
        b = best_row.iloc[0].to_dict()
        mape = b.get("mape", np.nan)
        smape = b.get("smape", np.nan)
        mae = b.get("mae", np.nan)
        rmse = b.get("rmse", np.nan)

        if pd.notna(mape) and mape > 20:
            recs.append("**MAPE > 20%** → reavaliar feature set / granularidade ou testar modelos adicionais.")
        if pd.notna(smape) and smape > 20:
            recs.append("**sMAPE alto** → pode haver escala/heterocedasticidade; testar **log/Box-Cox**.")
        if pd.notna(mae) and pd.notna(rmse) and rmse > 1.2 * mae:
            recs.append("**RMSE ≫ MAE** → penalidade forte a grandes erros → investigar outliers.")

    # recomendações pelo MPS
    if rupture_rate > 10:
        recs.append("**Rupturas (>10%)** → elevar estoque de segurança, capacidade ou lead time menor.")
    if avg_inventory > 0 and holding_cost > shortage_total_cost and holding_cost > ordering_cost:
        recs.append("**Custo de manutenção domina** → avaliar reduzir lotes (L4L) ou revisar SS.")
    if shortage_total_cost > holding_cost:
        recs.append("**Custo de falta elevado** → revisar mix de produção / capacidade / SS por risco.")
    if lot_policy == "FX" and orders_count == 0 and (mps_df["gross_requirements"] > 0).any():
        recs.append("**FX sem ordens**: revisar tamanho de lote e lead time (pode estar travando as liberações).")

    if not recs:
        st.success("Sem alertas críticos — parâmetros atuais parecem coerentes para este cenário.")
    else:
        st.markdown("\n".join(f"- {r}" for r in recs))

# =========================
# Navegação
# =========================
st.divider()
c1, c2 = st.columns(2)
with c1:
    st.page_link("pages/05_Inputs_MPS.py", label="⬅️ Voltar: Inputs do MPS", icon="⚙️")
with c2:
    st.page_link("pages/04_Previsao.py", label="🔮 Ajustar Previsão", icon="🔁")
