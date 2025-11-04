# pages/07_Dashboard_Conclusao.py
from __future__ import annotations
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Conclusão", page_icon="✅", layout="wide")
st.title("✅ Conclusão (Painel de Decisão)")

# -----------------------------
# Helpers
# -----------------------------
def _safe_num(x, nd=2):
    try:
        v = float(x)
        if np.isnan(v): return "—"
        if abs(v) >= 1000:
            return f"{v:,.0f}".replace(",", ".")
        return f"{v:.{nd}f}"
    except Exception:
        return "—"

def _to_ts(x):
    # aceita 'Set/25', Timestamp ou string ISO
    if isinstance(x, pd.Timestamp):
        return x
    s = str(x)
    # label tipo "Set/25"
    _PT = {"Jan":1,"Fev":2,"Mar":3,"Abr":4,"Mai":5,"Jun":6,"Jul":7,"Ago":8,"Set":9,"Out":10,"Nov":11,"Dez":12}
    if "/" in s and len(s) in (6,7):
        mon, yy = s.split("/")
        try:
            return pd.Timestamp(year=2000+int(yy), month=_PT.get(mon.title(), 1), day=1)
        except Exception:
            pass
    # ISO fallback
    try:
        return pd.to_datetime(s, errors="coerce")
    except Exception:
        return pd.NaT

def _kpi(label, value, help_text=None, key=None):
    c = st.container()
    with c:
        st.metric(label, value)
        if help_text:
            st.caption(help_text)
    return c

# -----------------------------
# Recuperos de memória
# -----------------------------
ss = st.session_state
res = ss.get("last_result")                   # resultado da previsão (dict-like com attrs)
fcst_df = ss.get("forecast_df")              # df salvo na pág. 04 (ds,y)
hist_df_norm = ss.get("ts_df_norm")          # upload normalizado (ds,y)
mps_tbl_display = ss.get("mps_table")        # tabela do MPS (a de exibição)
mps_detail = ss.get("mps_detail")            # detalhe do core (se você decidir guardar)
# Recupera tabela de experimentos (forma robusta)
exp_df = None
for key in ["experiments_df", "experiments_table", "pipeline_experiments"]:
    val = ss.get(key)
    if isinstance(val, pd.DataFrame) and not val.empty:
        exp_df = val
        break


tabs = st.tabs(["📊 Acurácia", "🧭 Vieses", "🏭 MPS & KPIs", "💡 Recomendações"])

# ======================================================
# TAB 1 — ACURÁCIA (cards enxutos + gráfico limpo)
# ======================================================
with tabs[0]:
    st.subheader("Desempenho dos modelos de previsão")

    # ----------------- Pega modelo campeão -----------------
    champion = {}
    if res is not None and hasattr(res, "attrs"):
        champion = res.attrs.get("champion", {}) or {}

    modelo_nome = champion.get("model", "Desconhecido")
    preprocess = champion.get("preprocess", "—")
    model_params = champion.get("model_params", "—")

    st.markdown(f"**🏆 Modelo Campeão:** {modelo_nome}")
    st.caption(f"Pré-processamento: `{preprocess}` — Parâmetros: `{model_params}`")

    st.markdown("---")

    # ----------------- Gráfico Real x Previsão -----------------
    hist = None
    if isinstance(hist_df_norm, pd.DataFrame) and {"ds","y"}.issubset(hist_df_norm.columns):
        hist = hist_df_norm.copy()
        hist["ds"] = hist["ds"].apply(_to_ts)
        hist = hist.dropna(subset=["ds"]).rename(columns={"y":"Real"})

    prev = None
    if isinstance(fcst_df, pd.DataFrame) and {"ds","y"}.issubset(fcst_df.columns):
        prev = fcst_df.copy()
        prev["ds"] = prev["ds"].apply(_to_ts)
        prev = prev.dropna(subset=["ds"]).rename(columns={"y":"Previsão"})

    if hist is None:
        st.info("Sem histórico em memória. Gere o upload na página **01_Upload**.")
    else:
        plot_df = pd.DataFrame({"ds": hist["ds"], "série": "Real", "valor": hist["Real"]})
        if prev is not None and len(prev) > 0:
            plot_df = pd.concat([
                plot_df,
                pd.DataFrame({"ds": prev["ds"], "série": "Previsão", "valor": prev["Previsão"]})
            ], ignore_index=True)

        import altair as alt
        chart = (
            alt.Chart(plot_df)
            .mark_line()
            .encode(
                x=alt.X("ds:T", title="Mês"),
                y=alt.Y("valor:Q", title="Quantidade"),
                color=alt.Color(
                    "série:N",
                    scale=alt.Scale(domain=["Real","Previsão"], range=["#1e3a8a", "#60a5fa"]),
                    legend=alt.Legend(title=None, orient="top")
                ),
                tooltip=[
                    alt.Tooltip("ds:T", title="Período"),
                    alt.Tooltip("série:N", title="Série"),
                    alt.Tooltip("valor:Q", title="Valor", format=",.0f"),
                ]
            )
            .properties(height=360, width="container")
            .interactive()
        )
        st.altair_chart(chart, use_container_width=True)

    st.markdown("---")

    # ----------------- Métricas e Avaliação -----------------
    mae = champion.get("MAE")
    smape = champion.get("sMAPE")
    rmse = champion.get("RMSE")
    mape = champion.get("MAPE")

    st.markdown("### 📊 Métricas de desempenho")

    def _avaliar_mae(v):
        if v is None or np.isnan(v): return "—"
        if v < 10: return "Excelente precisão (erro médio muito baixo)."
        elif v < 30: return "Boa precisão — previsão próxima da realidade."
        elif v < 60: return "Precisão moderada — há flutuações relevantes."
        else: return "Erro alto — revisar modelo e possíveis outliers."

    def _avaliar_smape(v):
        if v is None or np.isnan(v): return "—"
        if v < 10: return "Muito bom (erro percentual simétrico muito baixo)."
        elif v < 20: return "Bom desempenho geral."
        elif v < 40: return "Erro moderado — previsão aceitável, mas pode melhorar."
        else: return "Erro alto — previsão instável ou sazonalidade não capturada."

    def _avaliar_rmse(v):
        if v is None or np.isnan(v): return "—"
        return "RMSE mede a **dispersão dos erros** — quanto menor, mais consistente a previsão."

    def _avaliar_mape(v):
        if v is None or np.isnan(v): return "—"
        if v < 10: return "Excelente (erro médio abaixo de 10%)."
        elif v < 20: return "Bom (erro entre 10–20%)."
        elif v < 30: return "Atenção — erro considerável, revisar tendência."
        else: return "Ruim — erro alto, revisar modelo e dados."

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MAE", _safe_num(mae))
    c2.metric("sMAPE (%)", _safe_num(smape))
    c3.metric("RMSE", _safe_num(rmse))
    c4.metric("MAPE (%)", _safe_num(mape))

    st.markdown("#### 🧠 Interpretação das métricas")
    st.caption(f"**MAE:** {_avaliar_mae(mae)}")
    st.caption(f"**sMAPE:** {_avaliar_smape(smape)}")
    st.caption(f"**RMSE:** {_avaliar_rmse(rmse)}")
    st.caption(f"**MAPE:** {_avaliar_mape(mape)}")

    st.divider()

# ======================================================
# TAB 2 — VIESES (com conclusão automática)
# ======================================================
with tabs[1]:
    st.subheader("Diagnóstico de vieses da previsão")

    st.markdown("""
    > O **viés** de previsão mede se o modelo tende a **superestimar** ou **subestimar** os valores reais ao longo do tempo.  
    > Quando o viés é **positivo**, o modelo prevê valores maiores do que o realizado; quando é **negativo**, prevê valores menores.  
    > Um modelo sem viés apresenta erros que oscilam em torno de zero, indicando previsões equilibradas sem tendência sistemática.
    """)


    # 1) Tenta recuperar um backtest com ds, y_true, y_pred
    bt = None
    if res is not None and hasattr(res, "attrs"):
        for k in ["backtest", "oos_eval", "cv_last", "val_df", "fitted_df"]:
            obj = res.attrs.get(k)
            if isinstance(obj, pd.DataFrame) and {"ds", "y_true", "y_pred"}.issubset(obj.columns):
                bt = obj[["ds", "y_true", "y_pred"]].copy()
                bt["ds"] = pd.to_datetime(bt["ds"], errors="coerce")
                bt = bt.dropna(subset=["ds"]).sort_values("ds")
                break

    # 2) Sem backtest, não há diagnóstico honesto de viés
    if bt is None or bt.empty:
        st.info(
            "Não encontrei um **backtest** com `y_true` e `y_pred` no resultado da previsão. "
            "Sem esses dados não é possível calcular vieses históricos. "
            "Se quiser, podemos adicionar cross-validation ao pipeline para habilitar essa aba."
        )
    else:
        # Métricas de viés
        bt["erro"] = bt["y_pred"] - bt["y_true"]
        bias_abs = float(bt["erro"].mean()) if bt["erro"].notna().any() else np.nan

        # viés relativo (%): média de (erro/real) ignorando reais = 0
        with np.errstate(divide="ignore", invalid="ignore"):
            pct_vec = np.where(bt["y_true"] != 0, bt["erro"] / bt["y_true"], np.nan)
        bias_pct = float(np.nanmean(pct_vec) * 100.0)

        # MAE do backtest para escalar a conclusão
        mae_bt = float(np.nanmean(np.abs(bt["erro"]))) if bt["erro"].notna().any() else np.nan

        # KPIs lado a lado
        c1, c2, c3 = st.columns(3, gap="large")
        with c1:
            _kpi("Viés (nível)", _safe_num(bias_abs), "Média de (previsto − real)")
        with c2:
            _kpi("Viés (%)", _safe_num(bias_pct), "Média de (previsto − real)/real × 100")
        with c3:
            _kpi("MAE (backtest)", _safe_num(mae_bt), "Média do |erro| no período de teste")


        st.caption(
            "Interpretação: valores **positivos** indicam **superestimação**; negativos, **subestimação**. "
            "Quanto mais próximo de 0, menor o viés."
        )

        # 3) Gráfico do erro (azul escuro) + linha de referência zero
        import altair as alt
        linha_zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="#9ca3af").encode(y="y:Q")
        ch = (
            alt.Chart(bt[["ds", "erro"]])
            .mark_line(color="#1e3a8a")
            .encode(
                x=alt.X("ds:T", title="Período"),
                y=alt.Y("erro:Q", title="Erro (previsto − real)"),
                tooltip=[
                    alt.Tooltip("ds:T", title="Período"),
                    alt.Tooltip("erro:Q", title="Erro", format=",.2f"),
                ],
            )
            .properties(height=300, width="container")
            .interactive()
        )
        st.altair_chart(linha_zero + ch, use_container_width=True)

        # 4) Conclusão automática (baseada no viés vs. MAE)
        conclusao = ""
        if np.isfinite(bias_abs) and np.isfinite(mae_bt) and mae_bt > 0:
            # limiar: 5% do MAE → sem viés material; senão aponta direção
            if abs(bias_abs) < 0.05 * mae_bt:
                conclusao = "✅ **Sem viés sistemático relevante.** Os erros oscilam ao redor de zero."
            elif bias_abs > 0:
                conclusao = "⚠️ **Viés positivo (superestimação).** Em média o modelo prevê acima do realizado."
            else:
                conclusao = "⚠️ **Viés negativo (subestimação).** Em média o modelo prevê abaixo do realizado."
        else:
            conclusao = "ℹ️ **Não foi possível calcular uma conclusão automática** (dados insuficientes)."

        st.markdown(f"**Conclusão automática:** {conclusao}")

        # 5) Explicação fixa e objetiva
        st.markdown(
            """
**Como ler o gráfico acima:**  
- A linha mostra o **erro** em cada período (previsto − real).  
- **Acima de 0** → o modelo **superestimou**; **abaixo de 0** → **subestimou**.  
- Quando os pontos ficam próximos de 0 e alternam entre positivo/negativo, **não há viés sistemático**.  
- Deslocamentos persistentes para cima/baixo sugerem **viés** e pedem recalibração do modelo (parâmetros, sazonalidade ou tendência).
            """
        )


# ======================================================
# TAB 3 — MPS & Custos (EPQ alinhado ao material — sem custo de produção)
# ======================================================
with tabs[2]:
    st.subheader("MPS — Custos e Resumo (somente leitura dos inputs)")

    # ---------- Guardas ----------
    mps_inputs = st.session_state.get("mps_inputs", {})
    if not isinstance(mps_inputs, dict) or not mps_inputs:
        st.info("Não encontrei os **inputs do MPS**. Preencha e salve na página **05_Inputs_MPS**.")
        st.page_link("pages/05_Inputs_MPS.py", label="⚙️ Ir para 05_Inputs_MPS")
        st.stop()

    def _get_df_from_state(keys):
        for k in keys:
            obj = st.session_state.get(k, None)
            if isinstance(obj, pd.DataFrame) and not obj.empty:
                return obj
        return None
    mps_tbl_display = _get_df_from_state(["mps_tbl_display", "mps_table"])
    if mps_tbl_display is None:
        st.info("Não há tabela do MPS na memória. Gere o MPS na página **06_MPS** e volte.")
        st.page_link("pages/06_MPS.py", label="📅 Ir para 06_MPS")
        st.stop()

    # ---------- Helpers ----------
    def _safe(v, nd=0):
        try:
            if np.isnan(v): return "—"
        except Exception:
            pass
        try:
            return f"{float(v):,.{nd}f}".replace(",", "X").replace(".", ",").replace("X", ".")
        except Exception:
            return str(v)

    def _as_float(x, default=0.0):
        try:
            if x is None: return float(default)
            if isinstance(x, str) and x.strip() == "": return float(default)
            return float(x)
        except Exception:
            return float(default)

    def _find_row(df: pd.DataFrame, candidates: list[str]):
        idx_norm = df.index.astype(str).str.strip().str.lower()
        for cand in candidates:
            m = (idx_norm == cand.strip().lower())
            if m.any():
                pos = np.where(m)[0][0]
                label = df.index[pos]
                return df.loc[label]
        return None

    # ---------- Leitura segura ----------
    time_base = mps_inputs.get("time_base", "por mês")
    A = _as_float(mps_inputs.get("A", mps_inputs.get("order_cost", 0.0)), 0.0)
    v = _as_float(mps_inputs.get("v", mps_inputs.get("unit_cost", 0.0)), 0.0)

    h_mode = mps_inputs.get("h_mode", "Informar H diretamente")
    if h_mode == "Informar H diretamente":
        H_in = _as_float(mps_inputs.get("H", 0.0), 0.0)
        H_m = H_in if time_base == "por mês" else H_in / 12.0
        r_show = _as_float(mps_inputs.get("r", None), 0.0)
    else:
        r_val = _as_float(mps_inputs.get("r", 0.0), 0.0)
        H_calc = r_val * v
        H_m = H_calc if time_base == "por mês" else H_calc / 12.0
        r_show = r_val

    D_m = _as_float(mps_inputs.get("D_month", mps_inputs.get("D", 0.0)), 0.0)
    p_m = _as_float(mps_inputs.get("p_month", mps_inputs.get("p", 0.0)), 0.0)
    if time_base == "por ano":
        D_m = D_m / 12.0
        p_m = p_m / 12.0

    if D_m <= 0 and "forecast_df" in st.session_state:
        _fc = st.session_state["forecast_df"][["ds", "y"]].copy()
        D_m = float(np.nanmean(_fc["y"].values)) if len(_fc) else 0.0

    HORIZ_MESES = max(1, len(mps_tbl_display.columns))

    row_ruptura = _find_row(mps_tbl_display, ["Ruptura", "falta", "backlog", "não atendido"])
    total_ruptura = float(np.nansum(np.clip(row_ruptura.values.astype(float), 0, None))) if row_ruptura is not None else 0.0
    pi_shortage = _as_float(mps_inputs.get("pi_shortage", 0.0), 0.0)

    # ---------- EPQ ----------
    epq_viavel = (p_m > D_m) and (H_m > 0) and (A >= 0)

    if epq_viavel:
        fator = (1.0 - (D_m / p_m))
        try:
            Q_star = np.sqrt((2.0 * A * D_m) / (H_m * fator))
        except Exception:
            Q_star = np.nan
        I_med = 0.5 * Q_star * fator if (Q_star and Q_star > 0) else 0.0

        # Custos relevantes (por mês)
        C_setup_mes = (A * D_m / Q_star) if (Q_star and Q_star > 0) else 0.0
        C_hold_mes  = H_m * I_med

    else:
        Q_star = np.nan
        C_setup_mes = C_hold_mes = 0.0

    cost_encomendar = C_setup_mes * HORIZ_MESES
    cost_manter     = C_hold_mes * HORIZ_MESES
    cost_ruptura    = total_ruptura * pi_shortage
    cost_total      = cost_encomendar + cost_manter + cost_ruptura

    # ---------- Expander com variáveis ----------
    with st.expander("ℹ️ Variáveis (05_Inputs_MPS) e parâmetros calculados"):
        a1, a2, a3, a4 = st.columns(4)
        a1.metric("A (setup)", _safe(A, 2))
        a2.metric("v (valor unit.)", _safe(v, 2))
        a3.metric("H_mensal (R$/un·mês)", _safe(H_m, 4))
        a4.metric("π (ruptura)", _safe(pi_shortage, 2))
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("D (unid/mês)", _safe(D_m, 2))
        b2.metric("p (unid/mês)", _safe(p_m, 2))
        b3.metric("r (taxa man.)", _safe(r_show, 4))
        b4.metric("Q* (EPQ)", _safe(Q_star, 2))

        st.latex(r"Q^* = \sqrt{\frac{2AD}{H(1-D/p)}}")
        st.latex(r"I_{médio} = \frac{Q}{2}\left(1 - \frac{D}{p}\right)")
        st.latex(r"C_{setup} = \frac{AD}{Q}, \quad C_{hold} = H \cdot I_{médio}")
        st.caption("Custos acima são **por mês**; multiplicamos pelo **nº de meses do horizonte**.")

    # ---------- Layout visual ----------
    st.markdown("""
    <style>
    .kpi-card {background:#fff;border:1px solid #e5e7eb;border-radius:12px;padding:14px 16px;margin-bottom:12px}
    .kpi-top {display:flex;align-items:center;gap:6px;color:#374151;font-size:0.95rem}
    .kpi-help {color:#6b7280;cursor:help;font-size:0.95rem}
    .kpi-value {font-size:1.85rem;font-weight:700;line-height:1.1;margin-top:6px}
    .kpi-sub {color:#6b7280;font-size:0.85rem;margin-top:4px}
    </style>
    """, unsafe_allow_html=True)

    def _fmt_money(x, nd=2):
        try:
            return f"{float(x):,.{nd}f}".replace(",", "X").replace(".", ",").replace("X", ".")
        except Exception:
            return str(x)

    def _help_span(txt: str) -> str:
        safe = (txt or "").replace('"', "&quot;")
        return f'<span class="kpi-help" title="{safe}">ⓘ</span>'

    Q_lbl      = _safe(Q_star, 2)
    D_lbl      = _safe(D_m, 2)
    p_lbl      = _safe(p_m, 2)
    H_lbl      = _safe(H_m, 4)
    A_lbl      = _safe(A, 2)
    months_lbl = f"{int(HORIZ_MESES)}"

    tip_setup = (
        "Custo de Setup = A × D / Q. "
        f"Parâmetros usados: A={A_lbl}, D={D_lbl}, Q={Q_lbl}, meses={months_lbl}."
    )
    tip_hold = (
        "Custo de Manter = H × (Q/2) × (1 − D/p). "
        f"Parâmetros usados: H={H_lbl}, Q={Q_lbl}, D={D_lbl}, p={p_lbl}, meses={months_lbl}."
    )
    tip_rupt = "Custo de ruptura: Σ(Ruptura) × π."
    tip_total = "Custo total relevante = Setup + Manter + Ruptura."

    L, R = st.columns(2)

    with L:
        st.markdown(
            f"""<div class="kpi-card">
                <div class="kpi-top">Custo de encomendar (R$){_help_span(tip_setup)}</div>
                <div class="kpi-value">{_fmt_money(cost_encomendar, 2)}</div>
                <div class="kpi-sub">Usa A, D, Q e meses</div>
            </div>""",
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""<div class="kpi-card">
                <div class="kpi-top">Custo de manter (R$){_help_span(tip_hold)}</div>
                <div class="kpi-value">{_fmt_money(cost_manter, 2)}</div>
                <div class="kpi-sub">Usa H, Q, D, p e meses</div>
            </div>""",
            unsafe_allow_html=True,
        )

    with R:
        st.markdown(
            f"""<div class="kpi-card">
                <div class="kpi-top">Custo de ruptura (R$){_help_span(tip_rupt)}</div>
                <div class="kpi-value">{_fmt_money(cost_ruptura, 2)}</div>
                <div class="kpi-sub">Σ(Ruptura) × π</div>
            </div>""",
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""<div class="kpi-card">
                <div class="kpi-top">Custo total relevante{_help_span(tip_total)}</div>
                <div class="kpi-value">{_fmt_money(cost_total, 2)}</div>
                <div class="kpi-sub">Setup + Manter + Ruptura</div>
            </div>""",
            unsafe_allow_html=True,
        )

    if not epq_viavel:
        st.warning("EPQ não aplicável com os parâmetros atuais (é preciso **p > D** e **H > 0**). "
                   "Os custos de encomendar/manter mostrados ficarão zerados até ajustar os inputs.")

# ======================================================
# TAB 4 — Recomendações (texto curto e objetivo)
# ======================================================
with tabs[3]:
    st.subheader("Recomendações")
    recs = []

    # com base no campeão
    if champion:
        smape = champion.get("sMAPE")
        if smape is not None and isinstance(smape, (int, float)):
            if smape > 30:
                recs.append("sMAPE alto → considerar **mais dados**, **tratamento de outliers** e/ou **outro modelo**.")
            elif smape > 15:
                recs.append("sMAPE moderado → ajuste fino de **hiperparâmetros** e checagem de **sazonalidade**.")
            else:
                recs.append("sMAPE baixo → manter configuração atual e acompanhar periodicamente.")

    # sugestão de operação com MPS disponível
    if isinstance(mps_tbl_display, pd.DataFrame) and not mps_tbl_display.empty:
        try:
            estoque_final = int(mps_tbl_display.loc["Estoque Proj.", mps_tbl_display.columns[-1]])
            if estoque_final < 0:
                recs.append("Estoque projetado **negativo** no fim do horizonte → **antecipar** produção/compras.")
            elif estoque_final == 0:
                recs.append("Estoque projetado **zerado** no fim do horizonte → atenção a possíveis **rupturas**.")
        except Exception:
            pass

    if recs:
        st.markdown("\n".join(f"- {r}" for r in recs))
    else:
        st.markdown("- Sem recomendações automáticas no momento.")

    st.divider()
    st.page_link("pages/06_MPS.py", label="📅 Abrir MPS", icon="🗓️")
