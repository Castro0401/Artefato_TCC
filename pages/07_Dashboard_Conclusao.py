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
# TAB 3 — MPS & Custos (EPQ com Q do usuário; L4L por mês; sem custo de produção)
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

    # tabela do MPS (precisamos dela para L4L e para rupturas)
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

    # ---------- Leitura segura (05_Inputs_MPS) ----------
    time_base   = mps_inputs.get("time_base", "por mês")
    A           = _as_float(mps_inputs.get("A", mps_inputs.get("order_cost", 0.0)), 0.0)
    v           = _as_float(mps_inputs.get("v", mps_inputs.get("unit_cost", 0.0)), 0.0)  # não entra no custo, só exibição
    lot_policy  = mps_inputs.get("lot_policy_default", "FX")  # "FX" ou "L4L"
    Q_user      = int(_as_float(mps_inputs.get("lot_size_default", 1), 1))  # tamanho do lote quando FX

    # H (mensal)
    h_mode = mps_inputs.get("h_mode", "Informar H diretamente")
    if h_mode == "Informar H diretamente":
        H_in  = _as_float(mps_inputs.get("H", 0.0), 0.0)
        H_m   = H_in if time_base == "por mês" else H_in / 12.0
        r_show = _as_float(mps_inputs.get("r", None), 0.0)
    else:
        r_val  = _as_float(mps_inputs.get("r", 0.0), 0.0)
        H_calc = r_val * v
        H_m    = H_calc if time_base == "por mês" else H_calc / 12.0
        r_show = r_val

    # D e p sempre mensais (se vieram anuais, converte)
    D_m = _as_float(mps_inputs.get("D_month", mps_inputs.get("D", 0.0)), 0.0)
    p_m = _as_float(mps_inputs.get("p_month", mps_inputs.get("p", 0.0)), 0.0)
    if time_base == "por ano":
        D_m /= 12.0
        p_m /= 12.0

    # fallback: média da previsão salva
    if D_m <= 0 and "forecast_df" in st.session_state:
        _fc = st.session_state["forecast_df"][["ds", "y"]].copy()
        D_m = float(np.nanmean(_fc["y"].values)) if len(_fc) else 0.0

    HORIZ_MESES = max(1, len(mps_tbl_display.columns))

    # Ruptura (se existir)
    row_ruptura = _find_row(mps_tbl_display, ["Ruptura", "falta", "backlog", "não atendido"])
    total_ruptura = float(np.nansum(np.clip(row_ruptura.values.astype(float), 0, None))) if row_ruptura is not None else 0.0
    pi_shortage = _as_float(mps_inputs.get("pi_shortage", 0.0), 0.0)

    # Linha de recebimentos (para L4L e contagem de setups)
    row_qtd_mps = _find_row(mps_tbl_display, ["qtde. mps", "qtde mps", "quantidade mps", "mps qty"])
    qtd_mps_vals = row_qtd_mps.values.astype(float) if row_qtd_mps is not None else np.zeros(HORIZ_MESES)

    # ---------- Custos (Q do usuário) ----------
    # Condição estrutural do EPQ
    epq_estrutura_ok = (p_m > D_m) and (H_m > 0) and (A >= 0)

    if not epq_estrutura_ok:
        C_setup_total = 0.0
        C_hold_total  = 0.0
    else:
        if lot_policy == "FX":
            # EPQ com Q = lote informado
            Q = max(1, int(Q_user))
            C_setup_mes = (A * D_m / Q)
            C_hold_mes  = H_m * (Q / 2.0) * (1.0 - (D_m / p_m))
            C_setup_total = C_setup_mes * HORIZ_MESES
            C_hold_total  = C_hold_mes  * HORIZ_MESES
        else:
            # L4L: por mês com qi do MPS
            setups = int(np.nansum((qtd_mps_vals > 0).astype(int)))
            C_setup_total = A * setups
            fator = (1.0 - (D_m / p_m))
            qi_term = np.clip(qtd_mps_vals, 0, None) / 2.0
            C_hold_total = float(np.nansum(H_m * qi_term * fator))

    cost_ruptura = total_ruptura * pi_shortage
    cost_total   = C_setup_total + C_hold_total + cost_ruptura

    # ---------- Expander com variáveis ----------
    with st.expander("ℹ️ Variáveis (05_Inputs_MPS) e parâmetros usados"):
        a1, a2, a3, a4 = st.columns(4)
        a1.metric("A (setup)", _safe(A, 2))
        a2.metric("v (valor unit.)", _safe(v, 2))
        a3.metric("H mensal (R$/un·mês)", _safe(H_m, 2))
        a4.metric("π (ruptura)", _safe(pi_shortage, 2))
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("D (unid/mês)", _safe(D_m, 2))
        b2.metric("p (unid/mês)", _safe(p_m, 2))
        b3.metric("r (taxa man.)", _safe(r_show, 4))
        b4.metric("Política", "Lote Fixo" if lot_policy == "FX" else "Lote-a-Lote")
        if lot_policy == "FX":
            st.caption(f"Lote informado (Q): **{_safe(Q_user, 0)}**")
        else:
            st.caption(f"Setups no horizonte (meses com recebimento): **{int(np.nansum((qtd_mps_vals>0).astype(int)))}**")

        # Fórmulas renderizadas
        st.latex(r"C_{\text{setup,mês}}=\frac{A\,D_m}{Q}\quad\text{(FX)}")
        st.latex(r"C_{\text{hold,mês}}=H_m\cdot\frac{Q}{2}\left(1-\frac{D_m}{p_m}\right)\quad\text{(FX)}")
        st.latex(r"C_{\text{setup,total}}=\#\text{setups}\cdot A\quad\text{(L4L)}")
        st.latex(r"C_{\text{hold,total}}=\sum_i H_m\cdot\frac{q_i}{2}\left(1-\frac{D_m}{p_m}\right)\quad\text{(L4L)}")
        st.caption("Multiplicamos pelos **meses do horizonte** quando FX. Em L4L, somamos mês a mês.")

    # ---------- Layout visual (cards + tooltips) ----------
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

    months_lbl = f"{HORIZ_MESES:d}"
    tip_setup = (
        "FX: C_setup,mês = A·D_m/Q; total = C_setup,mês × meses. "
        "L4L: C_setup,total = nº de setups × A."
    )
    tip_hold = (
        "FX: C_hold,mês = H_m·(Q/2)·(1 − D_m/p_m); total = C_hold,mês × meses. "
        "L4L: C_hold,total = Σ H_m·(q_i/2)·(1 − D_m/p_m)."
    )
    tip_rupt = "C_rupt = Σ(Ruptura) × π (se sua tabela tiver a linha de Ruptura)."
    tip_total = "Custo total relevante = Setup + Manter + Ruptura."

    L, R = st.columns(2)

    with L:
        st.markdown(
            f"""<div class="kpi-card">
                <div class="kpi-top">Custo de setup (R$){_help_span(tip_setup)}</div>
                <div class="kpi-value">{_fmt_money(C_setup_total, 2)}</div>
                <div class="kpi-sub">{'FX (mensal × ' + months_lbl + ')' if lot_policy=='FX' else 'L4L (setups × A)'}</div>
            </div>""",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""<div class="kpi-card">
                <div class="kpi-top">Custo de manter (R$){_help_span(tip_hold)}</div>
                <div class="kpi-value">{_fmt_money(C_hold_total, 2)}</div>
                <div class="kpi-sub">{'FX (mensal × ' + months_lbl + ')' if lot_policy=='FX' else 'L4L (soma mensal)'}</div>
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

    if not epq_estrutura_ok:
        st.warning("Parâmetros inválidos para EPQ: é preciso **p > D** e **H > 0**. "
                   "Ajuste no **05_Inputs_MPS**.")

# =============================
# ATP — “dá pra atender novas demandas?”
# =============================
st.divider()
st.markdown("## 🧮 ATP — Capacidade de atender novas demandas (por mês)")

# 1) Obter ATP mensal (detail > atp  OU reconstruir de 'ATP(cum)' do display)
mps_detail = st.session_state.get("mps_detail", None)

def _monthly_atp_from_cum(df_disp: pd.DataFrame) -> pd.Series | None:
    try:
        idx_norm = df_disp.index.astype(str).str.strip().str.lower()
        m = (idx_norm == "atp(cum)".lower())
        if not m.any():
            return None
        row = df_disp.loc[df_disp.index[m][0]]
        cum = pd.to_numeric(row.values, errors="coerce")
        monthly = np.diff(np.r_[0, cum])
        return pd.Series(np.clip(monthly, 0, None), index=df_disp.columns)
    except Exception:
        return None

if isinstance(mps_detail, pd.DataFrame) and ("atp" in mps_detail.columns):
    atp_series = pd.to_numeric(mps_detail["atp"], errors="coerce").fillna(0)
    labels = st.session_state["forecast_df"]["ds"].tolist()
    PT_MON = ["Jan","Fev","Mar","Abr","Mai","Jun","Jul","Ago","Set","Out","Nov","Dez"]
    def _fmt(ts): 
        ts = pd.to_datetime(ts); return f"{PT_MON[ts.month-1]}/{ts.year%100:02d}"
    col_labels = [_fmt(x) for x in labels]
    atp_monthly = pd.Series(np.clip(atp_series.values, 0, None), index=col_labels)
else:
    atp_monthly = _monthly_atp_from_cum(mps_tbl_display)

if atp_monthly is None:
    st.info("Não encontrei o **ATP** no resultado. Gere o MPS novamente ou habilite o cálculo de ATP no core.")
else:
    # 2) Base e parâmetro de teste
    atp_df = pd.DataFrame({"Mês": atp_monthly.index, "ATP (unid/mês)": atp_monthly.values.astype(int)})
    extra = st.number_input(
        "Demanda extra hipotética (un/mês)", min_value=0, step=1, value=0,
        help="Valor fixo de nova demanda a testar em cada mês."
    )
    atp_df["Atende"] = atp_df["ATP (unid/mês)"] >= int(extra)

    # 3) Gráfico (cores intuitivas + linha de referência + rótulos)
    import altair as alt
    chart_data = atp_df.copy()
    chart_data["Demanda extra"] = extra

    color_scale = alt.Scale(
        domain=[True, False],
        range=["#16a34a", "#dc2626"]  # verde / vermelho
    )

    bars = alt.Chart(chart_data).mark_bar().encode(
        x=alt.X("Mês:N", title="Mês", sort=None),
        y=alt.Y("ATP (unid/mês):Q", title="ATP (unid/mês)", scale=alt.Scale(nice=True, zero=True)),
        color=alt.Color("Atende:N", title="Atende a extra?", scale=color_scale),
        tooltip=["Mês", alt.Tooltip("ATP (unid/mês):Q", format=",.0f"), "Atende"]
    )

    labels = alt.Chart(chart_data).mark_text(
        dy=-6, fontSize=11
    ).encode(
        x="Mês:N",
        y=alt.Y("ATP (unid/mês):Q", stack=None),
        text=alt.Text("ATP (unid/mês):Q", format=",.0f"),
        color=alt.value("#111827")
    )

    threshold = alt.Chart(chart_data).mark_rule(color="#0f172a", strokeDash=[6,4]).encode(
        x="Mês:N",
        y="Demanda extra:Q",
        tooltip=[alt.Tooltip("Demanda extra:Q", format=",.0f")]
    )

    st.altair_chart(
        (bars + labels + threshold).properties(
            height=340, width="container",
            title=f"ATP por mês (linha = demanda extra: {extra} un/mês)"
        ).configure_axis(labelFontSize=12, titleFontSize=12)
         .configure_legend(labelFontSize=12, titleFontSize=12),
        use_container_width=True
    )

    # 4) Espaço visual antes da tabela
    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    # 5) Tabela (com indicação visual)
    atp_df["✔ Atende?"] = atp_df["Atende"].map({True:"✅", False:"❌"})
    display_df = atp_df[["Mês", "ATP (unid/mês)", "✔ Atende?"]].rename(
        columns={"ATP (unid/mês)":"ATP (unid/mês)"}
    )
    st.dataframe(display_df, use_container_width=True, height=280)

    # 6) Resumo
    if extra > 0:
        meses_ok = display_df.loc[atp_df["Atende"], "Mês"].tolist()
        if meses_ok:
            st.success(f"Com **{extra} un/mês** de demanda extra, meses atendidos: {', '.join(meses_ok)}.")
        else:
            st.warning(f"Com **{extra} un/mês**, nenhum mês teria ATP suficiente.")
    else:
        st.caption("Ajuste a demanda extra acima para testar cenários.")


# ======================================================
# TAB 4 — Recomendações (assistente de decisões)
# ======================================================
with tabs[3]:
    st.subheader("Recomendações")

    # ----------------------------
    # Helpers
    # ----------------------------
    def _as_float(x, dv=0.0):
        try:
            if x is None: return float(dv)
            if isinstance(x, str) and x.strip()=="":
                return float(dv)
            return float(x)
        except Exception:
            return float(dv)

    def _money(x, nd=2):
        try:
            return f"{float(x):,.{nd}f}".replace(",", "X").replace(".", ",").replace("X", ".")
        except Exception:
            return str(x)

    def _safe_series(df, candidates):
        if not isinstance(df, pd.DataFrame) or df.empty:
            return None
        idx = df.index.astype(str).str.strip().str.lower()
        for cand in candidates:
            m = (idx == cand.strip().lower())
            if m.any():
                return df.loc[idx[m].index[0]].astype(float)
        return None

    # ----------------------------
    # Carrega insumos do estado
    # ----------------------------
    mps_inputs  = st.session_state.get("mps_inputs", {}) or {}
    mps_table   = st.session_state.get("mps_tbl_display") or st.session_state.get("mps_table")
    mps_detail  = st.session_state.get("mps_detail")  # se existir, ótimo; senão, seguimos só com mps_table
    champion    = st.session_state.get("model_champion", {}) or st.session_state.get("champion", {})

    # Parâmetros EPQ/inputs
    time_base   = mps_inputs.get("time_base", "por mês")
    A           = _as_float(mps_inputs.get("A", mps_inputs.get("order_cost", 0.0)))
    v           = _as_float(mps_inputs.get("v", mps_inputs.get("unit_cost", 0.0)))
    D_base      = _as_float(mps_inputs.get("D", 0.0))        # coerente com base escolhida
    p_base      = _as_float(mps_inputs.get("p", 0.0))
    pi_shortage = _as_float(mps_inputs.get("pi_shortage", mps_inputs.get("shortage_cost", 0.0)))
    h_mode      = mps_inputs.get("h_mode", "Informar H diretamente")
    if h_mode == "Informar H diretamente":
        H_in   = _as_float(mps_inputs.get("H", 0.0))
        H_mensal = H_in if time_base=="por mês" else (H_in/12.0)
    else:
        r_val    = _as_float(mps_inputs.get("r", 0.0))
        H_mensal = (r_val * v) if time_base=="por mês" else ((r_val * v)/12.0)

    # Converte D,p p/ mensal se vieram anuais
    D_m = D_base if time_base=="por mês" else (D_base/12.0)
    p_m = p_base if time_base=="por mês" else (p_base/12.0)

    # Horizonte (nº de meses) — via tabela do MPS
    HORIZ = len(mps_table.columns) if isinstance(mps_table, pd.DataFrame) else 0
    HORIZ = max(1, HORIZ)

    # Política/lote do usuário (Q_user se FX)
    lot_policy = mps_inputs.get("lot_policy_default", "FX")
    Q_user = _as_float(mps_inputs.get("lot_size_default", None), None) if lot_policy=="FX" else None

    # Ruptura total (se existir na tabela)
    row_rupt = _safe_series(mps_table, ["Ruptura", "falta", "backlog", "não atendido"])
    total_rupt = float(np.nansum(np.clip(row_rupt.values, 0, None))) if row_rupt is not None else 0.0

    # ----------------------------
    # CUSTOS — cenário base (usando Q_user se houver; se L4L, usamos Qtde. MPS p/ contar setups e estoque)
    # Fórmulas (mês):
    #  - Setup (FX): C_setup,m = A * D / Q_user
    #  - Manter (FX, EPQ): C_hold,m = H * (Q_user/2)*(1 - D/p), se p>D
    #  - L4L: setup = A * (#meses com "Qtde. MPS" > 0) / HORIZ; manter ≈ H * média(Estoque Proj.⁺)
    #  - Ruptura = Σ(Ruptura) * π  (no horizonte)
    # ----------------------------
    epq_ok = (p_m > D_m) and (H_mensal > 0)

    if lot_policy == "FX" and Q_user and Q_user > 0:
        C_setup_m = (A * D_m / Q_user)
        C_hold_m  = (H_mensal * (Q_user/2.0) * max(0.0, (1.0 - (D_m/p_m)))) if epq_ok else 0.0
    else:
        # L4L/indefinido: estimativa a partir da tabela
        row_mps    = _safe_series(mps_table, ["Qtde. MPS", "qtde mps", "mps qty"])
        row_stock  = _safe_series(mps_table, ["Estoque Proj.", "estoque projetado", "estoque"])
        pedidos_m  = int(np.nansum((row_mps.values > 0).astype(int))) if row_mps is not None else 0
        media_est  = float(np.nanmean(np.clip(row_stock.values, 0, None))) if row_stock is not None else 0.0
        C_setup_m  = A * (pedidos_m / HORIZ)
        C_hold_m   = H_mensal * media_est

    cost_setup   = C_setup_m  * HORIZ
    cost_hold    = C_hold_m   * HORIZ
    cost_rupt    = total_rupt * pi_shortage
    cost_total   = cost_setup + cost_hold + cost_rupt

    # ----------------------------
    # Q* (EPQ) para comparação (se aplicável)
    # ----------------------------
    if epq_ok and A>0:
        try:
            fator = (1.0 - (D_m/p_m))
            Q_star = np.sqrt((2.0*A*D_m) / (H_mensal * fator))
        except Exception:
            Q_star = np.nan
    else:
        Q_star = np.nan

    # ----------------------------
    # ATP — resumo de risco por mês
    # ----------------------------
    def _build_atp_df():
        # Prioriza mps_detail['atp']; senão tenta linha "ATP(cum)" e diferencia
        if isinstance(mps_detail, pd.DataFrame) and "atp" in mps_detail.columns:
            vals = mps_detail["atp"].astype(float).values
            idx  = mps_detail.index
        elif isinstance(mps_table, pd.DataFrame):
            atp_cum = _safe_series(mps_table, ["ATP(cum)", "atp(cum)", "atp cum"])
            if atp_cum is not None:
                cum = atp_cum.values.astype(float)
                vals = np.diff(np.r_[0, cum])  # diferença
                idx  = mps_table.columns
            else:
                return None
        else:
            return None
        lab = []
        PT = ["Jan","Fev","Mar","Abr","Mai","Jun","Jul","Ago","Set","Out","Nov","Dez"]
        for c in pd.Index(idx):
            try:
                ts = pd.to_datetime(c)
                lab.append(f"{PT[ts.month-1]}/{ts.year%100:02d}")
            except Exception:
                lab.append(str(c))
        return pd.DataFrame({"Mês": lab, "ATP_mês": vals})

    atp_df = _build_atp_df()

    # ----------------------------
    # HEADER com “quick KPIs”
    # ----------------------------
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Custo total (R$)", _money(cost_total))
    c2.metric("Holding no total", f"{(cost_hold/cost_total*100):.0f}%" if cost_total>0 else "—")
    c3.metric("Q (usuário)", f"{int(Q_user)}" if (Q_user and lot_policy=="FX") else ("L4L" if lot_policy=="L4L" else "—"))
    c4.metric("Q* (EPQ)", f"{int(round(Q_star))}" if not np.isnan(Q_star) else "—")

    # ----------------------------
    # CARDS de recomendações (regras simples)
    # ----------------------------
    st.markdown("""
    <style>
      .rec-grid {display:grid; grid-template-columns: repeat(2, minmax(0,1fr)); gap:12px; margin-top:8px;}
      .rec-card {border:1px solid #e5e7eb; border-radius:12px; padding:12px 14px; background:#fff;}
      .rec-title {font-weight:600; margin-bottom:6px;}
      .rec-sub {color:#6b7280; font-size:0.9rem;}
    </style>
    """, unsafe_allow_html=True)

    rec_html = []

    # 1) Capacidade
    if p_m <= D_m:
        rec_html.append(
            '<div class="rec-card"><div class="rec-title">⚠️ Capacidade insuficiente</div>'
            '<div class="rec-sub">p ≤ D → EPQ inválido. Avalie nivelamento, aumento de p ou smoothing da demanda.</div></div>'
        )

    # 2) Lote vs Q*
    if not np.isnan(Q_star) and (Q_user and lot_policy=="FX"):
        gap = (Q_user - Q_star)/Q_star
        if abs(gap) >= 0.15:
            sentido = "acima" if gap>0 else "abaixo"
            rec_html.append(
                f'<div class="rec-card"><div class="rec-title">📦 Lote {sentido} do ótimo</div>'
                f'<div class="rec-sub">Q={int(Q_user)} está {abs(gap)*100:.0f}% {sentido} do Q*≈{int(round(Q_star))}. '
                'Aproxime Q do ótimo para equilibrar setup × holding.</div></div>'
            )

    # 3) Custo dominante
    if cost_total>0:
        shares = {"Setup":cost_setup/cost_total, "Holding":cost_hold/cost_total, "Ruptura":cost_rupt/cost_total}
        dom = max(shares, key=shares.get)
        dica = {
            "Holding":"Reduzir H (financeiro/armazenagem), negociar espaço, aproximar Q de Q*.",
            "Setup":"Atacar A (SMED, padronização), avaliar lotes um pouco maiores.",
            "Ruptura":"Elevar SS (z), antecipar produção nos meses críticos."
        }[dom]
        rec_html.append(
            f'<div class="rec-card"><div class="rec-title">💡 Custo dominante: {dom}</div>'
            f'<div class="rec-sub">{dica}</div></div>'
        )

    # 4) ATP meses críticos
    if atp_df is not None and not atp_df.empty:
        crit = atp_df.loc[atp_df["ATP_mês"]<=0, "Mês"].tolist()
        if crit:
            rec_html.append(
                f'<div class="rec-card"><div class="rec-title">📉 Risco de atendimento</div>'
                f'<div class="rec-sub">Meses com ATP ≤ 0: {", ".join(crit[:4])}'
                f'{"…" if len(crit)>4 else ""}. Considere antecipar liberações.</div></div>'
            )

    if rec_html:
        st.markdown('<div class="rec-grid">' + "".join(rec_html) + '</div>', unsafe_allow_html=True)
    else:
        st.info("Sem alertas relevantes no cenário atual.")

    st.divider()

    # ----------------------------
    # What-if rápido (teste de Q) – não recalcula MPS, só custos teóricos
    # ----------------------------
    st.markdown("### What-if rápido de lote (Q)")
    colW1, colW2 = st.columns([1.2, 1])
    with colW1:
        base_q = int(Q_user) if (Q_user and lot_policy=="FX") else (int(round(Q_star)) if not np.isnan(Q_star) else 100)
        q_test = st.slider("Testar Q (apenas para análise de custos)", max(1, base_q//2), base_q*2, base_q, step=1,
                           help="Usa fórmulas de setup/holding por mês; não altera o MPS.")
    with colW2:
        if p_m > 0 and D_m > 0:
            C_setup_m_test = A * D_m / q_test
            C_hold_m_test  = H_mensal * (q_test/2.0) * max(0.0, (1.0 - (D_m/p_m))) if epq_ok else 0.0
            delta = (C_setup_m_test + C_hold_m_test)*HORIZ - (cost_setup + cost_hold)
            cor = "🟢" if delta < 0 else ("🟡" if abs(delta) < 1e-6 else "🔴")
            st.metric("Δ Custo (setup+holding) no horizonte", _money(delta), help="Negativo é economia.")
            st.caption(f"{cor} Setup: {_money(C_setup_m_test*HORIZ)} | Holding: {_money(C_hold_m_test*HORIZ)}")

    st.divider()

    # ----------------------------
    # ATP — mini heatmap + legenda
    # ----------------------------
    st.markdown("### ATP por mês (checagem rápida)")
    if atp_df is not None and not atp_df.empty:
        df_show = atp_df.copy()
        df_show["Status"] = np.where(df_show["ATP_mês"]>0, "OK", "Crítico")
        # espaço visual
        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        st.dataframe(
            df_show.style.apply(
                lambda s: ["background-color:#dcfce7" if (val>0) else "background-color:#fee2e2"
                           for val in df_show["ATP_mês"]],
                axis=0
            ),
            use_container_width=True, height=min(300, 40*(len(df_show)+1))
        )
        st.caption("Verde: ATP>0 (atende). Vermelho: ATP≤0 (não atende).")
    else:
        st.info("Não foi possível montar o ATP com os dados atuais.")


