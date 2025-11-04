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
# TAB 3 — MPS & Custos (somente leitura dos inputs)
# ======================================================
with tabs[2]:
    st.subheader("MPS — Custos e Resumo (somente leitura dos inputs)")

    # ---------- Guardas ----------
    mps_inputs = st.session_state.get("mps_inputs", {})
    if not isinstance(mps_inputs, dict) or not mps_inputs:
        st.info("Não encontrei os **inputs do MPS**. Preencha e salve na página **05_Inputs_MPS**.")
        st.page_link("pages/05_Inputs_MPS.py", label="⚙️ Ir para 05_Inputs_MPS (preencher custos)")
        st.stop()

    def _get_df_from_state(keys: list[str]):
        for k in keys:
            obj = st.session_state.get(k, None)
            if isinstance(obj, pd.DataFrame) and not obj.empty:
                return obj
        return None

    mps_tbl_display = _get_df_from_state(["mps_tbl_display", "mps_table"])
    if mps_tbl_display is None:
        st.info("Não há tabela do MPS na memória. Gere o MPS na página **06_MPS** e volte.")
        st.page_link("pages/06_MPS.py", label="📅 Ir para 06_MPS (Plano Mestre de Produção)")
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

    # ---------- Linhas usadas ----------
    row_qtde_mps  = _find_row(mps_tbl_display, ["Qtde. MPS", "qtde mps", "quantidade mps", "mps qty"])
    row_estoque   = _find_row(mps_tbl_display, ["Estoque Proj.", "estoque projetado", "estoque"])
    row_ruptura   = _find_row(mps_tbl_display, ["Ruptura", "falta", "backlog", "não atendido"])

    total_prod = float(np.nansum(row_qtde_mps.values.astype(float))) if row_qtde_mps is not None else 0.0
    order_count = int(np.nansum((row_qtde_mps.values.astype(float) > 0).astype(int))) if row_qtde_mps is not None else 0
    estoque_mes = row_estoque.values.astype(float) if row_estoque is not None else np.zeros(len(mps_tbl_display.columns))
    total_estoque = float(np.nansum(np.clip(estoque_mes, 0, None)))
    total_ruptura = float(np.nansum(np.clip(row_ruptura.values.astype(float), 0, None))) if row_ruptura is not None else 0.0
    estoque_final = float(estoque_mes[-1]) if estoque_mes.size else np.nan

    # ---------- Inputs de custo (robustos) ----------
    v = _as_float(mps_inputs.get("v", None), _as_float(mps_inputs.get("unit_cost", 0.0), 0.0))
    A = _as_float(mps_inputs.get("A", None), _as_float(mps_inputs.get("order_cost", 0.0), 0.0))
    time_base = mps_inputs.get("time_base", "por mês")  # "por mês" ou "por ano"
    h_mode = mps_inputs.get("h_mode", "Informar H diretamente")
    H_direct = mps_inputs.get("H", None)
    r = mps_inputs.get("r", None)
    pi_shortage = _as_float(mps_inputs.get("pi_shortage", None), _as_float(mps_inputs.get("shortage_cost", 0.0), 0.0))

    # D e p (apenas para referência visual no topo; se não houver, mostramos '—')
    D_month = _as_float(mps_inputs.get("D_month", mps_inputs.get("D", None)), 0.0)
    p_month = _as_float(mps_inputs.get("p_month", mps_inputs.get("p", None)), 0.0)
    if mps_inputs.get("time_base", "por mês") == "por ano":
        # se o usuário cadastrou por ano, convertemos para mostrar a taxa mensal
        D_month = D_month / 12.0 if D_month else 0.0
        p_month = p_month / 12.0 if p_month else 0.0

    # Converte H para base mensal (para custos)
    if h_mode == "Informar H diretamente":
        H_mensal = _as_float(H_direct, 0.0)
        if time_base == "por ano":
            H_mensal = H_mensal / 12.0
        r_show = _as_float(r, 0.0)  # apenas para exibir no topo se houver
    else:
        r_val = _as_float(r, 0.0)
        H_calc = r_val * v
        H_mensal = H_calc if time_base == "por mês" else (H_calc / 12.0)
        r_show = r_val

    # ---------- Cabeçalho: variáveis usadas (lembrança) ----------
    with st.expander("ℹ️ Variáveis & entradas usadas (vindas do 05_Inputs_MPS)"):
        a1, a2, a3, a4 = st.columns(4)
        a1.metric("A (setup/encomenda)", _safe(A, 2))
        a2.metric("v (valor unit.)", _safe(v, 2))
        a3.metric("H_mensal (R$/un·mês)", _safe(H_mensal, 4))
        a4.metric("π (custo ruptura)", _safe(pi_shortage, 2))
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("D (média mensal)", _safe(D_month, 2))
        b2.metric("p (produção mensal)", _safe(p_month, 2))
        b3.metric("r (taxa man.)", _safe(r_show, 4))
        b4.metric("Base informada", time_base)

        st.caption(
            "Convenções (EPQ): **A**=custo fixo por setup/encomenda; **D**=taxa de demanda; **p**=taxa de produção; "
            "**v**=valor unitário; **H**=custo de manter por unidade e por período (aqui exibido em base **mensal**); "
            "**r**=taxa de manutenção (quando utilizado para calcular H via **H = r·v**); **π**=custo de ruptura."
        )

    # ---------- Cálculo dos custos ----------
    cost_produzir   = total_prod * v
    cost_encomendar = order_count * A
    cost_manter     = total_estoque * H_mensal
    cost_ruptura    = total_ruptura * pi_shortage
    cost_total      = cost_produzir + cost_encomendar + cost_manter + cost_ruptura

    # ---------- Layout (sem tabela) ----------
    st.markdown("### KPIs principais")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Estoque final (último mês)", _safe(estoque_final, 0))
    k2.metric("Total produzido (Σ Qtde. MPS)", _safe(total_prod, 0))
    k3.metric("Meses com pedido (contagem)", f"{order_count:d}")
    k4.metric("Ruptura (Σ unidades)", _safe(total_ruptura, 0))

    st.markdown("### Decomposição de custos")
    cA, cB = st.columns(2)
    with cA:
        st.metric("Custo de produzir (R$)", _safe(cost_produzir, 2))
        st.caption(f"Fórmula: **Σ(Qtde. MPS) × v** → { _safe(total_prod,0) } × { _safe(v,2) }")
        st.metric("Custo de manter (R$)", _safe(cost_manter, 2))
        st.caption(f"Fórmula: **Σ(Estoque_mês) × H_mensal** → { _safe(total_estoque,0) } × { _safe(H_mensal,4) }")
    with cB:
        st.metric("Custo de encomendar (R$)", _safe(cost_encomendar, 2))
        st.caption(f"Fórmula: **(#meses com pedido) × A** → { order_count } × { _safe(A,2) }")
        st.metric("Custo de ruptura (R$)", _safe(cost_ruptura, 2))
        st.caption(f"Fórmula: **Σ(Ruptura) × π** → { _safe(total_ruptura,0) } × { _safe(pi_shortage,2) }")

    st.markdown("#### Custo relevante total")
    st.metric("Total (R$)", _safe(cost_total, 2))
    st.caption(
        "Observação: se **H** foi informado por **ano**, convertemos para mês dividindo por 12. "
        "Se você optou por **r** e **v**, usamos **H = r·v** (e aplicamos a mesma conversão de base quando necessário)."
    )

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
