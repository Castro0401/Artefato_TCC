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


# =========================
# TAB 3 — MPS (custos & resumo, sem tabela)
# =========================
with tabs[2]:
    import numpy as np
    import pandas as pd

    st.subheader("MPS — Custos e Resumo")

    # ---------- helpers ----------
    def _fmt(v, nd=0):
        try:
            if np.isnan(v): return "—"
        except Exception:
            pass
        try:
            return f"{float(v):,.{nd}f}".replace(",", "X").replace(".", ",").replace("X", ".")
        except Exception:
            return str(v)

    def _num(dct, key, fb=0.0):
        v = dct.get(key, None)
        if v is None: return float(fb)
        try:
            if isinstance(v, (np.floating, np.integer)): return float(v)
            if hasattr(v, "item"): return float(v.item())
            if isinstance(v, str):
                s = v.strip().replace(".", "").replace(",", ".")
                return float(s) if s else float(fb)
            return float(v)
        except Exception:
            return float(fb)

    def _find_row(df: pd.DataFrame, names: list[str]):
        idx = df.index.astype(str).str.strip().str.lower()
        for cand in names:
            c = str(cand).strip().lower()
            eq = (idx == c)
            if eq.any(): return df.iloc[int(np.where(eq)[0][0])]
            ct = idx.str.contains(c, regex=False)
            if ct.any(): return df.iloc[int(np.where(ct)[0][0])]
        return None

    # ---------- entrada necessária ----------
    mps_inputs = st.session_state.get("mps_inputs", {})
    if not isinstance(mps_inputs, dict) or not mps_inputs:
        st.info("Preencha e salve os **inputs de custos** na página 05_Inputs_MPS.")
        st.page_link("pages/05_Inputs_MPS.py", label="⚙️ Ir para 05_Inputs_MPS")
        st.stop()

    mps_tbl_display = st.session_state.get("mps_tbl_display") or st.session_state.get("mps_table")
    if not isinstance(mps_tbl_display, pd.DataFrame) or mps_tbl_display.empty:
        st.info("Gere o **MPS** na página 06_MPS e volte aqui.")
        st.page_link("pages/06_MPS.py", label="📅 Ir para 06_MPS (Plano Mestre de Produção)")
        st.stop()

    # ---------- custos (EPQ) ----------
    time_base = mps_inputs.get("time_base", "por mês")  # "por mês" | "por ano"

    # v (custo unitário) com fallback legado
    v = _num(mps_inputs, "v", _num(mps_inputs, "unit_cost", 0.0))
    A = _num(mps_inputs, "A", _num(mps_inputs, "order_cost", 0.0))                 # custo por setup/pedido
    pi = _num(mps_inputs, "pi_shortage", _num(mps_inputs, "shortage_cost", 0.0))   # custo de ruptura por unid

    # H mensal: direto ou r·v; se anual, converte p/ mês
    H_direct = mps_inputs.get("H", None)
    r_val = mps_inputs.get("r", None)
    if H_direct is not None:
        H = _num(mps_inputs, "H", 0.0)
        H_m = H if time_base == "por mês" else (H / 12.0)
        origem_H = "H informado"
    elif r_val is not None:
        r = _num(mps_inputs, "r", 0.0)
        r_m = r if time_base == "por mês" else (r / 12.0)
        H_m = r_m * v
        origem_H = "H = r · v"
    else:
        # compatibilidade com “holding_rate” (%/mês)
        holding_rate_pct = _num(mps_inputs, "holding_rate", 0.0)
        H_m = (holding_rate_pct / 100.0) * v
        origem_H = "H = (holding_rate%/mês) · v"

    # ---------- extrai linhas-chave do MPS ----------
    row_receipts = _find_row(mps_tbl_display, ["Qtde. MPS", "Qtde MPS", "Quantidade MPS", "planned order receipts"])
    row_releases = _find_row(mps_tbl_display, ["Início MPS", "Releases", "planned order releases"])
    row_stock    = _find_row(mps_tbl_display, ["Estoque Proj.", "Estoque Projetado", "projected", "on hand"])
    row_shortage = _find_row(mps_tbl_display, ["Ruptura", "Falta", "Backlog", "Não atendido", "shortage"])

    # produção total: preferir receipts; se não houver, usar releases
    if row_receipts is not None:
        prod_series = row_receipts.values.astype(float)
    elif row_releases is not None:
        prod_series = row_releases.values.astype(float)
    else:
        prod_series = np.zeros(len(mps_tbl_display.columns))

    total_prod   = float(np.nansum(np.clip(prod_series, 0, None)))
    order_count  = int(np.nansum((prod_series > 0).astype(int)))
    stock_series = row_stock.values.astype(float) if row_stock is not None else np.zeros(len(mps_tbl_display.columns))
    total_stock  = float(np.nansum(np.clip(stock_series, 0, None)))
    estoque_fin  = float(stock_series[-1]) if stock_series.size else np.nan
    total_short  = float(np.nansum(np.clip(row_shortage.values.astype(float), 0, None))) if row_shortage is not None else 0.0

    # ---------- custos ----------
    custo_produzir   = total_prod * v
    custo_encomendar = order_count * A
    custo_manter     = total_stock * H_m
    custo_ruptura    = total_short * pi
    custo_total      = custo_produzir + custo_encomendar + custo_manter + custo_ruptura

    # ---------- avisos úteis ----------
    if v <= 0:
        st.warning("**Custo unitário (v)** está 0. Defina em 05_Inputs_MPS (seção EPQ) para calcular corretamente o **Custo de produzir**.")
    if total_prod <= 0:
        st.info("A produção total está 0 neste horizonte (não encontrei 'Qtde. MPS' nem 'Início MPS' com valores positivos).")

    # ---------- layout: cards 2×2 + total ----------
    st.markdown("""
    <style>
    .kcards{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px;margin-top:6px;}
    .kcard{border:1px solid #e5e7eb;border-radius:14px;padding:14px 16px;background:#fafafa;}
    .kcard h4{margin:0 0 6px 0;font-size:0.95rem;color:#374151;}
    .kval{font-size:1.6rem;font-weight:700;line-height:1.1;margin-bottom:4px;}
    .kdesc{color:#6b7280;font-size:0.85rem;}
    .ktotal{border:2px solid #111827;background:#111827;color:#fff;border-radius:16px;padding:16px 18px;margin-top:10px;}
    .ktotal .kval{font-size:1.8rem;}
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="kcards">', unsafe_allow_html=True)
    st.markdown(f'''
        <div class="kcard">
            <h4>Custo de produzir</h4>
            <div class="kval">R$ {_fmt(custo_produzir, 2)}</div>
            <div class="kdesc">Σ(quantidade produzida) × <b>v</b></div>
        </div>
    ''', unsafe_allow_html=True)
    st.markdown(f'''
        <div class="kcard">
            <h4>Custo de encomendar</h4>
            <div class="kval">R$ {_fmt(custo_encomendar, 2)}</div>
            <div class="kdesc">nº de meses com pedido × <b>A</b></div>
        </div>
    ''', unsafe_allow_html=True)
    st.markdown(f'''
        <div class="kcard">
            <h4>Custo de manter</h4>
            <div class="kval">R$ {_fmt(custo_manter, 2)}</div>
            <div class="kdesc">Σ(estoque mensal) × <b>H</b> (mensal)</div>
        </div>
    ''', unsafe_allow_html=True)
    st.markdown(f'''
        <div class="kcard">
            <h4>Custo de ruptura</h4>
            <div class="kval">R$ {_fmt(custo_ruptura, 2)}</div>
            <div class="kdesc">Σ(unid. não atendidas) × <b>π</b></div>
        </div>
    ''', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(f'''
        <div class="ktotal">
            <h4 style="margin:0 0 6px 0;">Custo relevante total</h4>
            <div class="kval">R$ {_fmt(custo_total, 2)}</div>
            <div class="kdesc">Produzir + Encomendar + Manter + Ruptura</div>
        </div>
    ''', unsafe_allow_html=True)

    # ---------- KPIs de apoio ----------
    st.markdown("### Indicadores do horizonte")
    k1, k2, k3 = st.columns(3)
    k1.metric("Total produzido", _fmt(total_prod, 0))
    k2.metric("Meses com pedido", f"{order_count:d}")
    k3.metric("Estoque final", _fmt(estoque_fin, 0))

    # ---------- como calculamos ----------
    st.markdown("### Como calculamos")
    bullets = [
        f"**Custo de produzir** = Σ(Qtde. produzida) × **v** = {_fmt(total_prod,0)} × R$ {_fmt(v,2)} = **R$ {_fmt(custo_produzir,2)}**.",
        f"**Custo de encomendar** = nº de meses com pedido × **A** = {order_count} × R$ {_fmt(A,2)} = **R$ {_fmt(custo_encomendar,2)}**.",
        f"**Custo de manter** = Σ(estoque mensal) × **H** (mensal). **H (mensal)** veio de **{origem_H}**; H(mês) = R$ {_fmt(H_m,2)}.",
        f"**Custo de ruptura** = Σ(rupturas) × **π** = {_fmt(total_short,0)} × R$ {_fmt(pi,2)} = **R$ {_fmt(custo_ruptura,2)}**.",
    ]
    st.markdown("- " + "\n- ".join(bullets))



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
