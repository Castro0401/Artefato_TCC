# pages/07_Dashboard_Conclusao.py
from __future__ import annotations
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Conclusão", page_icon="✅", layout="wide")
st.title("✅ 07 — Conclusão (Painel de Decisão)")

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

    cL, cR = st.columns(2)
    with cL:
        st.page_link("pages/05_Inputs_MPS.py", label="⬅️ Voltar: Inputs do MPS", icon="⚙️")
    with cR:
        st.page_link("pages/04_Previsao.py", label="🛠️ Ajustar Previsão", icon="🧪")

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
# TAB 3 — MPS & KPIs (decisão + custos + ATP)
# ======================================================
with tabs[2]:
    st.subheader("KPIs do MPS")

    if not isinstance(mps_tbl_display, pd.DataFrame) or mps_tbl_display.empty:
        st.info("Não há tabela do MPS na memória. Gere o MPS na página **06_MPS** e volte.")
        st.page_link("pages/06_MPS.py", label="📅 Ir para 06_MPS (Plano Mestre de Produção)")
    else:
        # -------------------------------
        # Seleção de colunas (apenas datas)
        # -------------------------------
        date_cols, date_labels = [], []
        for c in mps_tbl_display.columns:
            ts = _to_ts(c)
            if not pd.isna(ts):
                date_cols.append(c)
                date_labels.append(ts.strftime("%b/%y").title().replace(".", ""))  # Set/25
        idx_map = dict(zip(date_cols, date_labels))

        def _row(name):
            if name in mps_tbl_display.index:
                s = mps_tbl_display.loc[name, date_cols]
                return pd.Series(pd.to_numeric(s, errors="coerce").astype(float).values,
                                 index=pd.to_datetime([_to_ts(c) for c in date_cols]))
            return None

        q_mps   = _row("Qtde. MPS")
        estoque = _row("Estoque Proj.")
        atp     = _row("ATP")  # se existir direto
        atp_cum = _row("ATP(cum)")

        if atp is None and atp_cum is not None:
            # ATP por período = diferença do acumulado
            atp = atp_cum.diff().fillna(atp_cum)

        # -------------------------------
        # Parâmetros econômicos (com defaults)
        # Se preferir mover para a página 05_Inputs_MPS, basta ler de st.session_state.
        # -------------------------------
        with st.expander("⚙️ Parâmetros econômicos (edite se necessário)", expanded=False):
            colA, colB, colC, colD = st.columns(4)
            unit_cost = colA.number_input("Custo unitário de produção/compra (R$)", min_value=0.0, value=float(st.session_state.get("mps_unit_cost", 1.0)), step=0.1)
            hold_rate = colB.number_input("Custo de manter estoque (% ao mês)", min_value=0.0, value=float(st.session_state.get("mps_hold_rate", 2.0)), step=0.1)
            hold_abs  = colC.number_input("OU custo de estoque (R$/unid·mês)", min_value=0.0, value=float(st.session_state.get("mps_hold_abs", 0.0)), step=0.1, help="Se > 0, ignora o percentual.")
            stockout_c = colD.number_input("Custo de falta (R$/unid)", min_value=0.0, value=float(st.session_state.get("mps_stockout", 10.0)), step=0.5)
            setup_c = st.number_input("Custo de setup (R$/lote de MPS)", min_value=0.0, value=float(st.session_state.get("mps_setup", 0.0)), step=1.0, help="Multiplica pelo nº de períodos com produção > 0.")

        # Guarda (opcional)
        st.session_state.update(dict(mps_unit_cost=unit_cost, mps_hold_rate=hold_rate,
                                     mps_hold_abs=hold_abs, mps_stockout=stockout_c, mps_setup=setup_c))

        # -------------------------------
        # KPIs de custo
        # -------------------------------
        # Produção: custo simples por unidade produzida
        prod_cost = np.nan
        if q_mps is not None:
            prod_cost = float(np.nansum(np.maximum(q_mps.values, 0)) * unit_cost)

        # Manutenção de estoque: soma estoque projetado positivo * custo por unidade·mês
        hold_cost = np.nan
        if estoque is not None:
            per_unit_hold = (hold_abs if hold_abs > 0 else (unit_cost * (hold_rate/100.0)))
            hold_cost = float(np.nansum(np.maximum(estoque.values, 0)) * per_unit_hold)

        # Falta: estoque negativo acumulado convertido em unidades em falta
        stockout_cost = np.nan
        if estoque is not None:
            faltas_unid = float(np.nansum(np.abs(np.minimum(estoque.values, 0))))
            stockout_cost = faltas_unid * stockout_c

        # Setup: nº de períodos com produção > 0
        setup_cost = np.nan
        if q_mps is not None:
            setups = int(np.nansum((q_mps.values > 0).astype(int)))
            setup_cost = setups * setup_c

        # Total relevante
        total_cost = np.nansum([x for x in [prod_cost, hold_cost, stockout_cost, setup_cost] if np.isfinite(x)])

        # KPIs de nível (sem "total planejado", conforme pedido)
        estoque_final = int(estoque.iloc[-1]) if estoque is not None and len(estoque) else np.nan
        rupturas = int(np.nansum((estoque.values < 0).astype(int))) if estoque is not None else np.nan

        st.markdown("### Resumo econômico e operacional")
        k1, k2, k3, k4, k5 = st.columns(5)
        _kpi("Custo de produção (R$)", _safe_num(prod_cost, 0), "∑ Qtde. MPS × custo unitário")
        _kpi("Custo de estoque (R$)", _safe_num(hold_cost, 0), "∑ estoque+ × custo por unid·mês")
        _kpi("Custo de falta (R$)", _safe_num(stockout_cost, 0), "Unidades em falta × custo de falta")
        _kpi("Custo de setup (R$)", _safe_num(setup_cost, 0), "Períodos com produção × custo de setup")
        _kpi("Custo relevante total (R$)", _safe_num(total_cost, 0), "Soma dos custos acima")

        k6, k7 = st.columns(2)
        _kpi("Estoque projetado no fim", _safe_num(estoque_final, 0))
        _kpi("Nº de períodos com ruptura", _safe_num(rupturas, 0))

        st.caption("Dica: se preferir, movemos esses parâmetros para a página **Inputs do MPS** e os tornamos persistentes por produto.")

        # -------------------------------
        # Explorador de ATP (atendimento de demandas extras)
        # -------------------------------
        st.markdown("### Explorador de ATP — atendimento de demandas extras")
        if atp is None:
            st.info("Não encontrei a linha **ATP** (ou **ATP(cum)**) no MPS para calcular a folga mensal.")
        else:
            # Entrada: demanda extra fixa por mês
            extra = st.slider("Demanda extra (unidades por mês)", min_value=0, max_value=int(max(100, np.nanmax(atp.values))), value=0, step=1)

            df_atp = pd.DataFrame({
                "ds": atp.index,
                "ATP": atp.values,
                "Atende_extra?": (atp.values >= extra) if extra > 0 else np.ones_like(atp.values, dtype=bool),
            })
            df_atp["Sobra"]   = np.where(df_atp["ATP"] - extra >= 0, df_atp["ATP"] - extra, 0)
            df_atp["Déficit"] = np.where(df_atp["ATP"] - extra < 0,  -(df_atp["ATP"] - extra), 0)

            # KPIs do explorador
            col_a, col_b, col_c = st.columns(3)
            _kpi("Meses que atendem 100%", _safe_num(int(df_atp["Atende_extra?"].sum()), 0))
            _kpi("Sobra total (unid)", _safe_num(float(df_atp["Sobra"].sum()), 0))
            _kpi("Déficit total (unid)", _safe_num(float(df_atp["Déficit"].sum()), 0))

            # Gráfico (barras azul-escuro)
            import altair as alt
            ch_atp = (
                alt.Chart(df_atp)
                .mark_bar(color="#1e3a8a")
                .encode(
                    x=alt.X("ds:T", title="Mês"),
                    y=alt.Y("ATP:Q", title="ATP (unidades)"),
                    tooltip=[
                        alt.Tooltip("ds:T", title="Período"),
                        alt.Tooltip("ATP:Q", title="ATP", format=",.0f"),
                        alt.Tooltip("Sobra:Q", format=",.0f"),
                        alt.Tooltip("Déficit:Q", format=",.0f"),
                        alt.Tooltip("Atende_extra?:N", title="Atende extra?")
                    ]
                )
                .properties(height=260, width="container")
                .interactive()
            )
            st.altair_chart(ch_atp, use_container_width=True)

            # Tabela compacta (opcional)
            with st.expander("Ver detalhes por mês", expanded=False):
                show = df_atp.copy()
                show["Mês"] = show["ds"].dt.strftime("%b/%y").str.title()
                st.dataframe(
                    show[["Mês","ATP","Sobra","Déficit","Atende_extra?"]]
                    .rename(columns={"Atende_extra?":"Atende?"}),
                    use_container_width=True, height=240
                )

    st.divider()


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
