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

    # ----------------- pega campeão + métricas -----------------
    champion = {}
    if res is not None and hasattr(res, "attrs"):
        champion = res.attrs.get("champion", {}) or {}

    # KPIs do campeão (se existirem)
    c1, c2, c3, c4 = st.columns(4)
    _kpi("MAE",        _safe_num(champion.get("MAE")),        "Erro Médio Absoluto", key="mae")
    _kpi("sMAPE (%)",  _safe_num(champion.get("sMAPE")),      "Erro percentual simétrico", key="smape")
    _kpi("RMSE",       _safe_num(champion.get("RMSE")),       "Raiz do erro quadrático médio", key="rmse")
    _kpi("MAPE (%)",   _safe_num(champion.get("MAPE")),       "Erro percentual médio", key="mape")

    st.markdown("---")

    # ----------------- gráfico Real x Previsão -----------------
    # histórico: da memória do upload
    hist = None
    if isinstance(hist_df_norm, pd.DataFrame) and {"ds","y"}.issubset(hist_df_norm.columns):
        hist = hist_df_norm.copy()
        hist["ds"] = hist["ds"].apply(_to_ts)
        hist = hist.dropna(subset=["ds"]).rename(columns={"y":"Real"})

    # previsão: da memória salva na 04
    prev = None
    if isinstance(fcst_df, pd.DataFrame) and {"ds","y"}.issubset(fcst_df.columns):
        prev = fcst_df.copy()
        prev["ds"] = prev["ds"].apply(_to_ts)
        prev = prev.dropna(subset=["ds"]).rename(columns={"y":"Previsão"})

    if hist is None:
        st.info("Sem histórico em memória. Gere o upload na página **01_Upload**.")
    else:
        # monta long para plot
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

    st.markdown("—")
    # Download dos experimentos (não exibir tabela gigante aqui)
    if isinstance(exp_df, pd.DataFrame) and len(exp_df) > 0:
        st.download_button(
            "⬇️ Baixar todos os experimentos (CSV)",
            data=exp_df.to_csv(index=False).encode("utf-8"),
            file_name="experimentos_previsao.csv",
            mime="text/csv",
            help="CSV com todas as combinações testadas, métricas e parâmetros."
        )
    else:
        st.caption("Sem tabela de experimentos em memória. Gere na página de **Previsão** e volte.")

    st.divider()
    cL, cR = st.columns(2)
    with cL:
        st.page_link("pages/05_Inputs_MPS.py", label="⬅️ Voltar: Inputs do MPS", icon="⚙️")
    with cR:
        st.page_link("pages/04_Previsao.py", label="🛠️ Ajustar Previsão", icon="🧪")

# ======================================================
# TAB 2 — VIESES (com fallback)
# ======================================================
with tabs[1]:
    st.subheader("Diagnóstico de vieses da previsão")

    # Tentamos montar uma base com y_true x y_pred.
    # 1) Se o pipeline guardou 'backtest' em attrs:
    bt = None
    if res is not None and hasattr(res, "attrs"):
        # procura formatos comuns
        for k in ["backtest", "oos_eval", "cv_last", "val_df", "fitted_df"]:
            obj = res.attrs.get(k)
            if isinstance(obj, pd.DataFrame) and {"ds","y_true","y_pred"}.issubset(obj.columns):
                bt = obj[["ds","y_true","y_pred"]].copy()
                bt["ds"] = pd.to_datetime(bt["ds"])
                break

    # 2) Caso não tenha backtest, não dá pra avaliar viés de maneira honesta.
    if bt is None:
        st.info(
            "Não encontrei um **backtest** com `y_true` e `y_pred` no resultado da previsão. "
            "Sem esses dados não é possível calcular vieses históricos. "
            "Se quiser, podemos adicionar cross-validation ao pipeline para habilitar essa aba."
        )
    else:
        bt = bt.sort_values("ds")
        bt["erro"] = bt["y_pred"] - bt["y_true"]
        bias_abs = float(bt["erro"].mean()) if bt["erro"].notna().any() else np.nan
        pct = np.where(bt["y_true"] != 0, bt["erro"] / bt["y_true"], np.nan)
        bias_pct = float(np.nanmean(pct)) * 100.0

        c1, c2 = st.columns(2)
        _kpi("Viés (nível)", _safe_num(bias_abs), "média de (previsto − real)")
        _kpi("Viés (%)", _safe_num(bias_pct), "média de (previsto − real)/real × 100")

        st.caption(
            "Interpretação: valores **positivos** indicam **superestimação**; negativos, **subestimação**. "
            "Quanto mais próximo de 0, menor o viés."
        )

        # Curva dos erros
        import altair as alt
        ch = (
            alt.Chart(bt[["ds","erro"]])
            .mark_line(color="#525252")
            .encode(x="ds:T", y="erro:Q", tooltip=["ds:T", alt.Tooltip("erro:Q", format=",.2f")])
            .properties(height=280, width="container")
            .interactive()
        )
        st.altair_chart(ch, use_container_width=True)

# ======================================================
# TAB 3 — MPS & KPIs (robusto)
# ======================================================
with tabs[2]:
    st.subheader("KPIs do MPS")

    if not isinstance(mps_tbl_display, pd.DataFrame) or mps_tbl_display.empty:
        st.info("Não há tabela do MPS na memória. Gere o MPS na página **06_MPS** e volte.")
        st.page_link("pages/06_MPS.py", label="📅 Ir para 06_MPS (Plano Mestre de Produção)")
    else:
        # formata cabeçalhos de datas para Mês/Ano
        new_cols = []
        for c in mps_tbl_display.columns:
            ts = _to_ts(c)
            if pd.isna(ts):
                new_cols.append(str(c))
            else:
                new_cols.append(ts.strftime("%b/%y").title().replace(".", ""))  # Set/25 etc.
        mps_show = mps_tbl_display.copy()
        mps_show.columns = new_cols

        # exibe
        st.dataframe(mps_show, use_container_width=True, height=320)

        # KPIs simples (exemplo)
        try:
            estoque_final = int(mps_tbl_display.loc["Estoque Proj.", mps_tbl_display.columns[-1]])
        except Exception:
            estoque_final = np.nan
        try:
            tot_receb = int(mps_tbl_display.loc["Qtde. MPS"].sum())
        except Exception:
            tot_receb = np.nan
        try:
            atp_ultimo = int(mps_tbl_display.loc["ATP(cum)"].iloc[-1])
        except Exception:
            atp_ultimo = np.nan

        st.markdown("### Resumo")
        k1, k2, k3 = st.columns(3)
        _kpi("Estoque Projetado (final do horizonte)", _safe_num(estoque_final, 0))
        _kpi("Total planejado (Qtde. MPS)", _safe_num(tot_receb, 0))
        _kpi("ATP acumulado (último período)", _safe_num(atp_ultimo, 0))

        st.caption("KPIs adicionais (cobertura, OTIF simulado, rupturas projetadas etc.) podem ser incluídos conforme sua regra de negócio.")

    st.divider()
    cL, cR = st.columns(2)
    with cL:
        st.page_link("pages/05_Inputs_MPS.py", label="⬅️ Voltar: Inputs do MPS", icon="⚙️")
    with cR:
        st.page_link("pages/04_Previsao.py", label="🛠️ Ajustar Previsão", icon="🧪")

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
