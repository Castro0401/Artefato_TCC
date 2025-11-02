# pages/07_Dashboard_Conclusao.py
from __future__ import annotations

import pandas as pd
import numpy as np
import streamlit as st
import altair as alt

st.set_page_config(page_title="Conclusão", page_icon="✅", layout="wide")
st.title("✅ 07 — Conclusão (Painel de Decisão)")

# =============================================================================
# Helpers
# =============================================================================
_PT = {"Jan":1,"Fev":2,"Mar":3,"Abr":4,"Mai":5,"Jun":6,"Jul":7,"Ago":8,"Set":9,"Out":10,"Nov":11,"Dez":12}
_REV_PT = {v:k for k, v in _PT.items()}

def to_month_begin(v) -> pd.Timestamp | pd.NaT:
    """Converte rótulos 'Set/25' ou strings de data para Timestamp no 1º dia do mês."""
    if isinstance(v, pd.Timestamp):
        return pd.Timestamp(year=v.year, month=v.month, day=1)
    s = str(v)
    if "/" in s and len(s) <= 6:
        try:
            mon, yy = s.split("/")
            return pd.Timestamp(year=2000 + int(yy), month=_PT[mon], day=1)
        except Exception:
            return pd.NaT
    try:
        dt = pd.to_datetime(s, errors="coerce")
        if pd.isna(dt):
            return pd.NaT
        return pd.Timestamp(year=dt.year, month=dt.month, day=1)
    except Exception:
        return pd.NaT

def fmt_month_label(ts: pd.Timestamp) -> str:
    """Formata Timestamp -> 'Set/25' (mês/ano 2 dígitos)."""
    return f"{_REV_PT.get(int(ts.month), '???')}/{str(ts.year)[-2:]}"

def _get_exp_df() -> pd.DataFrame | None:
    """Busca a tabela de experimentos com segurança (sem usar 'or' com DataFrame)."""
    for key in ("experiments_df", "experiments_table", "pipeline_experiments"):
        obj = st.session_state.get(key)
        if isinstance(obj, pd.DataFrame) and len(obj) > 0:
            # normaliza algumas colunas comuns (se existirem)
            df = obj.copy()
            if "sMAPE" in df.columns:
                # assegura tipo numérico
                df["sMAPE"] = pd.to_numeric(df["sMAPE"], errors="coerce")
            if "MAE" in df.columns:
                df["MAE"] = pd.to_numeric(df["MAE"], errors="coerce")
            if "RMSE" in df.columns:
                df["RMSE"] = pd.to_numeric(df["RMSE"], errors="coerce")
            return df
    return None

def _get_real_series() -> pd.DataFrame | None:
    """Traz série real da sessão e padroniza a coluna 'ds' para datetime."""
    if "ts_df_monthly" in st.session_state and isinstance(st.session_state["ts_df_monthly"], pd.DataFrame):
        df = st.session_state["ts_df_monthly"].copy()
        if "ds" in df.columns and "y" in df.columns:
            df["ds"] = pd.to_datetime(df["ds"], errors="coerce").map(to_month_begin)
            return df[["ds", "y"]].dropna()
    if "ts_df_norm" in st.session_state and isinstance(st.session_state["ts_df_norm"], pd.DataFrame):
        df = st.session_state["ts_df_norm"].copy()
        if "ds" in df.columns and "y" in df.columns:
            df["ds"] = df["ds"].map(to_month_begin)
            return df[["ds", "y"]].dropna()
    return None

def _get_forecast_df() -> pd.DataFrame | None:
    """Traz previsão salva p/ MPS (ds,y) e padroniza 'ds' para datetime."""
    f = st.session_state.get("forecast_df")
    if isinstance(f, pd.DataFrame) and {"ds","y"}.issubset(f.columns):
        out = f.copy()
        out["ds"] = pd.to_datetime(out["ds"], errors="coerce").map(to_month_begin)
        return out.dropna()
    # fallback: tentar extrair do last_result.attrs
    res = st.session_state.get("last_result")
    if res is not None and hasattr(res, "attrs"):
        for key in ("forecast","forecast_df","yhat","pred","prediction"):
            if key in res.attrs:
                obj = res.attrs[key]
                if isinstance(obj, pd.DataFrame) and {"ds","yhat"}.issubset(obj.columns):
                    out = obj.rename(columns={"yhat":"y"})[["ds","y"]].copy()
                    out["ds"] = pd.to_datetime(out["ds"], errors="coerce").map(to_month_begin)
                    return out.dropna()
                if isinstance(obj, pd.Series):
                    out = pd.DataFrame({"ds": obj.index, "y": obj.values})
                    out["ds"] = pd.to_datetime(out["ds"], errors="coerce").map(to_month_begin)
                    return out.dropna()
    return None

def kpi_card(label: str, value: str, small: bool = False):
    st.markdown(
        f"""
        <div style="display:flex;flex-direction:column;gap:2px;">
          <small style="color:#6b7280;font-size:0.85rem;">{label}</small>
          <div style="font-size:{'1.2rem' if small else '1.6rem'};font-weight:600;line-height:1.1;">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# =============================================================================
# Tabs
# =============================================================================
tab_acc, tab_mps, tab_rec = st.tabs(["Acurácia", "MPS & KPIs", "Recomendações"])

# =============================================================================
# 1) Acurácia
# =============================================================================
with tab_acc:
    st.subheader("Desempenho dos modelos de previsão")

    # Experimentos (vêm da página 04)
    exp_df = _get_exp_df()
    if exp_df is None:
        st.info("Sem tabela de experimentos em memória. Gere na página de **Previsão** e volte aqui.")
        st.page_link("pages/04_Previsao.py", label="🧙 Ir para 04_Previsao")
    else:
        # KPIs resumidos do experimento vencedor (se houver indicação na sessão)
        champ = {}
        res = st.session_state.get("last_result")
        if res is not None and hasattr(res, "attrs"):
            champ = res.attrs.get("champion", {}) or {}

        c1, c2, c3, c4 = st.columns(4)
        def _fmt(x):
            try:
                return f"{float(x):.4g}"
            except Exception:
                return "—"
        c1.kpi = kpi_card("MAE", _fmt(champ.get("MAE")))
        c2.kpi = kpi_card("sMAPE (%)", _fmt(champ.get("sMAPE")))
        c3.kpi = kpi_card("RMSE", _fmt(champ.get("RMSE")))
        c4.kpi = kpi_card("MAPE (%)", _fmt(champ.get("MAPE")))

        # Gráfico Real x Previsão
        real = _get_real_series()
        prev = _get_forecast_df()
        if (real is None) or (prev is None) or real.empty or prev.empty:
            st.info("Não foi possível exibir o gráfico Real × Previsão: dados ausentes ou inválidos.")
        else:
            df_long = pd.concat(
                [
                    pd.DataFrame({"ds": real["ds"], "valor": real["y"], "série": "Real"}),
                    pd.DataFrame({"ds": prev["ds"], "valor": prev["y"], "série": "Previsão"}),
                ],
                ignore_index=True,
            )
            chart = (
                alt.Chart(df_long.reset_index(drop=True))
                .mark_line()
                .encode(
                    x=alt.X("ds:T", title="Mês"),
                    y=alt.Y("valor:Q", title="Quantidade"),
                    color=alt.Color(
                        "série:N",
                        scale=alt.Scale(domain=["Real", "Previsão"], range=["#1e3a8a", "#60a5fa"]),
                        legend=alt.Legend(title=None, orient="top"),
                    ),
                    tooltip=[
                        alt.Tooltip("ds:T", title="Período"),
                        alt.Tooltip("série:N", title="Série"),
                        alt.Tooltip("valor:Q", title="Valor", format=",.0f"),
                    ],
                )
                .properties(height=320, width="container")
                .interactive()
            )
            st.altair_chart(chart, use_container_width=True)

# =============================================================================
# 2) MPS & KPIs
# =============================================================================
with tab_mps:
    st.subheader("KPIs do MPS")

    # Caso você tenha salvo algo na sessão na página 06 (opcional):
    #   - "mps_df"       : DataFrame detalhado
    #   - "mps_display"  : tabela agregada de visualização
    #   - "mps_labels"   : lista de timestamps dos meses
    mps_df = st.session_state.get("mps_df")
    mps_display = st.session_state.get("mps_display")
    mps_labels = st.session_state.get("mps_labels")

    if isinstance(mps_display, pd.DataFrame) and not mps_display.empty:
        # formata cabeçalhos como mês/ano curtos
        cols = []
        for c in mps_display.columns:
            try:
                dt = pd.to_datetime(c, errors="coerce")
                cols.append(fmt_month_label(dt) if pd.notna(dt) else str(c))
            except Exception:
                cols.append(str(c))
        mps_show = mps_display.copy()
        mps_show.columns = cols
        st.dataframe(mps_show, use_container_width=True, height=280)
    else:
        st.info("Não há tabela do MPS na memória. Gere o MPS na página **06_MPS** e volte.")
        st.page_link("pages/06_MPS.py", label="📅 Ir para 06_MPS (Plano Mestre de Produção)")

    # KPIs rápidos (se mps_df estiver disponível)
    if isinstance(mps_df, pd.DataFrame) and not mps_df.empty:
        # Exemplos de KPIs simples
        est_min = int(mps_df.get("projected_on_hand_end", pd.Series(dtype=float)).min(skipna=True) or 0)
        rupturas = int((mps_df.get("projected_on_hand_end", pd.Series(dtype=float)) < 0).sum())
        qtd_programada = int(mps_df.get("planned_order_receipts", pd.Series(dtype=float)).sum(skipna=True) or 0)

        c1, c2, c3 = st.columns(3)
        kpi_card("Estoque projetado mínimo", f"{est_min}")
        with c2:
            kpi_card("Períodos com ruptura (EOH < 0)", f"{rupturas}")
        with c3:
            kpi_card("Qtde total programada (MPS)", f"{qtd_programada:,}".replace(",", "."))

# =============================================================================
# 3) Recomendações
# =============================================================================
with tab_rec:
    st.subheader("Recomendações automáticas")
    bullets = []

    # Baseado nos experimentos
    exp_df = _get_exp_df()
    if exp_df is not None and "sMAPE" in exp_df.columns:
        try:
            best = float(exp_df["sMAPE"].min())
            if best > 30:
                bullets.append("sMAPE alto → considerar **ajustar pré-processamento** (log/Box-Cox/outliers) ou **ampliar grade** de modelos.")
            elif best > 15:
                bullets.append("sMAPE moderado → testar **mais réplicas de bootstrap** e revisar sazonalidade.")
            else:
                bullets.append("sMAPE baixo → manter configuração atual; avalie **robustez** com validação/bootstraps.")
        except Exception:
            pass
    else:
        bullets.append("Gere a previsão e os experimentos na aba **Previsão** para recomendações mais específicas.")

    # Comportamento da previsão vs real
    real = _get_real_series()
    prev = _get_forecast_df()
    if real is not None and prev is not None and not real.empty and not prev.empty:
        # Checagem simples de viés (últimos 6 pontos em comum)
        merged = pd.merge(real, prev, on="ds", how="inner", suffixes=("_real", "_prev"))
        if len(merged) >= 6:
            win = merged.tail(6).copy()
            err = (win["y_prev"] - win["y_real"]).mean()
            if err > 0:
                bullets.append("Previsão **otimista** nos últimos meses (tende a **superestimar**). Avalie ajuste de tendência.")
            elif err < 0:
                bullets.append("Previsão **conservadora** nos últimos meses (tende a **subestimar**). Avalie ajuste de tendência.")
            else:
                bullets.append("Previsão sem viés aparente nos últimos meses.")

    if bullets:
        st.markdown("\n".join(f"- {b}" for b in bullets))
    else:
        st.markdown("- Sem recomendações automáticas no momento.")

# =============================================================================
# Rodapé – navegação
# =============================================================================
st.divider()
c1, c2 = st.columns(2)
with c1:
    st.page_link("pages/05_Inputs_MPS.py", label="Voltar: Inputs do MPS")
with c2:
    st.page_link("pages/04_Previsao.py", label="Ajustar Previsão")
