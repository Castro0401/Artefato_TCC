# pages/07_Dashboard_Conclusao.py
from __future__ import annotations
import io
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------
# Título e abas
# -----------------------------
st.title("✅ 07 — Conclusão (Painel de Decisão)")

tabs = st.tabs(["📊 Acurácia", "🏭 MPS & KPIs", "💡 Recomendações"])

# Tenta importar Plotly (com fallback)
try:
    import plotly.express as px  # type: ignore
    _plotly_ok = True
except Exception:
    _plotly_ok = False


# Utilidades
def _fmt_month(x) -> str:
    """Formata datas como 'Mes/AA' de forma robusta."""
    try:
        dt = pd.to_datetime(x)
        return dt.strftime("%b/%y").title().replace(".", "")
    except Exception:
        return str(x)


def _link_row(left_label: str, page: str, label: str, icon: str):
    c1, c2 = st.columns([1, 4])
    with c1:
        st.markdown(f"**{left_label}**")
    with c2:
        st.page_link(page, label=label, icon=icon)


# ============================================================================
# TAB 1 — ACURÁCIA
# ============================================================================
with tabs[0]:
    st.subheader("Desempenho dos modelos de previsão")

    # Dados esperados em memória
    ts_df = st.session_state.get("ts_df_norm")            # ['ds','y'] historico
    fcst_df = st.session_state.get("forecast_df")         # ['ds','y'] previsão escolhida
    exp_df = (st.session_state.get("experiments_df")
              or st.session_state.get("experiments_table")
              or st.session_state.get("pipeline_experiments"))  # compatibilidade

    # 1) Tabela de experimentos (se houver)
    if isinstance(exp_df, pd.DataFrame) and not exp_df.empty:
        st.caption("Tabela de experimentos (topo). Baixe o CSV para detalhes completos.")
        # Mostra as N primeiras linhas de forma leve
        st.dataframe(exp_df.head(50), use_container_width=True, height=260)
        # Download CSV
        csv_bytes = exp_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Baixar experimentos (CSV)",
            data=csv_bytes,
            file_name="experimentos_previsao.csv",
            mime="text/csv",
            help="Baixa todos os experimentos gerados no Passo 2 — Previsão."
        )
    else:
        st.info("Sem tabela de experimentos em memória. Gere na página de **Previsão** e volte aqui.")
        _link_row("Ir:", "pages/04_Previsao.py", "Ir para 04_Previsao", "🔮")

    # 2) Gráfico Real × Previsão
    st.divider()
    st.subheader("Real × Previsão (linha do tempo)")

    if ts_df is None or fcst_df is None:
        st.info("Não foi possível exibir o gráfico: faltam dados de série histórica ou previsão.")
        _link_row("Ajustar:", "pages/04_Previsao.py", "Ajustar Previsão", "🛠️")
    else:
        try:
            hist = ts_df.copy()[["ds", "y"]].rename(columns={"y": "Real"})
            fut = fcst_df.copy()[["ds", "y"]].rename(columns={"y": "Previsão"})

            # 🔧 Correção do tipo: converte para datetime nas duas bases
            hist["ds"] = pd.to_datetime(hist["ds"], errors="coerce")
            fut["ds"] = pd.to_datetime(fut["ds"], errors="coerce")

            # Concatena e ordena
            both = pd.concat([hist, fut], ignore_index=True).sort_values("ds")

            if _plotly_ok:
                plot_df = (both
                           .melt(id_vars="ds", value_vars=["Real", "Previsão"],
                                 var_name="Série", value_name="Valor")
                           .dropna(subset=["ds", "Valor"]))
                fig = px.line(plot_df, x="ds", y="Valor", color="Série",
                              title="Real × Previsão",
                              labels={"ds": "Período", "Valor": "Quantidade"})
                fig.update_layout(legend_title_text="", height=420)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Plotly não está instalado neste ambiente. Instale `plotly` para visualizar o gráfico.")
        except Exception as e:
            st.info(f"Não foi possível exibir o gráfico Real × Previsão: {e}")

    st.divider()
    cL, cR = st.columns(2)
    with cL:
        st.page_link("pages/05_Inputs_MPS.py", label="⬅️ Voltar: Inputs do MPS", icon="🧰")
    with cR:
        st.page_link("pages/04_Previsao.py", label="🛠️ Ajustar Previsão", icon="🛠️")


# ============================================================================
# TAB 2 — MPS & KPIs
# ============================================================================
with tabs[1]:
    st.subheader("Produção planejada e disponibilidade (ATP)")

    # O MPS é calculado na página 06; aqui usamos o que estiver na sessão:
    # - forecast_df (previsão)
    # - mps_last_df (se a página 06 tiver salvo)
    # - ou mostramos instruções
    mps_df = st.session_state.get("mps_last_df") or st.session_state.get("mps_df")

    if not isinstance(mps_df, pd.DataFrame) or mps_df.empty:
        st.info(
            "Não encontrei o **MPS** em memória. Gere o MPS na aba **06_MPS** e volte aqui."
        )
        _link_row("Ir:", "pages/06_MPS.py", "Ir para 06_MPS (Plano Mestre de Produção)", "🗓️")
    else:
        # Exibir um resumo tabular leve
        cols_show = [c for c in [
            "ds",
            "gross_requirements",
            "projected_on_hand_end",
            "planned_order_receipts",
            "planned_order_releases",
            "atp",
        ] if c in mps_df.columns]

        df_show = mps_df[cols_show].copy()

        # Formata datas como Mês/Ano para visualização
        if "ds" in df_show.columns:
            df_show["Período"] = df_show["ds"].apply(_fmt_month)
            df_show = df_show.drop(columns=["ds"])
            # Reordena para deixar período na frente
            df_show = df_show[["Período"] + [c for c in df_show.columns if c != "Período"]]

        st.dataframe(df_show, use_container_width=True, height=320)

        # Gráfico ATP
        if _plotly_ok and "atp" in mps_df.columns:
            try:
                plot_atp = mps_df.copy()
                plot_atp["ds"] = pd.to_datetime(plot_atp["ds"], errors="coerce")
                plot_atp = plot_atp.dropna(subset=["ds"])
                fig_atp = px.bar(
                    plot_atp,
                    x="ds",
                    y="atp",
                    title="ATP por período",
                    labels={"ds": "Período", "atp": "Available-to-Promise"},
                )
                fig_atp.update_layout(height=360)
                st.plotly_chart(fig_atp, use_container_width=True)
            except Exception as e:
                st.info(f"Não foi possível gerar o gráfico de ATP: {e}")
        elif "atp" not in mps_df.columns:
            st.info("O MPS atual não possui coluna **atp**; gere novamente na página 06, se necessário.")
        else:
            st.info("Plotly não está instalado neste ambiente. Instale `plotly` para visualizar o gráfico de ATP.")

        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            st.page_link("pages/05_Inputs_MPS.py", label="⚙️ Ajustar Inputs do MPS", icon="⚙️")
        with c2:
            st.page_link("pages/06_MPS.py", label="🗓️ Recalcular MPS", icon="🗓️")


# ============================================================================
# TAB 3 — RECOMENDAÇÕES
# ============================================================================
with tabs[2]:
    st.subheader("Recomendações automáticas")

    # Usa diagnósticos e informações que já existem em sessão quando possível
    recs = []

    # Tipo de demanda (se salvo na análise detalhada)
    demand_type = st.session_state.get("demand_type")  # "Regular", "Intermittent", ...
    if demand_type:
        if demand_type in {"Intermittent", "Lumpy"}:
            recs.append("Aplicar **Croston/SBA/TSB** (demanda intermitente).")
        elif demand_type == "Erratic":
            recs.append("Demanda **errática**: suavização robusta/outlier handling e modelos sem sazonalidade rígida.")
        else:
            recs.append("Demanda **regular**: modelos clássicos (com/sem sazonalidade) tendem a funcionar.")

    # Transformações sugeridas (flags salvos na Análise Detalhada, se houver)
    hetero_flag = st.session_state.get("hetero_flag")
    if hetero_flag:
        recs.append("Sinais de **heterocedasticidade** → considerar **log** ou **Box-Cox**.")

    has_nonpositive = st.session_state.get("has_nonpositive")
    skew_val = st.session_state.get("skew_val")
    if has_nonpositive:
        recs.append("Há valores **≤ 0** → usar **Box-Cox** com deslocamento.")
    elif (skew_val is not None) and (skew_val == skew_val) and (skew_val > 0.5):
        recs.append("Distribuição **positiva** e **assimétrica** → **log(y)** recomendado.")

    # Força STL (se disponível)
    Ft = st.session_state.get("stl_F_trend")
    Fs = st.session_state.get("stl_F_seas")
    if Ft is not None and Ft == Ft and Ft < 0.2:
        recs.append("**Tendência fraca** (STL) → evitar modelos com tendência rígida.")
    if Fs is not None and Fs == Fs and Fs < 0.2:
        recs.append("**Sazonalidade fraca** (STL) → considerar modelos **sem sazonalidade**.")

    # Segurança para caso nada esteja na sessão
    if not recs:
        st.info("Sem recomendações automáticas no momento. Gere diagnósticos na aba **Análise Detalhada**.")
        _link_row("Ir:", "pages/03_Analise_Detalhada.py", "Análise Detalhada", "🧪")
    else:
        st.markdown("\n".join(f"- {r}" for r in recs))

    st.divider()
    cL, cR = st.columns(2)
    with cL:
        st.page_link("pages/05_Inputs_MPS.py", label="⬅️ Voltar: Inputs do MPS", icon="🧰")
    with cR:
        st.page_link("pages/06_MPS.py", label="🗓️ MPS", icon="🗓️")
