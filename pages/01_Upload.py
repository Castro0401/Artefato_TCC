# pages/01_Upload.py
# Página 1 — Upload da série temporal (mensal)
from __future__ import annotations
import pandas as pd
import numpy as np
import streamlit as st

st.title("📤 Upload da Série Temporal")

st.markdown("""
**Antes de enviar, observe:**
1. O arquivo deve ter **uma coluna de datas** (pode se chamar **ds, data, date, dt, mes, month, period...**)  
2. A **coluna de quantidades** deve ser a **coluna do produto** (ex.: *Cadeira de ripas*). O **nome da coluna será usado como nome do item** nas próximas etapas.  
3. É importante ter **pelo menos 50 observações**.  
4. O arquivo deve ser **Excel** (`.xlsx` ou `.xls`).  
""")

st.divider()
st.subheader("Fluxo deste artefato")
st.markdown("""
1. **Enviar série temporal** (Excel com 1 coluna de data e 1 coluna de quantidades).  
2. **Análise da série temporal** (decomposição, sazonalidade, tendências, outliers).
3. Gerar **previsão (6/8/12 meses)** com o melhor modelo.  
4. Construir **MPS** para apoiar o PCP.  
5. Exibir **dashboards** para auxílio na **tomada de decisões**.
""")

st.divider()
st.subheader("Envio do Excel (data + coluna do produto)")

file = st.file_uploader(
    "Selecione seu arquivo Excel",
    type=["xlsx", "xls"],
    help="Uma coluna de datas (ds, data, date, dt, mes...) e outra coluna com as quantidades do produto (o nome da coluna de quantidades será o nome do item)."
)

# ---- utilidades ----
DATE_ALIASES = {
    "ds","data","date","dt","mes","month","period","tempo","time","timestamp","periodo"
}

PT_MON = {1:"Jan",2:"Fev",3:"Mar",4:"Abr",5:"Mai",6:"Jun",7:"Jul",8:"Ago",9:"Set",10:"Out",11:"Nov",12:"Dez"}

def is_datetime_like(s: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(s):
        return True
    # muito comum virem como string reconhecível
    try:
        _ = pd.to_datetime(s, errors="raise")
        return True
    except Exception:
        return False

def pick_date_col(df: pd.DataFrame) -> str | None:
    cols_lower = {c.lower(): c for c in df.columns}
    # 1) tenta por alias
    for alias in DATE_ALIASES:
        if alias in cols_lower:
            return cols_lower[alias]
    # 2) tenta por dtype/conversão plausível
    datetime_candidates = [c for c in df.columns if is_datetime_like(df[c])]
    if datetime_candidates:
        return datetime_candidates[0]
    return None

def pick_value_col(df: pd.DataFrame, date_col: str | None) -> str | None:
    # escolhe a primeira coluna numérica (ou não-data) como quantidades
    candidates = [c for c in df.columns if c != date_col]
    if not candidates:
        return None
    # prioriza numéricas
    num_candidates = [c for c in candidates if pd.api.types.is_numeric_dtype(df[c])]
    if num_candidates:
        return num_candidates[0]
    # senão, tenta converter a 1ª candidata para numérico
    return candidates[0]

if file:
    try:
        df_in = pd.read_excel(file)
    except Exception as e:
        st.error(f"Não foi possível ler o Excel: {e}")
        st.stop()

    st.write("**Pré-visualização (primeiras linhas):**")
    st.dataframe(df_in.head(20), use_container_width=True)

    if df_in.empty or df_in.shape[1] < 2:
        st.error("O arquivo deve ter **ao menos 2 colunas**: uma de **data** e outra de **quantidades** do produto.")
        st.stop()

    # ---- detecção automática das colunas ----
    auto_date_col  = pick_date_col(df_in)
    auto_value_col = pick_value_col(df_in, auto_date_col)

    st.markdown("### Mapeamento das colunas")
    c1, c2 = st.columns(2)
    with c1:
        date_col = st.selectbox(
            "Selecione a coluna de **datas**",
            options=list(df_in.columns),
            index=(list(df_in.columns).index(auto_date_col) if auto_date_col in df_in.columns else 0)
        )
    with c2:
        # sugestão automática para a coluna de quantidades
        default_value_idx = 0
        if auto_value_col in df_in.columns:
            default_value_idx = list(df_in.columns).index(auto_value_col)
        value_col = st.selectbox(
            "Selecione a coluna de **quantidades (produto)**",
            options=[c for c in df_in.columns if c != date_col] or list(df_in.columns),
            index=default_value_idx if default_value_idx < len([c for c in df_in.columns if c != date_col]) else 0
        )

    # ---- validações básicas ----
    problems = []
    if date_col is None or value_col is None:
        problems.append("Não foi possível identificar coluna de data e/ou de quantidades.")
    if problems:
        st.error(" | ".join(problems))
        st.stop()

    # ---- normalização: datas e numérico ----
    df = df_in[[date_col, value_col]].copy()
    df.rename(columns={date_col: "ds", value_col: "y"}, inplace=True)

    # guarda o nome do produto = nome da coluna de valores original
    product_name = str(value_col)

    # coerções
    df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
    df["y"]  = pd.to_numeric(df["y"], errors="coerce")

    df = df.dropna(subset=["ds", "y"]).reset_index(drop=True)

    # checagem de tamanho mínimo na granularidade original
    if len(df) < 50:
        st.warning("A série tem menos de **50 observações** na granularidade original. Considere enviar um histórico maior para melhor robustez.")

    # ---- normaliza para mês (SEM SOMAR) ----
    df["month"] = df["ds"].dt.to_period("M").dt.to_timestamp()

    # Se já estiver mensal (1 linha por mês), não agrega
    if df["month"].is_unique:
        monthly = (
            df[["month", "y"]]
            .sort_values("month")
            .reset_index(drop=True)
        )
    else:
        st.warning("Detectei mais de uma linha no mesmo mês. Vou consolidar SEM somar.")

        agg_mode = st.selectbox(
            "Como consolidar valores quando houver duplicidade no mesmo mês?",
            options=["último", "primeiro", "média"],
            index=0
        )

        if agg_mode == "último":
            monthly = (
                df.sort_values("ds")
                .groupby("month", as_index=False)["y"].last()
                .sort_values("month")
            )
        elif agg_mode == "primeiro":
            monthly = (
                df.sort_values("ds")
                .groupby("month", as_index=False)["y"].first()
                .sort_values("month")
            )
        else:  # média
            monthly = (
                df.groupby("month", as_index=False)["y"].mean()
                .sort_values("month")
            )

    # ds deve permanecer como datetime (1º dia do mês)
    monthly = monthly.rename(columns={"month": "ds"})

    # label apenas para exibição (NÃO usar para cálculo)
    monthly["ds_label"] = monthly["ds"].apply(
        lambda ts: f"{PT_MON[ts.month]}/{str(ts.year)[-2:]}"
    )

    # exibição no Streamlit
    st.success(f"Arquivo válido! Série mensal preparada para **{product_name}** 👇")
    st.dataframe(monthly[["ds_label", "y"]], use_container_width=True)

    # salvar no estado COM ds datetime
    st.session_state["ts_df_norm"] = monthly[["ds", "y"]].copy()



    # ---- guarda no estado para as próximas etapas ----
    st.session_state["ts_df_norm"]   = monthly            # série mensal normalizada (ds,y)
    st.session_state["upload_ok"]    = True
    st.session_state["product_name"] = product_name       # nome do item/produto (virá do nome da coluna)

    # Invalida previsões anteriores (obriga a salvar nova no Passo 2)
    st.session_state.pop("forecast_df", None)
    st.session_state.pop("forecast_h", None)
    st.session_state["forecast_committed"] = False

    st.info(
        "Série carregada. Siga para a próxima página."
    )
    st.page_link("pages/02_Serie_Temporal.py", label="➡️ Seguir para Análise da Série Temporal")
else:
    st.info("Envie um Excel com **uma coluna de datas** e **uma coluna de quantidades do produto** (o nome da coluna de quantidades será usado como **nome do item**).")
