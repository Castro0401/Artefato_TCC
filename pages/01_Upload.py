# pages/01_upload.py
# Página 1 — Upload da série temporal (mensal)
from __future__ import annotations
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Passo 1 • Upload da Série", page_icon="📤", layout="wide")
st.title("📤 Passo 1: Upload da Série temporal do produto a ser analisado")

st.markdown("""
**Antes de enviar, observe:**
1. A série deve ter **coluna de datas `ds`** (mensal) e **coluna de quantidade `y`**.  
2. É importante ter **pelo menos 50 observações**.  
3. O arquivo deve ser **Excel** (`.xlsx` ou `.xls`).  
4. No Passo 2, o usuário poderá escolher o horizonte da previsão (**6, 8 ou 12 meses**), e isso alimentará o MPS (Passo 3).
""")

st.divider()
st.subheader("Fluxo deste artefato")
st.markdown("""
1. **Enviar série temporal** (Excel com `ds` e `y`).  
2. Gerar **previsão (6/8/12 meses)** com o melhor modelo (*em integração*).  
3. Construir **MPS** e **MRP** interativos para apoiar o PCP.  
4. Exibir **dashboards** e permitir **exportação**.
""")

st.divider()
st.subheader("Envio do Excel (ds, y)")

file = st.file_uploader(
    "Selecione seu arquivo Excel",
    type=["xlsx", "xls"],
    help="Use colunas: ds (data mensal) e y (quantidade). Ex.: ds=2023-01-01, y=300."
)

if file:
    try:
        df = pd.read_excel(file)
    except Exception as e:
        st.error(f"Não foi possível ler o Excel: {e}")
        st.stop()

    st.write("**Pré-visualização (primeiras linhas):**")
    st.dataframe(df.head(20), use_container_width=True)

    # Validações
    problems = []
    if not {"ds", "y"}.issubset(df.columns):
        problems.append("O arquivo precisa ter as colunas **ds** e **y**.")
    if len(df) < 50:
        problems.append("A série precisa ter **pelo menos 50 linhas**.")

    if problems:
        st.error(" | ".join(problems))
        st.stop()

    # Normaliza: garante datetime e numérico
    df = df.copy()
    df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna(subset=["ds", "y"]).reset_index(drop=True)

    # Agrega por mês (caso venha diária/semanal)
    monthly = (
        df.assign(month=df["ds"].dt.to_period("M").dt.to_timestamp())
          .groupby("month", as_index=False)["y"].sum()
          .sort_values("month")
    )
    # rótulo tipo Set/25
    pt = {1:"Jan",2:"Fev",3:"Mar",4:"Abr",5:"Mai",6:"Jun",7:"Jul",8:"Ago",9:"Set",10:"Out",11:"Nov",12:"Dez"}
    monthly["ds"] = monthly["month"].apply(lambda ts: f"{pt[ts.month]}/{str(ts.year)[-2:]}")
    monthly = monthly[["ds","y"]].reset_index(drop=True)

    st.success("Arquivo válido! Série mensal preparada 👇")
    st.dataframe(monthly, use_container_width=True)

    # Guarda para as próximas etapas
    st.session_state["ts_df_norm"] = monthly
    st.session_state["upload_ok"] = True

    # ✨ Invalida previsões anteriores (obriga a salvar uma nova no Passo 2)
    st.session_state.pop("forecast_df", None)
    st.session_state.pop("forecast_h", None)
    st.session_state["forecast_committed"] = False

    st.info("Série carregada. No **Passo 2**, escolha o horizonte (6/8/12 meses) e salve a previsão para liberar o **MPS**.")
    st.page_link("pages/02_Serie_Temporal.py", label="➡️ Seguir para Análise da Série Temporal")
else:
    st.info("Envie um Excel com colunas **ds** e **y** para continuar.")
