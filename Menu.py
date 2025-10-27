# Menu.py
from __future__ import annotations
import streamlit as st

st.set_page_config(page_title="Previsão & PCP", page_icon="🧭", layout="wide")

# --------- Home como FUNÇÃO (não arquivo) ---------
def home_page():
    st.title("🧭 Previsão & PCP")
    st.subheader("Integração entre Modelos de Previsão e PCP")

    st.markdown("""
    ### O que é  
    Artefato desenvolvido para **gerar previsões de demanda** a partir de modelos **clássicos e de *Machine Learning***, integrando os resultados às ferramentas tradicionais de **PCP**. O sistema também possibilita a criação de **dashboards executivos** que auxiliam a **análise de resultados** e a **tomada de decisão** de forma simples e visual.

---
    ### Benefícios
    - **Previsões consistentes** com métodos validados  
    - **Integração automática** entre previsão e MPS/MRP  
    - **Fluxo:** 🧾 DADOS → 🤖 PREVER → 🏭 PLANEJAR
    ---
    ### Principais outputs
    - 📈 Previsões (6, 8 ou 12 meses)  
    - 🗓️ MPS / 🧩 MRP  
    - 📊 Dashboards executivos
    """)

    st.divider()
    st.markdown("### Comece agora")
    if st.button("➡️ Iniciar — Passo 1 (Upload da Série)", type="primary"):
        try:
            st.switch_page("pages/01_Upload.py")
        except Exception:
            st.info("Use o menu lateral e clique em **Upload**.")

# --------- Navegação com títulos/ícones custom ---------
nav = st.navigation([
    st.Page(home_page,                          title="Início — Previsão & PCP", icon="🧭"),  # <- função
    st.Page("pages/01_Upload.py",               title="Upload",                   icon="📤"),
    st.Page("pages/02_Serie_Temporal.py",       title="Série Temporal",          icon="📈"),
    st.Page("pages/03_Analise_Detalhada.py",    title="Análise Detalhada",       icon="🔍"),
    st.Page("pages/04_Previsao.py",             title="Previsão",                 icon="🔮"),
    st.Page("pages/05_Inputs_MPS.py",           title="Inputs do MPS",           icon="⚙️"),
    st.Page("pages/06_MPS.py",                  title="MPS — Plano Mestre",      icon="🗓️"),
    st.Page("pages/07_Dashboard_Conclusao.py",  title="Dashboard & Conclusão",   icon="📊"),
])

nav.run()
