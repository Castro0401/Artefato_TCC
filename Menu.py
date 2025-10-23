# app_streamlit.py
# Rodar: streamlit run app_streamlit.py
from __future__ import annotations
import streamlit as st

st.set_page_config(page_title="Previsão & PCP", page_icon="🧭", layout="wide")

# =========================
# NAVIGAÇÃO EXPLÍCITA
# =========================
# Declare explicitamente TODAS as páginas que existem no seu projeto.
# Isso evita erros de caminhos inexistentes.
PAGES = st.navigation(
    pages=[
        st.Page("app_streamlit.py",               title="Início",            icon="🧭"),
        st.Page("pages/01_Upload.py",             title="Upload",            icon="📤"),
        st.Page("pages/02_Serie_Temporal.py",     title="Série Temporal",    icon="📈"),
        st.Page("pages/03_Analise_Detalhada.py",  title="Análise Detalhada", icon="🧪"),
        st.Page("pages/04_Previsao.py",           title="Previsão",          icon="🔮"),
        st.Page("pages/04_MPS.py",                title="MPS",               icon="🗓️"),
    ],
    position="sidebar",
    expanded=False,
)
# IMPORTANTE: rode a árvore de navegação no final do arquivo
# (depois do conteúdo da homepage) com PAGES.run()

# =========================
# HOME
# =========================
st.title("🧭 Previsão & PCP")
st.subheader("Integração entre Modelos de Previsão e Ferramentas de PCP")

st.markdown("""
### O que é  
Artefato para **gerar previsões de demanda** (métodos clássicos e ML) e integrar com **PCP** (MPS/MRP) e **dashboards**.

---

### Benefícios  
- **Apoio tecnológico** com métodos validados.  
- **Integração automática** entre previsão e planejamento.  
- **Fluxo intuitivo:** 🧾 **DADOS → 🤖 PREVER → 🏭 PLANEJAR**.

---

### Principais Outputs  
- 📈 **Previsão** para **6, 8 ou 12 meses**.  
- 🗓️ **MPS** e 🧩 **MRP** interativos.  
- 📊 **Dashboards** executivos.
""")

st.divider()
st.markdown("### Comece agora")
st.markdown("Envie a **série temporal** do produto e siga o fluxo.")

col1, col2 = st.columns([1, 4])
with col1:
    go = st.button("➡️ Iniciar - Passo 1 (Upload)", type="primary")
if go:
    try:
        st.switch_page("pages/01_Upload.py")
    except Exception:
        st.info("Se não abrir automaticamente, use o menu lateral: **Upload**.")

# Execução da navegação
PAGES.run()
