# Menu.py
# Página inicial - "Previsão & PCP"
# Rodar: streamlit run Menu.py

from __future__ import annotations
import streamlit as st

st.set_page_config(page_title="Previsão & PCP", page_icon="🧭", layout="wide")

st.navigation(
    pages=[
        st.Page("pages/01_Upload.py",            title="Upload",            icon="📤"),
        st.Page("pages/02_Serie_Temporal.py",    title="Série Temporal",    icon="📈"),
        st.Page("pages/03_Analise_Detalhada.py", title="Análise Detalhada", icon="🧪"),
        st.Page("pages/04_Previsao.py",          title="Previsão",          icon="🔮"),
        st.Page("pages/05_MPS.py",               title="MPS",               icon="🗓️"),
    ],
    position="sidebar",
    expanded=False,
)

# -------- Cabeçalho --------
st.title("🧭 Previsão & PCP")
st.subheader("Integração entre Modelos de Previsão e Planejamento e Controle da Produção (PCP)")

# -------- Texto principal --------
st.markdown("""
### O que é  
Artefato desenvolvido para **gerar previsões de demanda** a partir de modelos **clássicos e de *Machine Learning***, integrando os resultados às ferramentas tradicionais de **PCP**. O sistema também possibilita  a criação de **dashboards executivos** que auxiliam a **análise de resultados** e a **tomada de decisão** de forma simples e visual.

---

### Benefícios  
- **Apoio tecnológico** para geração de previsões consistentes e embasadas em métodos validados na literatura.  
- **Integração automática** entre previsão, planejamento (MPS/MRP) e indicadores.  
- **Fluxo contínuo e intuitivo:**  🧾 **DADOS → 🤖 PREVER → 🏭 PLANEJAR**.  

---

### Principais Outputs  
- 📈 **Previsão de demanda** para os próximos **6, 8 ou 12 meses**, identificando automaticamente o modelo mais adequado à série temporal.  
- 🗓️ **MPS** (Master Production Schedule) e 🧩 **MRP** (Material Requirements Planning) interativos.  
- 📊 **Dashboards executivos** para visualização consolidada dos resultados e apoio à decisão.  
""")

st.divider()

# -------- Navegação para próxima etapa --------
st.markdown("### Comece agora")
st.markdown(
    "A seguir, envie a **série temporal** do produto que deseja analisar. "
    "O sistema processará os dados, executará os modelos de previsão e gerará os planos MPS e MRP."
)

# Botão principal
col1, col2 = st.columns([1, 4])
with col1:
    go = st.button("➡️ Iniciar - Passo 1 (Upload da Série Temporal)", type="primary")

if go:
    try:
        st.switch_page("pages/01_Upload.py")
    except Exception:
        st.info("Se o botão não funcionar automaticamente, acesse o menu lateral e clique em **Passo 1 - Upload**.")

st.page_link("pages/01_upload.py", label="Ou clique aqui para abrir o **Passo 1** (Upload)", icon="📤")