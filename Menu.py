# Menu.py
# Rodar: streamlit run Menu.py

from __future__ import annotations
import streamlit as st
import os

# =========================
# Configuração da Home
# =========================
st.set_page_config(page_title="Previsão & PCP", page_icon="🧭", layout="wide")

# =========================
# Cabeçalho
# =========================
st.title("🧭 Previsão & PCP")
st.subheader("Integração entre Modelos de Previsão e Planejamento e Controle da Produção (PCP)")

# =========================
# Texto principal
# =========================
st.markdown("""
### O que é  
Artefato desenvolvido para **gerar previsões de demanda** a partir de modelos **clássicos e de *Machine Learning***, integrando os resultados às ferramentas tradicionais de **PCP**. O sistema também possibilita a criação de **dashboards executivos** que auxiliam a **análise de resultados** e a **tomada de decisão** de forma simples e visual.

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

# =========================
# Comece agora
# =========================
st.markdown("### Comece agora")
st.markdown(
    "Envie a **série temporal** do produto que deseja analisar. "
    "O sistema processará os dados, executará os modelos de previsão e gerará os planos MPS e MRP."
)

# Caminho da primeira página (deve existir exatamente assim)
TARGET_PAGE = "pages/01_Upload.py"

# Aviso rápido se a página alvo não existir (evita clique 'morto')
if not os.path.exists(TARGET_PAGE):
    st.warning(f"Página alvo não encontrada: `{TARGET_PAGE}`. Verifique o nome/ caminho do arquivo.")

# Botão principal
col1, _ = st.columns([1, 4])
with col1:
    go = st.button("➡️ Iniciar - Passo 1 (Upload da Série Temporal)", type="primary")

if go:
    try:
        st.switch_page(TARGET_PAGE)  # caminho EXATO para a página dentro de /pages
    except Exception as e:
        st.info("Não foi possível redirecionar automaticamente. Use o menu lateral e clique em **Upload**.")
        st.caption(f"(Detalhe técnico: {e})")

# =========================
# Notas importantes
# =========================
# 1) Não use st.navigation em nenhum arquivo.
# 2) Deixe st.set_page_config apenas neste Menu.py.
# 3) Garanta estrutura:
#    /SeuProjeto
#      Menu.py
#      /pages
#        01_Upload.py
#        02_Serie_Temporal.py
#        03_Analise_Detalhada.py
#        04_Previsao.py
#        05_MPS.py
# 4) Rode a partir da raiz do projeto: streamlit run Menu.py
