# Menu.py
# Página inicial - "Previsão & PCP"
# Rodar: streamlit run Menu.py

from __future__ import annotations
from pathlib import Path
import streamlit as st

st.set_page_config(page_title="Previsão & PCP", page_icon="🧭", layout="wide")

# ======================================================
# 🔒 1) NAVEGAÇÃO SEGURA (EVITA ERROS DE CAMINHO)
# ======================================================

def first_existing(paths: list[str]) -> str | None:
    """Retorna o primeiro caminho existente na lista."""
    for p in paths:
        if Path(p).exists():
            return p
    return None

missing_msgs = []

HOME_PAGE   = "Menu.py"
UPLOAD_PAGE = first_existing(["pages/01_Upload.py", "pages/01_upload.py"])
SERIE_PAGE  = first_existing(["pages/02_Serie_Temporal.py"])
ROBUST_PAGE = first_existing(["pages/03_Analise_Detalhada.py"])
PREV_PAGE   = first_existing(["pages/04_Previsao.py", "pages/03_Previsao.py"])
MPS_PAGE    = first_existing(["pages/05_MPS.py", "pages/03_MPS.py", "pages/03_mps.py"])

if not UPLOAD_PAGE: missing_msgs.append("• Passo 1 (Upload) não encontrado.")
if not SERIE_PAGE:  missing_msgs.append("• Passo 2 (Série Temporal) não encontrado.")
if not PREV_PAGE:   missing_msgs.append("• Passo 3 (Previsão) não encontrado.")
if not MPS_PAGE:    missing_msgs.append("• Passo 4 (MPS) não encontrado.")

# Define o menu lateral seguro
NAV = st.navigation(
    pages=[
        st.Page(HOME_PAGE,  title="Início",           icon="🧭"),
        *( [st.Page(UPLOAD_PAGE, title="Upload", icon="📤")] if UPLOAD_PAGE else [] ),
        *( [st.Page(SERIE_PAGE,  title="Série Temporal", icon="📈")] if SERIE_PAGE else [] ),
        *( [st.Page(ROBUST_PAGE, title="Análise Detalhada", icon="🧪")] if ROBUST_PAGE else [] ),
        *( [st.Page(PREV_PAGE,   title="Previsão", icon="🔮")] if PREV_PAGE else [] ),
        *( [st.Page(MPS_PAGE,    title="MPS", icon="🗓️")] if MPS_PAGE else [] ),
    ],
    position="sidebar",
    expanded=False,
)

# ======================================================
# 🧭 2) CONTEÚDO PRINCIPAL (mantido do seu original)
# ======================================================

st.title("🧭 Previsão & PCP")
st.subheader("Integração entre Modelos de Previsão e Ferramentas de Planejamento e Controle da Produção (PCP)")

st.markdown("""
### O que é  
Artefato desenvolvido para **gerar previsões de demanda** a partir de modelos **clássicos e de *Machine Learning***, integrando os resultados às ferramentas tradicionais de **PCP**.  
O sistema também possibilita a criação de **dashboards executivos** que auxiliam a **análise de resultados** e a **tomada de decisão** de forma simples e visual.

---

### Benefícios  
- **Apoio tecnológico** para geração de previsões consistentes e embasadas em métodos validados na literatura.  
- **Integração automática** entre previsão, planejamento (MPS/MRP) e indicadores.  
- **Fluxo contínuo e intuitivo:** 🧾 **DADOS → 🤖 PREVER → 🏭 PLANEJAR**.  

---

### Principais Outputs  
- 📈 **Previsão de demanda** para os próximos **6, 8 ou 12 meses**, identificando automaticamente o modelo mais adequado à série temporal.  
- 🗓️ **MPS** (Master Production Schedule) e 🧩 **MRP** (Material Requirements Planning) interativos.  
- 📊 **Dashboards executivos** para visualização consolidada dos resultados e apoio à decisão.
""")

st.divider()
st.markdown("### Comece agora")
st.markdown("""
A seguir, envie a **série temporal** do produto que deseja analisar.  
O sistema processará os dados, executará os modelos de previsão e gerará os planos MPS e MRP.
""")

col1, col2 = st.columns([1, 4])
with col1:
    go = st.button("➡️ Iniciar - Passo 1 (Upload da Série Temporal)", type="primary")

if go:
    if UPLOAD_PAGE:
        try:
            st.switch_page(UPLOAD_PAGE)
        except Exception:
            st.info("Se o botão não funcionar automaticamente, use o menu lateral: **Upload**.")
    else:
        st.error("Arquivo de Upload não encontrado. Verifique se '01_Upload.py' existe em /pages.")

# ======================================================
# ⚠️ 3) AVISOS DE CONFIGURAÇÃO (arquivos faltando)
# ======================================================
if missing_msgs:
    with st.expander("Avisos de configuração", expanded=True):
        for msg in missing_msgs:
            st.warning(msg)


