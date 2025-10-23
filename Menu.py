# Menu.py
# Rodar: streamlit run Menu.py

from __future__ import annotations
import os
import streamlit as st
from packaging import version

# =========================
# CONFIGURAÇÃO BÁSICA
# =========================
st.set_page_config(page_title="Previsão & PCP", page_icon="🧭", layout="wide")

APP_TITLE = "🧭 Previsão & PCP"
APP_SUBTITLE = "Integração entre Modelos de Previsão e Planejamento e Controle da Produção (PCP)"

# Defina os caminhos das páginas (NOMES DE ARQUIVOS SEM ACENTO/EMOJI!)
PAGES_DIR = "pages"
PAGES = [
    {"path": f"{PAGES_DIR}/01_Upload.py",            "title": "Upload",             "icon": "📤"},
    {"path": f"{PAGES_DIR}/02_Serie_Temporal.py",    "title": "Série Temporal",     "icon": "📈"},
    {"path": f"{PAGES_DIR}/03_Analise_Detalhada.py", "title": "Análise Detalhada",  "icon": "🔎"},
    {"path": f"{PAGES_DIR}/04_Previsao.py",          "title": "Previsão",           "icon": "📈"},
    {"path": f"{PAGES_DIR}/05_MPS.py",               "title": "MPS",                "icon": "🗓️"},
    #{"path": f"{PAGES_DIR}/06_MRP.py",               "title": "MRP",                "icon": "🧩"},
    #{"path": f"{PAGES_DIR}/07_Dashboard.py",         "title": "Dashboard",          "icon": "📊"},
]

# =========================
# VALIDAÇÕES DE ESTRUTURA
# =========================
missing = [p for p in PAGES if not os.path.exists(p["path"])]
if not os.path.exists(PAGES_DIR):
    st.warning(f"Diretório `{PAGES_DIR}/` não encontrado ao lado do Menu.py. Crie `{PAGES_DIR}/` e coloque as páginas lá.")
elif missing:
    st.warning("Algumas páginas configuradas não foram encontradas no disco:")
    for m in missing:
        st.write("•", m["path"])
    st.info("A navegação ainda funciona para as páginas existentes, mas confira os nomes/paths acima.")

# =========================
# CABEÇALHO
# =========================
st.title(APP_TITLE)
st.subheader(APP_SUBTITLE)

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

st.markdown("### Comece agora")
st.markdown(
    "A seguir, envie a **série temporal** do produto que deseja analisar. "
    "O sistema processará os dados, executará os modelos de previsão e gerará os planos MPS e MRP."
)

# =========================
# NAVEGAÇÃO (com fallback)
# =========================
# Preferência: usar st.navigation (Streamlit mais novo)
st_ver = version.parse(st.__version__)
has_navigation = hasattr(st, "navigation")  # disponível nas versões mais recentes
has_page_link = hasattr(st, "page_link")    # fallback elegante em versões ~1.24+

# Renderiza navegação lateral
def render_sidebar_links():
    with st.sidebar:
        st.header("Navegação")
        st.page_link("Menu.py", label="Home", icon="🧭")
        for pg in PAGES:
            if os.path.exists(pg["path"]):
                st.page_link(pg["path"], label=pg["title"], icon=pg["icon"])

# 1) Se existir st.navigation, registre as páginas (ele mesmo cria o menu)
if has_navigation:
    # Importante: registre SOMENTE as páginas do diretório /pages para não duplicar a home
    nav_pages = [st.Page(p["path"], title=p["title"], icon=p["icon"]) for p in PAGES if os.path.exists(p["path"])]
    st.navigation(pages=nav_pages)
else:
    # 2) Fallback usando st.page_link na sidebar
    if has_page_link:
        render_sidebar_links()
    else:
        # 3) Fallback "raiz": radio manual para versões bem antigas
        with st.sidebar:
            st.header("Navegação")
            choices = ["Home"] + [pg["title"] for pg in PAGES if os.path.exists(pg["path"])]
            choice = st.radio("Ir para:", choices, index=0)
            if choice != "Home":
                # Não há API nativa; informamos o link para clique
                sel = next(pg for pg in PAGES if pg["title"] == choice)
                st.markdown(f"[Abrir **{sel['title']}**](/?page={sel['path']})")

# =========================
# BOTÃO DE AÇÃO
# =========================
col1, col2 = st.columns([1, 4])
with col1:
    go = st.button("➡️ Iniciar - Passo 1 (Upload da Série Temporal)", type="primary")

if go:
    # Tente o switch_page se existir (novo) e caia para alternativas
    target = f"{PAGES_DIR}/01_Upload.py"
    try:
        if hasattr(st, "switch_page"):
            st.switch_page(target)
        elif has_page_link:
            # Mostra um link clicável como fallback imediato
            st.success("Versão do Streamlit sem `switch_page`. Clique abaixo para seguir:")
            st.page_link(target, label="Ir para Upload", icon="📤")
        else:
            # último recurso: sugerir menu lateral
            st.info("Não consegui redirecionar automaticamente. Acesse o menu lateral e clique em **Upload**.")
    except Exception as e:
        st.info("Se o botão não funcionar automaticamente, acesse o menu lateral e clique em **Upload**.")
        st.caption(f"(Detalhe técnico: {e})")
