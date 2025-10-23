# Menu.py
from __future__ import annotations
import os, glob, re
from pathlib import Path
import streamlit as st
from packaging import version

st.set_page_config(page_title="Previsão & PCP", page_icon="🧭", layout="wide")

APP_TITLE = "🧭 Previsão & PCP"
APP_SUBTITLE = "Integração entre Modelos de Previsão e Planejamento e Controle da Produção (PCP)"
PAGES_DIR = "pages"

# -------- descobrir páginas automaticamente (ordem 01_, 02_, ...)
def human_title(filename: str) -> str:
    base = Path(filename).stem                 # ex: "04_Previsao"
    base = re.sub(r"^\d+_", "", base)          # -> "Previsao"
    mapping = {
        "Previsao": "Previsão",
        "Serie_Temporal": "Série Temporal",
        "Analise_Detalhada": "Análise Detalhada",
        "Upload": "Upload",
        "MPS": "MPS",
        "MRP": "MRP",
        "Dashboard": "Dashboard",
    }
    return mapping.get(base, base.replace("_"," "))

def discover_pages():
    files = sorted(glob.glob(f"{PAGES_DIR}/*.py"))  # ordena: 01_, 02_...
    pages = []
    for f in files:
        title = human_title(Path(f).name)
        icon = "📄"
        if "Upload" in title: icon = "📤"
        elif "Série Temporal" in title or "Previsão" in title: icon = "📈"
        elif "Análise" in title: icon = "🔎"
        elif "MPS" in title: icon = "🗓️"
        elif "MRP" in title: icon = "🧩"
        elif "Dashboard" in title: icon = "📊"
        pages.append({"path": f, "title": title, "icon": icon})
    return pages

PAGES = discover_pages()

# -------- cabeçalho
st.title(APP_TITLE)
st.subheader(APP_SUBTITLE)

st.markdown("""
### O que é  
Artefato desenvolvido para **gerar previsões de demanda** a partir de modelos **clássicos e de *Machine Learning***, integrando os resultados às ferramentas tradicionais de **PCP**. O sistema também possibilita a criação de **dashboards executivos** que auxiliam a **análise de resultados** e a **tomada de decisão**.

---

### Benefícios  
- **Apoio tecnológico** para previsões consistentes.  
- **Integração automática** entre previsão, MPS/MRP e indicadores.  
- **Fluxo contínuo:** 🧾 **DADOS → 🤖 PREVER → 🏭 PLANEJAR**.  

---

### Principais Outputs  
- 📈 **Previsão** (6/8/12 meses) com seleção automática de modelo.  
- 🗓️ **MPS** e 🧩 **MRP** interativos.  
- 📊 **Dashboards** executivos.  
""")

st.divider()
st.markdown("### Comece agora")
st.markdown("Envie a **série temporal** do produto. O sistema processa, prevê e gera MPS/MRP.")

# -------- navegação (incluindo a Home!)
st_ver = version.parse(st.__version__)
has_navigation = hasattr(st, "navigation")
has_page_link = hasattr(st, "page_link")

if has_navigation:
    nav_pages = [  # inclua a própria Home para aparecer no menu e evitar confusão de página atual
        st.Page("Menu.py", title="Home", icon="🧭"),
        *[st.Page(p["path"], title=p["title"], icon=p["icon"]) for p in PAGES if os.path.exists(p["path"])],
    ]
    st.navigation(pages=nav_pages)
else:
    # fallback: navegação manual na sidebar
    with st.sidebar:
        st.header("Navegação")
        if has_page_link:
            st.page_link("Menu.py", label="Home", icon="🧭")
            for p in PAGES:
                if os.path.exists(p["path"]):
                    st.page_link(p["path"], label=p["title"], icon=p["icon"])
        else:
            # fallback mais antigo
            choices = ["Home"] + [p["title"] for p in PAGES if os.path.exists(p["path"])]
            choice = st.radio("Ir para:", choices, index=0)
            if choice != "Home":
                sel = next(p for p in PAGES if p["title"] == choice)
                st.markdown(f"[Abrir **{sel['title']}**](/?page={sel['path']})")

# -------- botão "Iniciar" com fallbacks reais de navegação
col1, col2 = st.columns([1, 4])
with col1:
    go = st.button("➡️ Iniciar - Passo 1 (Upload da Série Temporal)", type="primary")

if go:
    target = f"{PAGES_DIR}/01_Upload.py"
    try:
        if hasattr(st, "switch_page"):
            st.switch_page(target)  # caminho relativo, exatamente como registrado
        elif has_page_link:
            st.success("Clique para seguir para o Upload:")
            st.page_link(target, label="Ir para Upload", icon="📤")
        else:
            # último recurso: ajustar querystring e forçar rerun
            st.experimental_set_query_params(page=target)
            st.rerun()
    except Exception as e:
        st.info("Se não redirecionar, use o menu lateral e clique em **Upload**.")
        st.caption(f"(Detalhe técnico: {e})")
