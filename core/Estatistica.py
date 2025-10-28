# -*- coding: utf-8 -*-
"""
ADI & CV² — Versão 2 (saída mais estruturada)
---------------------------------------------
Lê Excel (colunas 'ds' e 'y'), calcula ADI e CV², classifica (Regular/Intermittent/Erratic/Lumpy)
e imprime TABELAS compactas + explicações.

Dicas p/ portar para Streamlit (NÃO implementar aqui):
- troque o caminho fixo por st.file_uploader;
- use st.dataframe() para as tabelas e st.markdown() para as explicações;
- se quiser um layout ainda mais bonito, formate as tabelas com Styler.
"""

import pandas as pd
import numpy as np
from textwrap import fill
import statsmodels.api as sm
from scipy.stats import shapiro
from scipy.stats import skew, normaltest, jarque_bera
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, kpss
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import boxcox, boxcox_normmax
from scipy.special import inv_boxcox
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ===== 1) Parâmetros e caminho =====
ARQUIVO = r"/Users/felipe/Desktop/Faculdade/TCC/Códigos VS Code/Streamlit/(2017-2025) Série Temporal - Prod Cod 7 (A).xlsx"
LIMIAR_ADI = 1.32
LIMIAR_CV2 = 0.49

# ===== 2) Funções utilitárias =====
def classificar_demanda(adi: float, cv2: float) -> str:
    if np.isinf(adi):
        return "Sem Demanda"
    if np.isnan(cv2):
        cv2 = 0.0
    if (adi < LIMIAR_ADI) and (cv2 < LIMIAR_CV2):
        return "Regular"
    if (adi >= LIMIAR_ADI) and (cv2 < LIMIAR_CV2):
        return "Intermittent"
    if (adi < LIMIAR_ADI) and (cv2 >= LIMIAR_CV2):
        return "Erratic"
    return "Lumpy"

EXPLICACOES_CURTAS = {
    "Regular":      "Demanda frequente e tamanhos pouco voláteis.",
    "Intermittent": "Muitos zeros (intermitência) mas tamanhos estáveis.",
    "Erratic":      "Poucos zeros (demanda frequente), porém tamanhos muito voláteis.",
    "Lumpy":        "Muitos zeros E tamanhos muito voláteis (caso mais difícil).",
    "Sem Demanda":  "Nenhuma ocorrência >0 no período analisado."
}

EXPLICACOES_LONGAS = {
    "Regular": (
        "Demanda frequente (ADI baixo) e baixa variabilidade nos tamanhos (CV² baixo). "
        "Geralmente bem atendida por métodos clássicos simples."
    ),
    "Intermittent": (
        "Demanda rara/intermitente (ADI alto), tamanhos relativamente estáveis (CV² baixo). "
        "Modelos específicos como Croston/SBA/TSB costumam ser mais adequados."
    ),
    "Erratic": (
        "Demanda ocorre com frequência (ADI baixo), mas o quanto se consome varia muito de uma ocorrência para outra (CV² alto). "
        "Requer políticas de estoque mais cautelosas e modelos que tolerem alta variabilidade."
    ),
    "Lumpy": (
        "Intermitência alta (ADI alto) combinada com tamanhos muito voláteis (CV² alto). "
        "É o cenário mais desafiador para previsão e reposição — Croston/SBA/TSB e revisão de políticas são recomendáveis."
    ),
    "Sem Demanda": (
        "A série não contém valores positivos no horizonte analisado. ADI é infinito e CV² é indefinido."
    ),
}

def calcular_adi_cv2(y: pd.Series) -> dict:
    y = y.astype(float).fillna(0.0)
    N = len(y)
    positivos = y[y > 0]
    Nz = len(positivos)
    frac_zeros = (N - Nz) / N if N > 0 else np.nan
    adi = (N / Nz) if Nz > 0 else np.inf

    if Nz >= 2:
        media = positivos.mean()
        desvio = positivos.std(ddof=1)
        cv2 = (desvio / media) ** 2 if media > 0 else np.nan
    elif Nz == 1:
        cv2 = 0.0
    else:
        cv2 = np.nan

    return dict(
        N=N, Nz=Nz, FracZeros=frac_zeros,
        ADI=adi, CV2=cv2, CV=np.sqrt(cv2) if pd.notna(cv2) else np.nan
    )

# ===== 3) Leitura =====
df = pd.read_excel(ARQUIVO)
esperadas = {"ds", "y"}
faltando = esperadas - set(df.columns)
if faltando:
    raise ValueError(f"Colunas esperadas {esperadas} não encontradas. Faltando: {faltando}")

df["ds"] = pd.to_datetime(df["ds"])
df = df.sort_values("ds").reset_index(drop=True)

# ===== 4) Métricas =====
m = calcular_adi_cv2(df["y"])
classe = classificar_demanda(m["ADI"], m["CV2"])
exp_curta = EXPLICACOES_CURTAS[classe]
exp_longa = EXPLICACOES_LONGAS[classe]

# ===== 5) Tabela 1 — Sumário das métricas =====
tabela_metricas = pd.DataFrame([{
    "Início": df["ds"].min().date(),
    "Fim": df["ds"].max().date(),
    "N períodos": m["N"],
    "N com demanda >0": m["Nz"],
    "% Zeros": round(100*m["FracZeros"], 2),
    "ADI": (round(m["ADI"], 4) if np.isfinite(m["ADI"]) else "∞"),
    "CV² (positivos)": (round(m["CV2"], 4) if pd.notna(m["CV2"]) else "n/d"),
    "CV": (round(m["CV"], 4) if pd.notna(m["CV"]) else "n/d"),
    "Classe": classe
}])

# ===== 6) Tabela 2 — Regras de classificação (referência visual) =====
tabela_regras = pd.DataFrame([
    {"Classe": "Regular",      "Condição": "ADI < 1,32  e  CV² < 0,49"},
    {"Classe": "Intermittent", "Condição": "ADI ≥ 1,32 e  CV² < 0,49"},
    {"Classe": "Erratic",      "Condição": "ADI < 1,32  e  CV² ≥ 0,49"},
    {"Classe": "Lumpy",        "Condição": "ADI ≥ 1,32 e  CV² ≥ 0,49"},
])

# ===== 7) Tabela 3 — O que a classe significa (curto) =====
tabela_classe = pd.DataFrame([{
    "Classe detectada": classe,
    "Significado (curto)": exp_curta
}])

# ===== 8) Impressão estruturada =====
pd.set_option("display.max_colwidth", 120)

print("\n=== ADI & CV² — Sumário das Métricas ===\n")
print(tabela_metricas.to_string(index=False))

print("\n=== Regras de Classificação (Syntetos & Boylan, 2005) ===\n")
print(tabela_regras.to_string(index=False))

print("\n=== Classe Detectada — Explicação Curta ===\n")
print(tabela_classe.to_string(index=False))

print("\nObservação (explicação detalhada):")
print(fill(f"- {exp_longa}", width=110))

print("\nNotas de interpretação geral:")
print("- ADI alto  → mais intermitência (muitos períodos com zero).")
print("- CV² alto  → tamanhos das demandas positivas muito voláteis (erraticidade).")
print("- Erratic   → baixa intermitência + alta volatilidade dos tamanhos.")
print("- Lumpy     → alta intermitência + alta volatilidade (mais difícil).")


# ===============================================================
# BLOCO 2 — HETEROCEDASTICIDADE E VARIÂNCIA CRESCENTE
# ===============================================================
"""
Neste bloco avaliamos:
- Se a série apresenta heterocedasticidade (variância não constante);
- Se há tendência de crescimento que amplie a dispersão dos valores;
- Se os resíduos seguem distribuição aproximadamente normal.

São usados:
1. Teste de Breusch–Pagan → heterocedasticidade formal.
2. Correlação entre média móvel e desvio padrão (proxy da variância crescente).
3. Shapiro–Wilk → normalidade (opcional, serve para observar assimetria).
"""

# ===== 1) Preparação =====
y = df["y"].astype(float).fillna(0.0).reset_index(drop=True)
t = np.arange(len(y))  # eixo temporal para regressão

# ===== 2) Teste de Breusch–Pagan =====
# modelo de regressão simples y ~ t
modelo = sm.OLS(y, sm.add_constant(t)).fit()
bp_test = sm.stats.diagnostic.het_breuschpagan(modelo.resid, modelo.model.exog)
bp_labels = ["Lagrange Multiplier", "p-valor LM", "F-Estatística", "p-valor F"]
bp_result = dict(zip(bp_labels, bp_test))

# ===== 3) Correlação entre média e desvio padrão móveis =====
rolling_mean = y.rolling(window=5, min_periods=3).mean()
rolling_std = y.rolling(window=5, min_periods=3).std()
corr_rolling = rolling_mean.corr(rolling_std)

# ===== 4) Teste de normalidade (Shapiro–Wilk) =====
shapiro_stat, shapiro_p = shapiro(y)

# ===== 5) Monta tabela principal =====
tabela_hetero = pd.DataFrame([{
    "Teste Breusch–Pagan (p-valor)": round(bp_result["p-valor F"], 6),
    "Correlação(média,std)": round(corr_rolling, 3),
    "Shapiro–Wilk (p-valor)": round(shapiro_p, 6)
}])

# ===== 6) Interpretações automáticas =====
interpret_bp = (
    "Evidência de heterocedasticidade (variância não constante)"
    if bp_result["p-valor F"] < 0.05 else
    "Sem evidência forte de heterocedasticidade"
)

interpret_corr = (
    "Correlação positiva alta — tendência de variância crescente ao longo do tempo"
    if corr_rolling > 0.5 else
    "Correlação baixa — sem forte padrão de crescimento da variância"
)

interpret_shapiro = (
    "Distribuição significativamente diferente da normal (assimetria presente)"
    if shapiro_p < 0.05 else
    "Distribuição aproximadamente normal"
)

# ===== 7) Impressão estruturada =====
print("\n=== Testes de Heterocedasticidade e Variância Crescente ===\n")
print(tabela_hetero.to_string(index=False))

print("\n=== Interpretação dos Indicadores ===")
print(f"- Breusch–Pagan → {interpret_bp}.")
print(f"- Correlação (rolling mean vs std) → {interpret_corr}.")
print(f"- Shapiro–Wilk → {interpret_shapiro}.")

print("\nNotas gerais:")
print("- Heterocedasticidade indica variância não constante, comum em séries com tendência ou sazonalidade.")
print("- Correlação alta entre média e desvio padrão reforça a hipótese de variância crescente (heterocedasticidade condicional).")
print("- Assimetria ou não normalidade também sugerem dispersão crescente e motivam transformações como log ou Box–Cox.")


# ===============================================================
# BLOCO 3 — ASSIMETRIA E POSITIVIDADE (diagnóstico para log)
# ===============================================================
"""
O que medimos aqui:
1) Positividade: mínimo da série e % de valores > 0.
2) Assimetria (Fisher–Pearson) da série original (y>0) e do log(y).
3) Normalidade (D’Agostino K² e Jarque–Bera) na escala original e na escala log.

Leituras gerais:
- skew > 0  → cauda longa à direita (assimetria positiva).
- |skew| ≈ 0 → distribuição aproximadamente simétrica.
- p < 0.05 em testes de normalidade → rejeita normalidade.
"""

# --- prepara série (não altera df original)
y_raw = df["y"].astype(float).copy()
y_min  = float(np.nanmin(y_raw))
pct_pos = float((y_raw > 0).mean())  # fração de positivos
n_total = int(y_raw.shape[0])

# filtramos positivos para medidas comparáveis com log (log só existe para y>0)
y_pos = y_raw[y_raw > 0].dropna()
n_pos = int(y_pos.shape[0])

# --- assimetria (original e log)
skew_raw = float(skew(y_pos, bias=False)) if n_pos >= 3 else np.nan
skew_log = float(skew(np.log(y_pos), bias=False)) if n_pos >= 3 else np.nan

# --- normalidade (D’Agostino K²) — requer n>=20 para boa potência
def k2_p(x):
    x = np.asarray(x)
    if np.sum(np.isfinite(x)) < 8:
        return np.nan  # muito poucos pontos para o teste
    stat, p = normaltest(x)
    return float(p)

k2_raw_p = k2_p(y_pos)
k2_log_p = k2_p(np.log(y_pos))

# --- Jarque–Bera (baseado em skew e kurtose)
def jb_p(x):
    x = np.asarray(x)
    if np.sum(np.isfinite(x)) < 8:
        return np.nan
    stat, p = jarque_bera(x)
    return float(p)

jb_raw_p = jb_p(y_pos)
jb_log_p = jb_p(np.log(y_pos))

# --- monta tabela
tabela_assim = pd.DataFrame([{
    "N_total": n_total,
    "N_positivos": n_pos,
    "Mín(y)": round(y_min, 4),
    "%>0": round(pct_pos, 3),
    "Skew(original)": None if np.isnan(skew_raw) else round(skew_raw, 3),
    "JB_p(original)": None if np.isnan(jb_raw_p) else round(jb_raw_p, 6),
    "K2_p(original)": None if np.isnan(k2_raw_p) else round(k2_raw_p, 6),
    "Skew(log)": None if np.isnan(skew_log) else round(skew_log, 3),
    "JB_p(log)": None if np.isnan(jb_log_p) else round(jb_log_p, 6),
    "K2_p(log)": None if np.isnan(k2_log_p) else round(k2_log_p, 6),
}])

# --- interpretações automáticas (curtas)
interp_pos = (
    "Série estritamente positiva (log aplicável sem deslocamento)."
    if (y_min > 0) else
    "Há zeros/negativos na série (log puro não definido; usar ajuste como log1p/Box–Cox com shift)."
)

def interp_skew(v):
    if np.isnan(v): return "Sem dados suficientes para estimar a assimetria."
    if abs(v) < 0.5:  return f"Assimetria fraca (|skew|={abs(v):.2f})."
    if abs(v) < 1.0:  return f"Assimetria moderada (|skew|={abs(v):.2f})."
    return f"Assimetria forte (|skew|={abs(v):.2f})."

def interp_norm(p, label):
    if np.isnan(p): return f"{label}: amostra muito pequena para o teste."
    return f"{label}: p={p:.4g} → " + ("rejeita normalidade" if p < 0.05 else "não rejeita normalidade")

msg_skew_raw = interp_skew(skew_raw)
msg_skew_log = interp_skew(skew_log)
msg_k2_raw   = interp_norm(k2_raw_p, "K² original")
msg_k2_log   = interp_norm(k2_log_p, "K² log")
msg_jb_raw   = interp_norm(jb_raw_p, "JB original")
msg_jb_log   = interp_norm(jb_log_p, "JB log")

# --- impressão estruturada
print("\n=== Assimetria e Positividade — Diagnóstico para Transformação Log ===\n")
print(tabela_assim.to_string(index=False))

print("\n=== Interpretação dos Indicadores ===")
print(f"- Positividade → {interp_pos}")
print(f"- Assimetria (original) → {msg_skew_raw}")
print(f"- Assimetria (log)      → {msg_skew_log}")
print(f"- Normalidade → {msg_k2_raw}; {msg_jb_raw}.")
print(f"- Normalidade (log) → {msg_k2_log}; {msg_jb_log}.")

print("\nNotas gerais:")
print("- Valores estritamente positivos e assimetria positiva forte são típicos de séries de demanda/receita.")
print("- Transformações monotônicas (log, Box–Cox) tendem a reduzir a assimetria e aproximar a normalidade.")
print("- Compare |skew| antes vs depois do log: redução substancial indica que o log normaliza melhor a distribuição.")


# ================================================================
# 4) Decomposição STL e Força da Tendência/Sazonalidade (Hyndman)
# ---------------------------------------------------------------
# O objetivo aqui é decompor a série (aditiva) e calcular:
#   - TrendStrength  = max(0, 1 - Var(R) / Var(T + R))
#   - SeasonalStrength = max(0, 1 - Var(R) / Var(S + R))
# onde Y = T + S + R (STL aditiva)
#
# Referência: Hyndman & Athanasopoulos (FPP2, Seção 6.7 "Measuring
# strength of trend and seasonality").
#
# Observação:
#  - Usamos STL com 'period' adequado (ex.: 12 para dados mensais).
#  - Este bloco NÃO decide modelo; apenas mede forças e reporta.
# ================================================================

# ---- 4.1 LER A SÉRIE (mesmo padrão das outras rotinas) ----
# ATENÇÃO: ajuste o caminho caso esteja rodando fora do seu Mac.
caminho_arquivo = r"/Users/felipe/Desktop/Faculdade/TCC/Códigos VS Code/Streamlit/(2017-2025) Série Temporal - Prod Cod 7 (A).xlsx"
df = pd.read_excel(caminho_arquivo)

# Garantir nomes e tipos
df = df.rename(columns={"ds": "ds", "y": "y"})  # mantém se já estiver certo
df["ds"] = pd.to_datetime(df["ds"])
df = df.sort_values("ds").reset_index(drop=True)

# Se sua série já é mensal com uma observação por mês, ótimo.
# Caso contrário, aqui está o "padrão" (soma por mês). Comente se não precisar:
# df = df.set_index("ds").resample("MS")["y"].sum().reset_index()

# Definir índice temporal
ts = df.set_index("ds")["y"].astype(float)

# ---- 4.2 DEFINIR 'period' (sazonalidade) para a STL ----
# Para dados mensais, use period = 12. Se for semanal, 52; diário com padrão semanal, 7; etc.
# Se quiser inferir, tente:
freq_inferida = pd.infer_freq(ts.index)
# Heurística: se for mensal (MS/M), usar 12; se None, manter 12 por padrão do seu projeto
if freq_inferida and freq_inferida.upper().startswith("M"):
    period = 12
elif freq_inferida and freq_inferida.upper().startswith("Q"):
    period = 4
elif freq_inferida and freq_inferida.upper().startswith("W"):
    period = 52
elif freq_inferida and freq_inferida.upper().startswith(("D", "B")):
    # diário; ajuste se houver padrão semanal
    period = 7
else:
    period = 12  # padrão mais comum no seu TCC

# ---- 4.3 DECOMPOSIÇÃO STL (aditiva) ----
stl = STL(ts, period=period, robust=True)  # robust=True lida melhor com outliers
res = stl.fit()
T = res.trend
S = res.seasonal
R = res.resid

# ---- 4.4 FORÇAS (Hyndman) ----
# Fórmulas:
# TrendStrength     = max(0, 1 - Var(R)/Var(T + R))
# SeasonalStrength  = max(0, 1 - Var(R)/Var(S + R))
# Notas:
#  - Usamos variância amostral (ddof=1).
#  - Se o denominador ficar ~0 (ex.: série quase sem T+R), protegemos com eps.

eps = 1e-12
var_R = np.var(R, ddof=1)

var_TplusR = np.var((T + R), ddof=1)
var_SplusR = np.var((S + R), ddof=1)

trend_strength = max(0.0, 1.0 - (var_R / max(var_TplusR, eps)))
season_strength = max(0.0, 1.0 - (var_R / max(var_SplusR, eps)))

# ---- 4.5 RESUMO E EXPLICAÇÃO ESTRUTURADA ----
inicio = ts.index.min().date()
fim = ts.index.max().date()
n = ts.shape[0]

summary_tbl = pd.DataFrame({
    "Início": [inicio],
    "Fim": [fim],
    "N períodos": [n],
    "Period(STL)": [period],
    "TrendStrength": [round(trend_strength, 3)],
    "SeasonalStrength": [round(season_strength, 3)]
})

print("\n=== Decomposição STL – Força da Tendência e Sazonalidade ===\n")
print(summary_tbl.to_string(index=False))

# ---- 4.6 INTERPRETAÇÃO (sem decidir modelo) ----
def qualifica_forca(x):
    # Regras simples para leitura (você já usou algo assim nos outros blocos):
    # <0.2 fraca | 0.2–0.5 moderada | >0.5 forte
    if x < 0.2:
        return "fraca"
    elif x < 0.5:
        return "moderada"
    else:
        return "forte"

qs_trend = qualifica_forca(trend_strength)
qs_season = qualifica_forca(season_strength)

print("\n=== Interpretação rápida (Hyndman, força em [0,1]) ===")
print(f"- Tendência: {qs_trend}  (TrendStrength = {trend_strength:.3f}).")
print(f"- Sazonalidade: {qs_season}  (SeasonalStrength = {season_strength:.3f}).")

print("\nNotas de leitura:")
print("- Valores mais próximos de 1 indicam componente mais dominante.")
print("- SeasonalStrength baixa sugere ausência de padrão sazonal relevante.")
print("- TrendStrength alta indica tendência pronunciada na série.")
print("- Esta seção é descritiva; a escolha do modelo será feita em outra etapa.")


# ================================
# 5) ADF & KPSS — Estacionariedade
# ================================

# --------------------------------
# Funções utilitárias (formatação)
# --------------------------------
def _fmt_p(p):
    try:
        return f"{p:.4f}" if p >= 0.0001 else f"{p:.1e}"
    except Exception:
        return str(p)

def _line():
    print()

# --------------------------------
# Leitura dos dados
# --------------------------------
# ATENÇÃO: ajuste o caminho caso necessário (o seu caminho Mac contém acentos e espaços; o pandas lida bem com isso).
caminho_arquivo = r"/Users/felipe/Desktop/Faculdade/TCC/Códigos VS Code/Streamlit/(2017-2025) Série Temporal - Prod Cod 7 (A).xlsx"

# Leitura simples: duas colunas — 'ds' (datas) e 'y' (quantidades)
df = pd.read_excel(caminho_arquivo)

# Higienização básica
df = df[['ds', 'y']].copy()
df['ds'] = pd.to_datetime(df['ds'])
df = df.sort_values('ds')
df['y'] = pd.to_numeric(df['y'], errors='coerce')
df = df.dropna(subset=['y'])
df = df.set_index('ds').asfreq(pd.infer_freq(df.set_index('ds').index) or 'D')  # se não inferir, assume diário
# Se houver lacunas geradas pela frequência, interpolamos suavemente (opcional, mas ajuda os testes)
if df['y'].isna().any():
    df['y'] = df['y'].interpolate(method='time')

y = df['y'].values

# --------------------------------
# Teste ADF (Dickey–Fuller Aumentado)
#   H0: série possui raiz unitária (NÃO estacionária)
#   Se p < 0.05 -> rejeita H0 -> estacionária
# --------------------------------
# Vamos rodar duas variações usuais:
# - 'c'   : com intercepto (nível)
# - 'ct'  : com intercepto e tendência (caso exista tendência)
adf_level = adfuller(y, regression='c', autolag='AIC')     # (stat, pvalue, usedlag, nobs, crit, icbest)
adf_trend = adfuller(y, regression='ct', autolag='AIC')

# --------------------------------
# Teste KPSS (Kwiatkowski–Phillips–Schmidt–Shin)
#   H0: série é estacionária
#   Se p < 0.05 -> rejeita H0 -> NÃO estacionária
# --------------------------------
# Também rodamos duas variações:
# - 'c'   : estacionariedade em nível
# - 'ct'  : estacionariedade em torno de uma tendência determinística
kpss_level_stat, kpss_level_p, kpss_level_lags, kpss_level_crit = kpss(y, regression='c', nlags='auto')   # (stat, pvalue, lags, crit)
kpss_trend_stat, kpss_trend_p, kpss_trend_lags, kpss_trend_crit = kpss(y, regression='ct', nlags='auto')

# --------------------------------
# Tabela de resultados — impressão estruturada
# --------------------------------
inicio = df.index.min().date()
fim    = df.index.max().date()
N      = len(df)

print("=== Testes ADF & KPSS — Diagnóstico de Estacionariedade ===")
print(f"Início         Fim           N obs")
print(f"{inicio}   {fim}   {N}")
_line()

print(">>> ADF — H0: 'possui raiz unitária' (não estacionária) — p < 0.05 → rejeita H0 (estacionária)")
print(f"{'Variação':<20}{'Estatística':>14}{'p-valor':>12}{'Lags':>8}{'N':>8}")
print(f"{'ADF (nível)':<20}{adf_level[0]:>14.3f}{_fmt_p(adf_level[1]):>12}{adf_level[2]:>8}{adf_level[3]:>8}")
print(f"{'ADF (tendência)':<20}{adf_trend[0]:>14.3f}{_fmt_p(adf_trend[1]):>12}{adf_trend[2]:>8}{adf_trend[3]:>8}")
_line()

print(">>> KPSS — H0: 'é estacionária' — p < 0.05 → rejeita H0 (não estacionária)")
print(f"{'Variação':<20}{'Estatística':>14}{'p-valor':>12}{'Lags':>8}")
print(f"{'KPSS (nível)':<20}{kpss_level_stat:>14.3f}{_fmt_p(kpss_level_p):>12}{kpss_level_lags:>8}")
print(f"{'KPSS (tendência)':<20}{kpss_trend_stat:>14.3f}{_fmt_p(kpss_trend_p):>12}{kpss_trend_lags:>8}")
_line()

# --------------------------------
# Interpretação automática (curta)
# --------------------------------
alpha = 0.05
adf_level_decision   = "rejeita H0 (estacionária)" if adf_level[1] < alpha else "não rejeita H0 (não estacionária)"
adf_trend_decision   = "rejeita H0 (estacionária)" if adf_trend[1] < alpha else "não rejeita H0 (não estacionária)"
kpss_level_decision  = "rejeita H0 (não estacionária)" if kpss_level_p < alpha else "não rejeita H0 (estacionária)"
kpss_trend_decision  = "rejeita H0 (não estacionária)" if kpss_trend_p < alpha else "não rejeita H0 (estacionária)"

print("=== Interpretação (curta) ===")
print(f"- ADF (nível):       {adf_level_decision}  | p = {_fmt_p(adf_level[1])}")
print(f"- ADF (tendência):   {adf_trend_decision}  | p = {_fmt_p(adf_trend[1])}")
print(f"- KPSS (nível):      {kpss_level_decision} | p = {_fmt_p(kpss_level_p)}")
print(f"- KPSS (tendência):  {kpss_trend_decision} | p = {_fmt_p(kpss_trend_p)}")
_line()

# --------------------------------
# Diagnóstico combinado (recomendação descritiva, sem executar transformação aqui)
# Padrões clássicos:
#   • ADF p>0.05 (não rejeita H0) e KPSS p<0.05 (rejeita H0)  → evidências de NÃO estacionariedade → considerar diferenciação Δy.
#   • ADF p<0.05 e KPSS p>0.05                                → evidências de estacionariedade.
#   • Ambos rejeitam ou ambos não rejeitam                    → caso ambíguo; avaliar tendência/sazonalidade e ACF/PACF.
adf_any_stationary  = (adf_level[1]   < alpha) or (adf_trend[1]   < alpha)
kpss_any_nonstat    = (kpss_level_p   < alpha) or (kpss_trend_p   < alpha)

if (not adf_any_stationary) and kpss_any_nonstat:
    combinado = "Padrão ADF↑ / KPSS↓ → Forte evidência de NÃO estacionariedade (considere Δy)."
elif adf_any_stationary and (not kpss_any_nonstat):
    combinado = "Padrão ADF↓ / KPSS↑ → Evidência de estacionariedade."
else:
    combinado = "Resultado misto/ambíguo → verifique tendência/sazonalidade e ACF/PACF; teste diferenciação."

print("=== Leitura combinada (ADF + KPSS) ===")
print(combinado)
_line()

print("Notas de leitura:")
print("- ADF testa raiz unitária (H0: não estacionária). p<0.05 → estacionária.")
print("- KPSS testa estacionariedade (H0: estacionária). p<0.05 → não estacionária.")
print("- A combinação ADF (alto p) + KPSS (baixo p) costuma indicar a necessidade de diferenciação (Δ).")


# ================================
# 6) FAC & FACP — Diagnóstico de Dependência Serial
# ================================
# IMPORTS PARA ESTA SEÇÃO
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox

# --------------------------------
# Preparação da série
# --------------------------------
y = df['y'].dropna()
inicio = df.index.min().date()
fim = df.index.max().date()
N = len(y)

# --------------------------------
# Cálculo das funções ACF e PACF
# --------------------------------
nlags = min(40, N // 10)
acf_vals = plot_acf(y, lags=nlags, alpha=0.05)
pacf_vals = plot_pacf(y, lags=nlags, alpha=0.05, method='ywm')

plt.show()

# --------------------------------
# Teste de Ljung-Box
#   H0: série é ruído branco (sem autocorrelação)
# --------------------------------
lb_test = acorr_ljungbox(y, lags=[12], return_df=True)
lb_pvalor = lb_test["lb_pvalue"].iloc[0]

# --------------------------------
# Impressão estruturada
# --------------------------------
print("=== FAC & FACP — Diagnóstico de Dependência Serial ===")
print(f"Início         Fim           N obs     Lags")
print(f"{inicio}   {fim}   {N}        {nlags}")
print()

print(">>> Teste Ljung–Box (lag 12)")
print(f"p-valor = {lb_pvalor:.4f}")
if lb_pvalor < 0.05:
    print("→ Rejeita H₀: há autocorrelação significativa (não é ruído branco).")
else:
    print("→ Não rejeita H₀: série aproxima-se de ruído branco.")
print()

print("=== Interpretação dos Gráficos ===")
print("- FAC (ACF): mostra a correlação entre observações separadas por k períodos.")
print("- FACP (PACF): mostra a correlação líquida após remover efeitos intermediários.")
print("- Pontos fora do intervalo azul (95%) indicam autocorrelação significativa.")
print("- Padrões usuais:")
print("    ▪ FAC decaindo lentamente → série não estacionária (possui memória longa).")
print("    ▪ FACP com 1–2 lags significativos → estrutura AR (autoregressiva).")
print("    ▪ FAC com 1–2 lags significativos → estrutura MA (média móvel).")
print("    ▪ Ambos decaindo juntos → processo misto ARMA/ARIMA.")
print()
print("Notas de leitura:")
print("- Ljung–Box confirma se há correlação remanescente global na série.")
print("- FAC e FACP serão usadas para definir a ordem (p, q) do modelo ARIMA.")


# ================================
# 7) Box–Cox λ (MLE) — Diagnóstico de Transformação
# ================================
# IMPORTS PARA ESTA SEÇÃO
from scipy.stats import boxcox, boxcox_normmax
from scipy.special import inv_boxcox
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------
# Preparação dos dados
# --------------------------------
y = df['y'].dropna()
y_pos = y[y > 0]  # Box-Cox requer valores positivos

# --------------------------------
# Estimação do lambda por Máxima Verossimilhança
# --------------------------------
lambda_mle = boxcox_normmax(y_pos, method='mle')
y_boxcox, fitted_lambda = boxcox(y_pos)

# --------------------------------
# Visualização do perfil de verossimilhança
# --------------------------------
lambdas = np.linspace(-1, 2, 200)
log_likelihood = [np.sum(np.log(np.abs(np.gradient(boxcox(y_pos, l))))) if l != 0 else np.nan for l in lambdas]

plt.figure(figsize=(7,4))
sns.lineplot(x=lambdas, y=log_likelihood, color="royalblue")
plt.axvline(lambda_mle, color='red', linestyle='--', label=f"λ MLE = {lambda_mle:.3f}")
plt.xlabel("Lambda (λ)")
plt.ylabel("Log-Verossimilhança")
plt.title("Perfil de Verossimilhança da Transformação Box–Cox")
plt.legend()
plt.show()

# --------------------------------
# Impressão dos resultados
# --------------------------------
print("=== Box–Cox λ (MLE) — Diagnóstico de Transformação ===")
print(f"Início       Fim         N observações")
print(f"{df.index.min().date()}   {df.index.max().date()}   {len(y_pos)}")
print()
print(f"λ (MLE estimado): {lambda_mle:.4f}")
print()

# Interpretação direta
if abs(lambda_mle) < 0.15:
    interpret = "≈ 0 → transformação logarítmica (log(y)) é adequada."
elif abs(lambda_mle - 1) < 0.15:
    interpret = "≈ 1 → transformação não é necessária (dados já em escala linear)."
else:
    interpret = f"≈ {lambda_mle:.2f} → transformação Box–Cox apropriada com λ={lambda_mle:.2f}."
print("=== Interpretação Automática ===")
print(f"λ (MLE) {interpret}")
print()
print("Notas gerais:")
print("- λ ≈ 0 → aplicar log(y).")
print("- λ ≈ 1 → manter dados originais.")
print("- λ entre 0 e 1 → aplicar Box–Cox com λ estimado (reduz heterocedasticidade e assimetria).")
