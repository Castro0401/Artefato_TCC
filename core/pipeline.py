# -*- coding: utf-8 -*-
"""
Pipeline unificado com LOGS: Transformações (original, log+ε, bootstrap FPP) + Modelos
(Croston, SBA, TSB, RandomForest, SARIMAX, LSTM opcional) com registro de experimentos.

Critério do campeão (FPP3 pgs anexadas): minimizar MAE (escala original). Desempates por RMSE.

Saídas:
  - .../experimentos_unificado.xlsx         -> aba "experiments" + aba "champion"
  - .../experimentos_unificado.csv
  - .../champion.csv

🔌 Streamlit hint:
- os prints via `log()` podem ser encaminhados para `st.status()` / `st.write()`
- o DataFrame final pode ser exibido com `st.dataframe()` e disponibilizado para download
"""

import os, time, warnings, itertools, sys, datetime as dt
from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional, Callable, Union

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import STL
from scipy.stats import boxcox, boxcox_normmax

warnings.filterwarnings("ignore")

# ============================
# LOGGING UTIL
# ============================
def _now():
    """Retorna horário hh:mm:ss para prefixar logs."""
    return dt.datetime.now().strftime("%H:%M:%S")

def log(msg: str):
    """
    Logger simples que imprime com timestamp.

    🔌 Streamlit hint:
    - troque por `st.write(msg)` ou acumule mensagens num buffer e mostre no app.
    """
    print(f"[{_now()}] {msg}", flush=True)

class Timer:
    """
    Context manager para medir tempo de blocos de código.
    Exemplo:
        with Timer("Treinando SARIMA"):
            ... código ...
    """
    def __init__(self, label: str):
        self.label = label
        self.t0 = None
    def __enter__(self):
        self.t0 = time.time()
        log(f"▶ {self.label} — início")
        return self
    def __exit__(self, exc_type, exc, tb):
        dt_s = time.time() - self.t0
        log(f"■ {self.label} — fim em {dt_s:.2f}s")

# ============================
# CONFIGS
# ============================
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Grades de hiperparâmetros dos modelos clássicos e ML
CROSTON_ALPHAS = [0.05, 0.1, 0.2, 0.3, 0.5]
SBA_ALPHAS     = [0.05, 0.1, 0.2, 0.3, 0.5]
TSB_ALPHA_GRID = [0.1, 0.3, 0.5]
TSB_BETA_GRID  = [0.1, 0.3, 0.5]

RF_LAGS_GRID = [3, 6, 12]                 # cria lags 1..k
RF_N_ESTIMATORS_GRID = [200, 500]
RF_MAX_DEPTH_GRID = [None, 5, 10]

# grade compacta para SARIMA; pode ser aberta em produção
SARIMA_GRID = {"p":[0,1,2], "d":[0,1], "q":[0,1,2], "P":[0,1], "D":[0,1], "Q":[0,1]}

# LSTM opcional: o código apenas roda se TensorFlow estiver presente
try:
    import tensorflow as tf  # type: ignore
    from tensorflow.keras.models import Sequential  # type: ignore
    from tensorflow.keras.layers import LSTM, Dense  # type: ignore
    KERAS_AVAILABLE = True
except Exception:
    KERAS_AVAILABLE = False

# ============================
# MÉTRICAS (calculadas SEMPRE na escala original quando há inversão)
# ============================
def eval_metrics(y_true, y_pred) -> Dict[str, float]:
    """
    Retorna MAE, MAPE, RMSE, sMAPE em dicionário.
    - MAPE segura para zeros (retorna NaN se não houver valores != 0).
    - sMAPE no formato 2|ŷ - y|/(|y|+|ŷ|); incluída apenas para referência.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        return {"MAE": np.nan, "MAPE": np.nan, "RMSE": np.nan, "sMAPE": np.nan}
    y_true = y_true[mask]; y_pred = y_pred[mask]
    def _mape(a, f):
        m = a != 0
        return np.nan if m.sum() == 0 else 100 * np.mean(np.abs((a[m] - f[m]) / a[m]))
    mae  = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    smap = float(100 * np.mean(2*np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-9)))
    mape = float(_mape(y_true, y_pred))
    return {"MAE": mae, "MAPE": mape, "RMSE": rmse, "sMAPE": smap}

# ============================
# CARREGAMENTO E PADRONIZAÇÃO DA SÉRIE
# ============================
def _load_series_from_excel(file_path: str, sheet_name=None, date_col=None, value_col=None) -> pd.Series:
    """
    Lê arquivo Excel, tenta inferir colunas de data/valor, agrega por mês e
    retorna Série mensal contínua (freq='MS'), preenchida por interpolação.
    """
    log(f"Lendo Excel: {file_path}")
    if sheet_name is None:
        df = pd.read_excel(file_path, sheet_name=0, engine="openpyxl")
    else:
        df = pd.read_excel(file_path, sheet_name=sheet_name, engine="openpyxl")

    # Se vier um dict (múltiplas abas), escolha a primeira não vazia
    if isinstance(df, dict):
        for _, sub in df.items():
            if isinstance(sub, pd.DataFrame) and not sub.empty:
                df = sub; break
    df = df.dropna(axis=1, how='all')

    # Inferência leve dos nomes de colunas
    if date_col is None or value_col is None:
        cand_date  = {"date","data","mes","mês","month","ds"}
        cand_value = {"valor","volume","qtd","quantidade","demand","demanda","y","value"}
        low = {c: str(c).strip().lower() for c in df.columns}
        if date_col is None:
            for c in df.columns:
                if low[c] in cand_date: date_col = c; break
        if value_col is None:
            for c in df.columns:
                if low[c] in cand_value: value_col = c; break
        if date_col is None or value_col is None:
            usable = [c for c in df.columns if df[c].notna().sum() > 0]
            if len(usable) < 2:
                raise ValueError("Planilha precisa ter ao menos 2 colunas (data e valor).")
            date_col  = date_col  or usable[0]
            value_col = value_col or usable[1]

    log(f"Colunas detectadas: data='{date_col}', valor='{value_col}'")

    # Conversões e limpeza
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    vals = pd.to_numeric(df[value_col], errors="coerce")
    # fallback para strings com pontuação BR
    if vals.isna().mean() > 0.2:
        vals = (df[value_col].astype(str).str.replace(".", "", regex=False).str.replace(",", ".", regex=False))
        vals = pd.to_numeric(vals, errors="coerce")
    df[value_col] = vals
    df = df.dropna(subset=[value_col]).sort_values(date_col)

    # Agrega por mês e cria índice MS contínuo
    df["_M"] = df[date_col].dt.to_period("M")
    s = df.groupby("_M")[value_col].sum()
    s.index = s.index.to_timestamp(how="start")
    s = s.asfreq("MS").interpolate("linear").bfill().ffill().astype(float)
    s.name = value_col
    log(f"Série mensal carregada: {len(s)} pontos, de {s.index.min().date()} a {s.index.max().date()}")
    return s

def ensure_monthly_series(df: pd.DataFrame, date_col: str = "ds", value_col: str = "y") -> pd.Series:
    """
    Converte um DataFrame com colunas (ds, y) para Série mensal (freq='MS'),
    agregando por mês e preenchendo lacunas simples.
    """
    s = df.copy()
    s[date_col] = pd.to_datetime(s[date_col], errors="coerce", dayfirst=True)
    s = s[[date_col, value_col]].dropna(subset=[date_col])
    s["__m"] = s[date_col].dt.to_period("M").dt.to_timestamp(how="start")
    s = s.groupby("__m", as_index=True)[value_col].sum().sort_index()
    full_idx = pd.date_range(s.index.min(), s.index.max(), freq="MS")
    s = s.reindex(full_idx).fillna(0.0).astype(float)
    s.name = value_col
    log(f"Série mensal padronizada: {len(s)} pontos")
    return s

def load_series(data_input: Union[str, pd.DataFrame, pd.Series], sheet_name=None, date_col=None, value_col=None) -> pd.Series:
    """
    Entrada flexível:
      - caminho Excel
      - DataFrame com ('ds','y') ou especificando `date_col`/`value_col`
      - Series com índice DatetimeIndex
    """
    if isinstance(data_input, pd.Series):
        s = data_input.copy()
        if s.index.freqstr != "MS": s = s.asfreq("MS")
        s = s.interpolate("linear").bfill().ffill().astype(float)
        log(f"Entrada: Series ({len(s)} pontos)")
        return s
    elif isinstance(data_input, pd.DataFrame):
        log("Entrada: DataFrame")
        if set(["ds","y"]).issubset(set(c.lower() for c in data_input.columns)):
            cols = {c.lower(): c for c in data_input.columns}
            df = data_input.rename(columns={cols["ds"]:"ds", cols["y"]:"y"})
        else:
            df = data_input.copy()
            if date_col is None or value_col is None:
                raise ValueError("Informe 'ds'/'y' ou date_col/value_col.")
        return ensure_monthly_series(df if 'ds' in df.columns else data_input, date_col=date_col or "ds", value_col=value_col or "y")
    else:
        return _load_series_from_excel(str(data_input), sheet_name, date_col, value_col)

# ============================
# TRANSFORMAÇÃO LOG + ε (ε escolhido para reduzir correlação nível-variância)
# ============================
def correlacao_media_desvio(series: pd.Series, window: int = 6) -> float:
    """
    Correlação entre média móvel e desvio-padrão móvel (proxy de heterocedasticidade).
    Quanto mais próximo de zero após a transformação, melhor a estabilização de variância.
    """
    m = series.rolling(window).mean(); s = series.rolling(window).std(ddof=0)
    valid = m.notna() & s.notna()
    return np.nan if valid.sum() < 3 else float(m[valid].corr(s[valid]))

def escolher_epsilon(y: pd.Series, window: int = 6) -> Tuple[float, float, float]:
    """
    Varre uma grade de ε proporcional ao menor positivo de y (após shift)
    e escolhe o que minimiza |corr(média móvel, desvio móvel)| no log.
    Retorna: (epsilon, score, shift aplicado)
    """
    shift = 0.0
    if (y <= 0).any(): shift = float(1 - y.min())  # garante positividade
    y_shift = y + shift
    min_pos = y_shift[y_shift > 0].min()
    base = np.array([0, 1e-6, 0.01, 0.05, 0.1, 0.5, 1.0])
    candidatos = np.unique(base * float(min_pos))
    melhor_eps, melhor_score = None, np.inf
    for eps in candidatos:
        y_log = np.log(y_shift + eps)
        score = abs(correlacao_media_desvio(y_log, window))
        if score < melhor_score: melhor_eps, melhor_score = float(eps), float(score)
    return float(melhor_eps), float(melhor_score), float(shift)

def make_log_transformers(s: pd.Series, window: int = 6):
    """
    Constrói funções forward/inverse para log(y+shift+ε) e seu inverso.
    Retorna também um texto com parâmetros para logging/tabela.
    """
    eps, score, shift = escolher_epsilon(s, window)
    log(f"Transformação LOG: epsilon={eps:.6g}, shift={shift:.6g}, score={score:.4g}")
    def fwd(x: pd.Series) -> pd.Series: return np.log(x.astype(float) + shift + eps)
    def inv(arr: np.ndarray) -> np.ndarray: return np.exp(np.asarray(arr, dtype=float)) - shift - eps
    params_txt = f"epsilon={eps:.6g}, shift={shift:.6g}, score={score:.4g}"
    return fwd, inv, params_txt

# ============================
# BOX–COX + STL + BOOTSTRAP (FPP-style)
# ============================
@dataclass
class BoxCoxParams: 
    lam: float; shift: float; note: str  # armazena λ (MLE), shift e observações

@dataclass
class DecompSTL:
    trend: pd.Series; seasonal: pd.Series; resid: pd.Series
    seasonal_window: int; trend_window: int; robust: bool

def inverse_boxcox(y_bc: np.ndarray, lam: float, shift: float) -> np.ndarray:
    """Inversão de Box–Cox para λ=0 (log) e λ≠0."""
    return (np.exp(y_bc) - shift) if np.isclose(lam, 0.0) else (np.power(lam*y_bc + 1.0, 1.0/lam) - shift)

def _make_odd(n: int) -> int: 
    """STL exige janelas ímpares; ajusta para o próximo ímpar."""
    return int(n) if int(n) % 2 == 1 else int(n) + 1

def _auto_windows(period: int, seasonal_hint: Optional[int] = None, trend_hint: Optional[int] = None) -> Tuple[int, int]:
    """Escolha conservadora de janelas para STL com base no período sazonal."""
    seasonal = _make_odd(max(7, (seasonal_hint or (period + 1))))
    trend = _make_odd(max(period + 1, (trend_hint or (2 * period + 1))))
    if trend <= period: trend = _make_odd(period + 1)
    return seasonal, trend

def fit_boxcox(y: pd.Series, small: float = 1e-6) -> Tuple[np.ndarray, BoxCoxParams]:
    """
    Aplica shift para positivar a série e estima λ por MLE (scipy) -> y_bc.
    Retorna série transformada e parâmetros (λ, shift).
    """
    y = y.astype(float); shift = max(0.0, -float(np.nanmin(y)) + small); y_pos = y + shift
    lam = float(boxcox_normmax(y_pos.values, method="mle"))
    y_bc = boxcox(y_pos.values, lmbda=lam)
    note = (f"Box–Cox MLE λ={lam:.3f}; shift={shift:.6g}")
    return y_bc, BoxCoxParams(lam=lam, shift=shift, note=note)

def decompose_stl(y_bc: np.ndarray, index: pd.DatetimeIndex, period: int = 12, robust: bool = True,
                  seasonal_hint: Optional[int] = None, trend_hint: Optional[int] = None) -> DecompSTL:
    """
    Decomposição STL em espaço Box–Cox: retorna componentes e janelas usadas.
    """
    y_bc_s = pd.Series(y_bc, index=index, name="y_bc")
    seasonal_w, trend_w = _auto_windows(period, seasonal_hint, trend_hint)
    stl = STL(y_bc_s, period=period, robust=robust, seasonal=seasonal_w, trend=trend_w)
    res = stl.fit()
    return DecompSTL(trend=pd.Series(res.trend, index=index, name="trend_bc"),
                     seasonal=pd.Series(res.seasonal, index=index, name="seas_bc"),
                     resid=pd.Series(res.resid, index=index, name="resid_bc"),
                     seasonal_window=seasonal_w, trend_window=trend_w, robust=robust)

def moving_block_bootstrap(resid: np.ndarray, block_size: int, rng: np.random.Generator) -> np.ndarray:
    """
    Bootstrap em blocos móveis: amostra blocos contíguos dos resíduos para preservar dependência temporal.
    """
    n = len(resid)
    if block_size <= 1: return rng.choice(resid, size=n, replace=True)
    starts = rng.integers(0, n, size=(int(np.ceil(n / block_size)) + 1))
    pieces = []
    for st in starts:
        idx = (np.arange(st, st + block_size) % n); pieces.append(resid[idx])
        if sum(map(len, pieces)) >= n: break
    return np.concatenate(pieces)[:n]

def bootstrap_series_list(s: pd.Series, period: int = 12, n_series: int = 20, block_size: int = 24,
                          robust: bool = True, seed: int = 42) -> Tuple[List[pd.Series], Dict[str, object]]:
    """
    Gera `n_series` réplicas sintéticas seguindo FPP:
      (1) Box–Cox (λ via MLE) + shift
      (2) STL robusta
      (3) Bootstrap dos resíduos (blocos)
      (4) Reconstrução e inversão de Box–Cox
    Retorna lista de séries e metadados úteis (λ, shift, janelas).
    """
    log(f"Gerando {n_series} réplicas bootstrap (block={block_size})…")
    idx = s.index
    y_bc, bc_params = fit_boxcox(s)
    stl = decompose_stl(y_bc, idx, period=period, robust=robust)
    rng = np.random.default_rng(seed)
    out: List[pd.Series] = []
    for i in range(1, n_series+1):
        resid_boot = moving_block_bootstrap(stl.resid.values, block_size, rng)
        y_bc_star = stl.trend.values + stl.seasonal.values + resid_boot
        y_star = inverse_boxcox(y_bc_star, lam=bc_params.lam, shift=bc_params.shift)
        y_star = np.clip(y_star, 0.0, None)
        out.append(pd.Series(y_star, index=idx, name=s.name))
        if i == 1 or i % 5 == 0 or i == n_series:
            log(f"… réplica {i}/{n_series} pronta")
    meta = {
        "n_series": n_series, "period": period, "block_size": block_size,
        "boxcox_lambda": bc_params.lam, "boxcox_shift": bc_params.shift,
        "stl_windows": {"seasonal": stl.seasonal_window, "trend": stl.trend_window},
    }
    log(f"Bootstrap concluído: {n_series} réplicas")
    return out, meta

# ============================
# MODELOS PARA DEMANDA INTERMITENTE (Croston/SBA/TSB)
# ============================
def _croston_core(y: np.ndarray, alpha: float = 0.1):
    """
    Núcleo de Croston para estimar componentes de tamanho (z) e intervalo (p).
    Retorna z, p e a série de previsões `f` one-step dentro da amostra.
    """
    y = np.asarray(y, dtype=float); n = len(y)
    z = np.zeros(n); p = np.zeros(n); f = np.zeros(n)
    nz = np.where(y > 0)[0]
    if len(nz) == 0: return z, p, f
    first = nz[0]; z[first] = y[first]; p[first] = 1
    f[:first+1] = z[first] / max(p[first], 1e-9)
    q = 0
    for t in range(first+1, n):
        if y[t] > 0:
            q += 1
            z[t] = z[t-1] + alpha * (y[t] - z[t-1])
            p[t] = p[t-1] + alpha * (q - p[t-1]); q = 0
        else:
            z[t] = z[t-1]; p[t] = p[t-1]; q += 1
        f[t] = z[t] / max(p[t], 1e-9)
    return z, p, f

def croston_forecast(y: np.ndarray, alpha: float, h: int):
    """Croston puro: previsão constante para horizonte h baseada no último f."""
    _, _, f = _croston_core(y, alpha); last_f = f[-1] if len(f) else 0.0
    return f, np.full(h, last_f)

def sba_forecast(y: np.ndarray, alpha: float, h: int):
    """SBA: correção de viés (1 - alpha/2) sobre Croston."""
    _, _, f = _croston_core(y, alpha); f_adj = f * (1 - alpha/2.0)
    last_f = f_adj[-1] if len(f_adj) else 0.0
    return f_adj, np.full(h, last_f)

def tsb_forecast(y: np.ndarray, alpha: float, beta: float, h: int):
    """
    TSB: suaviza separadamente a probabilidade de ocorrência (p) e o tamanho (z).
    Útil quando zeros e positivos se alternam.
    """
    y = np.asarray(y, dtype=float); n = len(y); p = np.zeros(n); z = np.zeros(n); f = np.zeros(n)
    p[0] = 1.0 if np.any(y>0) else 0.0; z[0] = y[y>0].mean() if np.any(y>0) else 0.0; f[0] = p[0]*z[0]
    for t in range(1, n):
        occ = 1.0 if y[t]>0 else 0.0
        p[t] = p[t-1] + beta*(occ - p[t-1])
        z[t] = z[t-1] + alpha*(((y[t] if occ==1.0 else z[t-1]) - z[t-1]))
        f[t] = p[t]*z[t]
    last_f = f[-1] if len(f) else 0.0
    return f, np.full(h, last_f)

# ============================
# SUPERVISÃO PARA RANDOM FOREST
# ============================
def make_supervised_from_series(s: pd.Series, lags: list) -> pd.DataFrame:
    """
    Constrói DataFrame com alvo 'y', lags 1..k e dummies de mês.
    Usado para RandomForest.
    """
    df = pd.DataFrame({"y": s.values}, index=s.index)
    for L in lags: df[f"lag_{L}"] = df["y"].shift(L)
    df["month"] = df.index.month
    df = pd.get_dummies(df, columns=["month"], drop_first=True).dropna()
    return df

# ============================
# LSTM (opcional)
# ============================
def _make_sequences(arr, window):
    """Cria janelas deslizantes (X,y) para séries normalizadas."""
    X, y = [], []
    for i in range(window, len(arr)): X.append(arr[i-window:i]); y.append(arr[i])
    return np.array(X), np.array(y)

def lstm_fit_predict(s: pd.Series, horizon: int, window: int, epochs: int, batch: int):
    """
    Treina LSTM minimalista para previsão one-shot do bloco de teste (último `horizon`).
    Retorna (y_test_invertida, y_pred_invertida, None, runtime).
    """
    if not KERAS_AVAILABLE: raise RuntimeError("TensorFlow/Keras não disponível.")
    values = s.values.reshape(-1,1)
    scaler = MinMaxScaler(); scaled = scaler.fit_transform(values)
    X, y = _make_sequences(scaled, window)
    if len(X) <= horizon: raise RuntimeError("Série insuficiente para LSTM no split.")
    X_train, X_test = X[:-horizon], X[-horizon:]; y_train, y_test = y[:-horizon], y[-horizon:]
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test  = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    model = Sequential([LSTM(64, input_shape=(window,1)), Dense(1)])
    model.compile(optimizer="adam", loss="mse")
    t0 = time.time(); model.fit(X_train, y_train, epochs=epochs, batch_size=batch, verbose=0)
    y_pred_test_scaled = model.predict(X_test, verbose=0); runtime = time.time() - t0
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1,1)).ravel()
    y_pred_inv = scaler.inverse_transform(y_pred_test_scaled).ravel()
    return y_test_inv, y_pred_inv, None, runtime

# ============================
# AVALIAÇÃO DE MODELOS EM UMA SÉRIE (para um dado pré-processamento)
# ============================
@dataclass
class ModelResult:
    """Linha de resultado que comporá a tabela final."""
    preprocess: str; preprocess_params: str; model: str; model_params: str
    mae: float; mape: float; rmse: float; smape_: float
    train_size: int; test_size: int; runtime_s: float

def evaluate_models_on_series(
    base_series: pd.Series, horizon: int, seasonal_period: int,
    preprocess_label: str, preprocess_params: str,
    forward_transform: Optional[Callable[[pd.Series], pd.Series]] = None,
    inverse_transform: Optional[Callable[[np.ndarray], np.ndarray]] = None
) -> List[ModelResult]:
    """
    Executa todos os modelos sobre `base_series`, já aplicando (ou não) uma transformação.
    Se `inverse_transform` é fornecida, as métricas são calculadas na escala original (recomendado).
    Retorna lista de ModelResult (uma linha por combinação de hiperparâmetros/modelo).
    """

    with Timer(f"Testes — {preprocess_label} ({preprocess_params})"):
        # 1) aplica transformação (se houver) e faz sanitização básica
        s_model = forward_transform(base_series) if forward_transform else base_series
        s_model = pd.Series(s_model.values, index=base_series.index, dtype=float)
        s_model = s_model.replace([np.inf, -np.inf], np.nan).interpolate("linear").bfill().ffill()
        if len(s_model.dropna()) < horizon + 2:
            raise ValueError("Série muito curta após preparo. Garanta pelo menos horizon+2 observações.")

        results: List[ModelResult] = []
        hist_all = s_model.iloc[:-horizon].values    # janela de treino
        test_vals = s_model.iloc[-horizon:].values   # janela de teste (holdout)

        def _metrics(y_true_mdl, y_pred_mdl):
            """
            Converte (se preciso) para a escala original antes de computar as métricas.
            """
            if inverse_transform:
                y_true = inverse_transform(y_true_mdl); y_pred = inverse_transform(y_pred_mdl)
            else:
                y_true, y_pred = y_true_mdl, y_pred_mdl
            return eval_metrics(y_true, y_pred)

        # ---- CROSTON
        log(f"→ Croston: {len(CROSTON_ALPHAS)} alphas")
        for j, alpha in enumerate(CROSTON_ALPHAS, 1):
            t0 = time.time()
            # walk-forward simples: a cada passo, prevê 1 e anexa o valor real
            hist = hist_all.copy(); preds = []
            for i in range(horizon):
                _, f1 = croston_forecast(hist, alpha, 1)
                preds.append(f1[0]); hist = np.append(hist, test_vals[i])
            y_pred = np.array(preds, dtype=float)
            if not np.all(np.isfinite(y_pred)):
                log(f"[WARN] {preprocess_label} | Croston alpha={alpha} -> y_pred inválido; pulando.")
                continue
            mets = _metrics(test_vals, y_pred); runtime = time.time() - t0
            results.append(ModelResult(preprocess_label, preprocess_params, "Croston", f"alpha={alpha}",
                                       mets["MAE"], mets["MAPE"], mets["RMSE"], mets["sMAPE"],
                                       len(s_model)-horizon, horizon, runtime))
            if j == 1 or j == len(CROSTON_ALPHAS):
                log(f"… Croston progresso {j}/{len(CROSTON_ALPHAS)}")

        # ---- SBA
        log(f"→ SBA: {len(SBA_ALPHAS)} alphas")
        for j, alpha in enumerate(SBA_ALPHAS, 1):
            t0 = time.time()
            hist = hist_all.copy(); preds = []
            for i in range(horizon):
                _, f1 = sba_forecast(hist, alpha, 1)
                preds.append(f1[0]); hist = np.append(hist, test_vals[i])
            y_pred = np.array(preds, dtype=float)
            if not np.all(np.isfinite(y_pred)):
                log(f"[WARN] {preprocess_label} | SBA alpha={alpha} -> y_pred inválido; pulando.")
                continue
            mets = _metrics(test_vals, y_pred); runtime = time.time() - t0
            results.append(ModelResult(preprocess_label, preprocess_params, "SBA", f"alpha={alpha}",
                                       mets["MAE"], mets["MAPE"], mets["RMSE"], mets["sMAPE"],
                                       len(s_model)-horizon, horizon, runtime))
            if j == 1 or j == len(SBA_ALPHAS):
                log(f"… SBA progresso {j}/{len(SBA_ALPHAS)}")

        # ---- TSB
        tot_tsb = len(TSB_ALPHA_GRID) * len(TSB_BETA_GRID)
        log(f"→ TSB: {tot_tsb} combinações (alpha x beta)")
        step = 0
        for alpha in TSB_ALPHA_GRID:
            for beta in TSB_BETA_GRID:
                step += 1
                t0 = time.time()
                hist = hist_all.copy(); preds = []
                for i in range(horizon):
                    _, f1 = tsb_forecast(hist, alpha, beta, 1)
                    preds.append(f1[0]); hist = np.append(hist, test_vals[i])
                y_pred = np.array(preds, dtype=float)
                if not np.all(np.isfinite(y_pred)):
                    log(f"[WARN] {preprocess_label} | TSB alpha={alpha}, beta={beta} -> y_pred inválido; pulando.")
                    continue
                mets = _metrics(test_vals, y_pred); runtime = time.time() - t0
                results.append(ModelResult(preprocess_label, preprocess_params, "TSB",
                                           f"alpha={alpha}, beta={beta}",
                                           mets["MAE"], mets["MAPE"], mets["RMSE"], mets["sMAPE"],
                                           len(s_model)-horizon, horizon, runtime))
                if step == 1 or step == tot_tsb:
                    log(f"… TSB progresso {step}/{tot_tsb}")

        # ---- RandomForest
        tot_rf = len(RF_LAGS_GRID) * len(RF_N_ESTIMATORS_GRID) * len(RF_MAX_DEPTH_GRID)
        log(f"→ RandomForest: {tot_rf} combinações (lags x n_estimators x max_depth)")
        cnt = 0
        for k in RF_LAGS_GRID:
            for n_est in RF_N_ESTIMATORS_GRID:
                for max_depth in RF_MAX_DEPTH_GRID:
                    cnt += 1
                    lags = list(range(1, k+1))
                    df_sup = make_supervised_from_series(s_model, lags)
                    if len(df_sup) <= horizon:
                        log(f"[WARN] {preprocess_label} | RF lags=1..{k} -> dados insuficientes; pulando.")
                        continue
                    y = df_sup["y"].values
                    X = df_sup.drop(columns=["y"]).values
                    # split simples: últimas `horizon` linhas para teste
                    X_train, X_test = X[:-horizon], X[-horizon:]
                    y_train, y_test = y[:-horizon], y[-horizon:]
                    t0 = time.time()
                    model = RandomForestRegressor(n_estimators=n_est, random_state=RANDOM_STATE, max_depth=max_depth)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test).astype(float); runtime = time.time() - t0
                    if not np.all(np.isfinite(y_pred)):
                        log(f"[WARN] {preprocess_label} | RF lags=1..{k} -> y_pred inválido; pulando.")
                        continue
                    mets = _metrics(y_test, y_pred)
                    results.append(ModelResult(preprocess_label, preprocess_params, "RandomForest",
                                               f"lags=1..{k}, n_estimators={n_est}, max_depth={max_depth}",
                                               mets["MAE"], mets["MAPE"], mets["RMSE"], mets["sMAPE"],
                                               len(s_model)-horizon, horizon, runtime))
                    if cnt == 1 or cnt == tot_rf:
                        log(f"… RF progresso {cnt}/{tot_rf}")

        # ---- SARIMAX
        combos = list(itertools.product(SARIMA_GRID["p"], SARIMA_GRID["d"], SARIMA_GRID["q"],
                                        SARIMA_GRID["P"], SARIMA_GRID["D"], SARIMA_GRID["Q"]))
        log(f"→ SARIMAX: {len(combos)} combinações")
        y_train = s_model.iloc[:-horizon]; y_test  = s_model.iloc[-horizon:]
        for i, (p,d,q,P,D,Q) in enumerate(combos, 1):
            try:
                t0 = time.time()
                model = SARIMAX(y_train, order=(p,d,q), seasonal_order=(P,D,Q,seasonal_period),
                                enforce_stationarity=False, enforce_invertibility=False)
                res = model.fit(disp=False)
                y_pred = res.get_forecast(steps=horizon).predicted_mean.values.astype(float)
                runtime = time.time() - t0
                if not np.all(np.isfinite(y_pred)):
                    log(f"[WARN] {preprocess_label} | SARIMAX({p},{d},{q})x({P},{D},{Q},{seasonal_period}) -> y_pred inválido; pulando.")
                    continue
                mets = _metrics(y_test.values, y_pred)
                params = f"order=({p},{d},{q}), seasonal=({P},{D},{Q},{seasonal_period}), AIC={res.aic:.2f}"
                results.append(ModelResult(preprocess_label, preprocess_params, "SARIMAX",
                                           params, mets["MAE"], mets["MAPE"], mets["RMSE"], mets["sMAPE"],
                                           len(s_model)-horizon, horizon, runtime))
                if i == 1 or i % 10 == 0 or i == len(combos):
                    log(f"… SARIMAX progresso {i}/{len(combos)}")
            except Exception as e:
                # falhas comuns: não convergência; parâmetros não invertíveis etc.
                log(f"[WARN] {preprocess_label} | SARIMAX({p},{d},{q})x({P},{D},{Q},{seasonal_period}) -> erro: {e}")

        # ---- LSTM (se disponível)
        if KERAS_AVAILABLE:
            lstm_combos = [(6,30,16),(12,30,16)]
            log(f"→ LSTM: {len(lstm_combos)} combinações")
            for c_idx, (window,epochs,batch) in enumerate(lstm_combos, 1):
                try:
                    y_test_inv, y_pred_inv, _, runtime = lstm_fit_predict(s_model, horizon, window, epochs, batch)
                    if inverse_transform:
                        # quando a série foi transformada (ex.: log), faz inversão antes de medir
                        y_true = inverse_transform(s_model.iloc[-horizon:].values)
                        y_pred = inverse_transform(y_pred_inv)
                        mets = eval_metrics(y_true, y_pred)
                    else:
                        mets = eval_metrics(y_test_inv, y_pred_inv)
                    params = f"window={window}, epochs={epochs}, batch={batch}, units=64"
                    results.append(ModelResult(preprocess_label, preprocess_params, "LSTM",
                                               params, mets["MAE"], mets["MAPE"], mets["RMSE"], mets["sMAPE"],
                                               len(s_model)-horizon, horizon, runtime))
                    if c_idx == 1 or c_idx == len(lstm_combos):
                        log(f"… LSTM progresso {c_idx}/{len(lstm_combos)}")
                except Exception as e:
                    log(f"[WARN] {preprocess_label} | LSTM window={window} -> erro: {e}")
        else:
            log("[LSTM] TensorFlow não encontrado; pulando LSTM.")

        log(f"✓ Concluídos testes: {preprocess_label} ({len(results)} linhas)")
        return results

# ============================
# RANKEAMENTO E SELEÇÃO DO CAMPEÃO
# ============================
def _simplicity_rank(model_name: str) -> int:
    """
    Ordem de simplicidade (para desempate final). Menor é melhor.
    🔌 ajuste livre se quiser priorizar métodos internos da empresa.
    """
    order = ["NaiveSeasonal","Croston","SBA","TSB","SARIMAX","RandomForest","LSTM"]
    try: return order.index(model_name)
    except ValueError: return len(order)

def select_champion(df: pd.DataFrame) -> pd.Series:
    """
    Critério principal: menor MAE (escala original).
    Desempates: menor RMSE -> menor (MAE+RMSE) -> modelo mais simples.
    Retorna a linha campeã como Series.
    """
    best = df[df["MAE"] == df["MAE"].min()].copy()
    best = best[best["RMSE"] == best["RMSE"].min()]
    if len(best) > 1:
        best["MAE_RMSE_SUM"] = best["MAE"] + best["RMSE"]
        best = best[best["MAE_RMSE_SUM"] == best["MAE_RMSE_SUM"].min()]
    if len(best) > 1:
        best["_simp"] = best["model"].apply(_simplicity_rank)
        best = best[best["_simp"] == best["_simp"].min()]
    return best.iloc[0]

# ============================
# ORQUESTRADOR GERAL
# ============================
def run_full_pipeline(
    data_input: Union[str, pd.DataFrame, pd.Series],
    sheet_name: Optional[str] = None, date_col: Optional[str] = None, value_col: Optional[str] = None,
    horizon: int = 6, seasonal_period: int = 12,
    do_original: bool = True, do_log: bool = True, do_bootstrap: bool = True,
    n_bootstrap: int = 20, bootstrap_block: int = 24,
    save_dir: Optional[str] = None
) -> pd.DataFrame:
    """
    Roda o pipeline completo sobre uma única série:
    - compara 3 pré-processamentos (original / log+ε / bootstrap FPP)
    - treina/avalia todos os modelos
    - compila a tabela e escolhe o campeão

    🔌 Streamlit hint:
    - Exponha `horizon`, `seasonal_period`, flags `do_*`, `n_bootstrap` e `bootstrap_block` como widgets.
    - Conecte `save_dir` a um diretório temporário e ofereça botões de download.
    """
    log("==== PIPELINE INICIADO ====")
    log(f"Params: horizon={horizon}, season={seasonal_period}, original={do_original}, log={do_log}, bootstrap={do_bootstrap}")
    if do_bootstrap: log(f"Bootstrap: n_replicas={n_bootstrap}, block={bootstrap_block}")
    log(f"LSTM disponível: {KERAS_AVAILABLE}")

    base_series = load_series(data_input, sheet_name=sheet_name, date_col=date_col, value_col=value_col)

    all_results: List[ModelResult] = []

    # ORIGINAL
    if do_original:
        log("Realizando testes da série ORIGINAL…")
        all_results += evaluate_models_on_series(
            base_series=base_series, horizon=horizon, seasonal_period=seasonal_period,
            preprocess_label="original", preprocess_params="-",
            forward_transform=None, inverse_transform=None
        )

    # LOG + ε
    if do_log:
        log("Preparando transformação LOG…")
        fwd, inv, params_txt = make_log_transformers(base_series, window=6)
        log("Realizando testes da série LOG-transformada…")
        all_results += evaluate_models_on_series(
            base_series=base_series, horizon=horizon, seasonal_period=seasonal_period,
            preprocess_label="log", preprocess_params=params_txt,
            forward_transform=fwd, inverse_transform=inv
        )

    # BOOTSTRAP
    if do_bootstrap:
        with Timer("Geração das réplicas sintéticas (bootstrap)"):
            series_list, meta = bootstrap_series_list(
                base_series, period=seasonal_period, n_series=n_bootstrap,
                block_size=bootstrap_block, robust=True, seed=RANDOM_STATE
            )
        log(f"{meta['n_series']} réplicas geradas (λ={meta['boxcox_lambda']:.3f}, shift={meta['boxcox_shift']:.6g})")

        for i, s_rep in enumerate(series_list, start=1):
            log(f"Realizando testes da SÉRIE SINTÉTICA {i}/{len(series_list)}…")
            params_txt = (f"replica={i}, block_size={bootstrap_block}, "
                          f"lambda={meta['boxcox_lambda']:.3f}, shift={meta['boxcox_shift']:.6g}")
            all_results += evaluate_models_on_series(
                base_series=s_rep, horizon=horizon, seasonal_period=seasonal_period,
                preprocess_label="bootstrap", preprocess_params=params_txt,
                forward_transform=None, inverse_transform=None
            )

    # Consolida a lista de ModelResult em DataFrame de experimentos
    rows = [{
        "preprocess": r.preprocess, "preprocess_params": r.preprocess_params,
        "model": r.model, "model_params": r.model_params,
        "MAE": r.mae, "MAPE": r.mape, "RMSE": r.rmse, "sMAPE": r.smape_,
        "Train Size": r.train_size, "Test Size": r.test_size, "Runtime (s)": r.runtime_s,
    } for r in all_results]

    df_out = pd.DataFrame(rows)

    # Seleção do CAMPEÃO segundo FPP3 (menor MAE; desempates RMSE, soma, simplicidade)
    champion = select_champion(df_out)
    log("===== CAMPEÃO (critério: menor MAE; desempates por RMSE/soma/simplicidade) =====")
    log(f"Preprocess: {champion['preprocess']} | Params: {champion['preprocess_params']}")
    log(f"Modelo: {champion['model']} | Hiperparâmetros: {champion['model_params']}")
    log(f"MAE={champion['MAE']:.6g} | RMSE={champion['RMSE']:.6g} | MAPE={champion['MAPE']:.6g} | sMAPE={champion['sMAPE']:.6g}")

    # Ordenação leve para visualização (não afeta o campeão já escolhido)
    df_out = df_out.sort_values(by=["preprocess","model","MAE","RMSE"]).reset_index(drop=True)

    # Persistência dos resultados
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        xlsx_path = os.path.join(save_dir, "experimentos_unificado.xlsx")
        csv_path  = os.path.join(save_dir, "experimentos_unificado.csv")
        champion_path = os.path.join(save_dir, "champion.csv")
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            df_out.to_excel(writer, sheet_name="experiments", index=False)
            pd.DataFrame([champion]).to_excel(writer, sheet_name="champion", index=False)
        df_out.to_csv(csv_path, index=False)
        pd.DataFrame([champion]).to_csv(champion_path, index=False)
        log(f"[OK] Resultados salvos em:\n - {xlsx_path}\n - {csv_path}\n - {champion_path}")

    log(f"==== PIPELINE FINALIZADO ====\nLinhas totais de experimentos: {len(df_out)}")
    log("Resumo rápido por preprocess:")
    resumo = df_out.groupby("preprocess").size().to_dict()
    for k, v in resumo.items():
        log(f"  • {k}: {v} linhas")

    # Guarda o campeão como atributo do DataFrame para acesso rápido no app
    df_out.attrs["champion"] = champion.to_dict()
    return df_out

# ============================
# EXECUÇÃO LOCAL (EXEMPLO)
# ============================
if __name__ == "__main__":
    # 🔌 Streamlit hint:
    # - No app, essas strings viram parâmetros (input file uploader e pasta de saída).
    CAMINHO = r"C:\Users\vitor\OneDrive\TCC\Códigos VSCODE\Séries Temporais\Série Temporal - Prod Cod 7 (A).xlsx"
    SAIDA   = r"C:\Users\vitor\OneDrive\TCC\Códigos VSCODE\Séries Temporais"

    with Timer("Rodando pipeline completo"):
        resultados = run_full_pipeline(
            data_input=CAMINHO,
            sheet_name=None, date_col=None, value_col=None,
            horizon=6, seasonal_period=12,
            do_original=True, do_log=True, do_bootstrap=True,
            n_bootstrap=20,         # ajuste livre
            bootstrap_block=24,     # referência p/ mensal
            save_dir=SAIDA
        )

    # Pré-visualização e log do campeão
    log("Prévia do resultado (top 10 linhas):")
    print(resultados.head(10).to_string(index=False))
    champ = resultados.attrs.get("champion", {})
    if champ:
        log("RESUMO CAMPEÃO:")
        for k, v in champ.items():
            log(f"  {k}: {v}")
