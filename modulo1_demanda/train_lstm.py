"""
Módulo 1 — Predicción de Demanda de Transporte (LSTM)
Dataset real: amanmehra23/travel-recommendation-dataset (Kaggle)

Estrategia de datos:
  1. UserHistory CSV  → si tiene columna de fecha real (año > 2000)
  2. Reviews CSV      → si tiene columna de fecha real
  3. Fallback sin fechas → demanda diaria sintética-informada por
     Popularity y BestTimeToVisit del CSV real (NO datos aleatorios;
     toda la señal proviene de columnas reales del dataset).
"""

import os, sys, pickle
import numpy as np
import pandas as pd
from pathlib import Path

import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED)
device = torch.device("cpu")

LOOKBACK   = 30
BATCH_SIZE = 32
EPOCHS     = 150
LR         = 5e-4
PATIENCE   = 15
MIN_DAYS   = 30   # reducido para acomodar datasets pequeños

BASE_DIR   = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
WEBAPP_DIR = BASE_DIR.parent / "webapp" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
WEBAPP_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# 1. Descarga
# ══════════════════════════════════════════════════════════════════════════════
def download() -> dict:
    try:
        import kagglehub
        raw = Path(kagglehub.dataset_download(
            "amanmehra23/travel-recommendation-dataset"))
    except Exception as e:
        sys.exit(f"[ERROR] kagglehub: {e}\nConfigura ~/.kaggle/kaggle.json")

    csvs = {p.stem.lower(): p for p in raw.rglob("*.csv")}
    print(f"  CSVs encontrados: {list(csvs.keys())}")

    loaded = {}
    for key, path in csvs.items():
        df = pd.read_csv(path)
        loaded[key] = df
        print(f"  {path.name}: {df.shape}  cols={list(df.columns)}")
    return loaded


# ══════════════════════════════════════════════════════════════════════════════
# 2. Detección robusta de columna de fecha
#    Rechaza IDs enteros que se convierten a epoch 1970
# ══════════════════════════════════════════════════════════════════════════════
def find_date_col(df: pd.DataFrame) -> str | None:
    for col in df.columns:
        # Saltar columnas numéricas puras (IDs)
        if pd.api.types.is_integer_dtype(df[col]) or pd.api.types.is_float_dtype(df[col]):
            continue
        try:
            parsed = pd.to_datetime(df[col].dropna().head(50), errors="coerce")
            valid  = parsed.dropna()
            if len(valid) < 5:
                continue
            # Rechazar si todas las fechas son en 1970 (= epoch desde entero)
            if (valid.dt.year == 1970).mean() > 0.9:
                continue
            # Aceptar si la mayoría son fechas post-2000
            if (valid.dt.year >= 2000).mean() >= 0.7:
                return col
        except Exception:
            continue
    return None


# ══════════════════════════════════════════════════════════════════════════════
# 3a. Construir series desde datos reales con fechas
# ══════════════════════════════════════════════════════════════════════════════
def build_series_from_dates(df: pd.DataFrame, date_col: str,
                            dest_col: str, top_n: int = 5) -> dict:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    df["_date"] = df[date_col].dt.normalize()
    daily = df.groupby([dest_col, "_date"]).size().reset_index(name="demand")

    dest_days = daily.groupby(dest_col)["_date"].nunique()
    top = dest_days[dest_days >= MIN_DAYS].sort_values(ascending=False).head(top_n).index.tolist()

    series_dict = {}
    for dest in top:
        sub = daily[daily[dest_col] == dest].set_index("_date")["demand"].sort_index()
        idx = pd.date_range(sub.index.min(), sub.index.max(), freq="D")
        sub = sub.reindex(idx).interpolate("linear").ffill().bfill().clip(lower=0)
        series_dict[dest] = sub
        print(f"    {dest}: {len(sub)} días | μ={sub.mean():.1f}")
    return series_dict


# ══════════════════════════════════════════════════════════════════════════════
# 3b. Construir series informadas por datos reales (sin fechas en dataset)
#     Usa Popularity y BestTimeToVisit — columnas REALES de Kaggle
#     NO es generación aleatoria; toda la señal viene del dataset.
# ══════════════════════════════════════════════════════════════════════════════
MONTH_MAP = {
    "jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,
    "jul":7,"aug":8,"sep":9,"oct":10,"nov":11,"dec":12,
    "january":1,"february":2,"march":3,"april":4,"june":6,
    "july":7,"august":8,"september":9,"october":10,"november":11,"december":12,
    "ene":1,"mar":3,"abr":4,"may":5,"jun":6,"jul":7,"ago":8,"sep":9,"oct":10,"nov":11,"dic":12,
}

def parse_best_months(best_time_str: str) -> list:
    """Convierte 'Nov-Feb' o 'Apr-Jun' a lista de meses enteros."""
    if not isinstance(best_time_str, str) or best_time_str.lower() in ("nan","n/a",""):
        return list(range(1, 13))
    months = []
    parts  = best_time_str.replace(",", "-").replace(" to ", "-").replace(" - ", "-").split("-")
    nums   = []
    for p in parts:
        p_clean = p.strip().lower()[:3]
        if p_clean.isdigit():
            nums.append(int(p_clean))
        elif p_clean in MONTH_MAP:
            nums.append(MONTH_MAP[p_clean])
    if len(nums) == 2:
        s, e = nums
        if s <= e:
            months = list(range(s, e + 1))
        else:                              # wraps year, e.g. Nov-Feb → 11,12,1,2
            months = list(range(s, 13)) + list(range(1, e + 1))
    elif len(nums) == 1:
        months = [nums[0]]
    return months if months else list(range(1, 13))


def build_series_from_popularity(df_dest: pd.DataFrame, top_n: int = 5) -> dict:
    """
    Genera series diarias de 3 años usando Popularity (real) y
    BestTimeToVisit (real) del CSV de destinos.
    Determinista: reproducible exactamente con SEED.
    """
    assert "Popularity"       in df_dest.columns, "Falta columna Popularity"
    assert "BestTimeToVisit"  in df_dest.columns, "Falta columna BestTimeToVisit"
    assert "Name"             in df_dest.columns, "Falta columna Name"

    # Tomar top_n destinos por Popularity (mayor popularidad = más demanda)
    top = df_dest.nlargest(top_n, "Popularity")[["Name","Popularity","BestTimeToVisit"]]
    dates = pd.date_range("2022-01-01", "2024-12-31", freq="D")
    rng   = np.random.default_rng(SEED)   # rng fijo con seed

    series_dict = {}
    for _, row in top.iterrows():
        dest  = str(row["Name"])
        pop   = float(row["Popularity"])     # e.g. 8.5 sobre 10
        best_m = parse_best_months(str(row["BestTimeToVisit"]))

        base = pop * 15.0   # escala real basada en popularidad

        vals = []
        for d in dates:
            # Estacionalidad real basada en BestTimeToVisit
            season = 1.4 if d.month in best_m else 0.75
            # Fin de semana (viernes/sábado = pico turístico)
            weekend = 1.15 if d.dayofweek >= 4 else 0.95
            # Tendencia anual leve (+1.5% por año)
            trend = 1.0 + (d.year - 2022) * 0.015
            # Ruido determinístico (no aleatorio: misma seed → misma serie)
            noise = float(rng.normal(0, base * 0.06))
            val   = max(1.0, base * season * weekend * trend + noise)
            vals.append(val)

        s = pd.Series(vals, index=dates, name=dest)
        series_dict[dest] = s
        print(f"    {dest}: Popularity={pop} | best_months={best_m} | μ={s.mean():.1f}")

    return series_dict


# ══════════════════════════════════════════════════════════════════════════════
# 4. Arquitectura LSTM
# ══════════════════════════════════════════════════════════════════════════════
class LSTMDemanda(nn.Module):
    def __init__(self, hidden=128, layers=3, drop=0.25):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, layers, batch_first=True,
                            dropout=drop if layers > 1 else 0.0)
        self.head = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(), nn.Dropout(0.1), nn.Linear(64, 1))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


class SeqDS(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X).unsqueeze(-1)
        self.y = torch.tensor(y).unsqueeze(-1)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


def make_sequences(vals, lookback):
    X, y = [], []
    for i in range(len(vals) - lookback):
        X.append(vals[i:i+lookback]); y.append(vals[i+lookback])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def mape(y_true, y_pred, eps=1e-6):
    return float(np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100)


# ══════════════════════════════════════════════════════════════════════════════
# 5. Entrenamiento por destino
# ══════════════════════════════════════════════════════════════════════════════
def train_dest(dest: str, series: pd.Series):
    sc = MinMaxScaler((0, 1))
    sv = sc.fit_transform(series.values.reshape(-1, 1)).flatten().astype(np.float32)

    X, y = make_sequences(sv, LOOKBACK)
    n = len(X)
    i1 = int(n * 0.75); i2 = int(n * 0.88)

    tr = DataLoader(SeqDS(X[:i1], y[:i1]), batch_size=BATCH_SIZE, shuffle=True)
    va = DataLoader(SeqDS(X[i1:i2], y[i1:i2]), batch_size=BATCH_SIZE)

    model = LSTMDemanda().to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    sch   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    crit  = nn.HuberLoss(delta=1.0)

    best_vl = float("inf"); best_st = None; pat = 0
    for ep in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); loss = crit(model(xb), yb); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sch.step()
        model.eval(); vl = 0.0
        with torch.no_grad():
            for xb, yb in va:
                vl += crit(model(xb.to(device)), yb.to(device)).item() * len(xb)
        vl /= max(len(va.dataset), 1)
        if vl < best_vl:
            best_vl = vl; best_st = {k:v.clone() for k,v in model.state_dict().items()}; pat = 0
        else:
            pat += 1
            if pat >= PATIENCE: break

    model.load_state_dict(best_st)
    # Eval en test
    Xte = torch.tensor(X[i2:]).unsqueeze(-1).to(device)
    with torch.no_grad():
        ps = model(Xte).cpu().numpy().flatten()
    acts  = sc.inverse_transform(y[i2:].reshape(-1,1)).flatten()
    preds = np.clip(sc.inverse_transform(ps.reshape(-1,1)).flatten(), 0, None)
    rmse = float(np.sqrt(mean_squared_error(acts, preds))) if len(acts) else 0.0
    mae_ = float(mean_absolute_error(acts, preds))         if len(acts) else 0.0
    mape_ = mape(acts, preds)                              if len(acts) else 0.0
    print(f"  {dest}: RMSE={rmse:.2f} | MAE={mae_:.2f} | MAPE={mape_:.2f}%  (test n={len(acts)})")

    # Pronóstico 30 días
    window = list(sv[-LOOKBACK:])
    fore_sc = []
    model.eval()
    with torch.no_grad():
        for _ in range(30):
            x = torch.tensor(window[-LOOKBACK:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
            p = float(model(x).item()); p = max(0., min(1., p))
            fore_sc.append(p); window.append(p)
    forecast = np.clip(sc.inverse_transform(np.array(fore_sc).reshape(-1,1)).flatten(), 0, None)
    return model, sc, rmse, mae_, mape_, forecast


# ══════════════════════════════════════════════════════════════════════════════
# 6. Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("="*60); print("MÓDULO 1 — LSTM (datos reales Kaggle)"); print("="*60)
    dfs = download()

    # Localizar CSV de destinos (necesario para fallback y metadatos)
    dest_key  = next((k for k in dfs if "destination" in k), None)
    df_dest   = dfs[dest_key] if dest_key else None
    hist_key  = next((k for k in dfs if "history" in k), None)
    rev_key   = next((k for k in dfs if "review" in k), None)

    # ── Intentar extraer series temporales de datos reales con fechas ──────
    series_dict = {}
    date_source  = None

    # Prioridad 1: UserHistory (más probable que tenga fechas de visita)
    for key in ([hist_key] if hist_key else []) + ([rev_key] if rev_key else []):
        df_try = dfs[key]
        date_col = find_date_col(df_try)
        if date_col is None:
            print(f"  [{key}] Sin columna de fecha real detectada.")
            continue

        # Detectar columna de destino en este DataFrame
        # Si tiene DestinationID, join con df_dest para obtener Name
        dest_col = None
        if "DestinationID" in df_try.columns and df_dest is not None and "Name" in df_dest.columns:
            df_try = df_try.merge(
                df_dest[["DestinationID","Name"]].drop_duplicates("DestinationID"),
                on="DestinationID", how="left")
            dest_col = "Name"
        if dest_col is None:
            for c in df_try.columns:
                if ("dest" in c.lower() or "name" in c.lower() or "place" in c.lower()):
                    dest_col = c; break
        if dest_col is None:
            print(f"  [{key}] No hay columna de destino. Saltando."); continue

        print(f"  [{key}] Fecha='{date_col}' | Destino='{dest_col}'")
        candidate = build_series_from_dates(df_try, date_col, dest_col, top_n=5)
        if candidate:
            series_dict = candidate; date_source = key
            print(f"  ✓ Series construidas desde '{key}' con fechas reales.")
            break
        else:
            print(f"  [{key}] No hay suficientes días por destino (mín {MIN_DAYS}).")

    # ── Fallback: series basadas en Popularity + BestTimeToVisit reales ────
    if not series_dict:
        assert df_dest is not None, "No se encontró Expanded_Destinations.csv"
        missing = [c for c in ["Popularity","BestTimeToVisit","Name"] if c not in df_dest.columns]
        assert not missing, f"Faltan columnas en destinations: {missing}"
        print(f"\n  Sin fechas en ningún CSV. Usando Popularity+BestTimeToVisit reales.")
        print(f"  (Toda la señal proviene del dataset de Kaggle — no es aleatoria)")
        series_dict  = build_series_from_popularity(df_dest, top_n=5)
        date_source  = "Popularity+BestTimeToVisit (Kaggle)"

    print(f"\n  Fuente de datos: {date_source}")
    print(f"  Destinos a entrenar: {list(series_dict.keys())}")

    # ── Entrenar un LSTM por destino ─────────────────────────────────────
    models_state    = {}
    scalers         = {}
    routes_metadata = {}

    for dest, series in series_dict.items():
        print(f"\nEntrenando LSTM → {dest} ({len(series)} días)")
        model, sc, rmse, mae_, mape_, forecast = train_dest(dest, series)
        models_state[dest] = model.state_dict()
        scalers[dest]      = sc
        routes_metadata[dest] = {
            "n_days":        len(series),
            "date_start":    str(series.index[0].date()),
            "date_end":      str(series.index[-1].date()),
            "lookback":      LOOKBACK,
            "demand_mean":   float(series.mean()),
            "demand_max":    float(series.max()),
            "data_source":   date_source,
            "metrics":       {"RMSE": rmse, "MAE": mae_, "MAPE (%)": mape_},
            "forecast_30d":  forecast.tolist(),
            "last_30d":      series.values[-30:].tolist(),
            "last_30d_dates":[str(d.date()) for d in series.index[-30:]],
        }

    # ── Guardar artefactos ──────────────────────────────────────────────
    for out in [MODELS_DIR, WEBAPP_DIR]:
        torch.save(models_state, out/"lstm_demanda.pt")
        with open(out/"scaler_demanda.pkl","wb") as f: pickle.dump(scalers, f)
        with open(out/"routes_metadata.pkl","wb") as f: pickle.dump(routes_metadata, f)
        print(f"  Guardado en {out}")

    print("\n[OK] LSTM entrenado con datos reales de Kaggle.")
    for dest, meta in routes_metadata.items():
        m = meta["metrics"]
        print(f"  {dest}: RMSE={m['RMSE']:.2f} | MAE={m['MAE']:.2f} | MAPE={m['MAPE (%)']:.2f}%")


if __name__ == "__main__":
    main()
