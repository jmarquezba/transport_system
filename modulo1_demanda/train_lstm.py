"""
Módulo 1 — Predicción de Demanda (Seq2Seq LSTM)
Arquitectura: Encoder-Decoder con predicción directa multi-paso.
Ventaja clave: predice los 30 días futuros en UN forward pass → sin drift.
"""
import sys, pickle
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

LOOKBACK      = 60    # días de historia como input
FORECAST_DAYS = 30    # días a predecir (salida directa, sin autoregresivo)
BATCH_SIZE    = 64
EPOCHS        = 200
LR            = 1e-3
PATIENCE      = 20
MIN_DAYS      = 30

BASE_DIR   = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
WEBAPP_DIR = BASE_DIR.parent / "webapp" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
WEBAPP_DIR.mkdir(parents=True, exist_ok=True)


# ── Descarga ──────────────────────────────────────────────────────────────────
def download():
    # Try local cache first (avoids network hang)
    cache_path = Path.home() / ".cache" / "kagglehub" / "datasets" / "amanmehra23" / "travel-recommendation-dataset" / "versions" / "1"
    if cache_path.exists():
        raw = cache_path
        print(f"  Usando cache local: {raw}")
    else:
        try:
            import kagglehub
            raw = Path(kagglehub.dataset_download(
                "amanmehra23/travel-recommendation-dataset"))
        except Exception as e:
            sys.exit(f"[ERROR] {e}")
    csvs = {p.stem.lower(): p for p in raw.rglob("*.csv")}
    print("  CSVs:", list(csvs.keys()))
    return {k: pd.read_csv(p) for k, p in csvs.items()}, csvs


# ── Detección de fecha robusta (rechaza IDs enteros) ──────────────────────────
def find_date_col(df):
    for col in df.columns:
        if pd.api.types.is_integer_dtype(df[col]) or \
           pd.api.types.is_float_dtype(df[col]):
            continue
        try:
            parsed = pd.to_datetime(df[col].dropna().head(50), errors="coerce")
            valid  = parsed.dropna()
            if len(valid) < 5: continue
            if (valid.dt.year == 1970).mean() > 0.9: continue
            if (valid.dt.year >= 2000).mean() >= 0.7:
                return col
        except: pass
    return None


# ── Construir series temporales ───────────────────────────────────────────────
MONTH_MAP = {"jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,"jul":7,
             "aug":8,"sep":9,"oct":10,"nov":11,"dec":12}

def parse_months(s):
    if not isinstance(s, str): return list(range(1,13))
    parts = s.replace(",","-").replace(" to ","-").split("-")
    nums  = [MONTH_MAP.get(p.strip().lower()[:3], 0) for p in parts]
    nums  = [n for n in nums if n]
    if len(nums) == 2:
        s2, e2 = nums
        return list(range(s2,13))+list(range(1,e2+1)) if s2>e2 else list(range(s2,e2+1))
    return nums or list(range(1,13))


def build_series(dfs, top_n=5):
    dest_key = next((k for k in dfs if "destination" in k), None)
    hist_key = next((k for k in dfs if "history"     in k), None)
    rev_key  = next((k for k in dfs if "review"      in k), None)
    df_dest  = dfs[dest_key] if dest_key else None
    series   = {}

    # Prioridad 1: fechas reales en history o reviews
    for key in ([hist_key] if hist_key else []) + ([rev_key] if rev_key else []):
        df = dfs[key]
        dc = find_date_col(df)
        if not dc: continue
        if "DestinationID" in df.columns and df_dest is not None and "Name" in df_dest.columns:
            df = df.merge(df_dest[["DestinationID","Name"]].drop_duplicates("DestinationID"),
                         on="DestinationID", how="left")
            dest_c = "Name"
        else:
            dest_c = next((c for c in df.columns if "dest" in c.lower() or "name" in c.lower()), None)
        if not dest_c: continue
        df[dc] = pd.to_datetime(df[dc], errors="coerce")
        df = df.dropna(subset=[dc])
        df["_d"] = df[dc].dt.normalize()
        daily = df.groupby([dest_c,"_d"]).size().reset_index(name="demand")
        days  = daily.groupby(dest_c)["_d"].nunique()
        top   = days[days >= MIN_DAYS].sort_values(ascending=False).head(top_n).index.tolist()
        for dest in top:
            sub = daily[daily[dest_c]==dest].set_index("_d")["demand"].sort_index()
            idx = pd.date_range(sub.index.min(), sub.index.max(), freq="D")
            s   = sub.reindex(idx).interpolate("linear").ffill().bfill().clip(lower=0)
            series[dest] = s
        if series:
            print(f"  Series desde '{key}' con fechas reales ({len(series)} destinos)")
            return series, "real_dates"

    # Fallback: series informadas por Popularity + BestTimeToVisit reales
    assert df_dest is not None and "Popularity" in df_dest.columns
    rng   = np.random.default_rng(SEED)
    dates = pd.date_range("2021-01-01", "2024-12-31", freq="D")
    # Dedup by Name first to get truly unique destinations
    df_dest_unique = df_dest.drop_duplicates(subset=["Name"]).nlargest(top_n, "Popularity")
    top5  = df_dest_unique[["Name","Popularity","BestTimeToVisit"]]
    for _, row in top5.iterrows():
        dest   = str(row["Name"])
        pop    = float(row["Popularity"])
        months = parse_months(str(row["BestTimeToVisit"]))
        base   = pop * 20.0
        vals   = []
        for d in dates:
            season  = 1.4 if d.month in months else 0.75
            weekend = 1.15 if d.dayofweek >= 4 else 0.95
            trend   = 1.0 + (d.year - 2021) * 0.02
            noise   = float(rng.normal(0, base * 0.05))
            vals.append(max(1.0, base * season * weekend * trend + noise))
        series[dest] = pd.Series(vals, index=dates)
    print(f"  Series desde Popularity+BestTimeToVisit reales ({len(series)} destinos)")
    return series, "popularity_informed"


# ── Features temporales (sin/cos encoding) ───────────────────────────────────
def make_features(series: pd.Series) -> np.ndarray:
    """
    Construye matriz [valor_norm, sin_mes, cos_mes, sin_dow, cos_dow].
    Los encodings permiten al LSTM aprender estacionalidad.
    """
    vals = series.values.astype(np.float32)
    idx  = series.index
    sin_m = np.sin(2 * np.pi * idx.month / 12).astype(np.float32)
    cos_m = np.cos(2 * np.pi * idx.month / 12).astype(np.float32)
    sin_d = np.sin(2 * np.pi * idx.dayofweek / 7).astype(np.float32)
    cos_d = np.cos(2 * np.pi * idx.dayofweek / 7).astype(np.float32)
    return np.stack([vals, sin_m, cos_m, sin_d, cos_d], axis=1)  # (T, 5)


# ── Dataset Seq2Seq ───────────────────────────────────────────────────────────
class Seq2SeqDS(Dataset):
    """
    X: ventana de LOOKBACK pasos con 5 features → (LOOKBACK, 5)
    Y: próximos FORECAST_DAYS valores escalados → (FORECAST_DAYS,)
    """
    def __init__(self, feat, scaled_vals, lookback, horizon):
        self.feat = feat          # (T, 5) features
        self.vals = scaled_vals   # (T,)   solo valor escalado
        self.lb   = lookback
        self.hz   = horizon
        # Escalar también las features de valor (col 0) — resto ya están en [-1,1]
        self.X, self.Y = [], []
        for i in range(len(scaled_vals) - lookback - horizon + 1):
            xi = np.copy(feat[i:i+lookback])
            xi[:, 0] = scaled_vals[i:i+lookback]   # reemplazar col 0 con valor escalado
            yi = scaled_vals[i+lookback: i+lookback+horizon]
            self.X.append(xi); self.Y.append(yi)
        self.X = torch.tensor(np.array(self.X, dtype=np.float32))
        self.Y = torch.tensor(np.array(self.Y, dtype=np.float32))
    def __len__(self):  return len(self.X)
    def __getitem__(self, i): return self.X[i], self.Y[i]


# ── Arquitectura Seq2Seq ──────────────────────────────────────────────────────
class Seq2SeqLSTM(nn.Module):
    """
    Encoder: LSTM procesa la ventana histórica.
    Decoder: capa MLP predice todos los FORECAST_DAYS en un solo paso.
    Ventaja: cero error acumulado vs. autoregresivo.
    """
    def __init__(self, input_size=5, hidden=128, layers=3,
                 forecast_steps=30, dropout=0.25):
        super().__init__()
        self.encoder = nn.LSTM(
            input_size, hidden, layers, batch_first=True,
            dropout=dropout if layers > 1 else 0.0)
        self.decoder = nn.Sequential(
            nn.Linear(hidden, 128),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, forecast_steps),
        )

    def forward(self, x):                       # x: (B, T, 5)
        _, (h, _) = self.encoder(x)
        return self.decoder(h[-1])              # (B, forecast_steps)


# ── Entrenamiento ─────────────────────────────────────────────────────────────
def train_dest(dest, series):
    feat = make_features(series)

    sc   = MinMaxScaler((0, 1))
    vals = sc.fit_transform(series.values.reshape(-1,1)).flatten().astype(np.float32)

    ds   = Seq2SeqDS(feat, vals, LOOKBACK, FORECAST_DAYS)
    n    = len(ds)
    i1   = int(n * 0.75); i2 = int(n * 0.88)

    tr   = DataLoader(torch.utils.data.Subset(ds, range(i1)),
                      batch_size=BATCH_SIZE, shuffle=True)
    va   = DataLoader(torch.utils.data.Subset(ds, range(i1, i2)),
                      batch_size=BATCH_SIZE)
    te   = torch.utils.data.Subset(ds, range(i2, n))

    model = Seq2SeqLSTM(forecast_steps=FORECAST_DAYS).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sch   = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=LR, steps_per_epoch=max(len(tr),1), epochs=EPOCHS)
    crit  = nn.HuberLoss(delta=1.0)

    best_vl = float("inf"); best_st = None; pat = 0
    for ep in range(1, EPOCHS+1):
        model.train()
        for xb, yb in tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step()
        model.eval(); vl = 0.0
        with torch.no_grad():
            for xb, yb in va:
                vl += crit(model(xb.to(device)), yb.to(device)).item()*len(xb)
        vl /= max(len(va.dataset), 1)
        if vl < best_vl:
            best_vl = vl; best_st = {k:v.clone() for k,v in model.state_dict().items()}; pat=0
        else:
            pat += 1
            if pat >= PATIENCE: break

    model.load_state_dict(best_st)

    # Eval en test: predecir desde cada ventana de test y comparar con real
    model.eval(); all_pred=[]; all_true=[]
    with torch.no_grad():
        for xb, yb in DataLoader(te, batch_size=64):
            pred = model(xb.to(device)).cpu().numpy()
            all_pred.append(pred); all_true.append(yb.numpy())
    all_pred = np.vstack(all_pred); all_true = np.vstack(all_true)
    # Desnormalizar para métricas (primera columna del forecast)
    p1 = sc.inverse_transform(all_pred[:,0].reshape(-1,1)).flatten()
    t1 = sc.inverse_transform(all_true[:,0].reshape(-1,1)).flatten()
    rmse = float(np.sqrt(mean_squared_error(t1,p1))) if len(p1) else 0.0
    mae_ = float(mean_absolute_error(t1,p1))          if len(p1) else 0.0
    mape_= float(np.mean(np.abs((t1-p1)/(t1+1e-6)))*100) if len(p1) else 0.0

    # Pronóstico 30 días: UNA sola inferencia (no autoregresivo)
    last_feat  = np.copy(feat[-LOOKBACK:])
    last_feat[:, 0] = vals[-LOOKBACK:]
    x_inf = torch.tensor(last_feat[None], dtype=torch.float32).to(device)
    with torch.no_grad():
        fore_sc = model(x_inf).cpu().numpy().flatten()
    forecast = np.clip(sc.inverse_transform(fore_sc.reshape(-1,1)).flatten(), 0, None)

    print(f"  {dest}: RMSE={rmse:.2f} MAE={mae_:.2f} MAPE={mape_:.2f}%  "
          f"[test n={len(p1)}, ep={ep}]")
    return model, sc, rmse, mae_, mape_, forecast


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("="*60)
    print("MODULO 1 - Seq2Seq LSTM (multi-step directo)")
    print("="*60)
    dfs, _ = download()
    series_dict, source = build_series(dfs, top_n=5)
    print(f"  {len(series_dict)} series | fuente: {source}")

    models_state = {}; scalers = {}; routes_meta = {}
    for dest, series in series_dict.items():
        print(f"\nEntrenando -> {dest} ({len(series)} dias)")
        m, sc, rmse, mae_, mape_, fore = train_dest(dest, series)
        models_state[dest] = m.state_dict(); scalers[dest] = sc
        routes_meta[dest] = {
            "n_days": len(series),
            "date_start": str(series.index[0].date()),
            "date_end":   str(series.index[-1].date()),
            "lookback": LOOKBACK, "forecast_days": FORECAST_DAYS,
            "demand_mean": float(series.mean()),
            "demand_max":  float(series.max()),
            "data_source": source,
            "metrics": {"RMSE":rmse,"MAE":mae_,"MAPE (%)":mape_},
            "forecast_30d":   fore.tolist(),
            "last_30d":       series.values[-30:].tolist(),
            "last_30d_dates": [str(d.date()) for d in series.index[-30:]],
        }

    for out in [MODELS_DIR, WEBAPP_DIR]:
        torch.save(models_state, out/"lstm_demanda.pt")
        pickle.dump(scalers,     open(out/"scaler_demanda.pkl","wb"))
        pickle.dump(routes_meta, open(out/"routes_metadata.pkl","wb"))
        print(f"  Guardado -> {out}")

    print("\n[OK] Seq2Seq LSTM entrenado.")
    for d,m in routes_meta.items():
        print(f"  {d}: RMSE={m['metrics']['RMSE']:.2f} MAPE={m['metrics']['MAPE (%)']:.2f}%")

if __name__ == "__main__":
    main()
