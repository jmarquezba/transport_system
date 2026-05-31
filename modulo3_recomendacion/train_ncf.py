"""
Módulo 3 — NCF: Sistema de Recomendación de Destinos de Viaje
Dataset real: amanmehra23/travel-recommendation-dataset (Kaggle)
999 reviews | 624 usuarios | 651 destinos

Estrategia adaptada a dataset sparse:
  - Leave-One-Out evaluation (último rating por usuario → test)
  - Negative sampling 4:1 con candidatos controlados
  - Hit Rate@K como métrica principal (más apropiada para sparse)
"""

import os, sys, pickle
import numpy as np
import pandas as pd
from pathlib import Path

import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder

SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED)
device = torch.device("cpu")

BATCH_SIZE = 128; EPOCHS = 60; LR = 0.001; EMB_DIM = 32

BASE_DIR   = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
WEBAPP_DIR = BASE_DIR.parent / "webapp" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
WEBAPP_DIR.mkdir(parents=True, exist_ok=True)


# ── 1. Descarga y carga ──────────────────────────────────────────────────────
def load_data():
    try:
        import kagglehub
        print("Descargando dataset desde Kaggle...")
        raw = Path(kagglehub.dataset_download("amanmehra23/travel-recommendation-dataset"))
    except Exception as e:
        sys.exit(f"[ERROR] kagglehub falló: {e}\nConfigura ~/.kaggle/kaggle.json")

    csvs = {p.stem.lower(): p for p in raw.rglob("*.csv")}
    print(f"  CSVs: {list(csvs.keys())}")

    rev_path  = next((p for k,p in csvs.items() if "review" in k), None)
    dest_path = next((p for k,p in csvs.items() if "destination" in k), None)
    hist_path = next((p for k,p in csvs.items() if "history" in k or "userhistory" in k), None)
    user_path = next((p for k,p in csvs.items() if "user" in k and "review" not in k
                      and "history" not in k), None)

    assert rev_path and dest_path, f"No se encontraron reviews o destinations. CSVs: {list(csvs.keys())}"

    df_reviews = pd.read_csv(rev_path)
    df_dest    = pd.read_csv(dest_path)
    df_hist    = pd.read_csv(hist_path)  if hist_path  else None
    df_users   = pd.read_csv(user_path)  if user_path  else None

    print(f"  Reviews: {df_reviews.shape} | Destinations: {df_dest.shape}")
    if df_hist  is not None: print(f"  UserHistory: {df_hist.shape} | cols={list(df_hist.columns)}")
    if df_users is not None: print(f"  Users: {df_users.shape}")

    return df_reviews, df_dest, df_hist, df_users


# ── 2. Preprocesamiento ──────────────────────────────────────────────────────
def preprocess(df_reviews, df_dest):
    # Detectar columnas user e item
    user_col = next(c for c in df_reviews.columns if "user" in c.lower())
    item_col = next(c for c in df_reviews.columns
                    if "dest" in c.lower() or "item" in c.lower() or "place" in c.lower())
    rating_col = next((c for c in df_reviews.columns
                       if "rating" in c.lower() or "score" in c.lower()), None)

    print(f"  user={user_col} | item={item_col} | rating={rating_col}")

    # Label positivo: si hay rating usa ≥3, si no hay rating cualquier interacción es 1
    if rating_col:
        df_reviews["label"] = (df_reviews[rating_col] >= 3).astype(int)
    else:
        df_reviews["label"] = 1

    # Encoders
    ue = LabelEncoder(); ie = LabelEncoder()
    df_reviews["user_enc"] = ue.fit_transform(df_reviews[user_col])
    df_reviews["item_enc"] = ie.fit_transform(df_reviews[item_col])

    n_users = df_reviews["user_enc"].nunique()
    n_items = df_reviews["item_enc"].nunique()
    print(f"  Usuarios: {n_users} | Items: {n_items} | Reviews: {len(df_reviews)}")

    # Construir catálogo de destinos
    dest_catalog = []
    if "Name" in df_dest.columns:
        for _, row in df_dest.iterrows():
            did = row.get("DestinationID", row.get(item_col, 0))
            if did in ie.classes_:
                enc_val = ie.transform([did])[0]
                dest_catalog.append({
                    "DestinationID": int(did),
                    "enc":           int(enc_val),
                    "Name":          str(row.get("Name", did)),
                    "Type":          str(row.get("Type", "N/A")),
                    "State":         str(row.get("State", "N/A")),
                    "Popularity":    float(row.get("Popularity", 0)),
                    "BestTimeToVisit": str(row.get("BestTimeToVisit", "N/A")),
                    "description":   f"Destino: {row.get('Name', did)}, {row.get('State','India')}.",
                })
    print(f"  Catálogo: {len(dest_catalog)} destinos mapeados")
    return df_reviews, ue, ie, n_users, n_items, dest_catalog, user_col, item_col


# ── 3. Leave-One-Out split ───────────────────────────────────────────────────
def loo_split(df_reviews):
    """
    Leave-One-Out: para cada usuario con ≥2 interacciones positivas,
    el ÚLTIMO item positivo es el item de test. El resto es training.
    """
    positives = df_reviews[df_reviews["label"] == 1].sort_values("user_enc")
    user_counts = positives.groupby("user_enc").size()

    train_rows, test_rows = [], []
    for u, grp in positives.groupby("user_enc"):
        if len(grp) >= 2:
            test_rows.append(grp.iloc[-1:])
            train_rows.append(grp.iloc[:-1])
        else:
            train_rows.append(grp)   # usuario con 1 sola interacción solo va a train

    df_test  = pd.concat(test_rows,  ignore_index=True) if test_rows  else pd.DataFrame()
    df_train_pos = pd.concat(train_rows, ignore_index=True)
    print(f"  Train positivos: {len(df_train_pos)} | Test (LOO): {len(df_test)}")
    return df_train_pos, df_test


def negative_sampling(df_pos, n_users_enc, n_items_enc, n_neg=4):
    """Genera n_neg negativos por cada positivo."""
    user_visited = df_pos.groupby("user_enc")["item_enc"].apply(set).to_dict()
    all_items    = set(range(n_items_enc))
    neg_rows = []
    for _, row in df_pos.iterrows():
        u = int(row["user_enc"])
        vis = user_visited[u]
        cands = list(all_items - vis)
        sample = np.random.choice(cands if len(cands) >= n_neg else list(all_items),
                                  size=n_neg, replace=False)
        for ni in sample:
            neg_rows.append({"user_enc": u, "item_enc": int(ni), "label": 0})
    return pd.concat([df_pos[["user_enc","item_enc","label"]],
                      pd.DataFrame(neg_rows)], ignore_index=True).sample(frac=1, random_state=SEED)


# ── 4. Arquitectura NCF ───────────────────────────────────────────────────────
class NCF(nn.Module):
    def __init__(self, n_users, n_items, emb_dim=32):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim*2, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64),        nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32),         nn.ReLU(),
            nn.Linear(32, 1),          nn.Sigmoid(),
        )
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)
        for l in self.mlp:
            if isinstance(l, nn.Linear):
                nn.init.xavier_uniform_(l.weight); nn.init.zeros_(l.bias)

    def forward(self, u, i):
        return self.mlp(torch.cat([self.user_emb(u), self.item_emb(i)], 1)).squeeze()


class IntDS(Dataset):
    def __init__(self, df):
        self.u = torch.tensor(df["user_enc"].values, dtype=torch.long)
        self.i = torch.tensor(df["item_enc"].values, dtype=torch.long)
        self.y = torch.tensor(df["label"].values,    dtype=torch.float32)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.u[i], self.i[i], self.y[i]


# ── 5. Entrenamiento ──────────────────────────────────────────────────────────
def train_ncf(df_train, n_users, n_items):
    model = NCF(n_users, n_items, EMB_DIM).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    crit  = nn.BCELoss()
    loader = DataLoader(IntDS(df_train), batch_size=BATCH_SIZE, shuffle=True)

    for ep in range(1, EPOCHS+1):
        model.train(); el=0.0
        for u,i,y in loader:
            u,i,y = u.to(device),i.to(device),y.to(device)
            opt.zero_grad(); loss=crit(model(u,i),y); loss.backward(); opt.step()
            el += loss.item()*len(u)
        el /= len(loader.dataset)
        if ep % 10 == 0 or ep == 1:
            print(f"  Epoch {ep:3d}/{EPOCHS} | Loss: {el:.4f}")
    return model


# ── 6. Evaluación Hit Rate@K (apropiada para LOO + sparse) ─────────────────
def evaluate_hit_rate(model, df_test, df_train_pos, n_items):
    """
    Hit Rate@K (HR@K): para cada usuario de test, score de su item real
    vs. K-1 negativos aleatorios. Si el item real está en top-K → hit.
    Mucho más estable que Precision@K con datasets sparse.
    """
    model.eval()
    all_items = list(range(n_items))
    user_train_items = df_train_pos.groupby("user_enc")["item_enc"].apply(set).to_dict()

    hr5=[]; hr10=[]; ndcg10=[]
    for _, row in df_test.iterrows():
        u      = int(row["user_enc"])
        pos_i  = int(row["item_enc"])
        excl   = user_train_items.get(u, set()) | {pos_i}
        negs   = [i for i in all_items if i not in excl]
        # 99 negativos + 1 positivo
        sample_neg = np.random.choice(negs, size=min(99, len(negs)), replace=False).tolist()
        candidates = sample_neg + [pos_i]

        u_t = torch.tensor([u]*len(candidates), dtype=torch.long)
        i_t = torch.tensor(candidates,            dtype=torch.long)
        with torch.no_grad():
            scores = model(u_t, i_t).numpy()
        ranked = np.argsort(scores)[::-1]
        pos_rank = int(np.where(ranked == len(candidates)-1)[0][0])  # rank of the positive item

        hr5.append(1 if pos_rank < 5  else 0)
        hr10.append(1 if pos_rank < 10 else 0)
        ndcg10.append(1.0/np.log2(pos_rank+2) if pos_rank < 10 else 0.0)

    return float(np.mean(hr5)), float(np.mean(hr10)), float(np.mean(ndcg10))


# ── 7. Main ───────────────────────────────────────────────────────────────────
def main():
    print("="*60); print("MÓDULO 3 — NCF (datos reales Kaggle)"); print("="*60)
    df_reviews, df_dest, df_hist, df_users = load_data()
    df_reviews, ue, ie, n_users, n_items, dest_catalog, user_col, item_col = \
        preprocess(df_reviews, df_dest)

    df_train_pos, df_test = loo_split(df_reviews)
    df_train = negative_sampling(df_train_pos, n_users, n_items, n_neg=4)
    print(f"  Train total (pos+neg): {len(df_train)} | Test LOO: {len(df_test)}")

    model = train_ncf(df_train, n_users, n_items)

    if len(df_test) > 0:
        hr5, hr10, ndcg10 = evaluate_hit_rate(model, df_test, df_train_pos, n_items)
        print(f"\nMétricas (Leave-One-Out, 99 negativos aleatorios):")
        print(f"  HR@5   = {hr5:.4f}  (Hit Rate en top-5)")
        print(f"  HR@10  = {hr10:.4f}  (Hit Rate en top-10)")
        print(f"  NDCG@10= {ndcg10:.4f}")
        metrics = {"HR@5":hr5, "HR@10":hr10, "NDCG@10":ndcg10,
                   "n_test_users":len(df_test), "eval_method":"Leave-One-Out (99 negatives)"}
    else:
        print("  Dataset demasiado sparse para eval LOO. Métricas no disponibles.")
        metrics = {"HR@5":0,"HR@10":0,"NDCG@10":0,"n_test_users":0}

    # Mostrar ejemplos de recomendaciones para 3 usuarios
    enc_to_dest = {d["enc"]: d for d in dest_catalog}
    sample_users = list(set(df_train_pos["user_enc"].values[:10]))[:3]
    for u_enc in sample_users:
        u_id = ue.inverse_transform([u_enc])[0]
        visited = set(df_train_pos[df_train_pos["user_enc"]==u_enc]["item_enc"].values)
        cands   = [i for i in range(n_items) if i not in visited]
        u_t = torch.tensor([u_enc]*len(cands), dtype=torch.long)
        i_t = torch.tensor(cands, dtype=torch.long)
        with torch.no_grad():
            scores = model(u_t, i_t).numpy()
        top5 = [cands[i] for i in np.argsort(scores)[::-1][:5]]
        print(f"\n  Usuario {u_id} → Top-5 recomendaciones:")
        for rank, ie_ in enumerate(top5, 1):
            d = enc_to_dest.get(ie_, {"Name": f"Item {ie_}", "Type": "?"})
            print(f"    {rank}. {d['Name']} ({d['Type']}) score={scores[cands.index(ie_)]:.4f}")

    # Guardar
    torch.save(model.state_dict(), MODELS_DIR/"ncf_model.pt")
    meta = {"user_encoder":ue, "item_encoder":ie, "n_users":n_users, "n_items":n_items,
            "dest_catalog":dest_catalog, "metrics":metrics}
    with open(MODELS_DIR/"ncf_metadata.pkl","wb") as f: pickle.dump(meta,f)

    for out in [MODELS_DIR, WEBAPP_DIR]:
        torch.save(model.state_dict(), out/"ncf_model.pt")
        with open(out/"ncf_metadata.pkl","wb") as f: pickle.dump(meta,f)
        print(f"  Guardado en {out}")

    print(f"\n[OK] NCF entrenado | HR@5={hr5:.4f} | HR@10={hr10:.4f} | NDCG@10={ndcg10:.4f}")

if __name__ == "__main__":
    main()
