import pickle, torch, numpy as np, torch.nn as nn
from pathlib import Path

MODELS_DIR = Path(r'c:\Users\santy\OneDrive\Escritorio\universidad\Semestre 9\Redes Neuronales\Trabajo3_IRNA\webapp\models')

class NCF(nn.Module):
    def __init__(self, n_users, n_items, emb_dim=16):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim * 2, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32),          nn.ReLU(),
            nn.Linear(32, 1),
        )
    def forward(self, user_ids, item_ids):
        return self.mlp(torch.cat([self.user_emb(user_ids),
                                   self.item_emb(item_ids)], dim=1)).squeeze()

with open(MODELS_DIR / 'ncf_metadata.pkl', 'rb') as f:
    meta = pickle.load(f)

print(f"n_users={meta['n_users']}, n_items={meta['n_items']}, type={meta.get('model_type')}")
print(f"metrics={meta['metrics']}")
print(f"dest_catalog len={len(meta['dest_catalog'])}")

model = NCF(meta['n_users'], meta['n_items'], emb_dim=16)
model.load_state_dict(torch.load(MODELS_DIR / 'ncf_model.pt', map_location='cpu'))
model.eval()
print("[OK] Model loaded successfully")

# Test 5 different users
test_users = [1, 50, 100, 300, 500]
user_enc = meta['user_encoder']
item_enc = meta['item_encoder']
dest_cat = meta['dest_catalog']

for uid in test_users:
    if uid not in user_enc.classes_:
        print(f"User {uid} not in dataset, skipping")
        continue
    ue = int(user_enc.transform([uid])[0])
    valid_items = []
    valid_encs = []
    for item in dest_cat:
        did = item['DestinationID']
        if did in item_enc.classes_:
            valid_items.append(item)
            valid_encs.append(int(item_enc.transform([did])[0]))

    u_t = torch.tensor([ue]*len(valid_encs), dtype=torch.long)
    i_t = torch.tensor(valid_encs, dtype=torch.long)
    with torch.no_grad():
        scores = model(u_t, i_t).numpy()

    # Deduplicate by name
    sorted_idx = np.argsort(scores)[::-1]
    seen = set()
    top = []
    for idx in sorted_idx:
        name = valid_items[idx]['Name']
        if name not in seen:
            seen.add(name)
            top.append((name, float(scores[idx])))
        if len(top) >= 5:
            break
    print(f"User {uid}: {[(n, f'{s:.2f}') for n, s in top]}")

print("\n[OK] Verification complete")
