"""
NCF Training Script - Rating Regression Approach
Uses MSE loss to predict actual ratings (1-5) instead of binary classification.
This preserves score granularity needed for personalized ranking.
"""
import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder

np.random.seed(42)
torch.manual_seed(42)

device = torch.device('cpu')
EMB_DIM = 16
EPOCHS = 200
LR = 0.005

class NCF(nn.Module):
    """NCF with regression output (no Sigmoid) for rating prediction."""
    def __init__(self, n_users, n_items, emb_dim=16):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # Raw output, NO sigmoid
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.user_emb.weight, std=0.1)
        nn.init.normal_(self.item_emb.weight, std=0.1)
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, user_ids, item_ids):
        u = self.user_emb(user_ids)
        i = self.item_emb(item_ids)
        x = torch.cat([u, i], dim=1)
        return self.mlp(x).squeeze()


def main():
    print("--- STARTING NCF TRAINING (REGRESSION APPROACH) ---")

    # ── 1. Load Kaggle dataset ──
    try:
        import kagglehub
        kaggle_path = kagglehub.dataset_download('amanmehra23/travel-recommendation-dataset')
    except Exception:
        kaggle_path = os.path.expanduser(
            '~/.cache/kagglehub/datasets/amanmehra23/travel-recommendation-dataset/versions/1'
        )

    reviews_file = os.path.join(kaggle_path, 'Final_Updated_Expanded_Reviews.csv')
    destinations_file = os.path.join(kaggle_path, 'Expanded_Destinations.csv')
    users_file = os.path.join(kaggle_path, 'Final_Updated_Expanded_Users.csv')
    history_file = os.path.join(kaggle_path, 'Final_Updated_Expanded_UserHistory.csv')

    df_reviews = pd.read_csv(reviews_file)
    df_dest = pd.read_csv(destinations_file)
    df_users = pd.read_csv(users_file)
    df_history = pd.read_csv(history_file)

    print(f"Reviews: {len(df_reviews)} | Destinations: {len(df_dest)} | Users: {len(df_users)} | History: {len(df_history)}")

    # ── 2. Combine ALL interactions (Reviews + History) to maximize signal ──
    # Reviews have Rating 1-5
    interactions_reviews = df_reviews[['UserID', 'DestinationID', 'Rating']].copy()

    # History has ExperienceRating 1-5
    interactions_history = df_history[['UserID', 'DestinationID', 'ExperienceRating']].copy()
    interactions_history.rename(columns={'ExperienceRating': 'Rating'}, inplace=True)

    df_all = pd.concat([interactions_reviews, interactions_history], ignore_index=True)
    # Merge with df_dest to get the Name of each destination
    df_all = df_all.merge(df_dest[['DestinationID', 'Name']], on='DestinationID', how='left')
    
    # If user reviewed/visited same destination multiple times, keep mean rating
    df_all = df_all.groupby(['UserID', 'Name'])['Rating'].mean().reset_index()
    print(f"Combined unique interactions: {len(df_all)}")

    # ── 3. Encode users and items ──
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()
    df_all['user_enc'] = user_encoder.fit_transform(df_all['UserID'])
    df_all['item_enc'] = item_encoder.fit_transform(df_all['Name'])

    n_users = len(user_encoder.classes_)
    n_items = len(item_encoder.classes_)
    print(f"Encoded users: {n_users}, Encoded items (Names): {n_items} -> {item_encoder.classes_}")

    # ── 4. Train/Test split (last 20% per user) ──
    train_rows, test_rows = [], []
    for _, group in df_all.groupby('user_enc'):
        n = len(group)
        n_test = max(1, int(n * 0.2))
        test_rows.append(group.iloc[-n_test:])
        train_rows.append(group.iloc[:-n_test])
    df_train = pd.concat(train_rows, ignore_index=True)
    df_test = pd.concat(test_rows, ignore_index=True)
    print(f"Train: {len(df_train)} | Test: {len(df_test)}")

    # ── 5. Tensors ──
    users_t = torch.tensor(df_train['user_enc'].values, dtype=torch.long)
    items_t = torch.tensor(df_train['item_enc'].values, dtype=torch.long)
    ratings_t = torch.tensor(df_train['Rating'].values, dtype=torch.float32)

    # ── 6. Train ──
    model = NCF(n_users, n_items, emb_dim=EMB_DIM).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

    model.train()
    for epoch in range(1, EPOCHS + 1):
        optimizer.zero_grad()
        preds = model(users_t, items_t)
        loss = criterion(preds, ratings_t)
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{EPOCHS} | MSE Loss: {loss.item():.4f}")

    # ── 7. Evaluate personalization ──
    model.eval()
    print("\n--- Verifying personalized recommendations ---")
    sample_users = list(user_encoder.classes_[:5])
    all_top5 = []
    for user_id in sample_users:
        user_enc = user_encoder.transform([user_id])[0]
        u_t = torch.tensor([user_enc] * n_items, dtype=torch.long)
        i_t = torch.tensor(list(range(n_items)), dtype=torch.long)
        with torch.no_grad():
            scores = model(u_t, i_t).numpy()
        top5 = np.argsort(scores)[::-1].tolist()
        all_top5.append(top5)
        top5_names = [str(item_encoder.classes_[idx]) for idx in top5]
        print(f"  User {user_id}: {top5_names} (scores: {[f'{scores[idx]:.2f}' for idx in top5]})")

    # Check diversity across users
    unique_top5 = len(set(tuple(t) for t in all_top5))
    print(f"\n  Unique Top-5 lists across {len(sample_users)} users: {unique_top5}/{len(sample_users)}")

    # ── 8. Build destination catalog (unique by Name) ──
    df_dest_unique = df_dest.drop_duplicates(subset=['Name']).copy()
    dest_catalog = []
    for _, row in df_dest_unique.iterrows():
        dest_catalog.append({
            'DestinationID': int(row['DestinationID']),
            'Name': str(row['Name']),
            'State': str(row['State']),
            'Type': str(row['Type']),
            'Popularity': float(row['Popularity']),
            'BestTimeToVisit': str(row['BestTimeToVisit']),
            'cost_usd': int(300 + row['Popularity'] * 100),
            'description': f"Explora {row['Name']} en {row['State']}. Destino de tipo {row['Type']} con popularidad {row['Popularity']:.1f}/10."
        })

    # ── 9. Save ──
    target_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'webapp', 'models'))
    os.makedirs(target_dir, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(target_dir, 'ncf_model.pt'))
    print(f"\nSaved: ncf_model.pt")

    # Parse user preferences from df_users
    df_users['Preferences'] = df_users['Preferences'].fillna('')
    user_preferences = {}
    for _, row in df_users.iterrows():
        uid = int(row['UserID'])
        prefs = [p.strip() for p in row['Preferences'].split(',') if p.strip()]
        user_preferences[uid] = prefs

    ncf_metadata = {
        'user_encoder': user_encoder,
        'item_encoder': item_encoder,
        'n_users': n_users,
        'n_items': n_items,
        'dest_catalog': dest_catalog,
        'model_type': 'regression',  # Flag so app.py knows
        'user_preferences': user_preferences,  # Add parsed preferences mapping
        'metrics': {
            'Final_MSE': float(loss.item()),
            'n_train': len(df_train),
            'n_test': len(df_test),
        }
    }
    with open(os.path.join(target_dir, 'ncf_metadata.pkl'), 'wb') as f:
        pickle.dump(ncf_metadata, f)
    print(f"Saved: ncf_metadata.pkl with user preferences")

    print("\n--- NCF TRAINING COMPLETED SUCCESSFULLY ---")


if __name__ == '__main__':
    main()
