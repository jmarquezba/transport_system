import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder

# Set seeds
np.random.seed(42)
torch.manual_seed(42)

# Load datasets
kaggle_path = r'C:\Users\santy\.cache\kagglehub\datasets\amanmehra23\travel-recommendation-dataset\versions\1'
df_reviews = pd.read_csv(os.path.join(kaggle_path, 'Final_Updated_Expanded_Reviews.csv'))
df_dest = pd.read_csv(os.path.join(kaggle_path, 'Expanded_Destinations.csv'))

# Fit encoders
user_encoder = LabelEncoder()
item_encoder = LabelEncoder()

df_reviews['user_enc'] = user_encoder.fit_transform(df_reviews['UserID'])
df_reviews['item_enc'] = item_encoder.fit_transform(df_reviews['DestinationID'])

n_users = len(user_encoder.classes_)
n_items = len(item_encoder.classes_)

# NCF for Rating Prediction (Regression)
class NCF_Regression(nn.Module):
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
            nn.Linear(32, 1) # Output raw rating value
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

model = NCF_Regression(n_users, n_items, emb_dim=16)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)

# Prepare inputs
users_t = torch.tensor(df_reviews['user_enc'].values, dtype=torch.long)
items_t = torch.tensor(df_reviews['item_enc'].values, dtype=torch.long)
ratings_t = torch.tensor(df_reviews['Rating'].values, dtype=torch.float32)

# Train
model.train()
for epoch in range(150):
    optimizer.zero_grad()
    preds = model(users_t, items_t)
    loss = criterion(preds, ratings_t)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 30 == 0:
        print(f"Epoch {epoch+1}/150 - MSE Loss: {loss.item():.4f}")

# Evaluate personalization for a few users
model.eval()
print("\n--- Top-5 Recommendations for different users ---")
for user_id in [327, 783, 12]:
    user_enc = user_encoder.transform([user_id])[0]
    u_tensor = torch.tensor([user_enc] * n_items, dtype=torch.long)
    i_tensor = torch.tensor(list(range(n_items)), dtype=torch.long)
    
    with torch.no_grad():
        preds = model(u_tensor, i_tensor).numpy()
        
    top_5_idx = np.argsort(preds)[::-1][:5]
    print(f"User {user_id}:")
    for rank, idx in enumerate(top_5_idx):
        item_id = item_encoder.classes_[idx]
        dest_name = df_dest[df_dest['DestinationID'] == item_id]['Name'].values[0]
        print(f"  Rank {rank+1}: DestinationID {item_id} ({dest_name}) - Predicted Rating: {preds[idx]:.2f}")
