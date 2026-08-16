import os
import zipfile
import urllib.request
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# Set random seeds
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# ==========================================
# 1. NEURAL NETWORK ARCHITECTURE
# ==========================================
class DeliveryTimePredictor(nn.Module):
    def __init__(self, input_dim):
        super(DeliveryTimePredictor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            
            nn.Linear(32, 1)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        return self.network(x)


# ==========================================
# 2. DATA DOWNLOAD & PREPROCESSING
# ==========================================
def download_data():
    zip_path = "dataset.zip"
    csv_path = "dataset.csv"

    if not os.path.exists(csv_path):
        print("📥 Downloading dataset (bypassing 403 Forbidden)...")
        url = "https://d2beiqkhq929f0.cloudfront.net/public_assets/assets/000/015/039/original/dataset.csv.zip?1663710760"
        
        # Headers to prevent 403 Permission Denied
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        
        with urllib.request.urlopen(req) as response, open(zip_path, 'wb') as out_file:
            out_file.write(response.read())

        print("📦 Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(".")
            
    return pd.read_csv(csv_path)

def main():
    df = download_data()

    print("🧹 Cleaning data and engineering features...")
    df['created_at'] = pd.to_datetime(df['created_at'])
    df['actual_delivery_time'] = pd.to_datetime(df['actual_delivery_time'])
    
    # Calculate target (minutes)
    df['delivery_time_min'] = (df['actual_delivery_time'] - df['created_at']).dt.total_seconds() / 60.0
    df = df.dropna(subset=['delivery_time_min'])
    df = df[(df['delivery_time_min'] >= 2) & (df['delivery_time_min'] <= 180)]

    # Feature Engineering
    df['order_hour'] = df['created_at'].dt.hour
    df['order_dayofweek'] = df['created_at'].dt.dayofweek
    df['is_weekend'] = (df['order_dayofweek'] >= 5).astype(int)

    df['busy_partner_ratio'] = df['total_busy_partners'] / (df['total_onshift_partners'] + 1e-5)
    df['partner_load_per_order'] = df['total_outstanding_orders'] / (df['total_onshift_partners'] + 1e-5)
    df['avg_item_price'] = df['subtotal'] / (df['total_items'] + 1e-5)

    num_cols = [
        'subtotal', 'total_items', 'num_distinct_items', 'min_item_price', 'max_item_price',
        'total_onshift_partners', 'total_busy_partners', 'total_outstanding_orders',
        'order_hour', 'order_dayofweek', 'is_weekend', 'busy_partner_ratio', 
        'partner_load_per_order', 'avg_item_price'
    ]
    cat_cols = ['market_id', 'store_primary_category', 'order_protocol']

    # Imputation
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    for col in cat_cols:
        df[col] = df[col].astype(str).fillna('Unknown')

    # Remove IQR Outliers
    q1 = df['delivery_time_min'].quantile(0.25)
    q3 = df['delivery_time_min'].quantile(0.75)
    iqr = q3 - q1
    df = df[(df['delivery_time_min'] >= q1 - 1.5 * iqr) & (df['delivery_time_min'] <= q3 + 1.5 * iqr)]

    # Split and scale
    X = df[num_cols + cat_cols]
    y = df['delivery_time_min'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
        ]
    )

    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    joblib.dump(preprocessor, 'preprocessor.joblib')
    print("💾 Saved preprocessor to 'preprocessor.joblib'")

    # Convert to Tensors
    X_train_t = torch.tensor(X_train_proc, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test_t = torch.tensor(X_test_proc, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=128, shuffle=True)

    input_dim = X_train_proc.shape[1]
    model = DeliveryTimePredictor(input_dim)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    print("🧠 Training PyTorch Neural Network...")
    epochs = 30
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_x.size(0)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"   Epoch [{epoch+1:02d}/{epochs:02d}] - Loss: {running_loss/len(train_loader.dataset):.4f}")

    # Evaluation
    model.eval()
    with torch.no_grad():
        preds = model(X_test_t).numpy().flatten()

    mae = np.mean(np.abs(y_test - preds))
    rmse = np.sqrt(np.mean((y_test - preds) ** 2))

    print(f"\n📊 Performance - MAE: {mae:.2f} mins | RMSE: {rmse:.2f} mins")

    torch.save(model.state_dict(), 'porter_nn_model.pth')
    print("💾 Saved PyTorch weights to 'porter_nn_model.pth'")

if __name__ == "__main__":
    main()