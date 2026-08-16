from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib

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
    def forward(self, x):
        return self.network(x)

app = FastAPI(title="Porter Delivery ETA Backend")

preprocessor = joblib.load('preprocessor.joblib')

# Calculate input dimension dynamically from preprocessor
dummy_df = pd.DataFrame([{
    'subtotal': 1000, 'total_items': 2, 'num_distinct_items': 2, 'min_item_price': 400,
    'max_item_price': 600, 'total_onshift_partners': 20, 'total_busy_partners': 10,
    'total_outstanding_orders': 15, 'order_hour': 19, 'order_dayofweek': 4,
    'is_weekend': 0, 'busy_partner_ratio': 0.5, 'partner_load_per_order': 0.75,
    'avg_item_price': 500, 'market_id': '1.0', 'store_primary_category': 'american',
    'order_protocol': '1.0'
}])
input_dim = preprocessor.transform(dummy_df).shape[1]

model = DeliveryTimePredictor(input_dim)
model.load_state_dict(torch.load('porter_nn_model.pth'))
model.eval()

class OrderRequest(BaseModel):
    market_id: str
    store_primary_category: str
    order_protocol: str
    subtotal: float
    total_items: int
    num_distinct_items: int
    min_item_price: float
    max_item_price: float
    total_onshift_partners: float
    total_busy_partners: float
    total_outstanding_orders: float
    created_at_hour: int
    created_at_dayofweek: int

@app.post("/predict")
def predict(order: OrderRequest):
    data = order.dict()
    data['order_hour'] = data.pop('created_at_hour')
    data['order_dayofweek'] = data.pop('created_at_dayofweek')
    data['is_weekend'] = 1 if data['order_dayofweek'] >= 5 else 0
    data['busy_partner_ratio'] = data['total_busy_partners'] / (data['total_onshift_partners'] + 1e-5)
    data['partner_load_per_order'] = data['total_outstanding_orders'] / (data['total_onshift_partners'] + 1e-5)
    data['avg_item_price'] = data['subtotal'] / (data['total_items'] + 1e-5)

    processed = preprocessor.transform(pd.DataFrame([data]))
    with torch.no_grad():
        tensor_in = torch.tensor(processed, dtype=torch.float32)
        prediction = model(tensor_in).item()

    return {"predicted_delivery_time_minutes": round(max(5.0, prediction), 2)}