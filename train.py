import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
import  joblib

# Define the URL to the dataset
dataset_url = "https://d2beiqkhq929f0.cloudfront.net/public_assets/assets/000/015/039/original/dataset.csv.zip"

# Load the dataset into a pandas DataFrame
df = pd.read_csv(dataset_url, compression='zip')

# The first 5 rows of the DataFrame
df.head()
# DataFrame information to check data types and non-null counts
df.info()

# Check for null values in each column
df.isnull().sum()
# Convert timestamp columns to datetime objects
df['created_at'] = pd.to_datetime(df['created_at'])
df['actual_delivery_time'] = pd.to_datetime(df['actual_delivery_time'])

# Calculate delivery time in minutes
df['delivery_time_minutes'] = (df['actual_delivery_time'] - df['created_at']).dt.total_seconds() / 60

# Display the first few rows with the new column
df[['created_at', 'actual_delivery_time', 'delivery_time_minutes']].head()

# Check for nulls again, especially for the new column
df.isnull().sum()
# Drop rows where delivery_time_minutes is null (these are the rows where actual_delivery_time was null)
df.dropna(subset=['delivery_time_minutes'], inplace=True)

# Impute missing numerical values with the mean
# Columns to impute: market_id, order_protocol, total_onshift_partners, total_busy_partners, total_outstanding_orders
for col in ['market_id', 'order_protocol', 'total_onshift_partners', 'total_busy_partners', 'total_outstanding_orders']:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].mean()) # Modified to avoid FutureWarning

# Impute missing categorical values for 'store_primary_category' with 'Unknown'
df['store_primary_category'] = df['store_primary_category'].fillna('Unknown') # Modified to avoid FutureWarning

# Verify that there are no more missing values
df.isnull().sum()
# Extract time-based features from 'created_at'
df['hour_of_day'] = df['created_at'].dt.hour
df['day_of_week'] = df['created_at'].dt.dayofweek
df['month'] = df['created_at'].dt.month

# Encode 'store_primary_category' using one-hot encoding
df = pd.get_dummies(df, columns=['store_primary_category'], prefix='category')

# Display the first few rows with the new features
df[['created_at', 'hour_of_day', 'day_of_week', 'month', 'category_american', 'category_mexican']].head()

# Define features (X) and target (y)
# Drop 'created_at', 'actual_delivery_time', 'store_id' as they are not direct features or have been processed
X = df.drop(columns=['created_at', 'actual_delivery_time', 'store_id', 'delivery_time_minutes'])
y = df['delivery_time_minutes']

# Identify numerical columns for scaling (excluding the newly created time features which are already numerical and one-hot encoded features)
numerical_cols = ['market_id', 'order_protocol', 'total_items', 'subtotal', 'num_distinct_items',
                  'min_item_price', 'max_item_price', 'total_onshift_partners', 'total_busy_partners', 'total_outstanding_orders']

# Ensure all numerical_cols are present in X before scaling
numerical_cols = [col for col in numerical_cols if col in X.columns]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numerical features
scaler = StandardScaler()
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

# Display the shapes of the datasets and a sample of the scaled training data
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")
X_train.head()


# Define the Neural Network model
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)  # Output layer for regression
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Display model summary
model.summary()

# Train the model
history = model.fit(
    X_train,
    y_train,
    epochs=20,  # You can adjust the number of epochs
    batch_size=256,
    validation_data=(X_test, y_test),
    verbose=1
)

# Evaluate the model on the test data
loss, mae = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss (Mean Squared Error): {loss:.2f}")
print(f"Test Mean Absolute Error: {mae:.2f}")



model.save("porter_nn_model.h5")
print("Training complete. Model and scaler saved successfully!")

# Save the StandardScaler
joblib.dump(scaler, 'scaler.pkl')
print('StandardScaler saved as scaler.pkl')

# Save the feature columns (column names of X_train) for consistent preprocessing in the API
joblib.dump(X_train.columns.tolist(), 'feature_columns.pkl')
print('Feature columns saved as feature_columns.pkl')