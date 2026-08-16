import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
import tensorflow as tf
import joblib

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('porter_nn_model.h5')
feature_columns = joblib.load('feature_columns.pkl')
# Load the scaler
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return 'Delivery Time Prediction API'

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)

        # Convert input data to DataFrame
        input_df = pd.DataFrame([data])

        if 'created_at' in input_df.columns:
            input_df['created_at'] = pd.to_datetime(input_df['created_at'])
            input_df['hour_of_day'] = input_df['created_at'].dt.hour
            input_df['day_of_week'] = input_df['created_at'].dt.dayofweek
            input_df['month'] = input_df['created_at'].dt.month
            input_df = input_df.drop(columns=['created_at'])
        else:

            pass # Assume hour_of_day, day_of_week, month are provided or handled upstream

        # One-hot encode 'store_primary_category'
        if 'store_primary_category' in input_df.columns:
            
            all_categories = [col.replace('category_', '') for col in feature_columns if col.startswith('category_')]
            for cat in all_categories:
                input_df[f'category_{cat}'] = (input_df['store_primary_category'] == cat).astype(int)
            input_df = input_df.drop(columns=['store_primary_category'])
        
        
        numerical_cols_for_inference = ['market_id', 'order_protocol', 'total_items', 'subtotal', 'num_distinct_items',
                                  'min_item_price', 'max_item_price', 'total_onshift_partners', 'total_busy_partners', 'total_outstanding_orders']
        for col in numerical_cols_for_inference:
            if col not in input_df.columns:
                input_df[col] = 0 # Or use a default/mean from training

        # Align columns with training data - this is crucial for one-hot encoded features
        # Create a DataFrame with all expected feature columns, filled with zeros
        processed_input = pd.DataFrame(0, index=[0], columns=feature_columns)
        
        # Fill in the values from the input
        for col in input_df.columns:
            if col in processed_input.columns:
                processed_input[col] = input_df[col]

        # Scale numerical features
        # Assuming `numerical_cols_for_inference` covers the numerical columns that were scaled
        processed_input[numerical_cols_for_inference] = scaler.transform(processed_input[numerical_cols_for_inference])

        # Make prediction
        prediction = model.predict(processed_input.values)
        
        return jsonify({'predicted_delivery_time_minutes': float(prediction[0][0])})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)