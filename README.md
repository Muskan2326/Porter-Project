# 🚚 Porter Intra-City Delivery Time Estimator

An end-to-end Machine Learning solution built to estimate intra-city delivery duration (in minutes) for Porter. This project utilizes a custom PyTorch Deep Learning Regression model, served via a FastAPI REST endpoint, and hosted with an interactive Streamlit web dashboard.

---

## 📌 Project Overview

Accurate delivery time prediction is critical for optimizing intra-city logistics operations. Key business objectives include:
- **Improving Customer Experience:** Displaying reliable real-time ETAs.
- **Optimizing Partner Allocation:** Efficiently assigning nearby delivery partners.
- **Identifying Bottlenecks:** Pinpointing high-congestion market areas and order fulfillment delays.
- **Resource Utilization:** Balancing supply-demand dynamics dynamically.

---

## 🏗️ Technical Architecture

- **Data Engineering & Preprocessing:** Handles missing values, performs IQR-based outlier removal, and generates derived metrics (e.g., `busy_partner_ratio`, `partner_load_per_order`, `avg_item_price`).
- **Deep Learning Model:** A 4-layer PyTorch Neural Network trained with Batch Normalization, Dropout (0.2), ReLU activations, and Kaiming Normal weight initialization.
- **Backend Service:** FastAPI REST endpoint (`/predict`) providing high-throughput model inference.
- **Frontend Dashboard:** Streamlit UI allowing real-time input parameter testing and delivery duration calculation.

---

## 🚀 How to Run locally (`localhost`)

Follow these steps to set up and execute the pipeline on your local machine.

### 1. Prerequisites
Ensure you have Python 3.9+ installed on your system.

### 2. Clone the Repository
```bash
git clone [https://github.com/Muskan2326/Porter-Project.git](https://github.com/Muskan2326/Porter-Project.git)
cd Porter-Project
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux / macOS / GitHub Codespaces:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
pip install -r requirements.txt
python train.py
uvicorn app:app --reload
streamlit run streamlit_app.py
