import streamlit as st
import requests

st.title("🚚 Porter Delivery ETA Estimator")

c1, c2 = st.columns(2)
with c1:
    market_id = st.selectbox("Market ID", ["1.0", "2.0", "3.0", "4.0", "5.0", "6.0"])
    store_cat = st.text_input("Store Category", "american")
    protocol = st.selectbox("Order Protocol", ["1.0", "2.0", "3.0", "4.0"])
    subtotal = st.number_input("Subtotal (cents)", value=2000)
    total_items = st.number_input("Total Items", value=3)
    num_distinct = st.number_input("Distinct Items", value=2)

with c2:
    min_price = st.number_input("Min Item Price", value=400)
    max_price = st.number_input("Max Item Price", value=1000)
    onshift = st.number_input("Onshift Partners", value=25)
    busy = st.number_input("Busy Partners", value=15)
    outstanding = st.number_input("Outstanding Orders", value=30)
    hour = st.slider("Hour of Day", 0, 23, 19)
    day = st.slider("Day of Week (0=Mon, 6=Sun)", 0, 6, 3)

if st.button("Calculate Delivery Time"):
    payload = {
        "market_id": market_id, "store_primary_category": store_cat, "order_protocol": protocol,
        "subtotal": subtotal, "total_items": total_items, "num_distinct_items": num_distinct,
        "min_item_price": min_price, "max_item_price": max_price, "total_onshift_partners": onshift,
        "total_busy_partners": busy, "total_outstanding_orders": outstanding,
        "created_at_hour": hour, "created_at_dayofweek": day
    }
    res = requests.post("http://127.0.0.1:8000/predict", json=payload)
    if res.status_code == 200:
        st.success(f"⏱️ **Estimated Delivery Time:** {res.json()['predicted_delivery_time_minutes']} Minutes")
    else:
        st.error("Error making prediction.")