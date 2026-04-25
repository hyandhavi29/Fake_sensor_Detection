import streamlit as st
import numpy as np
import joblib

model = joblib.load("models/sensor_model.pkl")
scaler = joblib.load("models/scaler.pkl")

st.title("Fake Sensor Detection System")

temp = st.number_input("Temperature")
hum = st.number_input("Humidity")
gas = st.number_input("Gas")
press = st.number_input("Pressure")
volt = st.number_input("Voltage")
noise = st.number_input("Noise")

if st.button("Predict"):
    data = np.array([[temp, hum, gas, press, volt, noise]])
    data = scaler.transform(data)
    pred = model.predict(data)

    if pred[0] == 0:
        st.success("Real Sensor")
    else:
        st.error("Fake Sensor")