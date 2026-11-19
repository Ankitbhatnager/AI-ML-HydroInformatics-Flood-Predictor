# 🛰️ AI/ML HydroInformatics: Live Flood Risk Predictor

An end-to-end Machine Learning and Real-Time Data pipeline built using **Python, Scikit-Learn, Matplotlib, and Open-Meteo APIs**. This project was developed following the **ISRO-IIRS Online Course on "AI/ML for HydroInformatics"**.

## 🚀 Features
- **Real-Time Hydrological Data:** Fetches live rainfall, humidity, soil moisture, and river discharge ($m^3/s$) via Open-Meteo Weather & GloFAS Flood APIs.
- **ML Risk Assessment:** Uses a Random Forest Classifier to predict flood probability.
- **7-Day Forecast Visualization:** Generates dual-axis rainfall and river discharge trend charts.
- **Pan-India Coverage:** Supports major Indian rivers (Ganga, Brahmaputra, Kosi, Godavari, etc.) and cities.

## 🛠️ Tech Stack
- **Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-Learn, Matplotlib, Requests
- **APIs:** Open-Meteo Weather API, GloFAS Flood API, Geocoding API