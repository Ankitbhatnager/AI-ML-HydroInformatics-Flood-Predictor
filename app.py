import streamlit as st
import pandas as pd
import numpy as np
import requests
import random
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

st.set_page_config(page_title="AI/ML HydroInformatics Flood Predictor", page_icon="🛰️", layout="centered")

# 1. Model Training (Cached taaki app fast chale)
@st.cache_resource
def load_and_train_model():
    np.random.seed(42)
    n = 1000
    feature_cols = ['Rainfall_mm', 'River_Discharge_m3s', 'Soil_Moisture', 'Humidity_%']

    df = pd.DataFrame({
        'Rainfall_mm': np.random.uniform(0, 150, n),
        'River_Discharge_m3s': np.random.uniform(10, 2000, n),
        'Soil_Moisture': np.random.uniform(0.05, 0.50, n),
        'Humidity_%': np.random.uniform(20, 100, n)
    })
    df['Flood_Risk'] = np.where((df['Rainfall_mm'] > 60) & (df['River_Discharge_m3s'] > 900), 1, 0)

    X_train, X_test, y_train, y_test = train_test_split(
        df[feature_cols], df['Flood_Risk'], test_size=0.2, random_state=42, stratify=df['Flood_Risk']
    )

    model = RandomForestClassifier(max_depth=None, min_samples_split=2, n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    report = classification_report(y_test, model.predict(X_test))
    return model, feature_cols, report

model, feature_cols, report = load_and_train_model()

# Major Indian River Coordinates
hydro_locations = {
    "Ganga River (Varanasi, UP)": (25.3176, 83.0130),
    "Brahmaputra River (Guwahati, Assam)": (26.1833, 91.7458),
    "Kosi River (Supaul, Bihar)": (26.1260, 86.6050),
    "Yamuna River (Delhi)": (28.6139, 77.2485),
    "Godavari River (Rajahmundry, AP)": (17.0005, 81.7799),
    "Krishna River (Vijayawada, AP)": (16.5062, 80.6480),
    "Narmada River (Bharuch, Gujarat)": (21.7051, 72.9959),
    "Mahanadi River (Cuttack, Odisha)": (20.4625, 85.8830),
    "Alaknanda/Ganga (Rishikesh, Uttarakhand)": (30.0869, 78.2676)
}

def analyze_location(name, lat, lon):
    try:
        # Safe params dictionary (Kabhi bhi & symbol corrupt nahi hoga)
        w_res = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "current": "precipitation,relative_humidity_2m,soil_moisture_0_to_1cm",
                "daily": "precipitation_sum",
                "forecast_days": 7,
                "timezone": "auto"
            },
            timeout=5
        ).json()

        f_res = requests.get(
            "https://flood-api.open-meteo.com/v1/flood",
            params={
                "latitude": lat,
                "longitude": lon,
                "daily": "river_discharge",
                "forecast_days": 7
            },
            timeout=5
        ).json()
    except Exception as e:
        st.error(f"❌ Network/API Error: {e}")
        return

    curr = w_res.get("current", {})
    rain = curr.get("precipitation", 0.0)
    hum = curr.get("relative_humidity_2m", 50.0)
    soil = curr.get("soil_moisture_0_to_1cm", 0.2)

    discharge_list = f_res.get("daily", {}).get("river_discharge", [50.0]*7)
    discharge_list = [d if d is not None else 50.0 for d in discharge_list]
    discharge = discharge_list[0]

    input_df = pd.DataFrame([[rain, discharge, soil, hum]], columns=feature_cols)
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1] * 100

    st.subheader(f"📍 {name}")
    st.caption(f"Coordinates: ({lat:.4f}, {lon:.4f})")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rainfall", f"{rain} mm")
    c2.metric("River Discharge", f"{discharge:.1f} m³/s")
    c3.metric("Soil Moisture", f"{soil:.3f}")
    c4.metric("Humidity", f"{hum}%")

    if pred == 1:
        st.error(f"⚠️ HIGH FLOOD RISK DETECTED ({prob:.1f}% probability)")
    else:
        st.success(f"✅ SAFE / LOW RISK ({prob:.1f}% flood probability)")

    # FIX 2: 100% Guaranteed Graph Rendering (Dark Theme + Safe Array Length)
    daily_w = w_res.get("daily", {})
    dates = daily_w.get("time", [])
    raw_rain = daily_w.get("precipitation_sum", [])
    daily_rain = [r if r is not None else 0.0 for r in raw_rain]

    if dates:
        n_days = len(dates)
        clean_discharge = (discharge_list + [50.0] * n_days)[:n_days]

        fig, ax1 = plt.subplots(figsize=(8, 3.8))
        fig.patch.set_facecolor('#0e1117')
        ax1.set_facecolor('#0e1117')

        ax1.bar(dates, daily_rain, color='#38bdf8', alpha=0.8, label='Rainfall (mm)', width=0.4)
        ax1.set_ylabel('Rainfall (mm)', color='#38bdf8', fontweight='bold')
        ax1.set_ylim(0, max(daily_rain) + 10)
        ax1.tick_params(axis='x', rotation=30, colors='white')
        ax1.tick_params(axis='y', colors='#38bdf8')
        ax1.grid(True, linestyle='--', alpha=0.2, color='white')

        ax2 = ax1.twinx()
        ax2.plot(dates, clean_discharge, color='#f87171', marker='o', linewidth=2.5, label='Discharge (m³/s)')
        ax2.set_ylabel('River Discharge (m³/s)', color='#f87171', fontweight='bold')
        ax2.set_ylim(0, max(clean_discharge) * 1.2 + 10)
        ax2.tick_params(axis='y', colors='#f87171')

        for spine in ax1.spines.values():
            spine.set_color('#334155')
        for spine in ax2.spines.values():
            spine.set_color('#334155')

        plt.title(f"7-Day Hydrological Forecast - {name}", color='white', fontweight='bold', pad=12)
        fig.tight_layout()
        st.pyplot(fig)
    else:
        st.info("ℹ️ 7-Day forecast data is currently unavailable for this location.")

# --- Streamlit UI (Iske neeche ka part bilkul same rahega) ---
st.title("🛰️ AI/ML HydroInformatics Flood Predictor")
st.write("Real-time hydrological monitoring & flood risk prediction powered by **Open-Meteo & GloFAS APIs**.")

query = st.text_input("🔍 Enter City, State, or River Name (or leave as 'random'):", value="Assam")

col_btn1, col_btn2 = st.columns(2)
run_search = col_btn1.button("Analyze Location", type="primary")
run_random = col_btn2.button("🎲 Random Indian River")

if run_random:
    chosen_name = random.choice(list(hydro_locations.keys()))
    lat, lon = hydro_locations[chosen_name]
    analyze_location(chosen_name, lat, lon)
elif run_search or query:
    q = query.strip() if query.strip() else "random"
    if q.lower() == "random":
        chosen_name = random.choice(list(hydro_locations.keys()))
        lat, lon = hydro_locations[chosen_name]
        analyze_location(chosen_name, lat, lon)
    else:
        matches = [k for k in hydro_locations if q.lower() in k.lower()]
        if matches:
            chosen_name = matches[0]
            lat, lon = hydro_locations[chosen_name]
            analyze_location(chosen_name, lat, lon)
        else:
            try:
                geo = requests.get(f"https://geocoding-api.open-meteo.com/v1/search?name={q}&count=1", timeout=5).json()
                if "results" in geo:
                    lat, lon = geo["results"][0]["latitude"], geo["results"][0]["longitude"]
                    full_name = f"{geo['results'][0]['name']}, {geo['results'][0].get('admin1', '')} ({geo['results'][0].get('country', '')})"
                    analyze_location(full_name, lat, lon)
                else:
                    st.warning(f"❌ '{q}' location not found.")
            except Exception as e:
                st.error(f"❌ Geocoding search failed: {e}")

with st.expander("📊 View ML Model Evaluation Metrics"):
    st.code(report)