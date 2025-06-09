import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns


st.set_page_config(page_title="LSTM Outage Predictor", layout="wide")
st.title("⚡ LSTM Outage Probability Predictor")

uploaded_file = st.file_uploader("📤 Upload weather data (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("📊 Uploaded Data Preview:", df.head())

    try:
        scaler = joblib.load("models/lstm_scaler.joblib")
        model = load_model("models/lstm_model.h5")

        features = ["TMAX", "TMIN", "PRCP", "AWND", "month", "dayofweek"]
        df[features] = scaler.transform(df[features])

        def create_sequences(data, window_size=7):
            X = []
            for i in range(len(data) - window_size):
                X.append(data[i:i+window_size, :])
            return np.array(X)

        X_seq = create_sequences(df[features].values)
        y_pred = model.predict(X_seq).flatten()

        df_results = df.iloc[-len(y_pred):].copy()
        df_results["predicted_outage_prob"] = y_pred

        st.subheader("🔮 Predicted Outage Probabilities")
        st.dataframe(df_results[["date", "predicted_outage_prob"]])

        # --- Visualization 1: Line Plot ---
        st.subheader("📈 Time Series of Predicted Probabilities")
        fig1, ax1 = plt.subplots()
        ax1.plot(df_results["date"], df_results["predicted_outage_prob"], color='red', marker='o', linestyle='-')
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Outage Probability")
        ax1.set_title("Predicted Outage Probability Over Time")
        plt.xticks(rotation=45)
        st.pyplot(fig1)

        # --- Visualization 2: Histogram ---
        st.subheader("📊 Distribution of Outage Probabilities")
        fig2, ax2 = plt.subplots()
        sns.histplot(df_results["predicted_outage_prob"], bins=20, kde=True, ax=ax2, color="skyblue")
        ax2.set_title("Histogram of Predicted Probabilities")
        st.pyplot(fig2)

        # --- Visualization 3: Boxplot (optional) ---
        st.subheader("📦 Boxplot of Predicted Probabilities")
        fig3, ax3 = plt.subplots()
        sns.boxplot(y=df_results["predicted_outage_prob"], ax=ax3)
        ax3.set_title("Boxplot of Predictions")
        st.pyplot(fig3)

    except Exception as e:
        st.error(f"❌ Error in prediction or visualization:\n{e}")
