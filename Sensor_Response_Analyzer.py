import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import BytesIO

# 10 samples per second = 0.1 second per sample
SAMPLING_INTERVAL = 0.1

st.set_page_config(
    page_title="Sensor Response Analyzer",
    layout="wide"
)

st.title("Sensor Response Analyzer")

uploaded = st.file_uploader(
    "Upload CSV or Excel file",
    type=["csv", "xlsx"]
)


def calc_metrics(df):

    df.columns = [str(c).lower().strip() for c in df.columns]

    if "signal" not in df.columns:
        raise ValueError("Column 'signal' not found")

    signal = pd.to_numeric(
        df["signal"],
        errors="coerce"
    ).to_numpy()

    signal = signal[~np.isnan(signal)]

    if len(signal) == 0:
        raise ValueError("No valid signal data found")

    time = np.arange(len(signal)) * SAMPLING_INTERVAL

    baseline = np.mean(signal[:50])

    peak = np.max(signal)

    peak_idx = np.argmax(signal)

    peak_time = float(time[peak_idx])

    amplitude = peak - baseline

    def crossing(frac):

        target = baseline + amplitude * frac

        idx = np.where(signal >= target)[0]

        if len(idx) == 0:
            return np.nan

        return float(time[
