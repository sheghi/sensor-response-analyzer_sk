import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import BytesIO

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

    if "time" not in df.columns:
        raise ValueError("Column 'time' not found")

    if "signal" not in df.columns:
        raise ValueError("Column 'signal' not found")

    # Convert signal to numeric
    signal = pd.to_numeric(
        df["signal"],
        errors="coerce"
    )

    # Convert Excel time column to datetime
    time_dt = pd.to_datetime(
        df["time"].astype(str),
        errors="coerce"
    )

    mask = (
        signal.notna()
        & time_dt.notna()
    )

    signal = signal[mask].to_numpy()
    time_dt = time_dt[mask]

    if len(signal) == 0:
        raise ValueError("No valid data found")

    # Convert to elapsed seconds
    time = (
        time_dt - time_dt.iloc[0]
    ).dt.total_seconds().to_numpy()

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

        return float(time[idx[0]])

    t10 = crossing(0.10)
    t50 = crossing(0.50)
    t90 = crossing(0.90)
    t95 = crossing(0.95)

    rise_time = (
        t90 - t10
        if not np.isnan(t10)
        and not np.isnan(t90)
        else np.nan
    )

    rms_noise = float(
        np.std(signal[:50])
    )

    return {
        "Baseline": baseline,
        "Peak": peak,
        "Peak Time (s)": peak_time,
        "Amplitude": amplitude,
        "T10 (s)": t10,
        "T50 (s)": t50,
        "T90 (s)": t90,
        "T95 (s)": t95,
        "Rise Time (s)": rise_time,
        "RMS Noise": rms_noise
    }


if uploaded:

    try:

        if uploaded.name.endswith(".csv"):
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_excel(uploaded)

        metrics = calc_metrics(df)

        st.subheader("Metrics")

        cols = st.columns(4)

        for i, (name, value) in enumerate(metrics.items()):

            cols[i % 4].metric(
                name,
                f"{value:.4f}"
            )

        # Plot using elapsed seconds
        plot_df = df.copy()

        plot_df.columns = [
            str(c).lower().strip()
            for c in plot_df.columns
        ]

        plot_df["time"] = pd.to_datetime(
            plot_df["time"].astype(str),
            errors="coerce"
        )

        plot_df = plot_df.dropna(
            subset=["time", "signal"]
        )

        plot_df["elapsed_seconds"] = (
            plot_df["time"]
            - plot_df["time"].iloc[0]
        ).dt.total_seconds()

        fig = px.line(
            plot_df,
            x="elapsed_seconds",
            y="signal",
            title="Sensor Response Curve"
        )

        fig.update_layout(
            xaxis_title="Time (s)",
            yaxis_title="Signal"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.subheader("Data Preview")
        st.dataframe(
            plot_df.head(50),
            use_container_width=True
        )

        export_df = pd.DataFrame(
            [metrics]
        )

        buffer = BytesIO()

        with pd.ExcelWriter(
            buffer,
            engine="openpyxl"
        ) as writer:

            export_df.to_excel(
                writer,
                index=False,
                sheet_name="Results"
            )

        st.download_button(
            "Download Results (Excel)",
            data=buffer.getvalue(),
            file_name="sensor_analysis_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    except Exception as e:

        st.error(f"Error: {e}")

else:

    st.info("Upload a file to begin analysis.")
