import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import BytesIO

st.set_page_config(
    page_title="Sensor Response Analyzer",
    layout="wide"
)

st.title("Multi-Cycle Sensor Response Analyzer")

uploaded = st.file_uploader(
    "Upload Excel or CSV file",
    type=["xlsx", "csv"]
)


def prepare_time(df):

    try:
        t = pd.to_datetime(df["time"].astype(str))
        elapsed = (t - t.iloc[0]).dt.total_seconds()
        return elapsed.to_numpy()

    except Exception:
        return np.arange(len(df))


def detect_peaks(signal):

    smooth = (
        pd.Series(signal)
        .rolling(window=15, center=True)
        .mean()
        .bfill()
        .ffill()
    )

    threshold = np.percentile(signal, 80)

    raw_peaks = []

    for i in range(1, len(smooth) - 1):

        if (
            smooth.iloc[i] > smooth.iloc[i - 1]
            and smooth.iloc[i] > smooth.iloc[i + 1]
            and smooth.iloc[i] > threshold
        ):
            raw_peaks.append(i)

    peaks = []

    for p in raw_peaks:

        if len(peaks) == 0:
            peaks.append(p)

        elif p - peaks[-1] > 300:
            peaks.append(p)

    return np.array(peaks)


def calc_cycle(signal, time, peak_idx, cycle_no):

    left = max(0, peak_idx - 250)
    right = min(len(signal) - 1, peak_idx + 250)

    baseline_region = signal[left:peak_idx]

    if len(baseline_region) < 10:
        return None

    baseline = np.min(baseline_region)

    peak = signal[peak_idx]

    amplitude = peak - baseline

    if amplitude <= 0:
        return None

    rise_signal = signal[left:peak_idx + 1]
    rise_time = time[left:peak_idx + 1]

    def crossing(frac):

        target = baseline + frac * amplitude

        idx = np.where(rise_signal >= target)[0]

        if len(idx) == 0:
            return np.nan

        return float(rise_time[idx[0]])

    t10 = crossing(0.10)
    t50 = crossing(0.50)
    t90 = crossing(0.90)
    t95 = crossing(0.95)

    response_time = np.nan

    if not np.isnan(t10) and not np.isnan(t90):
        response_time = t90 - t10

    return {
        "Cycle": cycle_no,
        "Baseline": baseline,
        "Peak": peak,
        "Amplitude": amplitude,
        "T10 (s)": t10,
        "T50 (s)": t50,
        "T90 (s)": t90,
        "T95 (s)": t95,
        "Rise Time (s)": response_time
    }


if uploaded:

    try:

        if uploaded.name.endswith(".csv"):
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_excel(uploaded)

        df.columns = [
            str(c).lower().strip()
            for c in df.columns
        ]

        if "time" not in df.columns:
            st.error("Column 'time' not found")
            st.stop()

        if "signal" not in df.columns:
            st.error("Column 'signal' not found")
            st.stop()

        signal = pd.to_numeric(
            df["signal"],
            errors="coerce"
        )

        valid = signal.notna()

        signal = signal[valid].to_numpy()

        df = df.loc[valid]

        time = prepare_time(df)

        peaks = detect_peaks(signal)

        if len(peaks) == 0:
            st.error("No cycles detected.")
            st.stop()

        results = []

        for cycle_no, peak_idx in enumerate(peaks, start=1):

            res = calc_cycle(
                signal,
                time,
                peak_idx,
                cycle_no
            )

            if res is not None:
                results.append(res)

        if len(results) == 0:
            st.error("No valid cycles found.")
            st.stop()

        results_df = pd.DataFrame(results)

        st.subheader("Cycle Results")

        st.dataframe(
            results_df,
            use_container_width=True
        )

        numeric_cols = [
            c for c in results_df.columns
            if c != "Cycle"
        ]

        summary_df = pd.DataFrame({
            "Metric": numeric_cols,
            "Average": [
                results_df[c].mean()
                for c in numeric_cols
            ],
            "Std Dev": [
                results_df[c].std()
                for c in numeric_cols
            ]
        })

        st.subheader("Average Results")

        st.dataframe(
            summary_df,
            use_container_width=True
        )

        plot_df = pd.DataFrame({
            "Time (s)": time,
            "Signal": signal
        })

        fig = px.line(
            plot_df,
            x="Time (s)",
            y="Signal",
            title="Sensor Response"
        )

        fig.add_scatter(
            x=time[peaks],
            y=signal[peaks],
            mode="markers",
            name="Detected Peaks"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        buffer = BytesIO()

        with pd.ExcelWriter(
            buffer,
            engine="openpyxl"
        ) as writer:

            results_df.to_excel(
                writer,
                sheet_name="Cycle Results",
                index=False
            )

            summary_df.to_excel(
                writer,
                sheet_name="Summary",
                index=False
            )

        st.download_button(
            "Download Analysis",
            data=buffer.getvalue(),
            file_name="sensor_cycle_analysis.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    except Exception as e:

        st.error(f"Error: {e}")

else:

    st.info("Upload a file to begin analysis.")
