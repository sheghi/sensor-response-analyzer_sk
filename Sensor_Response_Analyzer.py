import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import BytesIO

# 10 samples = 1 second
SAMPLE_TO_SECONDS = 0.1

st.set_page_config(
    page_title="Multi-Cycle Sensor Analyzer",
    layout="wide"
)

st.title("Multi-Cycle Sensor Response Analyzer")

uploaded = st.file_uploader(
    "Upload Excel or CSV",
    type=["xlsx", "csv"]
)


def find_cycles(signal):

    baseline = np.percentile(signal, 10)
    peak = np.percentile(signal, 90)

    threshold = baseline + 0.5 * (peak - baseline)

    active = signal > threshold

    starts = np.where(
        np.diff(active.astype(int)) == 1
    )[0]

    ends = np.where(
        np.diff(active.astype(int)) == -1
    )[0]

    if len(starts) == 0 or len(ends) == 0:
        return []

    if ends[0] < starts[0]:
        ends = ends[1:]

    n = min(len(starts), len(ends))

    return list(zip(starts[:n],
def cycle_metrics(signal, start, end, cycle_no):

    baseline_region = signal[max(0, start - 50):start]

    if len(baseline_region) == 0:
        return None

    baseline = np.mean(baseline_region)

    seg = signal[start:end]

    if len(seg) < 20:
        return None

    peak = np.max(seg)

    amplitude = peak - baseline

    if amplitude <= 0:
        return None

    peak_idx = np.argmax(seg)

    time = np.arange(len(signal)) * SAMPLE_TO_SECONDS

    seg_time = time[start:end]

    def rise_cross(frac):

        target = baseline + frac * amplitude

        idx = np.where(seg >= target)[0]

        if len(idx) == 0:
            return np.nan

        return seg_time[idx[0]]

    t10 = rise_cross(0.10)
    t50 = rise_cross(0.50)
    t90 = rise_cross(0.90)
    t95 = rise_cross(0.95)

    rise_time = (
        t90 - t10
        if not np.isnan(t10)
        and not np.isnan(t90)
        else np.nan
    )

    return {
        "Cycle": cycle_no,
        "Baseline": baseline,
        "Peak": peak,
        "Amplitude": amplitude,
        "T10 (s)": t10,
        "T50 (s)": t50,
        "T90 (s)": t90,
        "T95 (s)": t95,
        "Rise Time (s)": rise_time
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

        if "signal" not in df.columns:
            st.error("Column 'signal' not found.")
            st.stop()

        signal = pd.to_numeric(
            df["signal"],
            errors="coerce"
        )

        signal = signal.dropna().to_numpy()

        cycles = find_cycles(signal)

        if len(cycles) == 0:
            st.error("No exposure cycles detected.")
            st.stop()

        results = []

        for i, (start, end) in enumerate(cycles, start=1):

            m = cycle_metrics(
                signal,
                start,
                end,
                i
            )

            if m is not None:
                results.append(m)

        results_df = pd.DataFrame(results)

        st.subheader("Cycle-by-Cycle Results")
        st.dataframe(
            results_df,
            use_container_width=True
        )

        summary_df = pd.DataFrame({
            "Metric": [
                c for c in results_df.columns
                if c != "Cycle"
            ],
            "Average": [
                results_df[c].mean()
                for c in results_df.columns
                if c != "Cycle"
            ],
            "Std Dev": [
                results_df[c].std()
                for c in results_df.columns
                if c != "Cycle"
            ]
        })

        st.subheader("Average Metrics")
        st.dataframe(
            summary_df,
            use_container_width=True
        )

        time = np.arange(
            len(signal)
        ) * SAMPLE_TO_SECONDS

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

        for start, end in cycles:

            fig.add_vrect(
                x0=start * SAMPLE_TO_SECONDS,
                x1=end * SAMPLE_TO_SECONDS,
                fillcolor="green",
                opacity=0.15,
                line_width=0
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
    st.info("Upload a file to begin.")
