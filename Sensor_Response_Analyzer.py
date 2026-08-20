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

        elapsed = (
            t - t.iloc[0]
        ).dt.total_seconds()

        # divide by 10 as requested
        return (elapsed / 10).to_numpy()

    except:
        return np.arange(len(df)) / 10


def detect_cycles(signal):

    baseline = np.percentile(signal, 10)
    plateau = np.percentile(signal, 90)

    threshold = baseline + 0.5 * (
        plateau - baseline
    )

    active = signal > threshold

    rising = np.where(
        (active[1:] == True)
        & (active[:-1] == False)
    )[0]

    falling = np.where(
        (active[1:] == False)
        & (active[:-1] == True)
    )[0]

    cycles = []

    for start in rising:

        candidates = falling[
            falling > start
        ]

        if len(candidates) == 0:
            continue

        end = candidates[0]

        # ignore short events/noise
        if end - start > 500:
            cycles.append(
                (start, end)
            )

    return cycles


def calc_cycle(
    signal,
    time,
    start,
    end,
    cycle_no
):

    baseline_region = signal[
        max(0, start - 300):start
    ]

    if len(baseline_region) < 50:
        return None

    baseline = np.median(
        baseline_region
    )

    segment = signal[start:end]

    seg_time = time[start:end]

    peak = np.max(segment)

    amplitude = peak - baseline

    if amplitude <= 0:
        return None

    def cross(frac):

        target = baseline + (
            frac * amplitude
        )

        idx = np.where(
            segment >= target
        )[0]

        if len(idx) == 0:
            return np.nan

        return float(
            seg_time[idx[0]]
        )

    t10 = cross(0.10)
    t50 = cross(0.50)
    t90 = cross(0.90)
    t95 = cross(0.95)

    if (
        np.isnan(t10)
        or np.isnan(t90)
    ):
        rise_time = np.nan
    else:
        rise_time = t90 - t10

    return {
        "Cycle": cycle_no,
        "Baseline": round(baseline, 4),
        "Peak": round(peak, 4),
        "Amplitude": round(amplitude, 4),
        "T10 (s)": round(t10, 2),
        "T50 (s)": round(t50, 2),
        "T90 (s)": round(t90, 2),
        "T95 (s)": round(t95, 2),
        "Rise Time (s)": round(rise_time, 2)
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
            st.error(
                "Column 'time' not found"
            )
            st.stop()

        if "signal" not in df.columns:
            st.error(
                "Column 'signal' not found"
            )
            st.stop()

        signal = pd.to_numeric(
            df["signal"],
            errors="coerce"
        )

        mask = signal.notna()

        signal = signal[
            mask
        ].to_numpy()

        df = df.loc[mask]

        time = prepare_time(df)

        cycles = detect_cycles(signal)

        st.write(
            f"Detected cycles: {len(cycles)}"
        )

        results = []

        for i, (
            start,
            end
        ) in enumerate(
            cycles,
            start=1
        ):

            result = calc_cycle(
                signal,
                time,
                start,
                end,
                i
            )

            if result:
                results.append(result)

        if len(results) == 0:

            st.error(
                "No valid cycles detected"
            )

            st.stop()

        results_df = pd.DataFrame(
            results
        )

        st.subheader(
            "Cycle Results"
        )

        st.dataframe(
            results_df,
            use_container_width=True
        )

        numeric_cols = [
            c
            for c in results_df.columns
            if c != "Cycle"
        ]

        summary_df = pd.DataFrame({
            "Metric":
                numeric_cols,
            "Average": [
                results_df[c].mean()
                for c in numeric_cols
            ],
            "Std Dev": [
                results_df[c].std()
                for c in numeric_cols
            ]
        })

        st.subheader(
            "Average Results"
        )

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

        for start, end in cycles:

            fig.add_vrect(
                x0=time[start],
                x1=time[end],
                fillcolor="green",
                opacity=0.2,
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

        st.error(str(e))

else:

    st.info(
        "Upload a file to begin analysis."
    )
