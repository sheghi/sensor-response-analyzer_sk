import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import BytesIO

st.set_page_config(
    page_title="Gas Sensor Response Analyzer",
    layout="wide"
)

st.title("Gas Sensor Response Analyzer")

uploaded = st.file_uploader(
    "Upload Excel or CSV",
    type=["xlsx", "csv"]
)


def prepare_time(df):

    try:

        t = pd.to_datetime(
            df["time"].astype(str)
        )

        elapsed = (
            t - t.iloc[0]
        ).dt.total_seconds()

        # your correction
        return (elapsed / 10).to_numpy()

    except:

        return np.arange(len(df)) / 10


def detect_cycles(signal):

    baseline = np.percentile(signal, 10)
    plateau = np.percentile(signal, 90)

    threshold = baseline + (
        0.5 * (plateau - baseline)
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

        end_candidates = falling[
            falling > start
        ]

        if len(end_candidates) == 0:
            continue

        end = end_candidates[0]

        if end - start > 500:
            cycles.append(
                (start, end)
            )

    return cycles


def first_cross_above(
    signal,
    time,
    target
):

    idx = np.where(
        signal >= target
    )[0]

    if len(idx) == 0:
        return np.nan

    return time[idx[0]]


def first_cross_below(
    signal,
    time,
    target
):

    idx = np.where(
        signal <= target
    )[0]

    if len(idx) == 0:
        return np.nan

    return time[idx[0]]


def analyse_cycle(
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

    plateau_region = signal[
        max(start, end - 300):end
    ]

    stable = np.median(
        plateau_region
    )

    delta = stable - baseline

    if delta <= 0:
        return None

    rise_signal = signal[start:end]
    rise_time = time[start:end]

    t0_rise = rise_time[0]

    target30 = baseline + (
        0.30 * delta
    )

    target60 = baseline + (
        0.60 * delta
    )

    target90 = baseline + (
        0.90 * delta
    )

    resp30 = (
        first_cross_above(
            rise_signal,
            rise_time,
            target30
        )
        - t0_rise
    )

    resp60 = (
        first_cross_above(
            rise_signal,
            rise_time,
            target60
        )
        - t0_rise
    )

    resp90 = (
        first_cross_above(
            rise_signal,
            rise_time,
            target90
        )
        - t0_rise
    )

    recovery_signal = signal[end:]
    recovery_time = time[end:]

    if len(recovery_signal) < 50:
        return None

    t0_rec = recovery_time[0]

    rec90_target = baseline + (
        0.90 * delta
    )

    rec60_target = baseline + (
        0.60 * delta
    )

    rec30_target = baseline + (
        0.30 * delta
    )

    rec10_target = baseline + (
        0.10 * delta
    )

    rec90 = (
        first_cross_below(
            recovery_signal,
            recovery_time,
            rec90_target
        )
        - t0_rec
    )

    rec60 = (
        first_cross_below(
            recovery_signal,
            recovery_time,
            rec60_target
        )
        - t0_rec
    )

    rec30 = (
        first_cross_below(
            recovery_signal,
            recovery_time,
            rec30_target
        )
        - t0_rec
    )

    rec10 = (
        first_cross_below(
            recovery_signal,
            recovery_time,
            rec10_target
        )
        - t0_rec
    )

    return {
        "Cycle": cycle_no,
        "T30 Response (s)": round(resp30, 2),
        "T60 Response (s)": round(resp60, 2),
        "T90 Response (s)": round(resp90, 2),
        "T90 Recovery (s)": round(rec90, 2),
        "T60 Recovery (s)": round(rec60, 2),
        "T30 Recovery (s)": round(rec30, 2),
        "T10 Recovery (s)": round(rec10, 2)
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
            st.error("time column not found")
            st.stop()

        if "signal" not in df.columns:
            st.error("signal column not found")
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

        for i, (start, end) in enumerate(
            cycles,
            start=1
        ):

            res = analyse_cycle(
                signal,
                time,
                start,
                end,
                i
            )

            if res is not None:
                results.append(res)

        if len(results) == 0:

            st.error(
                "No valid cycles detected."
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
            file_name="sensor_response_analysis.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    except Exception as e:

        st.error(str(e))

else:

    st.info(
        "Upload a file to begin analysis."
    )
