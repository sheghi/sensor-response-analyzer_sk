import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from io import BytesIO


# ==================================================
# PAGE SETTINGS
# ==================================================

st.set_page_config(
    page_title="Gas Sensor Response Analyzer",
    layout="wide"
)

st.title("Gas Sensor Response Analyzer")

uploaded = st.file_uploader(
    "Upload Excel or CSV File",
    type=["xlsx", "csv"]
)


# ==================================================
# UTILITIES
# ==================================================

def prepare_time(df):

    try:

        time_col = pd.to_numeric(
            df["time"],
            errors="coerce"
        )

        if time_col.notna().all():

            return (
                (time_col - time_col.iloc[0])
                * 86400
            ).to_numpy()

    except Exception:
        pass

    return np.arange(len(df))


def smooth_signal(signal):

    return (
        pd.Series(signal)
        .rolling(
            window=31,
            center=True
        )
        .mean()
        .bfill()
        .ffill()
        .to_numpy()
    )


def interpolate_crossing(
    signal,
    time,
    target,
    rising=True
):

    for i in range(1, len(signal)):

        if rising:

            crossed = (
                signal[i - 1] < target
                and signal[i] >= target
            )

        else:

            crossed = (
                signal[i - 1] > target
                and signal[i] <= target
            )

        if crossed:

            x1 = time[i - 1]
            x2 = time[i]

            y1 = signal[i - 1]
            y2 = signal[i]

            if y2 == y1:
                return x1

            return x1 + (
                (target - y1)
                * (x2 - x1)
                / (y2 - y1)
            )

    return np.nan


# ==================================================
# CYCLE DETECTION
# ==================================================

def detect_cycles(signal):

    smoothed = smooth_signal(signal)

    gradient = np.gradient(smoothed)

    threshold = 2.5 * np.std(gradient)

    rise_points = np.where(
        gradient > threshold
    )[0]

    fall_points = np.where(
        gradient < -threshold
    )[0]

    if len(rise_points) == 0:
        return []

    min_gap = 50

    rises = [rise_points[0]]

    for p in rise_points[1:]:

        if p - rises[-1] > min_gap:
            rises.append(p)

    falls = [fall_points[0]]

    for p in fall_points[1:]:

        if p - falls[-1] > min_gap:
            falls.append(p)

    cycles = []

    for rise in rises:

        future_falls = [
            f for f in falls
            if f > rise
        ]

        if len(future_falls) == 0:
            continue

        cycles.append(
            (rise, future_falls[0])
        )

    return cycles


# ==================================================
# CYCLE ANALYSIS
# ==================================================

def analyse_cycle(
    signal,
    time,
    start,
    end,
    cycle_number
):

    smoothed = smooth_signal(signal)

    baseline = np.mean(
        smoothed[
            max(0, start - 150):start
        ]
    )

    plateau_start = start + int(
        (end - start) * 0.7
    )

    plateau = np.mean(
        smoothed[
            plateau_start:end
        ]
    )

    amplitude = plateau - baseline

    if abs(amplitude) < 0.01:
        return None

    response_signal = (

        smoothed[start:end]
        - baseline

    ) / amplitude

    response_time = time[start:end]

    t63_cross = interpolate_crossing(
        response_signal,
        response_time,
        0.63,
        True
    )

    t90_cross = interpolate_crossing(
        response_signal,
        response_time,
        0.90,
        True
    )

    t63 = (
        t63_cross - time[start]
    ) if not np.isnan(t63_cross) else np.nan

    t90 = (
        t90_cross - time[start]
    ) if not np.isnan(t90_cross) else np.nan

    recovery_signal = (

        smoothed[end:]
        - baseline

    ) / amplitude

    recovery_time = time[end:]

    t37_cross = interpolate_crossing(
        recovery_signal,
        recovery_time,
        0.37,
        False
    )

    t10_cross = interpolate_crossing(
        recovery_signal,
        recovery_time,
        0.10,
        False
    )

    t37 = (
        t37_cross - time[end]
    ) if not np.isnan(t37_cross) else np.nan

    t10 = (
        t10_cross - time[end]
    ) if not np.isnan(t10_cross) else np.nan

    return {

        "Cycle": cycle_number,
        "Baseline": round(baseline, 4),
        "Response 100%": round(plateau, 4),
        "Amplitude": round(amplitude, 4),
        "Gas ON (s)": round(time[start], 2),
        "Gas OFF (s)": round(time[end], 2),
        "T63 Response (s)": round(t63, 2)
        if not np.isnan(t63) else np.nan,
        "T90 Response (s)": round(t90, 2)
        if not np.isnan(t90) else np.nan,
        "T37 Recovery (s)": round(t37, 2)
        if not np.isnan(t37) else np.nan,
        "T10 Recovery (s)": round(t10, 2)
        if not np.isnan(t10) else np.nan

    }


# ==================================================
# MAIN APP
# ==================================================

if uploaded is not None:

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

            st.error(
                "Column 'signal' not found."
            )

            st.stop()

        signal = pd.to_numeric(
            df["signal"],
            errors="coerce"
        )

        mask = signal.notna()

        signal = signal[mask].to_numpy()

        df = df.loc[mask]

        time = prepare_time(df)

        cycles = detect_cycles(signal)

        st.success(
            f"Detected {len(cycles)} cycles"
        )

        results = []

        for i, (start, end) in enumerate(
            cycles,
            start=1
        ):

            result = analyse_cycle(
                signal,
                time,
                start,
                end,
                i
            )

            if result is not None:
                results.append(result)

        if len(results) == 0:

            st.error(
                "No valid cycles detected."
            )

            st.stop()

        results_df = pd.DataFrame(results)

        st.subheader(
            "Cycle Results"
        )

        st.dataframe(
            results_df,
            use_container_width=True
        )

        fig = go.Figure()

        fig.add_trace(

            go.Scatter(
                x=time,
                y=signal,
                mode="lines",
                name="Signal"
            )

        )

        for start, end in cycles:

            fig.add_vrect(
                x0=time[start],
                x1=time[end],
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
                sheet_name="Results",
                index=False
            )

        st.download_button(
            label="Download Analysis",
            data=buffer.getvalue(),
            file_name="sensor_response_analysis.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    except Exception as e:

        st.error(
            f"Error: {str(e)}"
        )

else:

    st.info(
        "Upload a file to begin analysis."
    )
