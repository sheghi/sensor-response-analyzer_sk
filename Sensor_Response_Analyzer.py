import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.signal import savgol_filter, find_peaks
from io import BytesIO


# =====================================================
# PAGE
# =====================================================

st.set_page_config(
    page_title="Gas Sensor Response Analyzer",
    layout="wide"
)

st.title("Gas Sensor Response Analyzer")

uploaded = st.file_uploader(
    "Upload Excel or CSV File",
    type=["xlsx", "csv"]
)


# =====================================================
# TIME
# =====================================================

def prepare_time(df):

    try:

        t = pd.to_numeric(
            df["time"],
            errors="coerce"
        )

        if t.notna().all():

            return (
                (t - t.iloc[0])
                * 86400
            ).to_numpy()

    except Exception:
        pass

    return np.arange(len(df))


# =====================================================
# SMOOTHING
# =====================================================

def smooth_signal(signal):

    n = len(signal)

    window = min(31, n - 1)

    if window % 2 == 0:
        window -= 1

    if window < 7:
        return signal

    return savgol_filter(
        signal,
        window_length=window,
        polyorder=3
    )


# =====================================================
# INTERPOLATION
# =====================================================

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

            y1 = signal[i - 1]
            y2 = signal[i]

            t1 = time[i - 1]
            t2 = time[i]

            if y2 == y1:
                return t1

            frac = (
                target - y1
            ) / (
                y2 - y1
            )

            return t1 + frac * (
                t2 - t1
            )

    return np.nan


# =====================================================
# CYCLE DETECTION
# =====================================================

def detect_cycles(signal):

    smoothed = smooth_signal(signal)

    gradient = np.gradient(smoothed)

    threshold = (
        2.5 *
        np.std(gradient)
    )

    rise_peaks, _ = find_peaks(
        gradient,
        height=threshold,
        distance=100
    )

    fall_peaks, _ = find_peaks(
        -gradient,
        height=threshold,
        distance=100
    )

    cycles = []

    f_idx = 0

    for rise in rise_peaks:

        while (
            f_idx < len(fall_peaks)
            and fall_peaks[f_idx] < rise
        ):
            f_idx += 1

        if f_idx >= len(fall_peaks):
            break

        cycles.append(
            (
                int(rise),
                int(fall_peaks[f_idx])
            )
        )

        f_idx += 1

    return cycles


# =====================================================
# ANALYSIS
# =====================================================

def analyse_cycle(
    signal,
    time,
    start,
    end,
    next_start,
    cycle_no
):

    smoothed = smooth_signal(signal)

    baseline = np.median(

        smoothed[
            max(0, start - 80):
            max(start - 20, 1)
        ]

    )

    plateau = np.median(

        smoothed[
            start + int(
                0.7 * (end - start)
            ):
            max(end - 10, start + 1)
        ]

    )

    delta = plateau - baseline

    if delta <= 0:
        return None

    # --------------------------
    # RESPONSE LEVELS
    # --------------------------

    level63 = (
        baseline
        + 0.63 * delta
    )

    level90 = (
        baseline
        + 0.90 * delta
    )

    response_signal = smoothed[
        start:end
    ]

    response_time = time[
        start:end
    ]

    t63_cross = interpolate_crossing(
        response_signal,
        response_time,
        level63,
        rising=True
    )

    t90_cross = interpolate_crossing(
        response_signal,
        response_time,
        level90,
        rising=True
    )

    t63 = np.nan
    t90 = np.nan

    if not np.isnan(t63_cross):

        t63 = (
            t63_cross
            - time[start]
        )

    if not np.isnan(t90_cross):

        t90 = (
            t90_cross
            - time[start]
        )

    # --------------------------
    # RECOVERY LEVELS
    # --------------------------

    level37 = (
        baseline
        + 0.37 * delta
    )

    level10 = (
        baseline
        + 0.10 * delta
    )

    recovery_signal = smoothed[
        end:next_start
    ]

    recovery_time = time[
        end:next_start
    ]

    t37_cross = interpolate_crossing(
        recovery_signal,
        recovery_time,
        level37,
        rising=False
    )

    t10_cross = interpolate_crossing(
        recovery_signal,
        recovery_time,
        level10,
        rising=False
    )

    t37 = np.nan
    t10 = np.nan

    if not np.isnan(t37_cross):

        t37 = (
            t37_cross
            - time[end]
        )

    if not np.isnan(t10_cross):

        t10 = (
            t10_cross
            - time[end]
        )

    return {

        "Cycle": cycle_no,

        "T63 Response (s)":
            round(t63, 2)
            if not np.isnan(t63)
            else np.nan,

        "T90 Response (s)":
            round(t90, 2)
            if not np.isnan(t90)
            else np.nan,

        "T37 Recovery (s)":
            round(t37, 2)
            if not np.isnan(t37)
            else np.nan,

        "T10 Recovery (s)":
            round(t10, 2)
            if not np.isnan(t10)
            else np.nan
    }


# =====================================================
# MAIN
# =====================================================

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

        signal = signal[
            mask
        ].to_numpy()

        df = df.loc[mask]

        time = prepare_time(df)

        cycles = detect_cycles(signal)

        results = []

        for i in range(
            len(cycles) - 1
        ):

            start, end = cycles[i]

            next_start = cycles[i + 1][0]

            row = analyse_cycle(
                signal,
                time,
                start,
                end,
                next_start,
                i + 1
            )

            if row is not None:

                results.append(row)

        if len(results) == 0:

            st.error(
                "No valid cycles detected."
            )

            st.stop()

        results_df = pd.DataFrame(results)

        avg = {

            "Cycle": "Average"

        }

        for col in results_df.columns[1:]:

            avg[col] = round(

                pd.to_numeric(
                    results_df[col],
                    errors="coerce"
                ).mean(),

                2

            )

        results_df = pd.concat(

            [
                results_df,
                pd.DataFrame([avg])
            ],

            ignore_index=True
        )

        st.subheader(
            "Results"
        )

        st.dataframe(
            results_df,
            hide_index=True,
            use_container_width=True
        )

        # --------------------------
        # Plot
        # --------------------------

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

        fig.update_layout(
            title="Sensor Signal",
            xaxis_title="Time (s)",
            yaxis_title="Signal"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # --------------------------
        # Export
        # --------------------------

        output = BytesIO()

        with pd.ExcelWriter(
            output,
            engine="openpyxl"
        ) as writer:

            results_df.to_excel(
                writer,
                sheet_name="Results",
                index=False
            )

        st.download_button(
            "Download Analysis",
            output.getvalue(),
            file_name="sensor_response_analysis.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    except Exception as e:

        st.error(
            f"Error: {e}"
        )

else:

    st.info(
        "Upload a file to begin analysis."
    )
