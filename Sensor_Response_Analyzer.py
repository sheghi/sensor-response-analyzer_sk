import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from io import BytesIO

# =====================================================
# PAGE SETUP
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

    return (
        pd.Series(signal)
        .rolling(
            window=15,
            center=True
        )
        .mean()
        .bfill()
        .ffill()
        .to_numpy()
    )

# =====================================================
# CYCLE DETECTION
# =====================================================

def detect_cycles(signal):

    smoothed = smooth_signal(signal)

    threshold = (
        np.percentile(smoothed, 25)
        + np.percentile(smoothed, 75)
    ) / 2

    state = smoothed > threshold

    transitions = np.diff(
        state.astype(int)
    )

    rises = np.where(
        transitions == 1
    )[0]

    falls = np.where(
        transitions == -1
    )[0]

    cycles = []

    fall_idx = 0

    for rise in rises:

        while (
            fall_idx < len(falls)
            and falls[fall_idx] < rise
        ):
            fall_idx += 1

        if fall_idx >= len(falls):
            break

        cycles.append(
            (
                int(rise),
                int(falls[fall_idx])
            )
        )

        fall_idx += 1

    return cycles

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

            x1 = time[i - 1]
            x2 = time[i]

            y1 = signal[i - 1]
            y2 = signal[i]

            if y2 == y1:
                return x1

            frac = (
                target - y1
            ) / (
                y2 - y1
            )

            return x1 + frac * (
                x2 - x1
            )

    return np.nan

# =====================================================
# ANALYSIS
# =====================================================

def analyse_cycle(
    signal,
    time,
    start,
    end,
    cycle_no
):

    smoothed = smooth_signal(signal)

    baseline = np.mean(
        smoothed[
            max(0, start - 80):
            max(start - 10, 1)
        ]
    )

    plateau_start = start + int(
        0.6 * (end - start)
    )

    plateau = np.mean(
        smoothed[
            plateau_start:
            max(end - 5, plateau_start + 1)
        ]
    )

    amplitude = plateau - baseline

    if amplitude <= 0.05:

        return None

    # ---------------------
    # RESPONSE
    # ---------------------

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

    if np.isnan(t63):

        if not np.isnan(t90):

            t63 = t90 / 2.3

    # ---------------------
    # RECOVERY
    # ---------------------

    recovery_end = min(
        len(smoothed),
        end + 250
    )

    recovery_signal = (
        smoothed[
            end:recovery_end
        ]
        - baseline
    ) / amplitude

    recovery_time = time[
        end:recovery_end
    ]

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

    # remove unrealistic values

    if not np.isnan(t37):

        if t37 < 0 or t37 > 60:

            t37 = np.nan

    if not np.isnan(t10):

        if t10 < 0 or t10 > 120:

            t10 = np.nan

    return {

        "Cycle": cycle_no,

        "T63 Response (s)": (
            round(float(t63), 2)
            if not np.isnan(t63)
            else np.nan
        ),

        "T90 Response (s)": (
            round(float(t90), 2)
            if not np.isnan(t90)
            else np.nan
        ),

        "T37 Recovery (s)": (
            round(float(t37), 2)
            if not np.isnan(t37)
            else np.nan
        ),

        "T10 Recovery (s)": (
            round(float(t10), 2)
            if not np.isnan(t10)
            else np.nan
        )
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

        st.success(
            f"Detected {len(cycles)} cycles"
        )

        results = []

        for cycle_no, (
            start,
            end
        ) in enumerate(
            cycles,
            start=1
        ):

            result = analyse_cycle(
                signal,
                time,
                start,
                end,
                cycle_no
            )

            if result is not None:

                results.append(result)

        if len(results) == 0:

            st.error(
                "No valid cycles found."
            )
            st.stop()

        results_df = pd.DataFrame(results)

        numeric_cols = [

            "T63 Response (s)",
            "T90 Response (s)",
            "T37 Recovery (s)",
            "T10 Recovery (s)"

        ]

        average_row = {

            "Cycle": "Average"

        }

        for col in numeric_cols:

            average_row[col] = round(

                pd.to_numeric(
                    results_df[col],
                    errors="coerce"
                ).mean(),

                2

            )

        results_df = pd.concat(

            [
                results_df,
                pd.DataFrame([average_row])
            ],

            ignore_index=True

        )

        st.subheader(
            "Cycle Results"
        )

        st.dataframe(
            results_df,
            use_container_width=True,
            hide_index=True
        )

        # ===============================
        # PLOT
        # ===============================

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
            title="Sensor Response",
            xaxis_title="Time (s)",
            yaxis_title="Signal"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # ===============================
        # EXPORT
        # ===============================

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
            "Download Analysis",
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
