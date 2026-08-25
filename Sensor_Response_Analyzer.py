import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from io import BytesIO

# =====================================================
# SETTINGS
# =====================================================

FIRST_GAS_ON = 234
GAS_DURATION = 1080
AIR_DURATION = 1080

# =====================================================
# PAGE
# =====================================================

st.set_page_config(
    page_title="Gas Sensor Response Analyzer",
    layout="wide"
)

st.title("Gas Sensor Response Analyzer")

uploaded = st.file_uploader(
    "Upload Excel or CSV file",
    type=["xlsx", "csv"]
)

# =====================================================
# TIME
# =====================================================

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

    except:
        pass

    return np.arange(len(df))


# =====================================================
# SMOOTHING
# =====================================================

def smooth_signal(signal):

    return (
        pd.Series(signal)
        .rolling(
            window=21,
            center=True
        )
        .mean()
        .bfill()
        .ffill()
        .to_numpy()
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

            t1 = time[i - 1]
            t2 = time[i]

            y1 = signal[i - 1]
            y2 = signal[i]

            if y1 == y2:
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
# TIMING-BASED CYCLES
# =====================================================

def detect_cycles_from_timing(time):

    period = (
        GAS_DURATION
        + AIR_DURATION
    )

    cycles = []

    cycle_no = 1

    gas_start = FIRST_GAS_ON

    while True:

        gas_end = (
            gas_start
            + GAS_DURATION
        )

        next_gas_start = (
            gas_start
            + period
        )

        if gas_end > time[-1]:

            break

        start_idx = np.argmin(
            np.abs(time - gas_start)
        )

        end_idx = np.argmin(
            np.abs(time - gas_end)
        )

        next_idx = np.argmin(
            np.abs(time - next_gas_start)
        )

        cycles.append(
            (
                cycle_no,
                start_idx,
                end_idx,
                next_idx
            )
        )

        cycle_no += 1

        gas_start += period

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
            max(0, start - 120):
            max(start - 20, 1)
        ]

    )

    plateau = np.median(

        smoothed[
            start + int(
                0.70 * (end - start)
            ):
            end - 10
        ]

    )

    delta = plateau - baseline

    if abs(delta) < 0.05:

        return None

    # ------------------------
    # RESPONSE
    # ------------------------

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

    # ------------------------
    # RECOVERY
    # ------------------------

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

        signal = signal[mask].to_numpy()

        df = df.loc[mask]

        time = prepare_time(df)

        cycles = detect_cycles_from_timing(
            time
        )

        results = []

        for (
            cycle_no,
            start,
            end,
            next_start
        ) in cycles:

            result = analyse_cycle(
                signal,
                time,
                start,
                end,
                next_start,
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

        avg_row = {

            "Cycle": "Average"

        }

        for column in results_df.columns[1:]:

            avg_row[column] = round(

                pd.to_numeric(
                    results_df[column],
                    errors="coerce"
                ).mean(),

                2

            )

        results_df = pd.concat(

            [
                results_df,
                pd.DataFrame([avg_row])
            ],

            ignore_index=True

        )

        st.subheader(
            "Cycle Results"
        )

        st.dataframe(
            results_df,
            hide_index=True,
            use_container_width=True
        )

        # ==================================
        # PLOT
        # ==================================

        fig = go.Figure()

        fig.add_trace(

            go.Scatter(
                x=time,
                y=signal,
                mode="lines",
                name="Signal"
            )

        )

        for (
            cycle_no,
            start,
            end,
            next_start
        ) in cycles:

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

        # ==================================
        # EXPORT
        # ==================================

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
    
