import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from io import BytesIO

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
# TIME CONVERSION
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

    except Exception:
        pass

    try:

        t = pd.to_datetime(df["time"])

        return (
            t - t.iloc[0]
        ).dt.total_seconds().to_numpy()

    except Exception:

        return np.arange(len(df))


# =====================================================
# SMOOTHING
# =====================================================

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


# =====================================================
# CYCLE DETECTION
# =====================================================

def detect_cycles(signal):

    smoothed = smooth_signal(signal)

    gradient = np.gradient(smoothed)

    threshold = 2.5 * np.std(gradient)

    rises = np.where(
        gradient > threshold
    )[0]

    falls = np.where(
        gradient < -threshold
    )[0]

    if len(rises) == 0 or len(falls) == 0:
        return []

    min_gap = max(
        50,
        int(len(signal) / 25)
    )

    rise_events = [rises[0]]

    for r in rises[1:]:

        if r - rise_events[-1] > min_gap:

            rise_events.append(r)

    fall_events = [falls[0]]

    for f in falls[1:]:

        if f - fall_events[-1] > min_gap:

            fall_events.append(f)

    cycles = []

    for rise in rise_events:

        candidates = [
            f for f in fall_events
            if f > rise
        ]

        if len(candidates) == 0:
            continue

        fall = candidates[0]

        if fall - rise > 20:

            cycles.append(
                (rise, fall)
            )

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

            return x1 + frac * (x2 - x1)

    return np.nan


# =====================================================
# CYCLE ANALYSIS
# =====================================================

def analyse_cycle(
    signal,
    time,
    start,
    end,
    cycle_no
):

    smoothed = smooth_signal(signal)

    gradient = np.gradient(smoothed)

    gas_on_idx = start
    gas_off_idx = end

    gas_on_time = time[gas_on_idx]
    gas_off_time = time[gas_off_idx]

    baseline = np.mean(

        smoothed[
            max(
                0,
                gas_on_idx - 150
            ):
            gas_on_idx
        ]

    )

    plateau_start = gas_on_idx + int(
        0.7 * (
            gas_off_idx - gas_on_idx
        )
    )

    plateau = np.mean(

        smoothed[
            plateau_start:
            gas_off_idx
        ]

    )

    amplitude = plateau - baseline

    if abs(amplitude) < 0.01:

        return None

    response_norm = (

        smoothed[
            gas_on_idx:
            gas_off_idx
        ] - baseline

    ) / amplitude

    response_time = time[
        gas_on_idx:
        gas_off_idx
    ]

    t63_cross = interpolate_crossing(
        response_norm,
        response_time,
        0.63,
        True
    )

    t90_cross = interpolate_crossing(
        response_norm,
        response_time,
        0.90,
        True
    )

    t63 = (
        t63_cross - gas_on_time
    ) if not np.isnan(t63_cross) else np.nan

    t90 = (
        t90_cross - gas_on_time
    ) if not np.isnan(t90_cross) else np.nan

    recovery_signal = smoothed[
        gas_off_idx:
    ]

    recovery_time = time[
        gas_off_idx:
    ]

    recovery_norm = (

        recovery_signal - baseline

    ) / amplitude

    t37_cross = interpolate_crossing(
        recovery_norm,
        recovery_time,
        0.37,
        False
    )

    t10_cross = interpolate_crossing(
        recovery_norm,
        recovery_time,
        0.10,
        False
    )

    t37 = (
        t37_cross - gas_off_time
    ) if not np.isnan(t37_cross) else np.nan

    t10 = (
        t10_cross - gas_off_time
    ) if not np.isnan(t10_cross) else np.nan

    return {

        "Cycle": cycle_no,
        "Baseline": round(float(baseline), 4),
        "Response 100%": round(float(plateau), 4),
        "Response 90%": round(float(plateau * 0.90), 4),
        "Response 63%": round(float(plateau * 0.63), 4),
        "Amplitude": round(float(amplitude), 4),

        "Gas ON (s)": round(
            float(gas_on_time),
            2
        ),

        "Gas OFF (s)": round(
            float(gas_off_time),
            2
        ),

        "T63 Response (s)": round(
            float(t63),
            2
        ) if not np.isnan(t63) else np.nan,

        "T90 Response (s)": round(
            float(t90),
            2
        ) if not np.isnan(t90) else np.nan,

        "T37 Recovery (s)": round(
            float(t37),
            2
        ) if not np.isnan(t37) else np.nan,

        "T10 Recovery (s)": round(
            float(t10),
            2
        ) if not np.isnan(t10) else np.nan
    }


# =====================================================
# MAIN APP
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
                "Column 'signal' not found"
            )

            st.stop()

        signal = pd.to_numeric(
            df["signal"],
            errors="coerce"
        )

        valid = signal.notna()

        signal = signal[
            valid
        ].to_numpy()

        df = df.loc[valid]

        time = prepare_time(df)

        cycles = detect_cycles(signal)

        st.success(
            f"Detected {len(cycles)} cycles"
        )

        results = []

        for n, (
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
                n
            )

            if result is not None:

                results.append(result)

        if len(results) == 0:

            st.error(
                "No valid cycles found."
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

        summary_df = pd.DataFrame({

            "Metric": [

                "T63 Response (s)",
                "T90 Response (s)",
                "T37 Recovery (s)",
                "T10 Recovery (s)"

            ],

            "Average": [

                results_df["T63 Response (s)"].mean(),
                results_df["T90 Response (s)"].mean(),
                results_df["T37 Recovery (s)"].mean(),
                results_df["T10 Recovery (s)"].mean()

            ],

            "Std Dev": [

                results_df["T63 Response (s)"].std(),
                results_df["T90 Response (s)"].std(),
                results_df["T37 Recovery (s)"].std(),
                results_df["T10 Recovery (s)"].std()

            ]

        })

        st.subheader(
            "Summary"
        )

        st.dataframe(
            summary_df,
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

        for i, (
            start,
            end
        ) in enumerate(cycles):

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

            label="Download Analysis",

            data=buffer.getvalue(),

            file_name="sensor_response_analysis.xlsx",

            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

        )

    except Exception as e:

        st.error(
            f"Error: {e}"
       
