import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
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

    if "time" in df.columns:

        t = pd.to_numeric(
            df["time"],
            errors="coerce"
        )

        if t.notna().all():

            return (
                (t - t.iloc[0]) * 86400
            ).to_numpy()

    return np.arange(len(df))

# =====================================================
# SMOOTHING
# =====================================================

def smooth_signal(signal):

    return (
        pd.Series(signal)
        .rolling(
            window=11,
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

            x1 = time[i - 1]
            x2 = time[i]

            y1 = signal[i - 1]
            y2 = signal[i]

            if y1 == y2:
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
# EVENT DETECTION
# =====================================================

def merge_events(events, gap=20):

    merged = []

    for e in events:

        if (
            len(merged) == 0
            or e - merged[-1] > gap
        ):
            merged.append(int(e))

    return merged


def detect_cycles(signal):

    smoothed = smooth_signal(signal)

    grad = np.gradient(smoothed)

    threshold = (
        3 * np.std(grad)
    )

    rises = np.where(
        grad > threshold
    )[0]

    falls = np.where(
        grad < -threshold
    )[0]

    rises = merge_events(
        rises,
        gap=20
    )

    falls = merge_events(
        falls,
        gap=20
    )

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
                rise,
                falls[fall_idx]
            )
        )

        fall_idx += 1

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

    max_response = np.max(
        smoothed[start:end]
    )

    delta = (
        max_response
        - baseline
    )

    if delta <= 0:
        return None

    level63 = (
        baseline
        + 0.63 * delta
    )

    level90 = (
        baseline
        + 0.90 * delta
    )

    response_signal = smoothed[start:end]
    response_time = time[start:end]

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

    level37 = (
        baseline
        + 0.37 * delta
    )

    level10 = (
        baseline
        + 0.10 * delta
    )

    recovery_signal = smoothed[end:next_start]
    recovery_time = time[end:next_start]

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

    return {
        "Cycle": cycle_no,
        "T63 Response (s)": (
            round(t63_cross - time[start], 2)
            if not np.isnan(t63_cross)
            else np.nan
        ),
        "T90 Response (s)": (
            round(t90_cross - time[start], 2)
            if not np.isnan(t90_cross)
            else np.nan
        ),
        "T37 Recovery (s)": (
            round(t37_cross - time[end], 2)
            if not np.isnan(t37_cross)
            else np.nan
        ),
        "T10 Recovery (s)": (
            round(t10_cross - time[end], 2)
            if not np.isnan(t10_cross)
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

        signal = pd.to_numeric(
            df["signal"],
            errors="coerce"
        )

        mask = signal.notna()

        signal = signal[mask].to_numpy()

        df = df.loc[mask]

        time = prepare_time(df)

        cycles = detect_cycles(signal)

        results = []

        for i in range(len(cycles) - 1):

            start, end = cycles[i]

            next_start = cycles[i + 1][0]

            result = analyse_cycle(
                signal,
                time,
                start,
                end,
                next_start,
                i + 1
            )

            if result is not None:
                results.append(result)

        results_df = pd.DataFrame(results)

        if len(results_df) > 0:

            avg = {"Cycle": "Average"}

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
            "Cycle Results"
        )

        st.dataframe(
            results_df,
            hide_index=True,
            use_container_width=True
        )

        # Cycle Viewer

        st.subheader(
            "Cycle Viewer"
        )

        cycle_choice = st.selectbox(
            "Select Cycle",
            list(range(1, len(cycles)))
        )

        start, end = cycles[
            cycle_choice - 1
        ]

        smoothed = smooth_signal(signal)

        baseline = np.median(
            smoothed[
                max(0, start - 80):
                max(start - 20, 1)
            ]
        )

        max_response = np.max(
            smoothed[start:end]
        )

        delta = (
            max_response
            - baseline
        )

        response = (
            smoothed[start:end]
            - baseline
        ) / delta

        response_time = (
            time[start:end]
            - time[start]
        )

        cycle_fig = go.Figure()

        cycle_fig.add_trace(
            go.Scatter(
                x=response_time,
                y=response,
                mode="lines"
            )
        )

        cycle_fig.add_hline(
            y=0.63,
            line_dash="dash"
        )

        cycle_fig.add_hline(
            y=0.90,
            line_dash="dash"
        )

        st.plotly_chart(
            cycle_fig,
            use_container_width=True
        )

        # Raw Signal

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

            fig.add_trace(
                go.Scatter(
                    x=[time[start]],
                    y=[signal[start]],
                    mode="markers",
                    marker=dict(
                        color="green",
                        size=10
                    ),
                    showlegend=False
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=[time[end]],
                    y=[signal[end]],
                    mode="markers",
                    marker=dict(
                        color="red",
                        size=10
                    ),
                    showlegend=False
                )
            )

        fig.update_layout(
            title="Detected Gas ON/OFF Events",
            template="plotly_white",
            height=600
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

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

        st.error(str(e))

else:

    st.info(
        "Upload a file to begin."
    )
