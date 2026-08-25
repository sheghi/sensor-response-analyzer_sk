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
                (t - t.iloc[0])
                * 86400
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

    baseline_start = max(
        0,
        start - 60
    )

    baseline_end = max(
        baseline_start + 5,
        start - 10
    )

    baseline = np.median(
        smoothed[
            baseline_start:
            baseline_end
        ]
    )

    plateau = np.percentile(
        smoothed[
            start + 15:
            end - 15
        ],
        95
    )

    delta = plateau - baseline

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
        True
    )

    t90_cross = interpolate_crossing(
        response_signal,
        response_time,
        level90,
        True
    )

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
        False
    )

    t10_cross = interpolate_crossing(
        recovery_signal,
        recovery_time,
        level10,
        False
    )

    return {

        "Cycle": cycle_no,

        "T63 Response (s)": round(
            t63_cross - time[start],
            2
        ) if not np.isnan(t63_cross) else np.nan,

        "T90 Response (s)": round(
            t90_cross - time[start],
            2
        ) if not np.isnan(t90_cross) else np.nan,

        "T37 Recovery (s)": round(
            t37_cross - time[end],
            2
        ) if not np.isnan(t37_cross) else np.nan,

        "T10 Recovery (s)": round(
            t10_cross - time[end],
            2
        ) if not np.isnan(t10_cross) else np.nan
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

        for i in range(
            len(cycles) - 1
        ):

            start, end = cycles[i]

            next_start = cycles[
                i + 1
            ][0]

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

        results_df = pd.DataFrame(results)

        avg_row = {

            "Cycle": "Average"

        }

        for col in results_df.columns[1:]:

            avg_row[col] = round(
                pd.to_numeric(
                    results_df[col],
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

        # =================================================
        # RESULTS TABLE
        # =================================================

        st.subheader(
            "Cycle Results"
        )

        st.dataframe(
            results_df,
            use_container_width=True,
            hide_index=True
        )

        # =================================================
        # OVERLAY PLOT
        # =================================================

        st.subheader(
            "Overlay of All Response Cycles"
        )

        smoothed = smooth_signal(signal)

        overlay = go.Figure()

        for i in range(
            len(cycles) - 1
        ):

            start, end = cycles[i]

            baseline = np.median(
                smoothed[
                    max(0, start - 60):
                    start - 10
                ]
            )

            plateau = np.percentile(
                smoothed[
                    start + 15:
                    end - 15
                ],
                95
            )

            delta = plateau - baseline

            if delta <= 0:
                continue

            response = (
                smoothed[start:end]
                - baseline
            ) / delta

            response_time = (
                time[start:end]
                - time[start]
            )

            overlay.add_trace(
                go.Scatter(
                    x=response_time,
                    y=response,
                    mode="lines",
                    name=f"Cycle {i+1}"
                )
            )

        overlay.add_hline(
            y=0.63,
            line_dash="dash",
            line_color="orange"
        )

        overlay.add_hline(
            y=0.90,
            line_dash="dash",
            line_color="red"
        )

        overlay.update_layout(
            height=600,
            template="plotly_white",
            xaxis_title="Time Since Gas ON (s)",
            yaxis_title="Normalised Response",
            yaxis=dict(
                range=[-0.1, 1.3]
            )
        )

        st.plotly_chart(
            overlay,
            use_container_width=True
        )

        # =================================================
        # RAW SIGNAL
        # =================================================

        st.subheader(
            "Detected ON/OFF Events"
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

        first_on = True
        first_off = True

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
                    name="Gas ON",
                    showlegend=first_on
                )
            )

            first_on = False

            fig.add_trace(
                go.Scatter(
                    x=[time[end]],
                    y=[signal[end]],
                    mode="markers",
                    marker=dict(
                        color="red",
                        size=10
                    ),
                    name="Gas OFF",
                    showlegend=first_off
                )
            )

            first_off = False

        fig.update_layout(
            template="plotly_white",
            xaxis_title="Time (s)",
            yaxis_title="Signal",
            height=600
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # =================================================
        # EXPORT
        # =================================================

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
            f"Error: {str(e)}"
        )

else:

    st.info(
        "Upload a file to begin analysis."
    )
