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
            window=11,
            center=True,
            min_periods=1
        )
        .mean()
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

            return (
                t1
                + frac * (t2 - t1)
            )

    return np.nan

# =====================================================
# CYCLE DETECTION
# =====================================================

def detect_cycles(signal):

    smoothed = smooth_signal(signal)

    grad = np.gradient(smoothed)

    threshold = (
        0.15
        * np.max(
            np.abs(grad)
        )
    )

    events = np.where(
        np.abs(grad) > threshold
    )[0]

    if len(events) == 0:

        return []

    merged = [events[0]]

    for e in events[1:]:

        if (
            e - merged[-1]
        ) > 20:

            merged.append(e)

    cycles = []

    i = 0

    while i < len(merged) - 1:

        start = merged[i]
        end = merged[i + 1]

        if (
            grad[start] > 0
            and grad[end] < 0
        ):

            cycles.append(
                (
                    int(start),
                    int(end)
                )
            )

        i += 2

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
            max(0, start - 50):
            max(start - 10, 1)
        ]

    )

    plateau = np.median(

        smoothed[
            start + int(
                0.30 * (end - start)
            ):
            start + int(
                0.70 * (end - start)
            )
        ]

    )

    delta = plateau - baseline

    if abs(delta) < 0.05:

        return None

    # =====================================
    # RESPONSE LEVELS
    # =====================================

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

    # =====================================
    # RECOVERY LEVELS
    # =====================================

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

    t63 = (
        t63_cross - time[start]
    ) if not np.isnan(
        t63_cross
    ) else np.nan

    t90 = (
        t90_cross - time[start]
    ) if not np.isnan(
        t90_cross
    ) else np.nan

    t37 = (
        t37_cross - time[end]
    ) if not np.isnan(
        t37_cross
    ) else np.nan

    t10 = (
        t10_cross - time[end]
    ) if not np.isnan(
        t10_cross
    ) else np.nan

    exposure_time = (
        time[end] - time[start]
    )

    return {

        "Cycle":
            cycle_no,

        "Exposure Time (s)":
            round(
                float(exposure_time),
                2
            ),

        "Baseline":
            round(
                float(baseline),
                4
            ),

        "Plateau":
            round(
                float(plateau),
                4
            ),

        "Response Size":
            round(
                float(delta),
                4
            ),

        "T63 Response (s)":
            round(
                float(t63),
                2
            ) if not np.isnan(
                t63
            ) else np.nan,

        "T90 Response (s)":
            round(
                float(t90),
                2
            ) if not np.isnan(
                t90
            ) else np.nan,

        "T37 Recovery (s)":
            round(
                float(t37),
                2
            ) if not np.isnan(
                t37
            ) else np.nan,

        "T10 Recovery (s)":
            round(
                float(t10),
                2
            ) if not np.isnan(
                t10
            ) else np.nan
    }
    # =====================================================
# MAIN
# =====================================================

if uploaded is not None:

    try:

        if uploaded.name.endswith(".csv"):

            df = pd.read_csv(
                uploaded
            )

        else:

            df = pd.read_excel(
                uploaded
            )

        df.columns = [
            str(c).lower().strip()
            for c in df.columns
        ]

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

        smoothed = smooth_signal(
            signal
        )

        cycles = detect_cycles(
            signal
        )

        st.success(
            f"Detected {len(cycles)} cycles"
        )

        # ==========================================
        # CYCLE TABLE
        # ==========================================

        st.subheader(
            "Detected Cycles"
        )

        cycle_table = pd.DataFrame(
            [
                {
                    "Cycle": i + 1,
                    "Start Index": start,
                    "End Index": end,
                    "Duration (points)": end - start,
                    "Start Time (s)": round(
                        time[start],
                        2
                    ),
                    "End Time (s)": round(
                        time[end],
                        2
                    )
                }
                for i, (start, end)
                in enumerate(cycles)
            ]
        )

        st.dataframe(
            cycle_table,
            use_container_width=True
        )

        # ==========================================
        # OVERVIEW PLOT
        # ==========================================

        st.subheader(
            "Cycle Detection Overview"
        )

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=time,
                y=signal,
                mode="lines",
                name="Raw Signal",
                line=dict(
                    color="lightgray"
                )
            )
        )

        fig.add_trace(
            go.Scatter(
                x=time,
                y=smoothed,
                mode="lines",
                name="Smoothed Signal",
                line=dict(
                    color="blue",
                    width=2
                )
            )
        )

        for i, (start, end) in enumerate(cycles):

            fig.add_vrect(
                x0=time[start],
                x1=time[end],
                fillcolor="green",
                opacity=0.08,
                line_width=0
            )

            fig.add_trace(
                go.Scatter(
                    x=[time[start]],
                    y=[smoothed[start]],
                    mode="markers+text",
                    text=[f"ON {i+1}"],
                    textposition="top center",
                    marker=dict(
                        color="green",
                        size=12,
                        symbol="triangle-up"
                    ),
                    showlegend=False
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=[time[end]],
                    y=[smoothed[end]],
                    mode="markers+text",
                    text=[f"OFF {i+1}"],
                    textposition="bottom center",
                    marker=dict(
                        color="red",
                        size=12,
                        symbol="triangle-down"
                    ),
                    showlegend=False
                )
            )

        fig.update_layout(
            title="Detected Cycles",
            xaxis_title="Time (s)",
            yaxis_title="Signal",
            height=800
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # ==========================================
        # GRADIENT DIAGNOSTIC
        # ==========================================

        st.subheader(
            "Gradient Diagnostic"
        )

        grad = np.gradient(
            smoothed
        )

        fig_grad = go.Figure()

        fig_grad.add_trace(
            go.Scatter(
                x=time,
                y=grad,
                mode="lines",
                name="Gradient"
            )
        )

        fig_grad.add_hline(
            y=0,
            line_dash="dash"
        )

        fig_grad.update_layout(
            title="Gradient vs Time",
            xaxis_title="Time (s)",
            yaxis_title="Gradient",
            height=450
        )

        st.plotly_chart(
            fig_grad,
            use_container_width=True
        )

        # ==========================================
        # CALCULATIONS
        # ==========================================

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

                results.append(
                    row
                )

        results_df = pd.DataFrame(
            results
        )

        if not results_df.empty:

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
                    pd.DataFrame(
                        [avg_row]
                    )
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

        # ==========================================
        # INDIVIDUAL CYCLE PLOTS
        # ==========================================

        st.subheader(
            "Individual Cycle Diagnostics"
        )

        for i, (start, end) in enumerate(cycles):

            pad = 50

            left = max(
                0,
                start - pad
            )

            right = min(
                len(signal),
                end + pad
            )

            fig_cycle = go.Figure()

            fig_cycle.add_trace(
                go.Scatter(
                    x=time[left:right],
                    y=signal[left:right],
                    mode="lines",
                    name="Signal"
                )
            )

            fig_cycle.add_vline(
                x=time[start],
                line_color="green",
                line_width=3
            )

            fig_cycle.add_vline(
                x=time[end],
                line_color="red",
                line_width=3
            )

            fig_cycle.update_layout(
                title=f"Cycle {i+1}",
                xaxis_title="Time (s)",
                yaxis_title="Signal",
                height=400
            )

            st.plotly_chart(
                fig_cycle,
                use_container_width=True
            )

        # ==========================================
        # NORMALIZED RESPONSE
        # ==========================================

        st.subheader(
            "Normalized Response Per Cycle"
        )

        for i in range(
            len(cycles) - 1
        ):

            start, end = cycles[i]

            next_start = cycles[
                i + 1
            ][0]

            baseline = np.median(
                smoothed[
                    max(
                        0,
                        start - 50
                    ):
                    max(
                        start - 10,
                        1
                    )
                ]
            )

            plateau = np.median(
                smoothed[
                    start + int(
                        0.30 * (end - start)
                    ):
                    start + int(
                        0.70 * (end - start)
                    )
                ]
            )

            if abs(
                plateau - baseline
            ) < 1e-12:

                continue

            norm = (
                smoothed[
                    start:next_start
                ]
                - baseline
            ) / (
                plateau - baseline
            )

            fig_norm = go.Figure()

            fig_norm.add_trace(
                go.Scatter(
                    x=time[
                        start:next_start
                    ] - time[start],
                    y=norm,
                    mode="lines",
                    name=f"Cycle {i+1}"
                )
            )

            fig_norm.update_layout(
                title=f"Normalized Cycle {i+1}",
                xaxis_title="Time (s)",
                yaxis_title="Normalized Response",
                yaxis=dict(
                    range=[0, 1.1]
                ),
                height=400
            )

            st.plotly_chart(
                fig_norm,
                use_container_width=True
            )

        # ==========================================
        # EXPORT
        # ==========================================

        if not results_df.empty:

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
