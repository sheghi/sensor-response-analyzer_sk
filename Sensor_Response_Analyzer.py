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

    return np.arange(
        len(df)
    )

# =====================================================
# SMOOTHING
# =====================================================

def smooth_signal(signal):

    return (

        pd.Series(signal)

        .rolling(
            window=9,
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

    for i in range(
        1,
        len(signal)
    ):

        if rising:

            crossed = (
                signal[i-1] < target
                and signal[i] >= target
            )

        else:

            crossed = (
                signal[i-1] > target
                and signal[i] <= target
            )

        if crossed:

            t1 = time[i-1]
            t2 = time[i]

            y1 = signal[i-1]
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
                + frac * (
                    t2 - t1
                )
            )

    return np.nan

# =====================================================
# MERGE
# =====================================================

def merge_events(
    events,
    gap=80
):

    if len(events) == 0:

        return []

    merged = [events[0]]

    for e in events[1:]:

        if (
            e - merged[-1]
        ) > gap:

            merged.append(
                int(e)
            )

    return merged

# =====================================================
# CYCLE DETECTION
# =====================================================

def detect_cycles(signal):

    smoothed = smooth_signal(
        signal
    )

    low_level = np.percentile(
        smoothed,
        20
    )

    high_level = np.percentile(
        smoothed,
        80
    )

    threshold = (
        low_level
        + high_level
    ) / 2

    state = (
        smoothed > threshold
    ).astype(int)

    changes = np.diff(state)

    rises = np.where(
        changes == 1
    )[0]

    falls = np.where(
        changes == -1
    )[0]

    rises = merge_events(
        rises,
        gap=80
    )

    falls = merge_events(
        falls,
        gap=80
    )

    cycles = []

    j = 0

    for rise in rises:

        while (
            j < len(falls)
            and falls[j] < rise
        ):

            j += 1

        if j < len(falls):

            duration = (
                falls[j]
                - rise
            )

            if duration > 80:

                cycles.append(
                    (
                        int(rise),
                        int(falls[j])
                    )
                )

            j += 1

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

    smoothed = smooth_signal(
        signal
    )

    # =====================================
    # BASELINE
    # =====================================

    baseline = np.median(

        smoothed[
            max(
                0,
                start - 40
            ):
            max(
                start - 5,
                1
            )
        ]

    )

    # =====================================
    # PLATEAU
    # =====================================

    plateau = np.median(

        smoothed[
            start + int(
                0.25 * (
                    end - start
                )
            ):
            start + int(
                0.75 * (
                    end - start
                )
            )
        ]

    )

    delta = (
        plateau - baseline
    )

    if abs(delta) < 0.01:

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

    # =====================================
    # CALCULATED TIMES
    # =====================================

    t63 = (
        t63_cross
        - time[start]
    ) if not np.isnan(
        t63_cross
    ) else np.nan

    t90 = (
        t90_cross
        - time[start]
    ) if not np.isnan(
        t90_cross
    ) else np.nan

    t37 = (
        t37_cross
        - time[end]
    ) if not np.isnan(
        t37_cross
    ) else np.nan

    t10 = (
        t10_cross
        - time[end]
    ) if not np.isnan(
        t10_cross
    ) else np.nan

    exposure_time = (
        time[end]
        - time[start]
    )

    recovery_time_total = (
        time[next_start]
        - time[end]
    )

    return {

        "Cycle":
            cycle_no,

        "Exposure Time (s)":
            round(
                float(
                    exposure_time
                ),
                2
            ),

        "Recovery Window (s)":
            round(
                float(
                    recovery_time_total
                ),
                2
            ),

        "Baseline":
            round(
                float(
                    baseline
                ),
                4
            ),

        "Plateau":
            round(
                float(
                    plateau
                ),
                4
            ),

        "Response Size":
            round(
                float(
                    delta
                ),
                4
            ),

        "T63 Response (s)":
            round(
                float(
                    t63
                ),
                2
            ) if not np.isnan(
                t63
            ) else np.nan,

        "T90 Response (s)":
            round(
                float(
                    t90
                ),
                2
            ) if not np.isnan(
                t90
            ) else np.nan,

        "T37 Recovery (s)":
            round(
                float(
                    t37
                ),
                2
            ) if not np.isnan(
                t37
            ) else np.nan,

        "T10 Recovery (s)":
            round(
                float(
                    t10
                ),
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

        time = prepare_time(
            df
        )

        smoothed = smooth_signal(
            signal
        )

        cycles = detect_cycles(
            signal
        )

        # =====================================
        # SUMMARY
        # =====================================

        st.success(
            f"Detected {len(cycles)} cycles"
        )

        st.write(
            f"Total points: {len(signal)}"
        )

        st.write(
            f"Final time: {round(time[-1], 1)} s"
        )

        # =====================================
        # CYCLE TABLE
        # =====================================

        st.subheader(
            "Detected Cycles"
        )

        cycle_table = pd.DataFrame(
            [
                {
                    "Cycle": i + 1,
                    "Start Index": start,
                    "End Index": end,
                    "Duration (s)": round(
                        time[end]
                        - time[start],
                        2
                    ),
                    "Start Time (s)": round(
                        time[start],
                        2
                    ),
                    "End Time (s)": round(
                        time[end],
                        2
                    )
                }
                for i, (
                    start,
                    end
                ) in enumerate(
                    cycles
                )
            ]
        )

        st.dataframe(
            cycle_table,
            use_container_width=True
        )

        # =====================================
        # OVERVIEW PLOT
        # =====================================

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

        for i, (
            start,
            end
        ) in enumerate(
            cycles
        ):

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
                    text=[
                        f"ON {i+1}"
                    ],
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
                    text=[
                        f"OFF {i+1}"
                    ],
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
            title="Detected ON/OFF Events",
            xaxis_title="Time (s)",
            yaxis_title="Signal",
            height=800
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # =====================================
        # CALCULATIONS
        # =====================================

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

                "Cycle":
                    "Average"

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
                hide_index=True,
                use_container_width=True
            )

        # =====================================
        # INDIVIDUAL CYCLE PLOTS
        # =====================================

        st.subheader(
            "Individual Cycle Diagnostics"
        )

        for i, (
            start,
            end
        ) in enumerate(
            cycles
        ):

            left = max(
                0,
                start - 30
            )

            right = min(
                len(signal),
                end + 30
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

        # =====================================
        # EXPORT
        # =====================================

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
