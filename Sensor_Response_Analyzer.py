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
# EVENT MERGING
# =====================================================

def merge_events(events, gap=50):

    merged = []

    for e in events:

        if (
            len(merged) == 0
            or e - merged[-1] > gap
        ):

            merged.append(int(e))

    return merged


# =====================================================
# CYCLE DETECTION
# =====================================================

def detect_cycles(signal):

    smoothed = smooth_signal(signal)

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
        + 0.5 * (
            high_level - low_level
        )
    )

    gas_on = (
        smoothed > threshold
    )

    changes = np.diff(
        gas_on.astype(int)
    )

    rises = np.where(
        changes == 1
    )[0]

    falls = np.where(
        changes == -1
    )[0]

    rises = merge_events(
        rises,
        gap=50
    )

    falls = merge_events(
        falls,
        gap=50
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

    smoothed = smooth_signal(signal)

    baseline = np.median(

        smoothed[
            max(0, start - 80):
            max(start - 20, 1)
        ]

    )

    plateau_start = (
        start
        + int(0.30 * (end - start))
    )

    plateau_end = (
        start
        + int(0.70 * (end - start))
    )

    plateau = np.median(

        smoothed[
            plateau_start:
            plateau_end
        ]

    )

    delta = (
        plateau - baseline
    )

    if abs(delta) < 0.05:
        return None

    # RESPONSE

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

    # RECOVERY

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
        t63_cross
        - time[start]
    ) if not np.isnan(t63_cross) else np.nan

    t90 = (
        t90_cross
        - time[start]
    ) if not np.isnan(t90_cross) else np.nan

    t37 = (
        t37_cross
        - time[end]
    ) if not np.isnan(t37_cross) else np.nan

    t10 = (
        t10_cross
        - time[end]
    ) if not np.isnan(t10_cross) else np.nan

    return {

        "Cycle": cycle_no,

        "T63 Response (s)":
            round(float(t63), 2),

        "T90 Response (s)":
            round(float(t90), 2),

        "T37 Recovery (s)":
            round(float(t37), 2),

        "T10 Recovery (s)":
            round(float(t10), 2)

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

            str(c)
            .lower()
            .strip()

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

        cycles = detect_cycles(
            signal
        )

        st.success(
            f"Detected {len(cycles)} cycles"
        )

        results = []

        for i in range(
            len(cycles) - 1
        ):

            start, end = cycles[i]

            next_start = (
                cycles[i + 1][0]
            )

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
        # MAIN PLOT
        # ==========================================

        smoothed = smooth_signal(
            signal
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

        fig.add_trace(

            go.Scatter(

                x=time,
                y=smoothed,
                mode="lines",
                name="Smoothed"

            )

        )

        for start, end in cycles:

            fig.add_vline(
                x=time[start],
                line_color="green"
            )

            fig.add_vline(
                x=time[end],
                line_color="red"
            )

        fig.update_layout(

            title="Detected ON/OFF Events",

            xaxis_title="Time (s)",

            yaxis_title="Signal"

        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # ==========================================
        # NORMALIZED CYCLE PLOTS
        # ==========================================

        st.subheader(
            "Normalized Response Per Cycle"
        )

        for i in range(
            len(cycles) - 1
        ):

            start, end = cycles[i]

            next_start = (
                cycles[i + 1][0]
            )

            baseline = np.median(

                smoothed[
                    max(0, start - 80):
                    max(start - 20, 1)
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

            norm = (

                smoothed[
                    start:next_start
                ]

                - baseline

            ) / (

                plateau - baseline

            )

            fig_cycle = go.Figure()

            fig_cycle.add_trace(

                go.Scatter(

                    x=time[
                        start:next_start
                    ] - time[start],

                    y=norm,

                    mode="lines",

                    name=f"Cycle {i+1}"

                )

            )

            fig_cycle.update_layout(

                title=f"Cycle {i+1}",

                xaxis_title="Time (s)",

                yaxis_title="Normalized Response",

                yaxis=dict(
                    range=[0, 1.1]
                )

            )

            st.plotly_chart(
                fig_cycle,
                use_container_width=True
            )

        # ==========================================
        # EXPORT
        # ==========================================

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
