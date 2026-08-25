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
# EVENT HELPERS
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


def refine_on(smoothed, idx):

    grad = np.gradient(smoothed)

    threshold = (
        0.05 * np.max(grad)
    )

    while (
        idx > 0
        and grad[idx] > threshold
    ):
        idx -= 1

    return idx


def refine_off(smoothed, idx):

    grad = np.gradient(smoothed)

    threshold = (
        0.05 *
        abs(np.min(grad))
    )

    while (
        idx > 0
        and abs(grad[idx]) > threshold
    ):
        idx -= 1

    return idx


# =====================================================
# DETECT CYCLES
# =====================================================

def detect_cycles(signal):

    smoothed = smooth_signal(signal)

    grad = np.gradient(smoothed)

    grad_std = np.std(grad)

    rise_candidates = np.where(
        grad > 3 * grad_std
    )[0]

    fall_candidates = np.where(
        grad < -3 * grad_std
    )[0]

    rises = merge_events(
        rise_candidates,
        gap=20
    )

    falls = merge_events(
        fall_candidates,
        gap=20
    )

    rises = [
        refine_on(smoothed, r)
        for r in rises
    ]

    falls = [
        refine_off(smoothed, f)
        for f in falls
    ]

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
            max(0, start - 40):
            max(start - 5, 1)
        ]

    )

    plateau = np.median(

        smoothed[
            start + int(
                0.75 * (end - start)
            ):
            max(end - 5, start + 1)
        ]

    )

    delta = plateau - baseline

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
        t63_cross - time[start]
    ) if not np.isnan(t63_cross) else np.nan

    t90 = (
        t90_cross - time[start]
    ) if not np.isnan(t90_cross) else np.nan

    t37 = (
        t37_cross - time[end]
    ) if not np.isnan(t37_cross) else np.nan

    t10 = (
        t10_cross - time[end]
    ) if not np.isnan(t10_cross) else np.nan

    return {

        "Cycle": cycle_no,

        "T63 Response (s)":
            round(float(t63), 2)
            if not np.isnan(t63)
            else np.nan,

        "T90 Response (s)":
            round(float(t90), 2)
            if not np.isnan(t90)
            else np.nan,

        "T37 Recovery (s)":
            round(float(t37), 2)
            if not np.isnan(t37)
            else np.nan,

        "T10 Recovery (s)":
            round(float(t10), 2)
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

        st.subheader(
            "Cycle Results"
        )

        st.dataframe(
            results_df,
            hide_index=True,
            use_container_width=True
        )

        # PLOT

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=time,
                y=signal,
                name="Signal",
                mode="lines"
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
                    name="Gas ON"
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
                    name="Gas OFF"
                )
            )

        fig.update_layout(
            title="Detected Gas ON/OFF Events",
            xaxis_title="Time (s)",
            yaxis_title="Signal"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # EXPORT

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
