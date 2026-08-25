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
                t - t.iloc[0]
            ) * 86400

    return pd.Series(
        np.arange(len(df))
    )

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
        0.05 *
        np.max(grad)
    )

    while idx > 0 and grad[idx] > threshold:

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

    threshold = (
        3 *
        np.std(grad)
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
            max(0, start - 60):
            start - 10
        ]
    )

    plateau = np.median(
        smoothed[
            start + 15:
            end - 15
        ]
    )

    delta = plateau - baseline

    if delta <= 0:
        return None

    level63 = baseline + 0.63 * delta
    level90 = baseline + 0.90 * delta

    t63_cross = interpolate_crossing(
        smoothed[start:end],
        time[start:end],
        level63,
        rising=True
    )

    t90_cross = interpolate_crossing(
        smoothed[start:end],
        time[start:end],
        level90,
        rising=True
    )

    level37 = baseline + 0.37 * delta
    level10 = baseline + 0.10 * delta

    t37_cross = interpolate_crossing(
        smoothed[end:next_start],
        time[end:next_start],
        level37,
        rising=False
    )

    t10_cross = interpolate_crossing(
        smoothed[end:next_start],
        time[end:next_start],
        level10,
        rising=False
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

    time = prepare_time(df).to_numpy()

    cycles = detect_cycles(signal)

    results = []

    for i in range(len(cycles) - 1):

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

    for c in results_df.columns[1:]:

        avg_row[c] = round(
            pd.to_numeric(
                results_df[c],
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

    st.subheader("Cycle Results")

    st.dataframe(
        results_df,
        use_container_width=True,
        hide_index=True
    )

    # Overlay plot

    st.subheader(
        "Overlay of All Cycles"
    )

    smoothed = smooth_signal(signal)

    overlay = go.Figure()

    for i in range(len(cycles) - 1):

        start, end = cycles[i]

        baseline = np.median(
            smoothed[
                max(0, start-60):
                start-10
            ]
        )

        plateau = np.median(
            smoothed[
                start+15:end-15
            ]
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
        line_dash="dash"
    )

    overlay.add_hline(
        y=0.90,
        line_dash="dash"
    )

    st.plotly_chart(
        overlay,
        use_container_width=True
    )

    # Raw signal plot

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

    st
