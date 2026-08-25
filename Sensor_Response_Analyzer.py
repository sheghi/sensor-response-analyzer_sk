import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
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


# ---------------------------------------------------
# TIME HANDLING
# ---------------------------------------------------

def prepare_time(df):

    time_col = pd.to_numeric(
        df["time"],
        errors="coerce"
    )

    if time_col.notna().all():

        # Excel fractional days to seconds

        return (
            (time_col - time_col.iloc[0])
            * 86400
        ).to_numpy()

    try:

        t = pd.to_datetime(df["time"])

        return (
            t - t.iloc[0]
        ).dt.total_seconds().to_numpy()

    except Exception:

        return np.arange(len(df))


# ---------------------------------------------------
# SMOOTHING
# ---------------------------------------------------

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


# ---------------------------------------------------
# CYCLE DETECTION
# ---------------------------------------------------

def detect_cycles(signal):

    smoothed = smooth_signal(signal)

    grad = np.gradient(smoothed)

    grad_std = np.std(grad)

    rise_points = np.where(
        grad > 2.5 * grad_std
    )[0]

    fall_points = np.where(
        grad < -2.5 * grad_std
    )[0]

    if len(rise_points) == 0:
        return []

    min_gap = max(
        50,
        len(signal) // 25
    )

    rises = [rise_points[0]]

    for p in rise_points[1:]:

        if p - rises[-1] > min_gap:
            rises.append(p)

    falls = []

    if len(fall_points):

        falls.append(fall_points[0])

        for p in fall_points[1:]:

            if p - falls[-1] > min_gap:
                falls.append(p)

    cycles = []

    for rise in rises:

        valid_falls = [
            f for f in falls
            if f > rise
        ]

        if len(valid_falls) == 0:
            continue

        fall = valid_falls[0]

        if fall - rise > 20:

            cycles.append(
                (rise, fall)
            )

    return cycles


# ---------------------------------------------------
# INTERPOLATION
# ---------------------------------------------------

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

            y1 = signal[i - 1]
            y2 = signal[i]

            t1 = time[i - 1]
            t2 = time[i]

            if y1 == y2:
                return t1

            frac = (
                target - y1
            ) / (
                y2 - y1
            )

            return t1 + frac * (t2 - t1)

    return np.nan


# ---------------------------------------------------
# CYCLE ANALYSIS
# ---------------------------------------------------

def analyse_cycle(
        signal,
        time,
        start,
        end,
        cycle_no
):

    smoothed = smooth_signal(signal)
    grad = np.gradient(smoothed)

    rise_window_start = max(
        0,
        start - 200
    )

    rise_window_end = min(
        len(signal),
        start + 200
    )

    rise_grad = grad[
        rise_window_start:
        rise_window_end
    ]

    max_grad_idx = (
            np.argmax(rise_grad)
            + rise_window_start
    )

    rise_threshold = (
            0.05
            * np.max(rise_grad)
    )

    gas_on_idx = max_grad_idx

    while (
            gas_on_idx > rise_window_start
            and grad[gas_on_idx]
            > rise_threshold
    ):
        gas_on_idx -= 1

    fall_window_start = max(
        gas_on_idx + 100,
        start
    )

    fall_window_end = min(
        len(signal),
        end + 300
    )

    fall_grad = grad[
        fall_window_start:
        fall_window_end
    ]

    min_grad_idx = (
            np.argmin(fall_grad)
            + fall_window_start
    )

    fall_threshold = (
            0.05
            * abs(np.min(fall_grad))
    )

    gas_off_idx = min_grad_idx

    while (
            gas_off_idx > fall_window_start
            and abs(grad[gas_off_idx])
            > fall_threshold
    ):
        gas_off_idx -= 1

    gas_on_time = time[gas_on_idx]
    gas_off_time = time[gas_off_idx]

    baseline = np.mean(
        smoothed[
        max(0, gas_on_idx - 200):
        gas_on_idx
        ]
    )

    plateau_start = gas_on_idx + int(
        0.7 *
        (gas_off_idx - gas_on_idx)
    )

    plateau = np.mean(
        smoothed[
        plateau_start:
        gas_off_idx
        ]
    )

    delta = plateau - baseline

    if abs(delta) < 0.01:
        return None

    response100 = plateau

    response90_display = (
            response100 * 0.90
    )

    response63_display = (
            response100 * 0.63
    )

    response_signal = (
                              smoothed[
                              gas_on_idx:
                              gas_off_idx
                              ] - baseline
                      ) / (
                              response100 - baseline
                      )

    response_time = time[
        gas_on_idx:
        gas_off_idx
    ]

    t63_cross = interpolate_crossing(
        response_signal,
        response_time,
        0.63,
        rising=True
    )

    t90_cross = interpolate_crossing(
        response_signal,
        response_time,
        0.90,
        rising=True
    )

    t63 = (
            t63_cross
            - gas_on_time
    ) if not np.isnan(t63_cross) else np.nan

    t90 = (
            t90_cross
            - gas_on_time
    ) if not np.isnan(t90_cross) else np.nan

    recovery_signal = smoothed[
                      gas_off_idx:
                      ]

    recovery_time = time[
                    gas_off_idx:
                    ]

    recovery_norm = (
                            recovery_signal
                            - baseline
                    ) / (
                            response100
                            - baseline
                    )

    t37_cross = interpolate_crossing(
        recovery_norm,
        recovery_time,
        0.37,
        rising=False
    )

    t10_cross = interpolate_crossing(
        recovery_norm,
        recovery_time,
        0.10,
        rising=False
    )

    t37 = (
            t37_cross
            - gas_off_time
    ) if not np.isnan(t37_cross) else np.nan

    t10 = (
            t10_cross
            - gas_off_time
    ) if not np.isnan(t10_cross) else np.nan

    return {

        "Cycle": cycle_no,
        "Baseline": round(float(baseline), 4),
        "Response 100%": round(float(response100), 4),
        "Response 90%": round(float(response90_display), 4),
        "Response 63%": round(float(response63_display), 4),
        "Amplitude": round(float(delta), 4),
        "Gas ON (s)": round(float(gas_on_time), 2),
        "Gas OFF (s)": round(float(gas_off_time), 2),

        "T63 Response (s)": round(float(t63), 2)
        if not np.isnan(t63)
        else np.nan,

        "T90 Response (s)": round(float(t90), 2)
        if not np.isnan(t90)
        else np.nan,

        "T37 Recovery (s)": round(float(t37), 2)
        if not np.isnan(t37)
        else np.nan,

        "T10 Recovery (s)": round(float(t10), 2)
        if not np.isnan(t10)
        else np.nan
    }


# ---------------------------------------------------
# MAIN APP
# ---------------------------------------------------

if uploaded:

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

        st.write(
            f"Detected cycles: {len(cycles)}"
        )

        results = []

        for cycle_no, (
                start,
                end
        ) in enumerate(
            cycles,
            start=1
        ):

            res = analyse_cycle(
                signal,
                time,
                start,
                end,
                cycle_no
            )

            if res is not None:
                results.append(res)

        if len(results) == 0:

            st.error(
                "No valid cycles detected."
            )
            st.stop()

        results_df = pd.DataFrame(results)

        st.subheader(
            "Cycle Results"
        )

        st.dataframe(
            results_df,
            use_container_width=
