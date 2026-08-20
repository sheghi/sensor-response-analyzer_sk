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
    "Upload Excel or CSV",
    type=["xlsx", "csv"]
)


def prepare_time(df):

    try:
        t = pd.to_datetime(
            df["time"].astype(str)
        )

        elapsed = (
            t - t.iloc[0]
        ).dt.total_seconds()

        return elapsed.to_numpy()

    except:
        return np.arange(len(df))


def smooth_signal(signal):

    return (
        pd.Series(signal)
        .rolling(window=21, center=True)
        .mean()
        .bfill()
        .ffill()
        .to_numpy()
    )


def detect_cycles(signal):

    smoothed = smooth_signal(signal)

    baseline = np.percentile(smoothed, 10)
    plateau = np.percentile(smoothed, 90)

    threshold = baseline + (
        0.2 * (plateau - baseline)
    )

    active = smoothed > threshold

    rising = np.where(
        (active[1:] == True) &
        (active[:-1] == False)
    )[0]

    falling = np.where(
        (active[1:] == False) &
        (active[:-1] == True)
    )[0]

    cycles = []

    for start in rising:

        candidates = falling[
            falling > start
        ]

        if len(candidates) == 0:
            continue

        end = candidates[0]

        if (end - start) > 300:
            cycles.append((start, end))

    return cycles


def find_crossing(sig, tm, target, mode):

    if mode == "above":
        idx = np.where(sig >= target)[0]
    else:
        idx = np.where(sig <= target)[0]

    if len(idx) == 0:
        return np.nan

    return tm[idx[0]]


def analyse_cycle(signal, time, start, end, cycle_no):

    smoothed = smooth_signal(signal)

    baseline = np.median(
        smoothed[max(0, start - 300):start]
    )

    plateau_start = start + int(
        0.5 * (end - start)
    )

    plateau_end = start + int(
        0.8 * (end - start)
    )

    stable = np.median(
        smoothed[
            plateau_start:plateau_end
        ]
    )

    delta = stable - baseline

    if delta <= 0:
        return None

    #
    # RESPONSE
    #

    gas_on_time = time[start]

    response_signal = smoothed[
        max(0, start - 100):end
    ]

    response_time = time[
        max(0, start - 100):end
    ]

    target30 = baseline + (
        0.30 * delta
    )

    target90 = baseline + (
        0.90 * delta
    )

    t30 = find_crossing(
        response_signal,
        response_time,
        target30,
        "above"
    )

    t90 = find_crossing(
        response_signal,
        response_time,
        target90,
        "above"
    )

    response30 = t30 - gas_on_time
    response90 = t90 - gas_on_time

    #
    # RECOVERY
    #

    gradient = np.gradient(smoothed)

    search_start = max(0, end - 500)
    search_end = min(
        len(signal),
        end + 100
    )

    grad_region = gradient[
        search_start:search_end
    ]

    gas_off_idx = (
        np.argmin(grad_region)
        + search_start
    )

    gas_off_idx = max(
        0,
        gas_off_idx - 150
    )

    gas_off_time = time[
        gas_off_idx
    ]

    recovery_signal = smoothed[
        gas_off_idx:min(
            len(signal),
            gas_off_idx + 1500
        )
    ]

    recovery_time = time[
        gas_off_idx:min(
            len(time),
            gas_off_idx + 1500
        )
    ]

    target60 = baseline + (
        0.60 * delta
    )

    target10 = baseline + (
        0.10 * delta
    )

    r60 = find_crossing(
        recovery_signal,
        recovery_time,
        target60,
        "below"
    )

    r10 = find_crossing(
        recovery_signal,
        recovery_time,
        target10,
        "below"
    )

    recovery60 = r60 - gas_off_time
    recovery10 = r10 - gas_off_time

    return {
        "Cycle": cycle_no,
        "T30 Response (s)": round(response30, 2),
        "T90 Response (s)": round(response90, 2),
        "T60 Recovery (s)": round(recovery60, 2),
        "T10 Recovery (s)": round(recovery10, 2)
    }


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

        signal = pd.to_numeric(
            df["signal"],
            errors="coerce"
        )

        mask = signal.notna()

        signal = signal[mask].to_numpy()

        df = df.loc[mask]

        time = prepare_time(df)

        cycles = detect_cycles(signal)

        st.write(
            f"Detected cycles: {len(cycles)}"
        )

        results = []

        for i, (start, end) in enumerate(
            cycles,
            start=1
        ):

            result = analyse_cycle(
                signal,
                time,
                start,
                end,
                i
            )

            if result is not None:
                results.append(result)

        if len(results) == 0:

            st.error(
                "No valid cycles detected."
            )

            st.stop()

        results_df = pd.DataFrame(
            results
        )

        st.subheader(
            "Cycle Results"
        )

        st.dataframe(
            results_df,
            use_container_width=True
        )

        summary_df = pd.DataFrame({
            "Metric": [
                "T30 Response (s)",
                "T90 Response (s)",
                "T60 Recovery (s)",
                "T10 Recovery (s)"
            ],
            "Average": [
                results_df["T30 Response (s)"].mean(),
                results_df["T90 Response (s)"].mean(),
                results_df["T60 Recovery (s)"].mean(),
                results_df["T10 Recovery (s)"].mean()
            ],
            "Std Dev": [
                results_df["T30 Response (s)"].std(),
                results_df["T90 Response (s)"].std(),
                results_df["T60 Recovery (s)"].std(),
                results_df["T10 Recovery (s)"].std()
            ]
        })

        st.subheader(
            "Average Results"
        )

        st.dataframe(
            summary_df,
            use_container_width=True
        )

        plot_df = pd.DataFrame({
            "Time (s)": time,
            "Signal": signal
        })

        fig = px.line(
            plot_df,
            x="Time (s)",
            y="Signal",
            title="Sensor Response"
        )

        for start, end in cycles:

            fig.add_vrect(
                x0=time[start],
                x1=time[end],
                fillcolor="green",
                opacity=0.15,
                line_width=0
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
            "Download Analysis",
            data=buffer.getvalue(),
            file_name="sensor_response_analysis.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    except Exception as e:

        st.error(str(e))

else:

    st.info("Upload a file to begin.")
