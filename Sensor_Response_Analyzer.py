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
        t = pd.to_datetime(df["time"].astype(str))

        elapsed = (
            t - t.iloc[0]
        ).dt.total_seconds()

        return (elapsed / 10).to_numpy()

    except:
        return np.arange(len(df)) / 10


def smooth_signal(signal):

    return (
        pd.Series(signal)
        .rolling(window=25, center=True)
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

    for rise in rising:

        candidates = falling[falling > rise]

        if len(candidates) == 0:
            continue

        fall = candidates[0]

        if fall - rise > 300:
            cycles.append((rise, fall))

    return cycles


def first_above(sig, tm, target):

    idx = np.where(sig >= target)[0]

    if len(idx) == 0:
        return np.nan

    return tm[idx[0]]


def first_below(sig, tm, target):

    idx = np.where(sig <= target)[0]

    if len(idx) == 0:
        return np.nan

    return tm[idx[0]]


def analyse_cycle(signal, time, start, end, cycle):

    smoothed = smooth_signal(signal)

    baseline = np.median(
        smoothed[max(0, start - 300):start]
    )

    plateau_start = start + int(
        0.40 * (end - start)
    )

    plateau_end = start + int(
        0.80 * (end - start)
    )

    stable = np.median(
        smoothed[plateau_start:plateau_end]
    )

    delta = stable - baseline

    if delta <= 0:
        return None

    gas_on = start
    gas_off = end

    rise_region = smoothed[
        max(0, start - 100):end
    ]

    rise_time = time[
        max(0, start - 100):end
    ]

    target30 = baseline + 0.30 * delta
    target60 = baseline + 0.60 * delta
    target90 = baseline + 0.90 * delta

    t30 = (
        first_above(
            rise_region,
            rise_time,
            target30
        )
        - time[gas_on]
    )

    t60 = (
        first_above(
            rise_region,
            rise_time,
            target60
        )
        - time[gas_on]
    )

    t90 = (
        first_above(
            rise_region,
            rise_time,
            target90
        )
        - time[gas_on]
    )

    recovery_region = smoothed[
        end:min(len(smoothed), end + 1000)
    ]

    recovery_time = time[
        end:min(len(time), end + 1000)
    ]

    target90r = baseline + 0.90 * delta
    target60r = baseline + 0.60 * delta
    target30r = baseline + 0.30 * delta
    target10r = baseline + 0.10 * delta

    r90 = (
        first_below(
            recovery_region,
            recovery_time,
            target90r
        )
        - time[gas_off]
    )

    r60 = (
        first_below(
            recovery_region,
            recovery_time,
            target60r
        )
        - time[gas_off]
    )

    r30 = (
        first_below(
            recovery_region,
            recovery_time,
            target30r
        )
        - time[gas_off]
    )

    r10 = (
        first_below(
            recovery_region,
            recovery_time,
            target10r
        )
        - time[gas_off]
    )

    return {
        "Cycle": cycle,
        "T30 Response (s)": round(t30, 2),
        "T60 Response (s)": round(t60, 2),
        "T90 Response (s)": round(t90, 2),
        "T90 Recovery (s)": round(r90, 2),
        "T60 Recovery (s)": round(r60, 2),
        "T30 Recovery (s)": round(r30, 2),
        "T10 Recovery (s)": round(r10, 2)
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

        for n, (start, end) in enumerate(
            cycles,
            start=1
        ):

            res = analyse_cycle(
                signal,
                time,
                start,
                end,
                n
            )

            if res:
                results.append(res)

        results_df = pd.DataFrame(results)

        st.subheader("Cycle Results")
        st.dataframe(results_df)

        summary_df = pd.DataFrame({
            "Metric": [
                c for c in results_df.columns
                if c != "Cycle"
            ],
            "Average": [
                results_df[c].mean()
                for c in results_df.columns
                if c != "Cycle"
            ],
            "Std Dev": [
                results_df[c].std()
                for c in results_df.columns
                if c != "Cycle"
            ]
        })

        st.subheader("Average Results")
        st.dataframe(summary_df)

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
            file_name="sensor_response_analysis.xlsx"
        )

    except Exception as e:
        st.error(str(e))

else:
    st.info("Upload a file to begin.")
