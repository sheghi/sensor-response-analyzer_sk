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


def prepare_time(df):

    try:

        t = pd.to_datetime(df["time"].astype(str))

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

    for rise in rising:

        candidates = falling[
            falling > rise
        ]

        if len(candidates) == 0:
            continue

        fall = candidates[0]

        if fall - rise > 300:
            cycles.append(
                (rise, fall)
            )

    return cycles


def analyse_cycle(signal, time, start, end, cycle_no):

    smoothed = smooth_signal(signal)

    baseline = np.median(
        smoothed[
            max(0, start - 300):start
        ]
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

    #########################
    # RESPONSE
    #########################

    gas_on_time = time[start]

    target30 = baseline + (
        0.30 * delta
    )

    target90 = baseline + (
        0.90 * delta
    )

    response_signal = smoothed[
        start:end
    ]

    response_time = time[
        start:end
    ]

    idx30 = np.where(
        response_signal >= target30
    )[0]

    idx90 = np.where(
        response_signal >= target90
    )[0]

    if len(idx30) == 0:
        t30 = np.nan
    else:
        t30 = (
            response_time[idx30[0]]
            - gas_on_time
        )

    if len(idx90) == 0:
        t90 = np.nan
    else:
        t90 = (
            response_time[idx90[0]]
            - gas_on_time
        )

    #########################
    # RECOVERY
    #########################

    gas_off_time = time[end]

    recovery_signal = smoothed[
        end:min(
            len(smoothed),
            end + 2000
        )
    ]

    recovery_time = time[
        end:min(
            len(time),
            end + 2000
        )
    ]

    target60 = baseline + (
        0.60 * delta
    )

    target10 = baseline + (
        0.10 * delta
    )

    idx60 = np.where(
        recovery_signal <= target60
    )[0]

    idx10 = np.where(
        recovery_signal <= target10
    )[0]

    if len(idx60) == 0:
        rec60 = np.nan
    else:
        rec60 = (
            recovery_time[idx60[0]]
            - gas_off_time
        )

    if len(idx10) == 0:
        rec10 = np.nan
    else:
        rec10 = (
            recovery_time[idx10[0]]
            - gas_off_time
        )

    return {
        "Cycle": cycle_no,
        "T30 Response (s)": round(t30, 2),
        "T90 Response (s)": round(t90, 2),
        "T60 Recovery (s)": round(rec60, 2),
        "T10 Recovery (s)": round(rec10, 2)
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

        if "signal" not in df.columns:
            st.error(
                "Column 'signal' not found"
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

    st.info("Upload a file to begin analysis.")
``
