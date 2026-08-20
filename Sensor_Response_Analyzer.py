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
        t = pd.to_datetime(
            df["time"].astype(str)
        )

        return (
            t - t.iloc[0]
        ).dt.total_seconds().to_numpy()

    except:
        return np.arange(len(df))


def smooth_signal(signal):

    return (
        pd.Series(signal)
        .rolling(
            window=21,
            center=True
        )
        .mean()
        .bfill()
        .ffill()
        .to_numpy()
    )


def detect_cycles(signal):

    smoothed = smooth_signal(signal)

    baseline = np.percentile(
        smoothed,
        10
    )

    plateau = np.percentile(
        smoothed,
        90
    )

    threshold = baseline + (
        0.2 * (plateau - baseline)
    )

    active = smoothed > threshold

    rises = np.where(
        (active[1:] == True)
        & (active[:-1] == False)
    )[0]

    falls = np.where(
        (active[1:] == False)
        & (active[:-1] == True)
    )[0]

    cycles = []

    for rise in rises:

        candidates = falls[
            falls > rise
        ]

        if len(candidates) == 0:
            continue

        fall = candidates[0]

        if (fall - rise) > 300:
            cycles.append(
                (rise, fall)
            )

    return cycles


def analyse_cycle(
    signal,
    time,
    start,
    end,
    cycle_no
):

    smoothed = smooth_signal(signal)

    baseline = np.median(
        smoothed[
            max(0, start - 300):start
        ]
    )

    plateau_start = start + int(
        0.4 * (end - start)
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

    # ------------------------
    # GAS ON DETECTION
    # ------------------------

    grad = np.gradient(smoothed)

    rise_window_start = max(
        0,
        start - 200
    )

    rise_window_end = min(
        len(smoothed),
        start + 200
    )

    rise_grad = grad[
        rise_window_start:rise_window_end
    ]

    gas_on_idx = (
        np.argmax(rise_grad)
        + rise_window_start
    )

    gas_on_time = time[
        gas_on_idx
    ]

    # ------------------------
    # RESPONSE
    # ------------------------

    target30 = baseline + (
        0.30 * delta
    )

    target90 = baseline + (
        0.90 * delta
    )

    response_signal = smoothed[
        gas_on_idx:end
    ]

    response_time = time[
        gas_on_idx:end
    ]

    idx30 = np.where(
        response_signal >= target30
    )[0]

    idx90 = np.where(
        response_signal >= target90
    )[0]

    t30 = np.nan
    t90 = np.nan

    if len(idx30) > 0:
        t30 = (
            response_time[idx30[0]]
            - gas_on_time
        )

    if len(idx90) > 0:
        t90 = (
            response_time[idx90[0]]
            - gas_on_time
        )

    # ------------------------
    # GAS OFF DETECTION
    # ------------------------

    fall_window_start = max(
        plateau_start,
        start
    )

    fall_window_end = min(
        len(smoothed),
        end + 300
    )

    fall_grad = grad[
        fall_window_start:fall_window_end
    ]

    gas_off_idx = (
        np.argmin(fall_grad)
        + fall_window_start
    )

    gas_off_time = time[
        gas_off_idx
    ]

    # ------------------------
    # RECOVERY
    # ------------------------

    recovery_signal = smoothed[
        gas_off_idx:
    ]

    recovery_time = time[
        gas_off_idx:
    ]

    # 60% recovered
    target60 = baseline + (
        0.40 * delta
    )

    # 90% recovered
    target10 = baseline + (
        0.10 * delta
    )

    idx60 = np.where(
        recovery_signal <= target60
    )[0]

    idx10 = np.where(
        recovery_signal <= target10
    )[0]

    t60 = np.nan
    t10 = np.nan

    if len(idx60) > 0:
        t60 = (
            recovery_time[idx60[0]]
            - gas_off_time
        )

    if len(idx10) > 0:
        t10 = (
            recovery_time[idx10[0]]
            - gas_off_time
        )

    return {
        "Cycle": cycle_no,
        "T30 Response (s)": round(
            float(t30),
            2
        ) if not np.isnan(t30) else np.nan,
        "T90 Response (s)": round(
            float(t90),
            2
        ) if not np.isnan(t90) else np.nan,
        "T60 Recovery (s)": round(
            float(t60),
            2
        ) if not np.isnan(t60) else np.nan,
        "T10 Recovery (s)": round(
            float(t10),
            2
        ) if not np.isnan(t10) else np.nan
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

        for cycle_no, (start, end) in enumerate(
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
                results_df[
                    "T30 Response (s)"
                ].mean(),
                results_df[
                    "T90 Response (s)"
                ].mean(),
                results_df[
                    "T60 Recovery (s)"
                ].mean(),
                results_df[
                    "T10 Recovery (s)"
                ].mean()
            ],
            "Std Dev": [
                results_df[
                    "T30 Response (s)"
                ].std(),
                results_df[
                    "T90 Response (s)"
                ].std(),
                results_df[
                    "T60 Recovery (s)"
                ].std(),
                results_df[
                    "T10 Recovery (s)"
                ].std()
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

    st.info(
        "Upload a file to begin analysis."
    )
