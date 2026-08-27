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

st.title(
    "Gas Sensor Response Analyzer"
)

uploaded = st.file_uploader(
    "Upload Excel or CSV File",
    type=["xlsx", "csv"]
)

# =====================================================
# TIME
# =====================================================

def prepare_time(df):

    t = pd.to_numeric(
        df["time"],
        errors="coerce"
    )

    if t.isna().all():

        return np.array([])

    t = t.dropna()

    return (
        (t - t.iloc[0])
        * 86400
    ).to_numpy()

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

            x1 = time[i-1]
            x2 = time[i]

            y1 = signal[i-1]
            y2 = signal[i]

            if y1 == y2:

                return x1

            frac = (
                target - y1
            ) / (
                y2 - y1
            )

            return (
                x1
                + frac * (
                    x2 - x1
                )
            )

    return np.nan

# =====================================================
# CYCLE DETECTION
# =====================================================

def detect_cycles(
    signal,
    time
):

    smoothed = smooth_signal(
        signal
    )

    low_level = np.percentile(
        smoothed,
        10
    )

    high_level = np.percentile(
        smoothed,
        90
    )

    threshold = (
        low_level
        + high_level
    ) / 2

    state = (
        smoothed > threshold
    ).astype(int)

    transitions = np.diff(
        state
    )

    rises = np.where(
        transitions == 1
    )[0]

    falls = np.where(
        transitions == -1
    )[0]

    cycles = []

    j = 0

    for rise in rises:

        while (
            j < len(falls)
            and falls[j] < rise
        ):
            j += 1

        if j >= len(falls):
            break

        fall = falls[j]

        duration = (
            time[fall]
            - time[rise]
        )

        # Expected ON ≈ 1080 s

        if (
            700
            <= duration
            <= 1400
        ):

            cycles.append(
                (
                    int(rise),
                    int(fall)
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

    # -------------------------------------
    # Baseline before exposure
    # -------------------------------------

    baseline = np.median(

        smoothed[
            max(0, start - 20):
            max(start - 3, 1)
        ]

    )

    # -------------------------------------
    # Plateau during exposure
    # -------------------------------------

    plateau_start = (
        start
        + int(
            0.30 * (
                end - start
            )
        )
    )

    plateau_end = (
        start
        + int(
            0.80 * (
                end - start
            )
        )
    )

    plateau = np.median(

        smoothed[
            plateau_start:
            plateau_end
        ]

    )

    response_size = (
        plateau
        - baseline
    )

    if abs(response_size) < 0.01:

        return None

    # -------------------------------------
    # Response levels
    # -------------------------------------

    level63 = (
        baseline
        + 0.63 * response_size
    )

    level90 = (
        baseline
        + 0.90 * response_size
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
        rising=True
    )

    t90_cross = interpolate_crossing(
        response_signal,
        response_time,
        level90,
        rising=True
    )

    # -------------------------------------
    # Recovery levels
    # -------------------------------------

    level37 = (
        baseline
        + 0.37 * response_size
    )

    level10 = (
        baseline
        + 0.10 * response_size
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
        rising=False
    )

    t10_cross = interpolate_crossing(
        recovery_signal,
        recovery_time,
        level10,
        rising=False
    )

    # -------------------------------------
    # Calculate times
    # -------------------------------------

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

    recovery_window = (
        time[next_start]
        - time[end]
    )

    return {

        "Cycle":
            cycle_no,

        "Exposure Time (s)":
            round(
                exposure_time,
                1
            ),

        "Recovery Window (s)":
            round(
                recovery_window,
                1
            ),

        "Baseline":
            round(
                baseline,
                4
            ),

        "Plateau":
            round(
                plateau,
                4
            ),

        "Response Size":
            round(
                response_size,
                4
            ),

        "T63 Response (s)":
            round(
                t63,
                1
            ) if not np.isnan(
                t63
            ) else np.nan,

        "T90 Response (s)":
            round(
                t90,
                1
            ) if not np.isnan(
                t90
            ) else np.nan,

        "T37 Recovery (s)":
            round(
                t37,
                1
            ) if not np.isnan(
                t37
            ) else np.nan,

        "T10 Recovery (s)":
            round(
                t10,
                1
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

        # =====================================
        # VALIDATE REQUIRED COLUMNS
        # =====================================

        if "time" not in df.columns:

            st.error(
                "Column 'time' not found."
            )
            st.stop()

        if "signal" not in df.columns:

            st.error(
                "Column 'signal' not found."
            )
            st.stop()

        signal = pd.to_numeric(
            df["signal"],
            errors="coerce"
        )

        time_raw = pd.to_numeric(
            df["time"],
            errors="coerce"
        )

        mask = (

            signal.notna()

            &

            time_raw.notna()

        )

        df = df.loc[
            mask
        ].copy()

        if len(df) == 0:

            st.error(
                "No valid rows remain after cleaning."
            )
            st.stop()

        signal = df[
            "signal"
        ].to_numpy()

        time = prepare_time(
            df
        )

        if len(time) == 0:

            st.error(
                "Unable to generate time axis."
            )
            st.stop()

        smoothed = smooth_signal(
            signal
        )

        cycles = detect_cycles(
            signal,
            time
        )

        # =====================================
        # DATA SUMMARY
        # =====================================

        st.success(
            f"Detected {len(cycles)} cycles"
        )

        st.write(
            f"Rows loaded: {len(df)}"
        )

        st.write(
            f"Sample interval: "
            f"{np.median(np.diff(time)):.1f} s"
        )

        st.write(
            f"Total duration: "
            f"{time[-1]/60:.1f} min"
        )

        # =====================================
        # CYCLE TABLE
        # =====================================

        st.subheader(
            "Detected Cycles"
        )

        if len(cycles) > 0:

            cycle_table = pd.DataFrame(

                [
                    {
                        "Cycle":
                            i + 1,

                        "Start Time (s)":
                            round(
                                float(
                                    time[start]
                                ),
                                1
                            ),

                        "End Time (s)":
                            round(
                                float(
                                    time[end]
                                ),
                                1
                            ),

                        "Duration (s)":
                            round(
                                float(
                                    time[end]
                                    - time[start]
                                ),
                                1
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

        else:

            st.warning(
                "No cycles detected."
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
                    color="lightgrey"
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

        for start, end in cycles:

            fig.add_vrect(
                x0=time[start],
                x1=time[end],
                fillcolor="green",
                opacity=0.08,
                line_width=0
            )

            fig.add_vline(
                x=time[start],
                line_color="green"
            )

            fig.add_vline(
                x=time[end],
                line_color="red"
            )

        fig.update_layout(

            title="Detected ON Periods",

            xaxis_title="Time (s)",

            yaxis_title="Signal",

            height=700

        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # =====================================
        # RESULTS
        # =====================================

        results = []

        if len(cycles) > 1:

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
