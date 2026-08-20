import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import BytesIO

st.set_page_config(page_title="Sensor Response Analyzer", layout="wide")

st.title("Sensor Response Analyzer")
st.markdown("Upload CSV or Excel files containing **time** and **signal** columns.")

uploaded = st.file_uploader("Upload file", type=["csv","xlsx"])


def calc_metrics(df):
    time = df['time'].to_numpy()
    signal = df['signal'].to_numpy()

    initial = np.mean(signal[:50])
    final = np.max(signal)
    span = final - initial

    def crossing(frac):
        target = initial + (final - initial) * frac
        idx = np.where(signal >= target)[0]

        if len(idx) == 0:
            return np.nan
        return float(time[idx[0]])

    t10 = crossing(0.10)
    t50 = crossing(0.50)
    t90 = crossing(0.90)
    t95 = crossing(0.95)

    rise = t90 - t10 if not np.isnan(t90) and not np.isnan(t10) else np.nan

    peak = np.max(signal)
    overshoot = ((peak-final)/abs(final))*100 if final != 0 else 0

    rms_noise = float(np.std(signal))

    tau = crossing(0.632)

    tol = abs(final) * 0.02
    settling = np.nan
    for i in range(len(signal)-1, -1, -1):
        if abs(signal[i]-final) > tol:
            settling = time[min(i+1, len(signal)-1)]
            break

    return {
        'T10': t10,
        'T50': t50,
        'T90': t90,
        'T95': t95,
        'Tau (63.2%)': tau,
        'Rise Time': rise,
        'Settling Time': settling,
        'Overshoot (%)': overshoot,
        'RMS Noise': rms_noise
    }

if uploaded:
    if uploaded.name.endswith('.csv'):
        df = pd.read_csv(uploaded)
    else:
        df = pd.read_excel(uploaded)

    df.columns = [c.lower().strip() for c in df.columns]

    if 'time' not in df.columns or 'signal' not in df.columns:
        st.error("File must contain columns named 'time' and 'signal'.")
    else:
        metrics = calc_metrics(df)

        st.subheader('Metrics')
        c1,c2,c3,c4 = st.columns(4)
        items=list(metrics.items())
        for idx,(k,v) in enumerate(items):
            [c1,c2,c3,c4][idx%4].metric(k, f'{v:.4f}' if pd.notna(v) else '-')

        fig = px.line(df, x='time', y='signal', title='Sensor Response Curve')
        st.plotly_chart(fig, use_container_width=True)

        st.subheader('Data Preview')
        st.dataframe(df.head(50), use_container_width=True)

        export_df = pd.DataFrame([metrics])
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            export_df.to_excel(writer, index=False, sheet_name='Results')

        st.download_button(
            'Download Results (Excel)',
            data=buffer.getvalue(),
            file_name='sensor_analysis_results.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
else:
    st.info('Upload a file to begin analysis.')
