import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import streamlit as st

# Título de la aplicación
st.title("Análisis de Referencias RRI")

# Subir archivo CSV
uploaded_file = st.file_uploader("Sube un archivo CSV", type=["csv"])

if uploaded_file is not None:
    try:
        # Leer el archivo CSV
        rri_df = pd.read_csv(uploaded_file, low_memory=False)

        # Convertir nombres de columnas a minúsculas
        rri_df.columns = rri_df.columns.str.lower()

        # Normalización de datos
        target_columns = [
            'atencion_origen', 'referencia_rechazada', 'referencia_efectiva',
            'retorno_cont_seguimiento', 'fecha_cita_destino', 'area_origen',
            'area_remision', 'paciente_notificado', 'referencia_pertinente'
        ]
        for col in target_columns:
            if col in rri_df.columns:
                rri_df[col] = rri_df[col].astype(str).str.lower().str.strip()

        # Convertir fechas a formato datetime
        rri_df['fecha_cita_destino'] = pd.to_datetime(rri_df['fecha_cita_destino'], errors='coerce')

        # Indicadores Calculados
        total_references_sent = len(rri_df)
        ce_rechazadas = rri_df[rri_df['referencia_rechazada'] == 'si']
        ce_agendadas = rri_df[rri_df['fecha_cita_destino'].notna()]
        ce_efectivas = rri_df[rri_df['referencia_efectiva'] == 'si']

        percent_ce_rechazadas = (len(ce_rechazadas) / total_references_sent) * 100 if total_references_sent > 0 else 0
        percent_ce_agendadas = (len(ce_agendadas) / total_references_sent) * 100 if total_references_sent > 0 else 0
        percent_ce_efectivas = (len(ce_efectivas) / total_references_sent) * 100 if total_references_sent > 0 else 0

        # Visualización de Indicadores
        st.subheader("Indicadores Calculados")
        st.write(f"% Referencias de CE Rechazadas: {percent_ce_rechazadas:.2f}%")
        st.write(f"% Referencias de CE Agendadas: {percent_ce_agendadas:.2f}%")
        st.write(f"% Referencias de CE efectivas: {percent_ce_efectivas:.2f}%")

        # Graficar Indicadores
        st.subheader("Gráficos de Indicadores")
        fig, ax = plt.subplots()
        indicators = {
            "CE Rechazadas": percent_ce_rechazadas,
            "CE Agendadas": percent_ce_agendadas,
            "CE Efectivas": percent_ce_efectivas
        }
        ax.barh(list(indicators.keys()), list(indicators.values()), color='skyblue')
        ax.set_xlabel("Porcentaje (%)")
        ax.set_title("Indicadores de Referencias")
        st.pyplot(fig)

        # Tendencias Mensuales
        st.subheader("Tendencias Mensuales")
        months = range(1, 13)
        ce_efectivas_trend = []

        for month in months:
            month_data = rri_df[rri_df['fecha_cita_destino'].dt.month == month]
            total_month_references = len(month_data)
            ce_month_efectivas = month_data[month_data['referencia_efectiva'] == 'si']

            percent_month_ce_efectivas = (len(ce_month_efectivas) / total_month_references) * 100 if total_month_references > 0 else 0
            ce_efectivas_trend.append(percent_month_ce_efectivas)

        plt.figure(figsize=(10, 6))
        plt.plot(range(1, 13), ce_efectivas_trend, marker='o', label="CE Efectivas")
        plt.xticks(ticks=range(1, 13), labels=["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"])
        plt.title("Tendencias de CE Efectivas")
        plt.ylabel("Porcentaje (%)")
        plt.grid(True)
        plt.legend()
        st.pyplot(plt)

        # Análisis Predictivo
        st.subheader("Proyección Futura")
        X = np.array(months).reshape(-1, 1)
        y = np.array(ce_efectivas_trend).reshape(-1, 1)

        poly = PolynomialFeatures(degree=3)
        X_poly = poly.fit_transform(X)

        model = LinearRegression()
        model.fit(X_poly, y)

        X_future = np.array(range(1, 19)).reshape(-1, 1)
        X_future_poly = poly.transform(X_future)
        y_future = model.predict(X_future_poly)

        plt.figure(figsize=(10, 6))
        plt.plot(months, ce_efectivas_trend, marker='o', label="Histórico")
        plt.plot(range(1, 19), y_future, linestyle='--', label="Proyección")
        plt.xticks(ticks=range(1, 19), labels=["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic", "Ene (F)", "Feb (F)", "Mar (F)", "Abr (F)", "May (F)", "Jun (F)"], rotation=45)
        plt.title("Proyección de CE Efectivas")
        plt.ylabel("Porcentaje (%)")
        plt.grid(True)
        plt.legend()
        st.pyplot(plt)

    except Exception as e:
        st.error(f"Error al procesar los datos: {e}")
else:
    st.info("Por favor, sube un archivo CSV para continuar.")
