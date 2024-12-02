import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import streamlit as st

# Título de la Aplicación
st.title("Sistema de Análisis de Indicadores de Referencias")

# Subida de Archivo CSV
uploaded_file = st.file_uploader("Sube un archivo CSV", type=["csv"])

if uploaded_file is not None:
    try:
        # Leer y procesar el archivo
        rri_df = pd.read_csv(uploaded_file, encoding="latin1", low_memory=False)
        rri_df.columns = rri_df.columns.str.lower()

        # Limpieza de datos
        columnas_objetivo = [
            "atencion_origen", "referencia_rechazada", "referencia_oportuna",
            "referencia_efectiva", "retorno_cont_seguimiento", "motivo_no_notificacion",
            "area_origen", "area_remision", "paciente_notificado", "referencia_pertinente"
        ]
        for col in columnas_objetivo:
            if col in rri_df.columns:
                rri_df[col] = rri_df[col].astype(str).str.lower().str.strip()

        rri_df['fecha_cita_destino'] = pd.to_datetime(rri_df['fecha_cita_destino'], errors='coerce')
        rri_df['paciente_notificado'] = rri_df['paciente_notificado'].replace(['nan', '', ' '], np.nan).str.strip()

        # Imputación de datos faltantes
        condicion_imputacion = (
            (rri_df['fecha_cita_destino'].notna()) &
            (rri_df['area_remision'] == 'consulta') &
            (rri_df['paciente_notificado'].isna())
        )
        rri_df.loc[condicion_imputacion, 'paciente_notificado'] = 'no'

        # Cálculo de Indicadores
        total_references_sent = len(rri_df)
        ce_rechazadas = rri_df[rri_df['referencia_rechazada'] == 'si']
        percent_ce_rechazadas = (len(ce_rechazadas) / total_references_sent) * 100 if total_references_sent > 0 else 0

        ce_total = rri_df[rri_df['area_origen'] == 'consulta externa']
        ce_no_rechazadas = ce_total[ce_total['referencia_rechazada'] != 'si']
        ce_agendadas = ce_no_rechazadas[ce_no_rechazadas['fecha_cita_destino'].notna()]
        percent_ce_agendadas = (len(ce_agendadas) / len(ce_no_rechazadas)) * 100 if len(ce_no_rechazadas) > 0 else 0

        ce_sin_notificacion = ce_agendadas[ce_agendadas['paciente_notificado'] == 'no']
        percent_ce_sin_notificacion = (len(ce_sin_notificacion) / len(ce_agendadas)) * 100 if len(ce_agendadas) > 0 else 0

        ce_efectivas = ce_total[ce_total['referencia_efectiva'] == 'si']
        percent_ce_efectivas = (len(ce_efectivas) / len(ce_total)) * 100 if len(ce_total) > 0 else 0

        ce_efectivas_con_retorno = ce_efectivas[ce_efectivas['retorno_cont_seguimiento'] == 'si']
        percent_ce_efectivas_con_retorno = (len(ce_efectivas_con_retorno) / len(ce_efectivas)) * 100 if len(ce_efectivas) > 0 else 0

        ce_no_agendadas = ce_total[ce_total['fecha_cita_destino'].isna()]
        percent_ce_no_agendadas = (len(ce_no_agendadas) / len(ce_no_rechazadas)) * 100 if len(ce_no_rechazadas) > 0 else 0

        emergencia_total = rri_df[rri_df['area_remision'] == 'emergencia']
        referencias_pertinentes = emergencia_total[emergencia_total['referencia_pertinente'].isin(['si', 'no'])]
        percent_referencias_emergencia_efectivas = (len(referencias_pertinentes) / len(emergencia_total)) * 100 if len(emergencia_total) > 0 else 0

        emergencia_con_retorno = emergencia_total[emergencia_total['retorno_cont_seguimiento'] == 'si']
        percent_referencias_emergencia_con_retorno = (len(emergencia_con_retorno) / len(referencias_pertinentes)) * 100 if len(referencias_pertinentes) > 0 else 0

        # Diccionario de Indicadores
        indicators = {
            "% Referencias de CE Rechazadas": percent_ce_rechazadas,
            "% Referencias de CE Agendadas": percent_ce_agendadas,
            "% Referencias de CE sin registro de notificación": percent_ce_sin_notificacion,
            "% Referencias de CE efectivas": percent_ce_efectivas,
            "% Referencias de CE efectivas con retorno": percent_ce_efectivas_con_retorno,
            "% Referencias enviadas a CE no agendadas": percent_ce_no_agendadas,
            "% Referencias de emergencia efectivas": percent_referencias_emergencia_efectivas,
            "% Referencias efectivas de emergencia con retorno": percent_referencias_emergencia_con_retorno,
            "Total de referencias enviadas": total_references_sent
        }

        # Mostrar Indicadores Calculados
        st.subheader("Indicadores Calculados")
        st.json(indicators)

        # Gráfico de Indicadores
        st.subheader("Gráfico de Indicadores")
        percentage_indicators = {k: v for k, v in indicators.items() if '%' in k}
        labels = list(percentage_indicators.keys())
        values = list(percentage_indicators.values())

        plt.figure(figsize=(10, 6))
        plt.barh(labels, values, color='skyblue')
        plt.xlabel("Porcentaje (%)")
        plt.title("Indicadores Clave")
        plt.tight_layout()
        plt.gca().invert_yaxis()
        st.pyplot(plt)

        # Gráfico de Proyecciones
        st.subheader("Proyección de Tendencias")

        # Configurar meses y valores
        months = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
        values_for_projection = list(percentage_indicators.values())[:12]

        # Asegurar que siempre haya datos, reemplazar valores faltantes con ceros
        if len(values_for_projection) < 12:
            values_for_projection += [0] * (12 - len(values_for_projection))

        # Crear datos para proyección
        X = np.arange(1, len(months) + 1).reshape(-1, 1)
        y = np.array(values_for_projection).reshape(-1, 1)

        # Entrenar modelo de regresión lineal
        model = LinearRegression()
        model.fit(X, y)

        # Predicción para meses futuros
        X_future = np.arange(1, len(months) + 7).reshape(-1, 1)
        y_pred = model.predict(X_future)

        # Crear gráfico
        plt.figure(figsize=(10, 6))
        plt.plot(months, y.flatten(), marker='o', label="Histórico")
        future_months = months + [f"Mes {i}" for i in range(13, 19)]
        plt.plot(future_months, y_pred.flatten(), linestyle='--', label="Predicción")
        plt.xlabel("Mes")
        plt.ylabel("Porcentaje")
        plt.title("Tendencias y Proyecciones")
        plt.legend()
        plt.tight_layout()
        st.pyplot(plt)

    except Exception as e:
        st.error(f"Error procesando el archivo: {e}")
else:
    st.info("Por favor, sube un archivo CSV para comenzar.")

