import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime

# Configurar título de la app
st.title("📊 Dashboard de Indicadores de Referencias CE")

# Subir archivo CSV
uploaded_file = st.file_uploader("📂 Subir archivo CSV", type=["csv"])

if uploaded_file is not None:
    # Cargar datos
    rri_df = pd.read_csv(uploaded_file, low_memory=False)
    rri_df.columns = rri_df.columns.str.lower()  # Convertir nombres de columnas a minúsculas

    # Normalizar columnas de texto
    columns_to_clean = ['atencion_origen', 'referencia_rechazada', 'referencia_oportuna', 'referencia_efectiva', 
                        'retorno_cont_seguimiento', 'motivo_no_notificacion', 'area_origen', 'area_remision', 
                        'posee_retorno', 'paciente_notificado', 'referencia_pertinente']
    
    for column in columns_to_clean:
        if column in rri_df.columns:
            rri_df[column] = rri_df[column].astype(str).str.lower().str.strip()

    # Convertir fechas
    rri_df['fecha_cita_destino'] = pd.to_datetime(rri_df['fecha_cita_destino'], errors='coerce')
    
    # Obtener la fecha actual
    fecha_hoy = datetime.datetime.today()

    # Limpieza de 'paciente_notificado'
    rri_df['paciente_notificado'] = rri_df['paciente_notificado'].replace(['nan', '', ' '], np.nan).str.strip()
    
    # Imputar "no" en 'paciente_notificado' cuando aplica
    condicion_imputacion = (
        (rri_df['fecha_cita_destino'].notna()) &
        (rri_df['area_remision'] == 'consulta') &
        (rri_df['paciente_notificado'].isna())
    )
    rri_df.loc[condicion_imputacion, 'paciente_notificado'] = 'no'

    # Calcular indicadores
    total_references_sent = len(rri_df)
    
    # 1. % Referencias de CE Rechazadas
    ce_rechazadas = rri_df[rri_df['referencia_rechazada'] == 'si']
    percent_ce_rechazadas = (len(ce_rechazadas) / total_references_sent) * 100 if total_references_sent > 0 else 0

    # 2. % Referencias de CE Agendadas
    ce_total = rri_df[rri_df['area_origen'] == 'consulta externa']
    ce_no_rechazadas = ce_total[ce_total['referencia_rechazada'] != 'si']
    ce_agendadas = ce_no_rechazadas[ce_no_rechazadas['fecha_cita_destino'].notna()]
    percent_ce_agendadas = (len(ce_agendadas) / len(ce_no_rechazadas)) * 100 if len(ce_no_rechazadas) > 0 else 0

    # 3. % Referencias de CE sin registro de notificación
    ce_agendadas = ce_total[(ce_total['area_remision'] == 'consulta') & (ce_total['fecha_cita_destino'].notna())]
    ce_sin_notificacion = ce_agendadas[ce_agendadas['paciente_notificado'] == 'no']
    percent_ce_sin_notificacion = (len(ce_sin_notificacion) / len(ce_agendadas)) * 100 if len(ce_agendadas) > 0 else 0

    # 4. % Referencias de CE efectivas (filtrado por fecha)
    rri_df_filtrado = rri_df[rri_df['fecha_cita_destino'] < fecha_hoy]
    ce_no_rechazadas_filtrado = rri_df_filtrado[rri_df_filtrado['referencia_rechazada'] != 'si']
    ce_efectivas_filtrado = ce_no_rechazadas_filtrado[ce_no_rechazadas_filtrado['referencia_efectiva'] == 'si']
    percent_ce_efectivas = (len(ce_efectivas_filtrado) / len(ce_no_rechazadas_filtrado)) * 100 if len(ce_no_rechazadas_filtrado) > 0 else 0

    # 5. % Referencias de CE efectivas con retorno
    rri_df_filtrado['posee_retorno'] = rri_df_filtrado['posee_retorno'].astype(str).str.lower().str.strip()
    ce_efectivas_con_retorno = ce_efectivas_filtrado[ce_efectivas_filtrado['posee_retorno'] == 'si']
    percent_ce_efectivas_con_retorno = (len(ce_efectivas_con_retorno) / len(ce_efectivas_filtrado)) * 100 if len(ce_efectivas_filtrado) > 0 else 0

    # 6. % Referencias enviadas a CE no agendadas
    ce_no_agendadas = ce_total[ce_total['fecha_cita_destino'].isna()]
    percent_ce_no_agendadas = (len(ce_no_agendadas) / len(ce_no_rechazadas)) * 100 if len(ce_no_rechazadas) > 0 else 0

    # 7. % Referencias a CE evaluadas como oportunas**
    ce_oportunas = ce_efectivas_filtrado[ce_efectivas_filtrado['referencia_oportuna'] == 'si']
    percent_ce_oportunas = (len(ce_oportunas) / len(ce_efectivas_filtrado)) * 100 if len(ce_efectivas_filtrado) > 0 else 0

    # 8. % Referencias a CE evaluadas como pertinentes**
    ce_pertinentes = ce_efectivas_filtrado[ce_efectivas_filtrado['referencia_pertinente'] == 'si']
    percent_ce_pertinentes = (len(ce_pertinentes) / len(ce_efectivas_filtrado)) * 100 if len(ce_efectivas_filtrado) > 0 else 0

    # Crear un diccionario con los indicadores
    indicators = {
        "% Referencias de CE Rechazadas": percent_ce_rechazadas,
        "% Referencias de CE Agendadas": percent_ce_agendadas,
        "% Referencias de CE sin notificación": percent_ce_sin_notificacion,
        "% Referencias de CE efectivas": percent_ce_efectivas,
        "% Referencias de CE efectivas con retorno": percent_ce_efectivas_con_retorno,
        "% Referencias enviadas a CE no agendadas": percent_ce_no_agendadas,
        "% Referencias a CE evaluadas como oportunas": percent_ce_oportunas,
        "% Referencias enviadas a CE evaluadas como pertinentes": percent_ce_pertinentes,
        "Total de referencias enviadas": total_references_sent
        
    }

    # Mostrar métricas en Streamlit
    st.subheader("📌 Indicadores Clave")
    for key, value in indicators.items():
        if "%" in key:
            st.metric(label=key, value=f"{value:.2f}%")
        else:
            st.metric(label=key, value=value)

    # Generar gráfico de indicadores en Streamlit
    st.subheader("📊 Gráfico de Indicadores")

    percentage_indicators = {k: v for k, v in indicators.items() if '%' in k}
    labels = list(percentage_indicators.keys())
    values = list(percentage_indicators.values())

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(labels, values)
    ax.set_xlabel("Porcentaje (%)")
    ax.set_title("Indicadores de Referencias")
    ax.invert_yaxis()
    st.pyplot(fig)

    # Mostrar DataFrame en Streamlit
    st.subheader("📋 Datos Procesados")
    st.dataframe(rri_df)
else:
    st.info("📥 Por favor, sube un archivo CSV para analizar los datos.")
