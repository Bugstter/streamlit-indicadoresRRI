import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime

# Configuración inicial
st.title("📊 Dashboard de Referencias - Consulta Externa y Emergencia")

# Sección de subida de archivos
st.subheader("📂 Subir Archivos CSV")

col1, col2 = st.columns(2)
with col1:
    uploaded_ce = st.file_uploader("📄 Subir archivo de **Consulta Externa**", type=["csv"])
with col2:
    uploaded_em = st.file_uploader("📄 Subir archivo de **Emergencia**", type=["csv"])

# Función para procesar archivos de Consulta Externa
def process_consulta_externa(file):
    if file is None:
        return None, None

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

    # Crear un diccionario con los indicadores
    indicators = {
        "% Referencias de CE Rechazadas": percent_ce_rechazadas,
        "% Referencias de CE Agendadas": percent_ce_agendadas,
        "% Referencias de CE sin notificación": percent_ce_sin_notificacion,
        "% Referencias de CE efectivas": percent_ce_efectivas,
        "% Referencias de CE efectivas con retorno": percent_ce_efectivas_con_retorno,
        "% Referencias enviadas a CE no agendadas": percent_ce_no_agendadas,
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

# Función para procesar archivos de Emergencia
def process_emergencia(file):
    if file is None:
        return None, None

    try:
        df = pd.read_csv(file, low_memory=False)
        df.columns = df.columns.str.lower()

        cols_to_clean = ['referencia_rechazada', 'referencia_oportuna', 'referencia_efectiva', 'area_origen', 
                         'area_remision', 'posee_retorno', 'paciente_notificado', 'referencia_pertinente']
        for col in cols_to_clean:
            if col in df.columns:
                df[col] = df[col].astype(str).str.lower().str.strip()

        df['fecha_cita_destino'] = pd.to_datetime(df['fecha_cita_destino'], errors='coerce')

        total_refs = len(df)
        if total_refs == 0:
            return None, None  

        # Cálculo de % Referencias Efectivas en Emergencia
        referencias_pertinentes = df[df['referencia_pertinente'].isin(['si', 'no'])]
        percent_pertinentes = (len(referencias_pertinentes) / len(df)) * 100 if len(df) > 0 else 0

        referencias_con_retorno = referencias_pertinentes[referencias_pertinentes['posee_retorno'] == 'si']
        percent_con_retorno = (len(referencias_con_retorno) / len(referencias_pertinentes)) * 100 if len(referencias_pertinentes) > 0 else 0

        referencias_oportunas = referencias_pertinentes[referencias_pertinentes['referencia_oportuna'] == 'si']
        percent_oportunas = (len(referencias_oportunas) / len(referencias_pertinentes)) * 100 if len(referencias_pertinentes) > 0 else 0

        referencias_efectivas = df[df['referencia_efectiva'] == 'si']
        percent_efectivas = (len(referencias_efectivas) / len(df)) * 100 if len(df) > 0 else 0

        indicators = {
            "% Referencias Efectivas": percent_efectivas,
            "% Referencias Evaluadas como Pertinentes": percent_pertinentes,
            "% Referencias con Retorno": percent_con_retorno,
            "% Referencias Evaluadas como Oportunas": percent_oportunas,
            "Total Referencias": total_refs
        }

        return df, indicators

    except Exception as e:
        st.error(f"⚠️ Error en archivo de Emergencia: {str(e)}")
        return None, None

# Procesar archivos
data_ce, indicators_ce = process_consulta_externa(uploaded_ce)
data_em, indicators_em = process_emergencia(uploaded_em)

# Verificar y mostrar resultados
if data_ce:
    st.subheader("📊 Indicadores - Consulta Externa")
    st.json(indicators_ce)
    
if data_em:
    st.subheader("📊 Indicadores - Emergencia")
    st.json(indicators_em)

# Comparación visual si ambos archivos están cargados
if data_ce and data_em:
    st.subheader("📊 Comparación de Indicadores")

    labels = list(indicators_ce.keys())[:-1]  # Excluir "Total Referencias"
    values_ce = [indicators_ce[k] for k in labels]
    values_em = [indicators_em[k] for k in labels]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(labels, values_ce, label="Consulta Externa", alpha=0.7)
    ax.barh(labels, values_em, label="Emergencia", alpha=0.7)
    ax.set_xlabel("Porcentaje (%)")
    ax.set_title("Comparación de Indicadores")
    ax.legend()
    ax.invert_yaxis()

    st.pyplot(fig)
