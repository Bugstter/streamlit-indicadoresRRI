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
        return None

    # Cargar datos
    df = pd.read_csv(file, low_memory=False)
    df.columns = df.columns.str.lower()

    # Normalizar datos
    cols_to_clean = ['referencia_rechazada', 'referencia_oportuna', 'referencia_efectiva', 'area_origen', 
                     'area_remision', 'posee_retorno', 'paciente_notificado', 'referencia_pertinente']
    
    for col in cols_to_clean:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().str.strip()

    # Convertir fechas
    df['fecha_cita_destino'] = pd.to_datetime(df['fecha_cita_destino'], errors='coerce')
    fecha_hoy = datetime.datetime.today()

    # Imputar "no" en 'paciente_notificado'
    cond_imputacion = (
        (df['fecha_cita_destino'].notna()) &
        (df['area_remision'] == 'consulta') &
        (df['paciente_notificado'].isna())
    )
    df.loc[cond_imputacion, 'paciente_notificado'] = 'no'

    # Calcular indicadores
    total_refs = len(df)
    refs_rechazadas = df[df['referencia_rechazada'] == 'si']
    percent_rechazadas = (len(refs_rechazadas) / total_refs) * 100 if total_refs > 0 else 0

    ce_total = df[df['area_origen'] == 'consulta externa']
    ce_no_rechazadas = ce_total[ce_total['referencia_rechazada'] != 'si']
    ce_agendadas = ce_no_rechazadas[ce_no_rechazadas['fecha_cita_destino'].notna()]
    percent_agendadas = (len(ce_agendadas) / len(ce_no_rechazadas)) * 100 if len(ce_no_rechazadas) > 0 else 0

    ce_sin_notificacion = ce_agendadas[ce_agendadas['paciente_notificado'] == 'no']
    percent_sin_notificacion = (len(ce_sin_notificacion) / len(ce_agendadas)) * 100 if len(ce_agendadas) > 0 else 0

    df_filtrado = df[df['fecha_cita_destino'] < fecha_hoy]
    ce_no_rechazadas_filtrado = df_filtrado[df_filtrado['referencia_rechazada'] != 'si']
    ce_efectivas_filtrado = ce_no_rechazadas_filtrado[ce_no_rechazadas_filtrado['referencia_efectiva'] == 'si']
    percent_efectivas = (len(ce_efectivas_filtrado) / len(ce_no_rechazadas_filtrado)) * 100 if len(ce_no_rechazadas_filtrado) > 0 else 0

    df_filtrado['posee_retorno'] = df_filtrado['posee_retorno'].astype(str).str.lower().str.strip()
    ce_efectivas_con_retorno = ce_efectivas_filtrado[ce_efectivas_filtrado['posee_retorno'] == 'si']
    percent_efectivas_con_retorno = (len(ce_efectivas_con_retorno) / len(ce_efectivas_filtrado)) * 100 if len(ce_efectivas_filtrado) > 0 else 0

    indicators = {
        "% Referencias Rechazadas": percent_rechazadas,
        "% Referencias Agendadas": percent_agendadas,
        "% Referencias sin Notificación": percent_sin_notificacion,
        "% Referencias Efectivas": percent_efectivas,
        "% Referencias Efectivas con Retorno": percent_efectivas_con_retorno,
        "Total Referencias": total_refs
    }
    
    return df, indicators

# Función para procesar archivos de Emergencia
def process_emergencia(file):
    if file is None:
        return None

    df = pd.read_csv(file, low_memory=False)
    df.columns = df.columns.str.lower()

    # Normalización de datos
    cols_to_clean = ['referencia_rechazada', 'referencia_oportuna', 'referencia_efectiva', 'area_origen', 
                     'area_remision', 'posee_retorno', 'paciente_notificado', 'referencia_pertinente']
    
    for col in cols_to_clean:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().str.strip()

    df['fecha_cita_destino'] = pd.to_datetime(df['fecha_cita_destino'], errors='coerce')

    emergencia_total = df[df['area_remision'] == 'emergencia']
    referencias_pertinentes = emergencia_total[emergencia_total['referencia_pertinente'].isin(['si', 'no'])]
    percent_emergencia_efectivas = (len(referencias_pertinentes) / len(emergencia_total)) * 100 if len(emergencia_total) > 0 else 0

    referencias_pertinentes_con_retorno = referencias_pertinentes[referencias_pertinentes["posee_retorno"] == "si"]
    percent_retorno = (len(referencias_pertinentes_con_retorno) / len(referencias_pertinentes)) * 100 if len(referencias_pertinentes) > 0 else 0

    ue_pertinentes = df[df['area_remision'] == 'emergencia']
    total_referencias_pertinentes = len(ue_pertinentes)
    num_referencias_oportunas = len(ue_pertinentes[ue_pertinentes["referencia_oportuna"] == "si"])
    percent_oportunas = (num_referencias_oportunas / total_referencias_pertinentes) * 100 if total_referencias_pertinentes > 0 else 0

    num_referencias_pertinentes = len(ue_pertinentes[ue_pertinentes["referencia_pertinente"] == "si"])
    percent_pertinentes = (num_referencias_pertinentes / total_referencias_pertinentes) * 100 if total_referencias_pertinentes > 0 else 0

    indicators = {
        "% Referencias de Emergencia Efectivas": percent_emergencia_efectivas,
        "% Referencias Evaluadas como Oportunas": percent_oportunas,
        "% Referencias Evaluadas como Pertinentes": percent_pertinentes,
        "% Referencias con Retorno": percent_retorno
    }
    
    return df, indicators

# Procesar archivos
data_ce, indicators_ce = process_consulta_externa(uploaded_ce)
data_em, indicators_em = process_emergencia(uploaded_em)

# Mostrar resultados
if data_ce:
    st.subheader("📊 Indicadores - Consulta Externa")
    st.json(indicators_ce)
    
if data_em:
    st.subheader("📊 Indicadores - Emergencia")
    st.json(indicators_em)
