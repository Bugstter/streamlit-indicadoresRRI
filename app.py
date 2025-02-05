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

# Función para procesar archivos
def process_file(file, tipo):
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

        # Indicadores generales
        referencias_efectivas = df[df['referencia_efectiva'] == 'si']
        percent_efectivas = (len(referencias_efectivas) / len(df)) * 100 if len(df) > 0 else 0

        referencias_pertinentes = df[df['referencia_pertinente'] == 'si']
        percent_pertinentes = (len(referencias_pertinentes) / len(df)) * 100 if len(df) > 0 else 0

        referencias_oportunas = df[df['referencia_oportuna'] == 'si']
        percent_oportunas = (len(referencias_oportunas) / len(df)) * 100 if len(df) > 0 else 0

        # Indicadores específicos por tipo
        if tipo == "consulta externa":
            ce_rechazadas = df[df['referencia_rechazada'] == 'si']
            percent_rechazadas = (len(ce_rechazadas) / total_refs) * 100 if total_refs > 0 else 0

            ce_total = df[df['area_origen'] == 'consulta externa']
            ce_no_rechazadas = ce_total[ce_total['referencia_rechazada'] != 'si']
            ce_agendadas = ce_no_rechazadas[ce_no_rechazadas['fecha_cita_destino'].notna()]
            percent_agendadas = (len(ce_agendadas) / len(ce_no_rechazadas)) * 100 if len(ce_no_rechazadas) > 0 else 0

            ce_sin_notificacion = ce_agendadas[ce_agendadas['paciente_notificado'] == 'no']
            percent_sin_notificacion = (len(ce_sin_notificacion) / len(ce_agendadas)) * 100 if len(ce_agendadas) > 0 else 0

            indicators = {
                "% Referencias Rechazadas": percent_rechazadas,
                "% Referencias Agendadas": percent_agendadas,
                "% Referencias Efectivas": percent_efectivas,
                "% Referencias sin Notificación": percent_sin_notificacion,
                "% Referencias Evaluadas como Oportunas": percent_oportunas,
                "% Referencias Evaluadas como Pertinentes": percent_pertinentes,
                "Total Referencias": total_refs
            }
        
        else:  # Emergencia
            referencias_pertinentes_con_retorno = referencias_pertinentes[referencias_pertinentes['posee_retorno'] == 'si']
            percent_con_retorno = (len(referencias_pertinentes_con_retorno) / len(referencias_pertinentes)) * 100 if len(referencias_pertinentes) > 0 else 0

            indicators = {
                "% Referencias de Emergencia Efectivas": percent_efectivas,
                "% Referencias Evaluadas como Oportunas": percent_oportunas,
                "% Referencias Evaluadas como Pertinentes": percent_pertinentes,
                "% Referencias con Retorno": percent_con_retorno,
                "Total Referencias": total_refs
            }

        return df, indicators

    except Exception as e:
        st.error(f"⚠️ Error en archivo de {tipo}: {str(e)}")
        return None, None

# Procesar archivos
data_ce, indicators_ce = process_file(uploaded_ce, "consulta externa")
data_em, indicators_em = process_file(uploaded_em, "emergencia")

# Mostrar resultados
if indicators_ce:
    st.subheader("📊 Indicadores - Consulta Externa")
    st.json(indicators_ce)

if indicators_em:
    st.subheader("📊 Indicadores - Emergencia")
    st.json(indicators_em)

# Comparación visual si ambos archivos están cargados
if indicators_ce and indicators_em:
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
