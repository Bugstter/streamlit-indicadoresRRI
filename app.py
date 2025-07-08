import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime

st.set_page_config(layout="wide")
st.title("📈 Análisis de Referencias por Área de Atención")

# Función común para cargar y limpiar datos
def cargar_datos(archivo):
    df = pd.read_csv(archivo, low_memory=False)
    df.columns = df.columns.str.lower()
    columnas_texto = ['atencion_origen', 'referencia_rechazada', 'referencia_oportuna', 'referencia_efectiva',
                      'retorno_cont_seguimiento', 'motivo_no_notificacion', 'area_origen', 'area_remision',
                      'posee_retorno', 'paciente_notificado', 'referencia_pertinente']
    for col in columnas_texto:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().str.strip()
    df['fecha_cita_destino'] = pd.to_datetime(df['fecha_cita_destino'], errors='coerce')
    return df

# Función de gráfico
def graficar_indicadores(indicadores, titulo):
    fig, ax = plt.subplots(figsize=(10, 5))
    porcentajes = {k: v for k, v in indicadores.items() if "%" in k}
    ax.barh(list(porcentajes.keys()), list(porcentajes.values()))
    ax.set_xlabel("Porcentaje (%)")
    ax.set_title(titulo)
    ax.invert_yaxis()
    st.pyplot(fig)

# Función indicadores CE
def calcular_indicadores_ce(df):
    fecha_hoy = datetime.datetime.today()

    df['paciente_notificado'] = df['paciente_notificado'].replace(['nan', '', ' '], np.nan).str.strip()
    df.loc[
        (df['fecha_cita_destino'].notna()) & 
        (df['area_remision'] == 'consulta') & 
        (df['paciente_notificado'].isna()), 
        'paciente_notificado'
    ] = 'no'

    total_references_sent = len(df)
    ce_rechazadas = df[df['referencia_rechazada'] == 'si']
    percent_ce_rechazadas = (len(ce_rechazadas) / total_references_sent) * 100 if total_references_sent > 0 else 0

    ce_total = df[df['area_origen'] == 'consulta externa']
    ce_no_rechazadas = ce_total[ce_total['referencia_rechazada'] != 'si']
    ce_agendadas = ce_no_rechazadas[ce_no_rechazadas['fecha_cita_destino'].notna()]
    percent_ce_agendadas = (len(ce_agendadas) / len(ce_no_rechazadas)) * 100 if len(ce_no_rechazadas) > 0 else 0

    ce_agendadas_notif = ce_total[(ce_total['area_remision'] == 'consulta') & (ce_total['fecha_cita_destino'].notna())]
    ce_sin_notificacion = ce_agendadas_notif[ce_agendadas_notif['paciente_notificado'] == 'no']
    percent_ce_sin_notificacion = (len(ce_sin_notificacion) / len(ce_agendadas_notif)) * 100 if len(ce_agendadas_notif) > 0 else 0

    df_filtrado = df[df['fecha_cita_destino'] < fecha_hoy]
    ce_no_rechazadas_filtrado = df_filtrado[df_filtrado['referencia_rechazada'] != 'si']
    ce_efectivas_filtrado = ce_no_rechazadas_filtrado[ce_no_rechazadas_filtrado['referencia_efectiva'] == 'si']
    percent_ce_efectivas = (len(ce_efectivas_filtrado) / len(ce_no_rechazadas_filtrado)) * 100 if len(ce_no_rechazadas_filtrado) > 0 else 0

    df_filtrado['posee_retorno'] = df_filtrado['posee_retorno'].astype(str).str.lower().str.strip()
    ce_efectivas_con_retorno = ce_efectivas_filtrado[ce_efectivas_filtrado['posee_retorno'] == 'si']
    percent_ce_efectivas_con_retorno = (len(ce_efectivas_con_retorno) / len(ce_efectivas_filtrado)) * 100 if len(ce_efectivas_filtrado) > 0 else 0

    referencias_no_rechazadas = df[df['referencia_rechazada'] == "no"]
    referencias_no_agendadas = referencias_no_rechazadas[referencias_no_rechazadas['fecha_cita_destino'].isna()]
    percent_ce_recibidas_no_agendadas = (len(referencias_no_agendadas) / len(referencias_no_rechazadas)) * 100 if len(referencias_no_rechazadas) > 0 else 0

    ce_oportunas = ce_efectivas_filtrado[ce_efectivas_filtrado['referencia_oportuna'] == 'si']
    percent_ce_oportunas = (len(ce_oportunas) / len(ce_efectivas_filtrado)) * 100 if len(ce_efectivas_filtrado) > 0 else 0

    ce_pertinentes = ce_efectivas_filtrado[ce_efectivas_filtrado['referencia_pertinente'] == 'si']
    percent_ce_pertinentes = (len(ce_pertinentes) / len(ce_efectivas_filtrado)) * 100 if len(ce_efectivas_filtrado) > 0 else 0

    indicadores = {
        "% Referencias de CE Rechazadas": percent_ce_rechazadas,
        "% Referencias de CE Agendadas": percent_ce_agendadas,
        "% Referencias de CE sin notificación": percent_ce_sin_notificacion,
        "% Referencias de CE efectivas": percent_ce_efectivas,
        "% Referencias de CE efectivas con retorno": percent_ce_efectivas_con_retorno,
        "% de referencias recibidas en CE no agendadas": percent_ce_recibidas_no_agendadas,
        "% Referencias a CE evaluadas como oportunas": percent_ce_oportunas,
        "% Referencias enviadas a CE evaluadas como pertinentes": percent_ce_pertinentes,
        "Total de referencias enviadas": total_references_sent
    }
    return indicadores

# Función indicadores UE
def calcular_indicadores_ue(df):
    emergencia_total = df[df['area_remision'] == 'emergencia']
    referencias_pertinentes = emergencia_total[emergencia_total['referencia_pertinente'].isin(['si', 'no'])]
    percent_referencias_emergencia_efectivas = (len(referencias_pertinentes) / len(emergencia_total)) * 100 if len(emergencia_total) > 0 else 0

    retorno = referencias_pertinentes[referencias_pertinentes["posee_retorno"] == "si"]
    percent_pertinentes_retorno = (len(retorno) / len(referencias_pertinentes)) * 100 if len(referencias_pertinentes) > 0 else 0

    total_referencias_pertinentes = len(referencias_pertinentes)
    oportunas = referencias_pertinentes[referencias_pertinentes["referencia_oportuna"] == "si"]
    pertinentes = referencias_pertinentes[referencias_pertinentes["referencia_pertinente"] == "si"]
    
    percent_ue_oportunas = (len(oportunas) / total_referencias_pertinentes) * 100 if total_referencias_pertinentes > 0 else 0
    percent_ue_pertinentes = (len(pertinentes) / total_referencias_pertinentes) * 100 if total_referencias_pertinentes > 0 else 0

    indicadores = {
        "% Referencias de emergencia efectivas": percent_referencias_emergencia_efectivas,
        "% Referencias enviadas a UE evaluadas como oportunas": percent_ue_oportunas,
        "% Referencias enviadas a UE evaluadas como pertinentes": percent_ue_pertinentes,
        "% Referencias pertinentes con retorno (Posee Retorno)": percent_pertinentes_retorno
    }
    return indicadores

# Tabs para CE y UE
tab1, tab2 = st.tabs(["🏥 Consulta Externa", "🚨 Unidad de Emergencia"])

with tab1:
    st.subheader("📂 Subir archivo para análisis de Consulta Externa")
    archivo_ce = st.file_uploader("Archivo CSV", type=["csv"], key="ce")
    if archivo_ce:
        df_ce = cargar_datos(archivo_ce)
        indicadores_ce = calcular_indicadores_ce(df_ce)
        st.write("### Indicadores Calculados")
        st.dataframe(pd.DataFrame.from_dict(indicadores_ce, orient='index', columns=['Valor']))
        graficar_indicadores(indicadores_ce, "Indicadores de Referencias en Consulta Externa")

with tab2:
    st.subheader("📂 Subir archivo para análisis de Unidad de Emergencia")
    archivo_ue = st.file_uploader("Archivo CSV", type=["csv"], key="ue")
    if archivo_ue:
        df_ue = cargar_datos(archivo_ue)
        indicadores_ue = calcular_indicadores_ue(df_ue)
        st.write("### Indicadores Calculados")
        st.dataframe(pd.DataFrame.from_dict(indicadores_ue, orient='index', columns=['Valor']))
        graficar_indicadores(indicadores_ue, "Indicadores de Referencias en Unidad de Emergencia")
