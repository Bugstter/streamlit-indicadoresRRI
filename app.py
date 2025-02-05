import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
import streamlit as st

#para activar el host: streamlit run "c:/Users/sis/Desktop/RRI front.py"

# Título de la aplicación
st.title("Análisis de Referencias RRI")

# Subir archivo CSV
uploaded_file = st.file_uploader("Sube un archivo CSV", type=["csv"])

# Inicializar rri_df como None para evitar errores de referencia
rri_df = None

if uploaded_file is not None:
    try:
        # Leer el archivo CSV
        rri_df = pd.read_csv(uploaded_file, low_memory=False)

# Convert all column names to lowercase for easier access
rri_df.columns = rri_df.columns.str.lower()

# Standardize column values to lowercase for consistent matching and remove spaces
for column in ['atencion_origen', 'referencia_rechazada', 'referencia_oportuna', 'referencia_efectiva', 'retorno_cont_seguimiento', 'motivo_no_notificacion', 'area_origen', 'area_remision', 'posee_retorno', 'paciente_notificado', 'referencia_pertinente']:
    if column in rri_df.columns:
        rri_df[column] = rri_df[column].astype(str).str.lower().str.strip()

# Convertir 'fecha_cita_destino' en datetime para asegurar que todas las fechas sean válidas
rri_df['fecha_cita_destino'] = pd.to_datetime(rri_df['fecha_cita_destino'], errors='coerce')

# Obtener la fecha de hoy
fecha_hoy = datetime.datetime.today()

# Limpieza de 'paciente_notificado': Convertir valores 'nan', espacios, y otros caracteres invisibles a NaN
# Reemplazar cualquier valor que sea 'nan' como string, espacios o vacíos con NaN
rri_df['paciente_notificado'] = rri_df['paciente_notificado'].replace(['nan', '', ' '], np.nan)
# Adicionalmente, quitar espacios en blanco y tabulaciones y convertir celdas vacías a NaN
rri_df['paciente_notificado'] = rri_df['paciente_notificado'].str.strip().replace('', np.nan)

# Imputar "no" en 'paciente_notificado' solo si 'fecha_cita_destino' tiene una fecha asignada y 'paciente_notificado' es nulo
condicion_imputacion = (
    (rri_df['fecha_cita_destino'].notna()) &
    (rri_df['area_remision'] == 'consulta') &
    (rri_df['paciente_notificado'].isna())
)
rri_df.loc[condicion_imputacion, 'paciente_notificado'] = 'no'

# Calculate indicators
# 1. % Referencias de CE Rechazadas
total_references_sent = len(rri_df)
ce_rechazadas = rri_df[rri_df['referencia_rechazada'] == 'si']
if total_references_sent > 0:
    percent_ce_rechazadas = (len(ce_rechazadas) / total_references_sent) * 100
else:
    percent_ce_rechazadas = 0

# 2. % Referencias de CE Agendadas
ce_total = rri_df[rri_df['area_origen'] == 'consulta externa']
ce_no_rechazadas = ce_total[ce_total['referencia_rechazada'] != 'si']
ce_agendadas = ce_no_rechazadas[ce_no_rechazadas['fecha_cita_destino'].notna()]
if len(ce_no_rechazadas) > 0:
    percent_ce_agendadas = (len(ce_agendadas) / len(ce_no_rechazadas)) * 100
else:
    percent_ce_agendadas = 0

# 3. % Referencias de CE sin registro de notificación
# Total de referencias enviadas a consulta externa con cita asignada en la columna "fecha_cita_destino" y con "area_remision" igual a "Consulta"
ce_agendadas = ce_total[(ce_total['area_remision'] == 'consulta') & (ce_total['fecha_cita_destino'].notna())]

# Total de referencias enviadas a consulta externa con paciente no notificado
ce_sin_notificacion = ce_agendadas[ce_agendadas['paciente_notificado'] == 'no']

# Cálculo del porcentaje de referencias sin notificación
if len(ce_agendadas) > 0:
    percent_ce_sin_notificacion = (len(ce_sin_notificacion) / len(ce_agendadas)) * 100
else:
    percent_ce_sin_notificacion = 0

# **Cálculo normal sin filtro de fecha**
ce_total = rri_df[rri_df['area_origen'] == 'consulta externa']
ce_no_rechazadas = ce_total[ce_total['referencia_rechazada'] != 'si']

# **Cálculo de referencias efectivas solo con fechas menores a hoy**
rri_df_filtrado = rri_df[rri_df['fecha_cita_destino'] < fecha_hoy]

ce_total_filtrado = rri_df_filtrado[rri_df_filtrado['area_origen'] == 'consulta externa']
ce_no_rechazadas_filtrado = ce_total_filtrado[ce_total_filtrado['referencia_rechazada'] != 'si']
ce_efectivas_filtrado = ce_no_rechazadas_filtrado[ce_no_rechazadas_filtrado['referencia_efectiva'] == 'si']

# 4. % Referencias de CE efectivas (con filtro de fecha)**
percent_ce_efectivas = (len(ce_efectivas_filtrado) / len(ce_no_rechazadas_filtrado)) * 100 if len(ce_no_rechazadas_filtrado) > 0 else 0

# 5. % Referencias de CE efectivas con retorno (con filtro de fecha)**
# Normalizar la columna 'posee_retorno' para eliminar inconsistencias
rri_df_filtrado['posee_retorno'] = rri_df_filtrado['posee_retorno'].astype(str).str.lower().str.strip()

# Definir total de referencias efectivas en CE después del filtro de fecha
total_ce_efectivas_filtrado = len(rri_df_filtrado[
    (rri_df_filtrado['area_origen'] == 'consulta externa') &
    (rri_df_filtrado['referencia_rechazada'] != 'si') &
    (rri_df_filtrado['referencia_efectiva'] == 'si')
])

# Filtrar referencias efectivas con retorno en CE después del filtro de fecha
ce_efectivas_con_retorno = rri_df_filtrado[
    (rri_df_filtrado['area_origen'] == 'consulta externa') &
    (rri_df_filtrado['referencia_rechazada'] != 'si') &
    (rri_df_filtrado['referencia_efectiva'] == 'si') &
    (rri_df_filtrado['posee_retorno'] == 'si')
]

# Calcular el porcentaje corregido
percent_ce_efectivas_con_retorno = (len(ce_efectivas_con_retorno) / total_ce_efectivas_filtrado) * 100 if total_ce_efectivas_filtrado > 0 else 0

# 6. % de referencias enviadas a CE no agendadas
# Total de referencias enviadas a consulta externa sin fecha de cita asignada y que no fueron rechazadas
ce_no_agendadas = rri_df[(rri_df['area_origen'] == 'consulta externa') & (rri_df['fecha_cita_destino'].isna())]
if len(rri_df[(rri_df['area_origen'] == 'consulta externa') & (rri_df['referencia_rechazada'] == 'no')]) > 0:
    percent_ce_no_agendadas = (len(ce_no_agendadas) / len(rri_df[(rri_df['area_origen'] == 'consulta externa') & (rri_df['referencia_rechazada'] == 'no')])) * 100
else:
    percent_ce_no_agendadas = 0

# 7. % Referencias de emergencia efectivas
# Total de referencias recibidas en "area_remision" con valor "emergencia" entre el total de referencias que tienen "referencia_pertinente" como "si" o "no"
emergencia_total = rri_df[rri_df['area_remision'] == 'emergencia']
referencias_pertinentes = emergencia_total[emergencia_total['referencia_pertinente'].isin(['si', 'no'])]
if len(referencias_pertinentes) > 0:
    percent_referencias_emergencia_efectivas = (len(referencias_pertinentes) / len(emergencia_total)) * 100
else:
    percent_referencias_emergencia_efectivas = 0

# 8. % Referencias de emergencia con retorno
# Total de referencias enviadas a emergencia con "retorno_cont_seguimiento" igual a "si" entre el total de referencias pertinentes enviadas a emergencia
emergencia_con_retorno = emergencia_total[(emergencia_total['retorno_cont_seguimiento'] == 'si')]
if len(referencias_pertinentes) > 0:
    percent_referencias_emergencia_con_retorno = (len(emergencia_con_retorno) / len(referencias_pertinentes)) * 100
else:
    percent_referencias_emergencia_con_retorno = 0

# 9. % Referencias Agendadas por establecimientos receptores
# Total de referencias recibidas en "area_remision" con valor "consulta" y "fecha_cita_destino" asignada entre el total de referencias que no fueron rechazadas
referencias_recibidas = rri_df[(rri_df['area_remision'] == 'consulta') & (rri_df['fecha_cita_destino'].notna())]
referencias_no_rechazadas = rri_df[rri_df['referencia_rechazada'] == 'no']
if len(referencias_no_rechazadas) > 0:
    percent_referencias_agendadas_por_establecimientos = (len(referencias_recibidas) / len(referencias_no_rechazadas)) * 100
else:
    percent_referencias_agendadas_por_establecimientos = 0

# 10. % de referencias recibidas en CE no agendadas
# Total de referencias recibidas en CE sin fecha de cita asignada entre el total de referencias recibidas en CE no rechazadas
ce_recibidas_no_agendadas = ce_no_rechazadas[ce_no_rechazadas['fecha_cita_destino'].isna()]
if len(ce_no_rechazadas) > 0:
    percent_ce_recibidas_no_agendadas = (len(ce_recibidas_no_agendadas) / len(ce_no_rechazadas)) * 100
else:
    percent_ce_recibidas_no_agendadas = 0

# 11. % referencias de emergencia efectivas
# Total de referencias recibidas en "area_remision" con valor "emergencia" y "referencia_pertinente" igual a "si" o "no" entre el total de referencias recibidas en emergencia
emergencia_recibidas = rri_df[rri_df['area_remision'] == 'emergencia']
emergencia_recibidas_efectivas = emergencia_recibidas[emergencia_recibidas['referencia_pertinente'].isin(['si', 'no'])]
if len(emergencia_recibidas) > 0:
    percent_referencias_emergencia_efectivas = (len(emergencia_recibidas_efectivas) / len(emergencia_recibidas)) * 100
else:
    percent_referencias_emergencia_efectivas = 0

# 12. % referencias de emergencia con retorno
# Total de referencias recibidas en "area_remision" con valor "emergencia" y "retorno_cont_seguimiento" igual a "si" entre el total de referencias con "referencia_pertinente" igual a "si" o "no"
emergencia_recibidas_con_retorno = emergencia_recibidas[emergencia_recibidas['retorno_cont_seguimiento'] == 'si']
if len(emergencia_recibidas_efectivas) > 0:
    percent_referencias_emergencia_con_retorno = (len(emergencia_recibidas_con_retorno) / len(emergencia_recibidas_efectivas)) * 100
else:
    percent_referencias_emergencia_con_retorno = 0

# 13. % Referencias a CE evaluadas como oportunas**
ce_oportunas = ce_efectivas_filtrado[ce_efectivas_filtrado['referencia_oportuna'] == 'si']
percent_ce_oportunas = (len(ce_oportunas) / len(ce_efectivas_filtrado)) * 100 if len(ce_efectivas_filtrado) > 0 else 0

# 14.% Referencias a CE evaluadas como pertinentes**
ce_pertinentes = ce_efectivas_filtrado[ce_efectivas_filtrado['referencia_pertinente'] == 'si']
percent_ce_pertinentes = (len(ce_pertinentes) / len(ce_efectivas_filtrado)) * 100 if len(ce_efectivas_filtrado) > 0 else 0

# **Indicadores de emergencia**
ue_efectivas = rri_df[rri_df['area_origen'] == 'unidad de emergencia']
ue_efectivas = ue_efectivas[ue_efectivas['referencia_efectiva'] == 'si']

# 15.Indicador 3: % Referencias a UE evaluadas como oportunas**
ue_oportunas = ue_efectivas[ue_efectivas['referencia_oportuna'] == 'si']
percent_ue_oportunas = (len(ue_oportunas) / len(ue_efectivas)) * 100 if len(ue_efectivas) > 0 else 0

# 16.Indicador 4: % Referencias a UE evaluadas como pertinentes**
ue_pertinentes = ue_efectivas[ue_efectivas['referencia_pertinente'] == 'si']
percent_ue_pertinentes = (len(ue_pertinentes) / len(ue_efectivas)) * 100 if len(ue_efectivas) > 0 else 0


# Print all the calculated indicators
print(f"% Referencias de CE Rechazadas: {percent_ce_rechazadas:.2f}%")
print(f"% Referencias de CE Agendadas: {percent_ce_agendadas:.2f}%")
print(f"% Referencias de CE sin registro de notificacion: {percent_ce_sin_notificacion:.2f}%")
print(f"% Referencias de CE efectivas: {percent_ce_efectivas:.2f}%")
print(f"% Referencias de CE efectivas con retorno (con citas pasadas): {percent_ce_efectivas_con_retorno:.2f}%")
print(f"% Referencias enviadas a CE no agendadas: {percent_ce_no_agendadas:.2f}%")
print(f"% Referencias Agendadas por establecimientos receptores: {percent_referencias_agendadas_por_establecimientos:.2f}%")
print(f"% de referencias recibidas en CE no agendadas: {percent_ce_recibidas_no_agendadas:.2f}%")
print(f"% Referencias enviadas a CE evaluadas como oportunas: {percent_ce_oportunas:.2f}%")
print(f"% Referencias enviadas a CE evaluadas como pertinentes: {percent_ce_pertinentes:.2f}%")
print(f"Total de referencias enviadas: {total_references_sent}")

# Create a dictionary to store the indicators and their values
indicators = {
    "% Referencias de CE Rechazadas": percent_ce_rechazadas,
    "% Referencias de CE Agendadas": percent_ce_agendadas,
    "% Referencias de CE sin registro de notificacion": percent_ce_sin_notificacion,
    "% Referencias de CE efectivas": percent_ce_efectivas,
    "% Referencias de CE efectivas con retorno": percent_ce_efectivas_con_retorno,
    "% Referencias enviadas a CE no agendadas": percent_ce_no_agendadas,
    "% Referencias de emergencia efectivas": percent_referencias_emergencia_efectivas,
    "% Referencias efectivas de emergencia con retorno": percent_referencias_emergencia_con_retorno,
    "% Referencias Agendadas por establecimientos receptores": percent_referencias_agendadas_por_establecimientos,
    "% de referencias recibidas en CE no agendadas": percent_ce_recibidas_no_agendadas,
    "% referencias de emergencia efectivas": percent_referencias_emergencia_efectivas,
    "% referencias de emergencia con retorno": percent_referencias_emergencia_con_retorno,
    "Total de referencias enviadas": total_references_sent # Added total references to the dictionary
}
# Filtrar indicadores de porcentaje para graficar, excluyendo el total de referencias
percentage_indicators = {k: v for k, v in indicators.items() if '%' in k}

# Configurar los datos para el gráfico
labels = list(percentage_indicators.keys())
values = list(percentage_indicators.values())

# Configurar el gráfico de barras
plt.figure(figsize=(10, 8))
plt.barh(labels, values)
plt.xlabel('Porcentaje (%)')
plt.title('Indicadores de Referencias')
plt.gca().invert_yaxis()  # Invertir el eje y para que el primer elemento esté arriba
plt.tight_layout()

# Mostrar el gráfico
plt.show()
