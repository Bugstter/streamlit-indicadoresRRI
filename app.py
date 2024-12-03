import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import streamlit as st

#para activar el host: streamlit run "c:/Users/sis/Desktop/RRI front.py"

# Título de la aplicación
st.title("Análisis de Referencias RRI")

# Subir archivo CSV
uploaded_file = st.file_uploader("Sube un archivo CSV", type=["csv"])

if uploaded_file is not None:
    try:
    # Leer el archivo CSV
    rri_df = pd.read_csv(uploaded_file, low_memory=False)

# Convert all column names to lowercase for easier access
rri_df.columns = rri_df.columns.str.lower()

# Standardize column values to lowercase for consistent matching and remove spaces
for column in ['atencion_origen', 'referencia_rechazada', 'referencia_oportuna', 'referencia_efectiva', 'retorno_cont_seguimiento', 'motivo_no_notificacion', 'area_origen', 'area_remision', 'paciente_notificado', 'referencia_pertinente']:
    if column in rri_df.columns:
        rri_df[column] = rri_df[column].astype(str).str.lower().str.strip()
         for col in required_columns:
            if col not in rri_df.columns:
                st.error(f"Falta la columna requerida: {col}")
                st.stop()

# Convertir 'fecha_cita_destino' en datetime para asegurar que todas las fechas sean válidas
rri_df['fecha_cita_destino'] = pd.to_datetime(rri_df['fecha_cita_destino'], errors='coerce')

# Limpieza de 'paciente_notificado': Convertir valores 'nan', espacios, y otros caracteres invisibles a NaN
rri_df['paciente_notificado'] = rri_df['paciente_notificado'].replace(['nan', '', ' '], np.nan)
rri_df['paciente_notificado'] = rri_df['paciente_notificado'].str.strip().replace('', np.nan)

# Imputar "no" en 'paciente_notificado' solo si 'fecha_cita_destino' tiene una fecha asignada y 'paciente_notificado' es nulo
condicion_imputacion = (
    (rri_df['fecha_cita_destino'].notna()) &
    (rri_df['area_remision'] == 'consulta') &
    (rri_df['paciente_notificado'].isna())
)
rri_df.loc[condicion_imputacion, 'paciente_notificado'] = 'no'

# Cálculo de indicadores
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

# 3. % Referencias de CE sin registro de notificacion
ce_agendadas = ce_total[(ce_total['area_remision'] == 'consulta') & (ce_total['fecha_cita_destino'].notna())]
ce_sin_notificacion = ce_agendadas[ce_agendadas['paciente_notificado'] == 'no']
if len(ce_agendadas) > 0:
    percent_ce_sin_notificacion = (len(ce_sin_notificacion) / len(ce_agendadas)) * 100
else:
    percent_ce_sin_notificacion = 0

# 4. % Referencias de CE efectivas
ce_efectivas = ce_total[ce_total['referencia_efectiva'] == 'si']
if len(ce_total) > 0:
    percent_ce_efectivas = (len(ce_efectivas) / len(ce_total)) * 100
else:
    percent_ce_efectivas = 0

# 5. % Referencias de CE efectivas con retorno
ce_efectivas_con_retorno = ce_efectivas[ce_efectivas['retorno_cont_seguimiento'] == 'si']
if len(ce_efectivas) > 0:
    percent_ce_efectivas_con_retorno = (len(ce_efectivas_con_retorno) / len(ce_efectivas)) * 100
else:
    percent_ce_efectivas_con_retorno = 0

# 6. % de referencias enviadas a CE no agendadas
ce_no_agendadas = rri_df[(rri_df['area_origen'] == 'consulta externa') & (rri_df['fecha_cita_destino'].isna())]
if len(rri_df[(rri_df['area_origen'] == 'consulta externa') & (rri_df['referencia_rechazada'] == 'no')]) > 0:
    percent_ce_no_agendadas = (len(ce_no_agendadas) / len(rri_df[(rri_df['area_origen'] == 'consulta externa') & (rri_df['referencia_rechazada'] == 'no')])) * 100
else:
    percent_ce_no_agendadas = 0

# 7. % Referencias de emergencia efectivas
emergencia_total = rri_df[rri_df['area_remision'] == 'emergencia']
referencias_pertinentes = emergencia_total[emergencia_total['referencia_pertinente'].isin(['si', 'no'])]
if len(referencias_pertinentes) > 0:
    percent_referencias_emergencia_efectivas = (len(referencias_pertinentes) / len(emergencia_total)) * 100
else:
    percent_referencias_emergencia_efectivas = 0

# 8. % Referencias de emergencia con retorno
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

st.write("### Indicadores Calculados")
st.write(f"% Referencias de CE Rechazadas: {percent_ce_rechazadas:.2f}%")
st.write(f"% Referencias de CE Agendadas: {percent_ce_agendadas:.2f}%")
st.write(f"% Referencias de CE sin registro de notificacion: {percent_ce_sin_notificacion:.2f}%")
st.write(f"% Referencias de CE efectivas: {percent_ce_efectivas:.2f}%")
st.write(f"% Referencias de CE efectivas con retorno: {percent_ce_efectivas_con_retorno:.2f}%")
st.write(f"% Referencias enviadas a CE no agendadas: {percent_ce_no_agendadas:.2f}%")
st.write(f"% Referencias de emergencia efectivas: {percent_referencias_emergencia_efectivas:.2f}%")
st.write(f"% Referencias efectivas de emergencia con retorno: {percent_referencias_emergencia_con_retorno:.2f}%")
st.write(f"% Referencias Agendadas por establecimientos receptores: {percent_referencias_agendadas_por_establecimientos:.2f}%")
st.write(f"% de referencias recibidas en CE no agendadas: {percent_ce_recibidas_no_agendadas:.2f}%")
st.write(f"% referencias de emergencia efectivas: {percent_referencias_emergencia_efectivas:.2f}%")
st.write(f"% referencias de emergencia con retorno: {percent_referencias_emergencia_con_retorno:.2f}%")
st.write(f"Total de referencias enviadas: {total_references_sent}")

# Crear un diccionario para almacenar los indicadores y sus valores
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
st.pyplot(plt)

#Creacion de graficas tendenciales y proyeccion
# Estandarizar valores de columnas para una coincidencia consistente
target_columns = ['atencion_origen', 'referencia_rechazada', 'referencia_oportuna', 'referencia_efectiva',
                  'retorno_cont_seguimiento', 'motivo_no_notificacion', 'area_origen', 'area_remision',
                  'paciente_notificado', 'referencia_pertinente']
for column in target_columns:
    if column in rri_df.columns:
        rri_df[column] = rri_df[column].astype(str).str.lower()

# Crear listas vacías para almacenar las tendencias mensuales
months = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
ce_efectivas_trend = []
ce_rechazadas_trend = []
ce_sin_notificacion_trend = []
ce_recibidas_no_agendadas_trend = []
ce_no_agendadas_trend = []

# Iterar sobre cada mes para calcular los indicadores mensuales
for month in range(1, 13):
    # Filtrar por mes
    month_data = rri_df[rri_df['fecha_remision'].str.contains(f'-{month:02d}-', na=False)]

    # Calcular los indicadores para cada mes
    total_references_sent = len(month_data)

    # % Referencias de CE efectivas
    ce_total = month_data[month_data['area_origen'] == 'consulta externa']
    ce_efectivas = ce_total[ce_total['referencia_efectiva'] == 'si']
    percent_ce_efectivas = (len(ce_efectivas) / len(ce_total)) * 100 if len(ce_total) > 0 else 0
    ce_efectivas_trend.append(percent_ce_efectivas)

    # % Referencias de CE Rechazadas
    ce_rechazadas = month_data[month_data['referencia_rechazada'] == 'si']
    percent_ce_rechazadas = (len(ce_rechazadas) / total_references_sent) * 100 if total_references_sent > 0 else 0
    ce_rechazadas_trend.append(percent_ce_rechazadas)

    # % Referencias de CE sin registro de notificación
    ce_sin_notificacion = ce_total[(ce_total['paciente_notificado'] == 'no') | (ce_total['paciente_notificado'].isna())]
    percent_ce_sin_notificacion = (len(ce_sin_notificacion) / len(ce_total)) * 100 if len(ce_total) > 0 else 0
    ce_sin_notificacion_trend.append(percent_ce_sin_notificacion)

    # % Referencias recibidas en CE no agendadas
    ce_no_rechazadas = ce_total[ce_total['referencia_rechazada'] != 'si']
    ce_recibidas_no_agendadas = ce_no_rechazadas[ce_no_rechazadas['fecha_cita_destino'].isna()]
    percent_ce_recibidas_no_agendadas = (len(ce_recibidas_no_agendadas) / len(ce_no_rechazadas)) * 100 if len(ce_no_rechazadas) > 0 else 0
    ce_recibidas_no_agendadas_trend.append(percent_ce_recibidas_no_agendadas)

    # % Referencias enviadas a CE no agendadas
    ce_no_agendadas = ce_total[ce_total['fecha_cita_destino'].isna()]
    percent_ce_no_agendadas = (len(ce_no_agendadas) / len(ce_total)) * 100 if len(ce_total) > 0 else 0
    ce_no_agendadas_trend.append(percent_ce_no_agendadas)

# Configurar el gráfico de líneas con los indicadores calculados
plt.figure(figsize=(12, 8))

plt.plot(months, ce_efectivas_trend, marker='o', linestyle='-', label="% Referencias de CE efectivas")
plt.plot(months, ce_rechazadas_trend, marker='o', linestyle='-', label="% Referencias de CE Rechazadas")
plt.plot(months, ce_sin_notificacion_trend, marker='o', linestyle='-', label="% Referencias de CE sin registro de notificación")
plt.plot(months, ce_recibidas_no_agendadas_trend, marker='o', linestyle='-', label="% Referencias recibidas en CE no agendadas")
plt.plot(months, ce_no_agendadas_trend, marker='o', linestyle='-', label="% Referencias enviadas a CE no agendadas")

plt.xlabel('Meses')
plt.ylabel('Porcentaje (%)')
plt.title('Tendencias de Indicadores de Referencias durante el Año')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Mostrar el gráfico
st.pyplot(plt)

# Agregar análisis predictivo utilizando un modelo de regresión polinomial para mejorar el ajuste
# Convertir los meses en valores numéricos para el análisisstreamlit run "c:/Users/sis/Desktop/RRI front.py"
X = np.array(range(1, 13)).reshape(-1, 1)
X_future = np.array(range(1, 19)).reshape(-1, 1)  # Predicción para los próximos 6 meses

# Función para ajustar y predecir con regresión polinomial
def polynomial_prediction(X, y, degree=3):
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)
    model = LinearRegression()
    model.fit(X_poly, y)
    X_future_poly = poly.transform(X_future)
    y_pred = model.predict(X_future_poly)
    r2 = r2_score(y, model.predict(X_poly))
    return y_pred, r2

# Predecir tendencia para "% Referencias de CE efectivas"
y_ce_efectivas = np.array(ce_efectivas_trend).reshape(-1, 1)
y_ce_efectivas_pred, r2_ce_efectivas = polynomial_prediction(X, y_ce_efectivas)

# Predecir tendencia para "% Referencias de CE Rechazadas"
y_ce_rechazadas = np.array(ce_rechazadas_trend).reshape(-1, 1)
y_ce_rechazadas_pred, r2_ce_rechazadas = polynomial_prediction(X, y_ce_rechazadas)

# Predecir tendencia para "% Referencias de CE sin registro de notificación"
y_ce_sin_notificacion = np.array(ce_sin_notificacion_trend).reshape(-1, 1)
y_ce_sin_notificacion_pred, r2_ce_sin_notificacion = polynomial_prediction(X, y_ce_sin_notificacion)

# Predecir tendencia para "% Referencias recibidas en CE no agendadas"
y_ce_recibidas_no_agendadas = np.array(ce_recibidas_no_agendadas_trend).reshape(-1, 1)
y_ce_recibidas_no_agendadas_pred, r2_ce_recibidas_no_agendadas = polynomial_prediction(X, y_ce_recibidas_no_agendadas)

# Predecir tendencia para "% Referencias enviadas a CE no agendadas"
y_ce_no_agendadas = np.array(ce_no_agendadas_trend).reshape(-1, 1)
y_ce_no_agendadas_pred, r2_ce_no_agendadas = polynomial_prediction(X, y_ce_no_agendadas)

# Configurar el gráfico de líneas con predicciones
plt.figure(figsize=(12, 8))

# Tendencia histórica
plt.plot(months, ce_efectivas_trend, marker='o', linestyle='-', label="% Referencias de CE efectivas (Histórico)")
plt.plot(months, ce_rechazadas_trend, marker='o', linestyle='-', label="% Referencias de CE Rechazadas (Histórico)")
plt.plot(months, ce_sin_notificacion_trend, marker='o', linestyle='-', label="% Referencias de CE sin registro de notificación (Histórico)")
plt.plot(months, ce_recibidas_no_agendadas_trend, marker='o', linestyle='-', label="% Referencias recibidas en CE no agendadas (Histórico)")
plt.plot(months, ce_no_agendadas_trend, marker='o', linestyle='-', label="% Referencias enviadas a CE no agendadas (Histórico)")

# Predicciones
months_future = months + ["Ene (Fut)", "Feb (Fut)", "Mar (Fut)", "Abr (Fut)", "May (Fut)", "Jun (Fut)"]
plt.plot(months_future, y_ce_efectivas_pred, linestyle='--', label=f"% Referencias de CE efectivas (Predicción, R2={r2_ce_efectivas:.2f})")
plt.plot(months_future, y_ce_rechazadas_pred, linestyle='--', label=f"% Referencias de CE Rechazadas (Predicción, R2={r2_ce_rechazadas:.2f})")
plt.plot(months_future, y_ce_sin_notificacion_pred, linestyle='--', label=f"% Referencias de CE sin registro de notificación (Predicción, R2={r2_ce_sin_notificacion:.2f})")
plt.plot(months_future, y_ce_recibidas_no_agendadas_pred, linestyle='--', label=f"% Referencias recibidas en CE no agendadas (Predicción, R2={r2_ce_recibidas_no_agendadas:.2f})")
plt.plot(months_future, y_ce_no_agendadas_pred, linestyle='--', label=f"% Referencias enviadas a CE no agendadas (Predicción, R2={r2_ce_no_agendadas:.2f})")

plt.xlabel('Meses')
plt.ylabel('Porcentaje (%)')
plt.title('Tendencias y Predicción de Indicadores de Referencias')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Mostrar el gráfico con predicciones
st.pyplot(plt)
