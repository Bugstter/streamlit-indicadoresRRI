import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime

st.set_page_config(layout="wide")
st.title("📈 Análisis de Referencias por Área de Atención")

# ──────────────────────────────────────────────────────────────
#  Funciones auxiliares
# ──────────────────────────────────────────────────────────────
def cargar_datos(archivo):
    """Carga CSV, homogeniza columnas y normaliza texto."""
    df = pd.read_csv(archivo, low_memory=False)
    df.columns = df.columns.str.lower()
    cols_txt = [
        'atencion_origen', 'referencia_rechazada', 'referencia_oportuna',
        'referencia_efectiva', 'retorno_cont_seguimiento', 'motivo_no_notificacion',
        'area_origen', 'area_remision', 'posee_retorno', 'paciente_notificado',
        'referencia_pertinente'
    ]
    for c in cols_txt:
        if c in df.columns:
            df[c] = df[c].astype(str).str.lower().str.strip()
    df['fecha_cita_destino'] = pd.to_datetime(df['fecha_cita_destino'],
                                              errors='coerce')
    return df

def graficar_porcentajes(df_indicadores, titulo):
    """Grafica únicamente la columna porcentaje de un DataFrame de indicadores."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(df_indicadores.index, df_indicadores['porcentaje'])
    ax.set_xlabel("Porcentaje (%)")
    ax.set_title(titulo)
    ax.invert_yaxis()
    st.pyplot(fig)

# ──────────────────────────────────────────────────────────────
#  Indicadores de Consulta Externa
# ──────────────────────────────────────────────────────────────
def calcular_indicadores_ce(df):
    hoy = datetime.datetime.today()

    # Imputar valores faltantes en 'paciente_notificado'
    df['paciente_notificado'] = (
        df['paciente_notificado'].replace(['nan', '', ' '], np.nan).str.strip()
    )
    cond_imput = (
        (df['fecha_cita_destino'].notna()) &
        (df['area_remision'] == 'consulta') &
        (df['paciente_notificado'].isna())
    )
    df.loc[cond_imput, 'paciente_notificado'] = 'no'

    total_env = len(df)

    # --- 1. CE rechazadas
    num_rech = len(df[df['referencia_rechazada'] == 'si'])
    den_rech = total_env
    pct_rech = (num_rech / den_rech * 100) if den_rech else 0

    # --- 2. CE agendadas (sobre no rechazadas)
    ce_total = df[df['area_origen'] == 'consulta externa']
    ce_no_rech = ce_total[ce_total['referencia_rechazada'] != 'si']
    num_agend = len(ce_no_rech[ce_no_rech['fecha_cita_destino'].notna()])
    den_agend = len(ce_no_rech)
    pct_agend = (num_agend / den_agend * 100) if den_agend else 0

    # --- 3. CE sin notificación
    ce_agend_notif = ce_total[
    (ce_total['area_remision'] == 'consulta') &
    (ce_total['fecha_cita_destino'].notna())
    ]
    
    num_sin_notif = len(ce_agend_notif[
    ce_agend_notif['paciente_notificado'].astype(str).str.strip().replace('nan', '').replace('None', '') == ""
])

    den_sin_notif = len(ce_agend_notif)
    pct_sin_notif = (num_sin_notif / den_sin_notif * 100) if den_sin_notif else 0

    # --- 4. CE efectivas
    df_pasadas = df[df['fecha_cita_destino'] < hoy]
    ce_nr_pas = df_pasadas[df_pasadas['referencia_rechazada'] != 'si']
    num_efec = len(ce_nr_pas[ce_nr_pas['referencia_efectiva'] == 'si'])
    den_efec = len(ce_nr_pas)
    pct_efec = (num_efec / den_efec * 100) if den_efec else 0

    # --- 5. CE efectivas con retorno
    num_ret = len(ce_nr_pas[(ce_nr_pas['referencia_efectiva'] == 'si') &
                            (ce_nr_pas['posee_retorno'] == 'si')])
    den_ret = num_efec
    pct_ret = (num_ret / den_ret * 100) if den_ret else 0

    # --- 6. CE no agendadas aun
    ref_no_rech = df[df['referencia_rechazada'] == 'no']
    num_no_agend = len(ref_no_rech[ref_no_rech['fecha_cita_destino'].isna()])
    den_no_agend = len(ref_no_rech)
    pct_no_agend = (num_no_agend / den_no_agend * 100) if den_no_agend else 0

    # --- 7. CE oportunas
    num_opor = len(ce_nr_pas[ce_nr_pas['referencia_oportuna'] == 'si'])
    den_opor = num_efec
    pct_opor = (num_opor / den_opor * 100) if den_opor else 0

    # --- 8. CE pertinentes
    num_pert = len(ce_nr_pas[ce_nr_pas['referencia_pertinente'] == 'si'])
    den_pert = num_efec
    pct_pert = (num_pert / den_pert * 100) if den_pert else 0

    indicadores = {
        "% CE rechazadas":        (num_rech, den_rech, pct_rech),
        "% CE agendadas":         (num_agend, den_agend, pct_agend),
        "% CE sin notificación":  (num_sin_notif, den_sin_notif, pct_sin_notif),
        "% CE efectivas":         (num_efec, den_efec, pct_efec),
        "% CE efectivas c/ret.":  (num_ret, den_ret, pct_ret),
        "% CE no agendadas":      (num_no_agend, den_no_agend, pct_no_agend),
        "% CE oportunas":         (num_opor, den_opor, pct_opor),
        "% CE pertinentes":       (num_pert, den_pert, pct_pert),
        "Total referencias enviadas": (total_env, "-", "-")
    }
    # Convertimos a DataFrame para un manejo uniforme
    df_ind = pd.DataFrame(indicadores, index=["numerador","denominador","porcentaje"]).T
    df_ind["porcentaje"] = pd.to_numeric(df_ind["porcentaje"], errors="coerce").round(2)
    return df_ind

# ──────────────────────────────────────────────────────────────
#  Indicadores de Unidad de Emergencia
# ──────────────────────────────────────────────────────────────
def calcular_indicadores_ue(df):
    ue_total = df[df['area_remision'] == 'emergencia']

    # 1. Efectivas (pertinentes sobre enviadas a UE)
    num_efec = len(ue_total[ue_total['referencia_pertinente'].isin(["si", "no"])])
    den_efec = len(ue_total)
    pct_efec = (num_efec / den_efec * 100) if den_efec else 0

    # 2. Retorno entre pertinentes
    ue_pert = ue_total[ue_total['referencia_pertinente'].isin(["si", "no"])]
    num_ret = len(ue_pert[ue_pert['posee_retorno'] == 'si'])
    den_ret = len(ue_pert)
    pct_ret = (num_ret / den_ret * 100) if den_ret else 0

    # 3. Oportunas sobre pertinentes
    num_opor = len(ue_pert[ue_pert['referencia_oportuna'] == 'si'])
    den_opor = len(ue_pert)
    pct_opor = (num_opor / den_opor * 100) if den_opor else 0

    # 4. Pertinentes sobre pertinentes+no pertinentes
    num_pert = len(ue_pert[ue_pert['referencia_pertinente'] == 'si'])
    den_pert = len(ue_pert)
    pct_pert = (num_pert / den_pert * 100) if den_pert else 0

    indicadores = {
        "% UE efectivas":          (num_efec, den_efec, pct_efec),
        "% UE con retorno":        (num_ret, den_ret, pct_ret),
        "% UE oportunas":          (num_opor, den_opor, pct_opor),
        "% UE pertinentes":        (num_pert, den_pert, pct_pert),
        "Total referencias enviadas a UE": (len(ue_total), "-", "-")
    }
    df_ind = pd.DataFrame(indicadores, index=["numerador","denominador","porcentaje"]).T
    df_ind["porcentaje"] = pd.to_numeric(df_ind["porcentaje"], errors="coerce").round(2)
    return df_ind

# ──────────────────────────────────────────────────────────────
#  Layout Streamlit (pestañas)
# ──────────────────────────────────────────────────────────────
tab_ce, tab_ue = st.tabs(["🏥 Consulta Externa", "🚨 Unidad de Emergencia"])

with tab_ce:
    st.subheader("📂 Subir archivo para Consulta Externa")
    archivo = st.file_uploader("CSV CE", type=["csv"], key="ce")
    if archivo:
        df_ce = cargar_datos(archivo)
        df_ind_ce = calcular_indicadores_ce(df_ce)
        st.write("### Indicadores de Consulta Externa")
        st.dataframe(df_ind_ce)
        graficar_porcentajes(df_ind_ce, "Indicadores CE (porcentaje)")

with tab_ue:
    st.subheader("📂 Subir archivo para Unidad de Emergencia")
    archivo = st.file_uploader("CSV UE", type=["csv"], key="ue")
    if archivo:
        df_ue = cargar_datos(archivo)
        df_ind_ue = calcular_indicadores_ue(df_ue)
        st.write("### Indicadores de Emergencia")
        st.dataframe(df_ind_ue)
        graficar_porcentajes(df_ind_ue, "Indicadores UE (porcentaje)")
