import pandas as pd
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

import dash
from dash import dcc, html

import sys
import os

# Agrega el directorio actual al path para poder importar
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import create_ageb_layout, register_ageb_callbacks

# Df a utilizar:

# Nueva base de datos de backbone 2026
backbone = pd.read_csv('/Users/alanrobles/Documents/ALAN/GEOSTATS/Codigo/clinicas_con_final.csv')

# Df utilizado para calcular el nse por ageb: censo + coneval + amai
nse = pd.read_csv('/Users/alanrobles/Documents/ALAN/GEOSTATS/csv_nse/df_nse.csv')

# Eliminar columnas del antiguo cálculo del nse en el df de backbone
col_inicio = 'POBTOT'
col_fin = 'NSE'

backbone = backbone.drop(columns=backbone.columns[
    backbone.columns.get_loc(col_inicio) : backbone.columns.get_loc(col_fin) + 1
])
id_backbone = set(backbone['CVEGEO'].dropna().unique())
id_nse = set(nse['CVEGEO'].dropna().unique())

solo_backbone = id_backbone - id_nse

# Juntar ambos dfs 
merged = backbone.merge(nse, on='CVEGEO', how='left')
merged.to_csv('DBcomnpleta.csv')
merged
## Gráfico 1

clinic_counts = merged['Clinic Size'].value_counts().reset_index()
clinic_counts.columns = ['Clinic Size', 'Count']

# === Gráfico 1 ===
fig_1 = px.pie(
    clinic_counts,
    names='Clinic Size',
    values='Count',
    title='Distribución de tamaños de clínicas',
    color='Clinic Size',
    color_discrete_map={
        'Small': 'red',
        'Medium': 'blue',
        'Micro': 'orange',
    },
    hover_data=['Count']
)

fig_1.update_traces(
    textposition='inside',
    textinfo='percent+label',
    hoverinfo='label+percent'
)
merged = merged.rename(columns={'Specialty of the \nMedical Equipment': 'Specialty of the Medical Equipment'})
merged = merged.rename(columns={'Number of \nConsultations per Month': 'Number of Consultations per Month'})
backbone = backbone.rename(columns={'Specialty of the \nMedical Equipment': 'Specialty of the Medical Equipment'})
backbone = backbone.rename(columns={'Number of \nConsultations per Month': 'Number of Consultations per Month'})
# Gráfico 2
top20mun_consultations = (
    merged
    .groupby(['State', 'Municipality', 'Clinic Size'])['Number of Consultations per Month']
    .mean()
    .reset_index()
    .sort_values(by='Number of Consultations per Month', ascending=False)
    .head(20)
)

top20mun_consultations['estado_mun'] = top20mun_consultations['State'] + ', ' + top20mun_consultations['Municipality']

fig_2 = px.bar(
    top20mun_consultations,
    x='estado_mun',
    y='Number of Consultations per Month',
    color='Clinic Size',
    color_discrete_sequence=px.colors.sequential.Viridis,
    title='Top 20 Promedio de Consultas por Municipio y Tamaño de Clínica',
    hover_data={
        'Number of Consultations per Month': ':.2f',
        'estado_mun': True,
        'Clinic Size': True
    }
)

fig_2.update_layout(
    xaxis_title='Municipio',
    yaxis_title='Promedio de Consultas por Mes',
    xaxis_tickangle=-45,
    legend_title_text='Tamaño de Clínica',
    title_font_size=16,
    xaxis_title_font_size=12,
    yaxis_title_font_size=12
)
# === FIGURA 3: MAPA 1 ===
fig_3 = px.scatter_mapbox(
    merged,
    lat='latitud',
    lon='longitud',
    hover_name='Contract',
    hover_data={
        'Clinic Size': True,
        'Number of Consultations per Month': ':.2f'
    },
    color='Clinic Size',
    size='Number of Consultations per Month',
    color_discrete_sequence=px.colors.sequential.Viridis,
    mapbox_style='open-street-map',
    zoom=4,
    height=600,
    title='Mapa de Clínicas Backbone por # Consultas por Mes'
)

fig_3.update_layout(
    title_font_size=18,
    margin=dict(l=20, r=20, t=60, b=20),
    mapbox=dict(center=dict(lat=23.6345, lon=-102.5528)),  # Centrado en México
)
# # === FIGURA 4: MAPA DE COBERTURA VULNERABLE ===
# merged['cobertura_pct'] = merged['Number of Consultations per Month'] / (merged['PSINDER'] + 1)

# fig_4 = px.scatter_mapbox(
#     merged,
#     lat='latitud',
#     lon='longitud',
#     color='cobertura_pct',
#     size='Number of Consultations per Month',
#     mapbox_style='open-street-map',
#     zoom=4,
#     height=600,
#     title='% de Cobertura Vulnerable Estimada por Clínica'
# )

# fig_4.update_layout(
#     title_font_size=18,
#     margin=dict(l=20, r=20, t=60, b=20),
#     mapbox=dict(center=dict(lat=23.6345, lon=-102.5528)),
# )
# === FIGURA 5: % PROMEDIO DE PACIENTES DE BAJO INGRESO POR TIPO DE CLÍNICA ===

# Limpiar la columna: extraer solo el primer número o manejar los valores problemáticos
def clean_percentage(value):
    if pd.isna(value):
        return None
    if isinstance(value, str):
        # Si es como "10%10%10%..." tomar solo el primer número
        if '%' in value and len(value.split('%')) > 2:
            return float(value.split('%')[0])
        # Si es "10%", quitar el % y convertir
        return float(value.replace('%', ''))
    return float(value)

merged['pct_low_income_clean'] = merged['% of Patients with Middle-Low to Low Income'].apply(clean_percentage)

# Ahora agrupar
grouped = merged.groupby('Clinic Size')['pct_low_income_clean'].mean().reset_index()

fig_5 = px.bar(
    grouped,
    x='Clinic Size',
    y='pct_low_income_clean',
    color='Clinic Size',
    color_discrete_sequence=['seagreen'],
    title='% Promedio de Pacientes con Ingresos Medio-Bajo o Bajo por Tipo de Clínica'
)

fig_5.update_layout(
    xaxis_title='Tamaño de Clínica',
    yaxis_title='Porcentaje (%)',
    title_font_size=16,
    xaxis_title_font_size=12,
    yaxis_title_font_size=12,
    showlegend=False
)
# 1. Convertir latitud y longitud a numéricos
def convertir_coordenada(coordenada):
    """Convierte coordenada a float, manejando strings con comas y NaN"""
    if pd.isna(coordenada):
        return None
    if isinstance(coordenada, str):
        # Reemplazar coma por punto (formato europeo)
        coordenada = coordenada.replace(',', '.')
        # Eliminar cualquier texto no numérico
        import re
        coordenada = re.findall(r'-?\d+\.?\d*', coordenada)
        if coordenada:
            return float(coordenada[0])
        return None
    try:
        return float(coordenada)
    except (ValueError, TypeError):
        return None

# Aplicar conversión
merged['latitud'] = merged['latitud'].apply(convertir_coordenada)
merged['longitud'] = merged['longitud'].apply(convertir_coordenada)
# === FIGURA 6: MAPA DE CLÍNICAS POR % DE PACIENTES DE BAJO INGRESO ===
fig_6 = px.scatter_mapbox(
    merged,
    lat='latitud',
    lon='longitud',
    hover_name='Contract',
    color='Clinic Size',
    size='pct_low_income_clean',
    mapbox_style='open-street-map',
    zoom=4,
    height=600,
    title='Mapa de Clínicas Backbone por % de Pacientes de Bajos Ingresos'
)

fig_6.update_layout(
    title_font_size=18,
    margin=dict(l=20, r=20, t=60, b=20),
    mapbox=dict(center=dict(lat=23.6345, lon=-102.5528))
)
# === FIGURA 7: CLÍNICAS QUE ATIENDEN PACIENTES CON SEGURO MÉDICO ===
clinic_counts = backbone['Attends Patients with Health Insurance'].value_counts().reset_index()
clinic_counts.columns = ['Atiende Seguro Médico', 'Cantidad']

fig_7 = px.pie(
    clinic_counts,
    names='Atiende Seguro Médico',
    values='Cantidad',
    color_discrete_sequence=px.colors.sequential.Viridis,
    title='Porcentaje de Clínicas que Atienden Pacientes con Seguro Médico'
)

fig_7.update_traces(textinfo='percent+label', pull=[0.05, 0])
fig_7.update_layout(title_font_size=16)
merged["Average Age of the SME (years)"] = (
    merged["Average Age of the SME (years)"]
    .str.replace(",", ".", regex=False)
    .astype(float)
)
fig_8 = px.histogram(
    merged,
    x='Average Age of the SME (years)',
    nbins=50,
    histfunc='count',
    title='Antigüedad de las Clínicas'
)
fig_8.update_traces(marker_color='indianred')
fig_8.update_layout(
    xaxis_title='Average Age of the SME (years)',
    yaxis_title='Frecuencia',
    title_font_size=16
)

fig_9 = px.box(
    merged,
    x='Specialty of the Medical Equipment',
    y='Average Age of the SME (years)',
    color='Specialty of the Medical Equipment',
    title='Especialidades con Clínicas con más antigüedad'
)
fig_9.update_layout(
    xaxis_title='Especialidad',
    yaxis_title='Promedio de Edad de SME',
    xaxis_tickangle=-45,
    title_font_size=16,
    xaxis_title_font_size=12,
    yaxis_title_font_size=12,
    showlegend=False
)
fig_10 = px.scatter(
    merged,
    x='Average Age of the SME (years)',
    y='Number of Consultations per Month',
    color='Specialty of the Medical Equipment',
    size='pct_low_income_clean',
    hover_name='Specialty of the Medical Equipment',
    title='Edad vs Consultas por Especialidad (tamaño = % Pacientes vulnerables)',
    labels={
        'Average Age of the SME (years)': 'Años de operación',
        'Number of Consultations per Month': 'Consultas mensuales',
        '% of Patients with Middle-Low to Low Income': '% Pacientes bajos recursos'
    }
)

fig_10
fig_11 = px.scatter(
    merged,
    x='Total Staff',
    y='Number of Consultations per Month',
    color='Clinic Size',
    size='Number of Consultations per Month',
    size_max=40,
    title='Relación entre Personal Médico y Consultas Mensuales'
)

fig_11.update_layout(
    xaxis_title='Personal Total',
    yaxis_title='Número de Consultas por Mes',
    title_font_size=16
)

fig_11_zoom = px.scatter(
    merged,
    x='Total Staff',
    y='Number of Consultations per Month',
    color='Clinic Size',
    size='Number of Consultations per Month',
    size_max=40,
    title='Relación entre Personal Médico y Consultas Mensuales (Zoom)'
)

fig_11_zoom.update_layout(
    xaxis_title='Personal Total',
    yaxis_title='Número de Consultas por Mes',
    title_font_size=16
)

fig_11_zoom.update_xaxes(range=[0, 70])
fig_11_zoom.update_yaxes(range=[0, 15000])

merged['consultas_por_staff'] = merged['Number of Consultations per Month'] / merged['Total Staff']
fig_12 = px.box(
    merged,
    x='Specialty of the Medical Equipment',
    y='consultas_por_staff',
    color='Specialty of the Medical Equipment',
    title='Consultas por Personal por Especialidad del Equipo Médico'
)
fig_12.update_layout(
    xaxis_title='Especialidad del Equipo Médico',
    yaxis_title='Consultas por Personal',
    xaxis_tickangle=-45,
    title_font_size=16,
    xaxis_title_font_size=12,
    yaxis_title_font_size=12,
    yaxis_showgrid=True,
    yaxis_gridcolor='lightgray',
    yaxis_gridwidth=0.5,
    showlegend=False
)

fig_13 = px.scatter_mapbox(
    merged,
    lat='latitud',
    lon='longitud',
    hover_name='Contract',
    color='nse_nivel_v2',
    mapbox_style='open-street-map',
    zoom=4,
    height=600,
    title='Mapa de las Clínicas Backbone por NSE'
)
fig_13.update_layout(
    title_font_size=18,
    margin=dict(l=20, r=20, t=60, b=20),
    mapbox=dict(center=dict(lat=23.6345, lon=-102.5528))
)

import numpy as np

# Preparar DataFrame con ratio
df = merged[['Contract', 'Clinic Size', 'Number of Consultations per Month', 'Total Staff', 'City', 'Specialty of the Medical Equipment']].copy()
df['operational_ratio'] = df['Number of Consultations per Month'] / df['Total Staff']
limits = {'Micro': 100, 'Small': 150, 'Medium': 200}
df['overloaded'] = df.apply(lambda x: x['operational_ratio'] > limits.get(x['Clinic Size'], np.inf), axis=1)
overloaded_clinics = df[df['overloaded']]

# fig_14 → Por City
city_counts = overloaded_clinics['City'].value_counts().head(10).reset_index()
city_counts.columns = ['City', 'Cantidad']

fig_14 = px.bar(
    city_counts,
    x='Cantidad',
    y='City',
    orientation='h',
    title='Clínicas Sobrecargadas por Ciudad',
    color='Cantidad',
    color_continuous_scale='Reds'
)
fig_14.update_layout(yaxis=dict(autorange="reversed"))  # Invertir eje y para barras horizontales

# fig_15 → Por Specialty
spec_counts = overloaded_clinics['Specialty of the Medical Equipment'].value_counts().head(10).reset_index()
spec_counts.columns = ['Especialidad', 'Cantidad']

fig_15 = px.bar(
    spec_counts,
    x='Cantidad',
    y='Especialidad',
    orientation='h',
    title='Clínicas Sobrecargadas por Especialidad',
    color='Cantidad',
    color_continuous_scale='Reds'
)
fig_15.update_layout(yaxis=dict(autorange="reversed"))
# === FIGURA 16: Distribución de NSE de las Clínicas ===
nse_counts = merged['nse_nivel_v2'].value_counts().reset_index()
nse_counts.columns = ['nse_nivel_v2', 'Cantidad']

fig_16 = px.pie(
    nse_counts,
    names='nse_nivel_v2',
    values='Cantidad',
    color_discrete_sequence=px.colors.sequential.Viridis,
    title='Distribución de NSE de las Clínicas'
)
fig_16.update_traces(textinfo='percent+label', pull=[0.05]*len(nse_counts))
fig_16.update_layout(title_font_size=16)

# Paso 1: contar cuántas clínicas hay por especialidad
counts = merged['Specialty of the Medical Equipment'].value_counts().reset_index()
counts.columns = ['Specialty of the Medical Equipment', 'Count']

# Paso 2: calcular porcentaje
counts['Percentage'] = 100 * counts['Count'] / counts['Count'].sum()

# Paso 3: agrupar las especialidades con <5% en "Others"
others = counts[counts['Percentage'] < 5]['Count'].sum()
main = counts[counts['Percentage'] >= 5].copy()

# agregar la categoría Others
if others > 0:
    main.loc[len(main)] = ['Others', others, 100 * others / counts['Count'].sum()]

fig_16
# Paso 4: graficar
fig_17 = px.pie(
    main,
    names='Specialty of the Medical Equipment',
    values='Count',
    title='Distribución de especialidades de clínicas',
    color='Specialty of the Medical Equipment',
    hover_data=['Percentage']
)

fig_17.update_traces(
    textposition='inside',
    textinfo='percent+label',
    hoverinfo='label+percent+name'
)
# === FIGURA 18 ===
clinicas_backbone = merged.rename(columns={'Specialty of the Medical Equipment': 'Specialty of the Medical Equipment'})
fig_18 = px.scatter_mapbox(
    clinicas_backbone,
    lat='latitud',
    lon='longitud',
    hover_name='Contract',
    hover_data={
        'Specialty of the Medical Equipment': True,
    },
    color='Specialty of the Medical Equipment',
    color_discrete_sequence=px.colors.sequential.Viridis,
    mapbox_style='open-street-map',
    zoom=4,
    height=600,
    title='Mapa de Clínicas Backbone por # Consultas por Mes'
)

fig_18.update_layout(
    title_font_size=18,
    margin=dict(l=20, r=20, t=60, b=20),
    mapbox=dict(center=dict(lat=23.6345, lon=-102.5528)),  # Centrado en México
)

fig_18.update_traces(
    marker=dict(size=12),
    selector=dict(mode='markers')
)

# Paso 1: calcular promedio de consultas por especialidad
promedios = (
    merged
    .groupby('Specialty of the Medical Equipment', as_index=False)
    ['Number of Consultations per Month']
    .mean()
)

# Paso 2: graficar los promedios
fig_19 = px.bar(
    promedios,
    x='Specialty of the Medical Equipment',
    y='Number of Consultations per Month',
    title='Promedio de Consultas por Especialidad',
    hover_data={
        'Number of Consultations per Month': ':.2f',
    },
    color='Specialty of the Medical Equipment',  # opcional: colorear por especialidad
)

# Paso 3: personalizar diseño
fig_19.update_layout(
    xaxis_title='Especialidad',
    yaxis_title='Promedio de Consultas por Mes',
    xaxis_tickangle=-45,
    showlegend=False,  # ocultar leyenda si no es necesaria
    title_font_size=16,
    xaxis_title_font_size=12,
    yaxis_title_font_size=12
)
nse_consultas = (
    merged.groupby('nse_nivel_v2', as_index=False)
    ['Number of Consultations per Month']
    .sum()
)

# Orden correcto AMAI
orden_amai = ['E', 'D', 'D+', 'C-', 'C', 'C+', 'AB']

nse_consultas['nse_nivel_v2'] = pd.Categorical(
    nse_consultas['nse_nivel_v2'],
    categories=orden_amai,
    ordered=True
)

nse_consultas = nse_consultas.sort_values('nse_nivel_v2')

fig_20 = px.bar(
    nse_consultas,
    x='nse_nivel_v2',
    y='Number of Consultations per Month',
    color='Number of Consultations per Month',
    color_continuous_scale='Viridis',
    title='Total de Consultas por NSE',
    text_auto='.2s'
)

fig_20.update_layout(
    xaxis_title='NSE',
    yaxis_title='Total de Consultas por Mes',
    title_font_size=16,
    showlegend=False
)

merged['precio_promedio'] = pd.to_numeric(merged['precio_promedio'], errors='coerce')

""" # === FIGURA 21: MAPA DE CLÍNICAS POR PRECIO PROMEDIO DE LA ZONA ===
# Permite identificar geográficamente en qué zonas de precio opera cada clínica.

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# =========================================
# Filtrar datos válidos
# =========================================
map_data = merged.dropna(subset=['precio_promedio']).copy()

# =========================================
# Definir ciudades
# Ajusta el nombre de la columna si es necesario
# =========================================
ciudades = {
    'CDMX': ['Ciudad de México', 'CDMX'],
    'Guadalajara': ['Guadalajara'],
    'Monterrey': ['Monterrey']
}

# =========================================
# Crear subplots
# =========================================
fig = make_subplots(
    rows=1,
    cols=3,
    specs=[[{'type': 'mapbox'},
            {'type': 'mapbox'},
            {'type': 'mapbox'}]],
    subplot_titles=['CDMX', 'Guadalajara', 'Monterrey']
)

# =========================================
# Centros aproximados
# =========================================
centros = {
    'CDMX': dict(lat=19.4326, lon=-99.1332),
    'Guadalajara': dict(lat=20.6597, lon=-103.3496),
    'Monterrey': dict(lat=25.6866, lon=-100.3161)
}

# =========================================
# Agregar cada mapa
# =========================================
for i, (nombre, valores) in enumerate(ciudades.items(), start=1):

    df_city = map_data[
        map_data['City'].isin(valores)
    ]

    fig.add_trace(
        go.Scattermapbox(
            lat=df_city['latitud'],
            lon=df_city['longitud'],
            mode='markers',
            marker=dict(
                size=10,
                color=df_city['precio_promedio'],
                colorscale='Viridis',
                showscale=(i == 3),  # solo una barra de color
                colorbar=dict(title='Precio Promedio')
            ),
            text=df_city['Contract'],
            customdata=df_city[
                ['precio_promedio',
                 'Clinic Size',
                 'nse_nivel_v2',
                 'Number of Consultations per Month']
            ],
            hovertemplate=(
                "<b>%{text}</b><br>" +
                "Precio promedio: %{customdata[0]:,.0f} MXN<br>" +
                "Tamaño clínica: %{customdata[1]}<br>" +
                "NSE: %{customdata[2]}<br>" +
                "Consultas/mes: %{customdata[3]:,.0f}<extra></extra>"
            )
        ),
        row=1,
        col=i
    )

# =========================================
# Configuración de cada mapa
# =========================================
for i, ciudad in enumerate(['CDMX', 'Guadalajara', 'Monterrey'], start=1):

    fig.update_layout(
        {
            f"mapbox{i if i > 1 else ''}": dict(
                style='open-street-map',
                center=centros[ciudad],
                zoom=9
            )
        }
    )

# =========================================
# Layout final
# =========================================
fig.update_layout(
    height=600,
    title='Precio Promedio por Zona en Clínicas Backbone',
    margin=dict(l=20, r=20, t=60, b=20)
) """


# %%
# === FIGURA 22: SCATTER — PRECIO PROMEDIO vs CONSULTAS POR MES ===
# Correlación entre el nivel socioeconómico de la zona (proxy: precio promedio)
# y la demanda de consultas. El tamaño de cada punto refleja el % de pacientes
# vulnerables, y el color diferencia el tamaño de clínica.

fig_22 = px.scatter(
    merged,
    x='precio_promedio',
    y='Number of Consultations per Month',
    color='Clinic Size',
    size='pct_low_income_clean',
    size_max=35,
    hover_name='Contract',
    hover_data={
        'precio_promedio': ':.0f',
        'Number of Consultations per Month': True,
        'pct_low_income_clean': ':.1f',
        'nse_nivel_v2': True
    },
    color_discrete_sequence=px.colors.sequential.Viridis,
    title='Precio Promedio de la Zona vs Consultas por Mes<br>'
          '<sup>Tamaño del punto = % Pacientes con ingresos bajos</sup>'
)

fig_22.update_layout(
    xaxis_title='Precio Promedio de la Zona (MXN)',
    yaxis_title='Consultas por Mes',
    legend_title_text='Tamaño de Clínica',
    title_font_size=16,
    xaxis_title_font_size=12,
    yaxis_title_font_size=12
)

fig_22_zoom = px.scatter(
    merged,
    x='precio_promedio',
    y='Number of Consultations per Month',
    color='Clinic Size',
    size='pct_low_income_clean',
    size_max=35,
    hover_name='Contract',
    hover_data={
        'precio_promedio': ':.0f',
        'Number of Consultations per Month': True,
        'pct_low_income_clean': ':.1f',
        'nse_nivel_v2': True
    },
    color_discrete_sequence=px.colors.sequential.Viridis,
    title='Precio Promedio de la Zona vs Consultas por Mes<br>'
          '<sup>Tamaño del punto = % Pacientes con ingresos bajos</sup>'
)

fig_22_zoom.update_layout(
    xaxis_title='Precio Promedio de la Zona (MXN)',
    yaxis_title='Consultas por Mes',
    legend_title_text='Tamaño de Clínica',
    title_font_size=16,
    xaxis_title_font_size=12,
    yaxis_title_font_size=12
)

fig_22_zoom.update_xaxes(range=[0, 2100])
fig_22_zoom.update_yaxes(range=[0, 4000])


# %%
# === FIGURA 24: BARRAS — PRECIO PROMEDIO MEDIANO POR TAMAÑO DE CLÍNICA ===
# Responde si las clínicas grandes tienden a ubicarse en zonas más caras.
# Se usa la mediana en lugar del promedio para mitigar el efecto de outliers.

# =========================================
# Mediana y media por tamaño de clínica
# =========================================
precio_por_tamano = (
    merged
    .groupby('Clinic Size')['precio_promedio']
    .agg(['median', 'mean'])
    .reset_index()
)

# =========================================
# Reestructurar para Plotly
# =========================================
precio_long = precio_por_tamano.melt(
    id_vars='Clinic Size',
    value_vars=['median', 'mean'],
    var_name='Métrica',
    value_name='Precio'
)

# =========================================
# Gráfico
# =========================================
fig_24 = px.bar(
    precio_long,
    x='Clinic Size',
    y='Precio',
    color='Métrica',
    barmode='group',
    text_auto=',.0f',
    title='Media y Mediana del Precio de Zona por Tamaño de Clínica',
    color_discrete_sequence=px.colors.sequential.Viridis
)

fig_24.update_layout(
    xaxis_title='Tamaño de Clínica',
    yaxis_title='Precio Promedio (MXN)',
    title_font_size=16,
    xaxis_title_font_size=12,
    yaxis_title_font_size=12
)

fig_24.update_traces(textposition='outside')
# =============================================================================
# BLOQUE B: DEMOGRAFÍA (GRUPOS DE EDAD Y GÉNERO) vs DEMANDA DE CONSULTAS
# =============================================================================

# --- Preprocesamiento: crear variables de proporción demográfica ---

# =========================================
# Convertir columnas a formato numérico
# =========================================
cols_demo = [
    'pob_60ymas',
    'pob_total',
    'pob_15a64',
    'pob_0a14',
    'pob_15a49_f',
    'pob_fem'
]

for col in cols_demo:
    merged[col] = pd.to_numeric(
        merged[col]
        .astype(str)
        .str.replace(',', '', regex=False)
        .str.strip(),
        errors='coerce'
    )

# =========================================
# Evitar divisiones entre cero
# =========================================
merged['pob_total'] = merged['pob_total'].replace(0, np.nan)
merged['pob_fem'] = merged['pob_fem'].replace(0, np.nan)

# Proporción de adultos mayores (60+) sobre población total
merged['pct_adultos_mayores'] = (
    merged['pob_60ymas'] / merged['pob_total']
).replace([np.inf, -np.inf], np.nan) * 100

# Proporción de población en edad productiva (15-64)
merged['pct_pob_15a64'] = (
    merged['pob_15a64'] / merged['pob_total']
).replace([np.inf, -np.inf], np.nan) * 100

# Proporción de población infantil (0-14)
merged['pct_pob_0a14'] = (
    merged['pob_0a14'] / merged['pob_total']
).replace([np.inf, -np.inf], np.nan) * 100

# Proporción de mujeres en edad fértil (15-49) sobre población femenina total
merged['pct_mujeres_fertiles'] = (
    merged['pob_15a49_f'] / merged['pob_fem']
).replace([np.inf, -np.inf], np.nan) * 100
# %%
# === FIGURA 26: SCATTER — % ADULTOS MAYORES vs CONSULTAS POR MES ===
# Las zonas con mayor proporción de adultos mayores deberían demandar
# más consultas. El color indica la especialidad de la clínica.

# =========================================
# Filtrar datos válidos
# =========================================
scatter_data = merged.dropna(
    subset=[
        'pct_adultos_mayores',
        'Number of Consultations per Month',
        'precio_promedio'
    ]
).copy()

# =========================================
# Scatter plot
# =========================================
fig_26 = px.scatter(
    scatter_data,
    x='pct_adultos_mayores',
    y='Number of Consultations per Month',
    color='Specialty of the Medical Equipment',
    size='precio_promedio',
    size_max=30,
    hover_name='Contract',
    hover_data={
        'pct_adultos_mayores': ':.1f',
        'Number of Consultations per Month': ':.0f',
        'Clinic Size': True,
        'precio_promedio': ':.0f'
    },
    color_discrete_sequence=px.colors.sequential.Viridis,
    title='% Adultos Mayores en la Zona vs Consultas por Mes<br>'
          '<sup>Tamaño del punto = Precio Promedio de la Zona</sup>'
)

fig_26.update_layout(
    xaxis_title='% Población de 60 años o más',
    yaxis_title='Consultas por Mes',
    legend_title_text='Especialidad',
    title_font_size=16,
    xaxis_title_font_size=12,
    yaxis_title_font_size=12
)

# %%
# === FIGURA 27: BOX PLOT — CONSULTAS POR MES POR NSE, SEGMENTADO POR TAMAÑO ===
# Combina NSE con tamaño de clínica para ver si el volumen de consultas
# varía según el nivel socioeconómico de la zona en cada segmento.

fig_27 = px.box(
    merged.dropna(subset=['nse_nivel_v2']),
    x='nse_nivel_v2',
    y='Number of Consultations per Month',
    color='Clinic Size',
    color_discrete_sequence=px.colors.sequential.Viridis,
    title='Consultas por Mes según NSE y Tamaño de Clínica',
    category_orders={'nse_nivel_v2': orden_amai}
)

fig_27.update_layout(
    xaxis_title='Nivel NSE (AMAI)',
    yaxis_title='Consultas por Mes',
    legend_title_text='Tamaño de Clínica',
    title_font_size=16,
    xaxis_title_font_size=12,
    yaxis_title_font_size=12
)

# =========================================================
# APP DASH - Layout centrado y ancho completo
# =========================================================
from dash import Dash, dcc, html, Input, Output
from flask_caching import Cache
import time  # solo para simular carga lenta
# ==========================
# Inicialización de Dash
# ==========================
app = Dash(__name__, suppress_callback_exceptions=True)
server = app.server

# Configurar caché
cache = Cache(app.server, config={
    'CACHE_TYPE': 'SimpleCache',
    'CACHE_DEFAULT_TIMEOUT': 600
})

# Lista de estados para el filtro global
state_options = [{'label': s, 'value': s} for s in sorted(merged['State'].unique())]
state_options.insert(0, {'label': 'Todos', 'value': 'Todos'})  # opción para mostrar todo

# ==========================
# Funciones para generar gráficos filtrados
# ==========================
def generate_fig1(df):
    return fig_1  

def generate_fig2(df):
    return fig_2

def generate_fig3(df):
    return fig_3

def generate_fig5(df):
    return fig_5

def generate_fig6(df):
    return fig_6

def generate_fig7(df):
    return fig_7

def generate_fig8(df):
    return fig_8

def generate_fig9(df):
    return fig_9

def generate_fig10(df):
    return fig_10

def generate_fig11(df):
    return fig_11

def generate_fig12(df):
    return fig_12

def generate_fig14(df):
    return fig_14

def generate_fig15(df):
    return fig_15

def generate_fig13(df):
    return fig_13

def generate_fig16(df):
    return fig_16

def generate_fig16(df):
    return fig_17

def generate_fig16(df):
    return fig_18

def generate_fig16(df):
    return fig_19

def generate_fig16(df):
    return fig_20

def generate_fig22(df):
    return fig_22


def generate_fig24(df):
    return fig_24


def generate_fig26(df):
    return fig_26

def generate_fig27(df):
    return fig_27


# ==========================
# Crear dashboards
# ==========================
@cache.memoize()

@cache.memoize()
def create_dashboards(estado_filtro=None, nse_filtro=None, especialidad_filtro=None):
    # Aplicar filtro a los DataFrames principales
    if estado_filtro and estado_filtro != "Todos":
        clinicas_filtradas = merged[backbone['State'] == estado_filtro].copy()
        merged_filtrado = merged[merged['State'] == estado_filtro].copy()
    else:
        clinicas_filtradas = merged.copy()
        merged_filtrado = merged.copy()
    
    # Aplicar filtro por NSE
    if nse_filtro and nse_filtro != "Todos":
        merged_filtrado = merged_filtrado[merged_filtrado['nse_nivel_v2'] == nse_filtro].copy()
    
    # Aplicar filtro por especialidad
    if especialidad_filtro and especialidad_filtro != "Todos":
        clinicas_filtradas = clinicas_filtradas[clinicas_filtradas['Specialty of the Medical Equipment'] == especialidad_filtro].copy()
        merged_filtrado = merged_filtrado[merged_filtrado['Specialty of the Medical Equipment'] == especialidad_filtro].copy()
    #Resto del código de creación de figuras permanece igual...
    # === Gráfico 1 - Actualizado para usar DataFrame filtrado ===
    clinic_counts = merged_filtrado['Clinic Size'].value_counts().reset_index()

    # === Gráfico 1 - Actualizado para usar DataFrame filtrado ===
    clinic_counts = merged_filtrado['Clinic Size'].value_counts().reset_index()
    clinic_counts.columns = ['Clinic Size', 'Count']

    fig_1 = px.pie(
        clinic_counts,
        names='Clinic Size',
        values='Count',
        title=f'Distribución de tamaños de clínicas{" - " + estado_filtro if estado_filtro and estado_filtro != "Todos" else ""}',
        color='Clinic Size',
        color_discrete_map={
            'Small': 'red',
            'Medium': 'blue',
            'Micro': 'orange',
        },
        hover_data=['Count']
    )
    fig_1.update_traces(
        textposition='inside',
        textinfo='percent+label',
        hoverinfo='label+percent'
    )

    # === Gráfico 2 - Actualizado ===
    top20mun_consultations = (
        clinicas_filtradas
        .groupby(['State', 'Municipality', 'Clinic Size'])['Number of Consultations per Month']
        .mean()
        .reset_index()
        .sort_values(by='Number of Consultations per Month', ascending=False)
        .head(20)
    )

    top20mun_consultations['estado_mun'] = top20mun_consultations['State'] + ', ' + top20mun_consultations['Municipality']

    fig_2 = px.bar(
        top20mun_consultations,
        x='estado_mun',
        y='Number of Consultations per Month',
        color='Clinic Size',
        color_discrete_sequence=px.colors.sequential.Viridis,
        title=f'Top 20 Promedio de Consultas por Municipio y Tamaño de Clínica{" - " + estado_filtro if estado_filtro and estado_filtro != "Todos" else ""}',
        hover_data={
            'Number of Consultations per Month': ':.2f',
            'estado_mun': True,
            'Clinic Size': True
        }
    )
    fig_2.update_layout(
        xaxis_title='Municipio',
        yaxis_title='Promedio de Consultas por Mes',
        xaxis_tickangle=-45,
        legend_title_text='Tamaño de Clínica',
        title_font_size=16,
        xaxis_title_font_size=12,
        yaxis_title_font_size=12
    )

    # === FIGURA 3: MAPA 1 - Actualizado ===
    fig_3 = px.scatter_mapbox(
        clinicas_filtradas,
        lat='latitud',
        lon='longitud',
        hover_name='Contract',
        hover_data={
            'Clinic Size': True,
            'Number of Consultations per Month': ':.2f'
        },
        color='Clinic Size',
        mapbox_style='open-street-map',
        zoom=4,
        height=600,
        title=f'Mapa de Clínicas Backbone por Tamaño de Clínica{" - " + estado_filtro if estado_filtro and estado_filtro != "Todos" else ""}'
    )

    fig_3.update_traces(marker=dict(size=18))

    fig_3.update_layout(
        title_font_size=18,
        margin=dict(l=20, r=20, t=60, b=20),
        mapbox=dict(center=dict(lat=23.6345, lon=-102.5528)),
    )

    """ # === FIGURA 4: MAPA DE COBERTURA VULNERABLE - Actualizado ===
    merged_filtrado['cobertura_pct'] = merged_filtrado['Number of Consultations per Month'] / (merged_filtrado['PSINDER'] + 1)

    fig_4 = px.scatter_mapbox(
        merged_filtrado,
        lat='latitud',
        lon='longitud',
        color='cobertura_pct',
        size='Number of Consultations per Month',
        mapbox_style='open-street-map',
        zoom=4,
        height=600,
        title=f'% de Cobertura Vulnerable Estimada por Clínica{" - " + estado_filtro if estado_filtro and estado_filtro != "Todos" else ""}'
    )

    fig_4.update_layout(
        title_font_size=18,
        margin=dict(l=20, r=20, t=60, b=20),
        mapbox=dict(center=dict(lat=23.6345, lon=-102.5528)),
    ) """

    # === FIGURA 5: % PROMEDIO DE PACIENTES DE BAJO INGRESO - Actualizado ===
    def clean_percentage(value):
        if pd.isna(value):
            return None
        if isinstance(value, str):
            if '%' in value and len(value.split('%')) > 2:
                return float(value.split('%')[0])
            return float(value.replace('%', ''))
        return float(value)

    clinicas_filtradas['pct_low_income_clean'] = (
        clinicas_filtradas['% of Patients with Middle-Low to Low Income']
        .apply(clean_percentage)
)

    grouped = clinicas_filtradas.groupby('Clinic Size')['pct_low_income_clean'].mean().reset_index()
    grouped = grouped.rename(columns={'pct_low_income_clean': '% of Patients with Middle-Low to Low Income'})

    fig_5 = px.bar(
        grouped,
        x='Clinic Size',
        y='% of Patients with Middle-Low to Low Income',
        color='Clinic Size',
        color_discrete_sequence=['seagreen'],
        title=f'% Promedio de Pacientes con Ingresos Medio-Bajo o Bajo por Tipo de Clínica{" - " + estado_filtro if estado_filtro and estado_filtro != "Todos" else ""}'
    )
    fig_5.update_layout(
        xaxis_title='Tamaño de Clínica',
        yaxis_title='Porcentaje (%)',
        title_font_size=16,
        xaxis_title_font_size=12,
        yaxis_title_font_size=12,
        showlegend=False
    )

    # === FIGURA 6: MAPA DE CLÍNICAS POR % DE PACIENTES DE BAJO INGRESO - Actualizado ===
    fig_6 = px.scatter_mapbox(
        clinicas_filtradas,
        lat='latitud',
        lon='longitud',
        hover_name='Contract',
        color='Clinic Size',
        size='pct_low_income_clean',
        mapbox_style='open-street-map',
        zoom=4,
        height=600,
        title=f'Mapa de Clínicas Backbone por % de Pacientes de Bajos Ingresos{" - " + estado_filtro if estado_filtro and estado_filtro != "Todos" else ""}'
    )
    fig_6.update_layout(
        title_font_size=18,
        margin=dict(l=20, r=20, t=60, b=20),
        mapbox=dict(center=dict(lat=23.6345, lon=-102.5528))
    )

    # === FIGURA 7: CLÍNICAS QUE ATIENDEN PACIENTES CON SEGURO MÉDICO - Actualizado ===
    clinic_counts = clinicas_filtradas['Attends Patients with Health Insurance'].value_counts().reset_index()
    clinic_counts.columns = ['Atiende Seguro Médico', 'Cantidad']

    fig_7 = px.pie(
        clinic_counts,
        names='Atiende Seguro Médico',
        values='Cantidad',
        color_discrete_sequence=px.colors.sequential.Viridis,
        title=f'Porcentaje de Clínicas que Atienden Pacientes con Seguro Médico{" - " + estado_filtro if estado_filtro and estado_filtro != "Todos" else ""}'
    )
    fig_7.update_traces(textinfo='percent+label', pull=[0.05, 0])
    fig_7.update_layout(title_font_size=16)

    # === FIGURA 8: HISTOGRAMA ANTIGÜEDAD - Actualizado ===
    fig_8 = px.histogram(
        clinicas_filtradas,
        x='Average Age of the SME (years)',
        nbins=20,
        title=f'Antigüedad de las Clínicas{" - " + estado_filtro if estado_filtro and estado_filtro != "Todos" else ""}'
    )
    fig_8.update_traces(marker_color='indianred')
    fig_8.update_layout(
        xaxis_title='Average Age of the SME (years)',
        yaxis_title='Frecuencia',
        title_font_size=16
    )

    # === FIGURA 9: BOXPLOT ESPECIALIDADES - Actualizado ===
    fig_9 = px.box(
        clinicas_filtradas,
        x='Specialty of the Medical Equipment',
        y='Average Age of the SME (years)',
        color='Specialty of the Medical Equipment',
        title=f'Especialidades con Clínicas con más antigüedad{" - " + estado_filtro if estado_filtro and estado_filtro != "Todos" else ""}'
    )
    fig_9.update_layout(
        xaxis_title='Especialidad',
        yaxis_title='Promedio de Edad de SME',
        xaxis_tickangle=-45,
        title_font_size=16,
        xaxis_title_font_size=12,
        yaxis_title_font_size=12,
        showlegend=False
    )

    # === FIGURA 10: SCATTER EDAD VS CONSULTAS - Actualizado ===
    fig_10 = px.scatter(
        merged_filtrado,
        x='Average Age of the SME (years)',
        y='Number of Consultations per Month',
        color='Specialty of the Medical Equipment',
        size='pct_low_income_clean',
        hover_name='Specialty of the Medical Equipment',
        title=f'Edad vs Consultas por Especialidad{" - " + estado_filtro if estado_filtro and estado_filtro != "Todos" else ""}',
        labels={
            'Average Age of the SME (years)': 'Años de operación',
            'Number of Consultations per Month': 'Consultas mensuales',
            '% of Patients with Middle-Low to Low Income': '% Pacientes bajos recursos'
        }
    )

    # === FIGURA 11: SCATTER PERSONAL VS CONSULTAS - Actualizado ===
    fig_11 = px.scatter(
        clinicas_filtradas,
        x='Total Staff',
        y='Number of Consultations per Month',
        color='Clinic Size',
        size='Number of Consultations per Month',
        size_max=40,
        title=f'Relación entre Personal Médico y Consultas Mensuales{" - " + estado_filtro if estado_filtro and estado_filtro != "Todos" else ""}'
    )
    fig_11.update_layout(
        xaxis_title='Personal Total',
        yaxis_title='Número de Consultas por Mes',
        title_font_size=16
    )

    # === FIGURA 12: BOX CONSULTAS POR PERSONAL - Actualizado ===
    clinicas_filtradas['consultas_por_staff'] = clinicas_filtradas['Number of Consultations per Month'] / clinicas_filtradas['Total Staff']

    fig_12 = px.box(
        clinicas_filtradas,
        x='Specialty of the Medical Equipment',
        y='consultas_por_staff',
        color='Specialty of the Medical Equipment',
        title=f'Consultas por Personal por Especialidad del Equipo Médico{" - " + estado_filtro if estado_filtro and estado_filtro != "Todos" else ""}'
    )
    fig_12.update_layout(
        xaxis_title='Especialidad del Equipo Médico',
        yaxis_title='Consultas por Personal',
        xaxis_tickangle=-45,
        title_font_size=16,
        xaxis_title_font_size=12,
        yaxis_title_font_size=12,
        yaxis_showgrid=True,
        yaxis_gridcolor='lightgray',
        yaxis_gridwidth=0.5,
        showlegend=False
    )

    # === FIGURA 13: MAPA NSE - Actualizado ===
    fig_13 = px.scatter_mapbox(
        clinicas_filtradas,
        lat='latitud',
        lon='longitud',
        hover_name='Contract',
        color='nse_nivel_v2',
        color_discrete_sequence = [
            "#3B528B",  # azul
            "#440154",  # morado profund
            "#FDE725",  # amarillo brillante
            "#21918C"  # verde turquesa
        ],
        mapbox_style='open-street-map',
        zoom=4,
        height=600,
        title=f'Mapa de Clínicas Backbone por NSE{" - " + estado_filtro if estado_filtro and estado_filtro != "Todos" else ""}'
    )

    fig_13.update_traces(marker=dict(size=18))

    fig_13.update_layout(
        title_font_size=18,
        margin=dict(l=20, r=20, t=60, b=20),
        mapbox=dict(center=dict(lat=23.6345, lon=-102.5528))
    )

    # === FIGURAS 14-15: CLÍNICAS SOBRECARGADAS - Actualizado ===
    df = clinicas_filtradas[['Contract', 'Clinic Size', 'Number of Consultations per Month', 'Total Staff', 'City', 'Specialty of the Medical Equipment']].copy()
    df['operational_ratio'] = df['Number of Consultations per Month'] / df['Total Staff']
    limits = {'Micro': 100, 'Small': 150, 'Medium': 200}
    df['overloaded'] = df.apply(lambda x: x['operational_ratio'] > limits.get(x['Clinic Size'], np.inf), axis=1)
    overloaded_clinics = df[df['overloaded']]

    # fig_14 → Por City
    city_counts = overloaded_clinics['City'].value_counts().head(10).reset_index()
    city_counts.columns = ['City', 'Cantidad']

    fig_14 = px.bar(
        city_counts,
        x='Cantidad',
        y='City',
        orientation='h',
        title=f'Clínicas Sobrecargadas por Ciudad{" - " + estado_filtro if estado_filtro and estado_filtro != "Todos" else ""}',
        color='Cantidad',
        color_continuous_scale='Reds'
    )
    fig_14.update_layout(yaxis=dict(autorange="reversed"))

    # fig_15 → Por Specialty
    spec_counts = overloaded_clinics['Specialty of the Medical Equipment'].value_counts().head(10).reset_index()
    spec_counts.columns = ['Especialidad', 'Cantidad']

    fig_15 = px.bar(
        spec_counts,
        x='Cantidad',
        y='Especialidad',
        orientation='h',
        title=f'Clínicas Sobrecargadas por Especialidad{" - " + estado_filtro if estado_filtro and estado_filtro != "Todos" else ""}',
        color='Cantidad',
        color_continuous_scale='Reds'
    )
    fig_15.update_layout(yaxis=dict(autorange="reversed"))

    # === FIGURA 16: Distribución de NSE - Actualizado ===
    nse_counts = clinicas_filtradas['nse_nivel_v2'].value_counts().reset_index()
    nse_counts.columns = ['nse_nivel_v2', 'Cantidad']

    fig_16 = px.pie(
        nse_counts,
        names='nse_nivel_v2',
        values='Cantidad',
        color_discrete_sequence=px.colors.sequential.Viridis,
        title=f'Distribución de NSE de las Clínicas{" - " + estado_filtro if estado_filtro and estado_filtro != "Todos" else ""}'
    )
    fig_16.update_traces(textinfo='percent+label', pull=[0.05]*len(nse_counts))
    fig_16.update_layout(title_font_size=16)

    # === FIGURA 17: Distribución de especialidades (agrupadas <5%) ===
    # Paso 1: contar cuántas clínicas hay por especialidad
    counts = merged_filtrado['Specialty of the Medical Equipment'].value_counts().reset_index()
    counts.columns = ['Specialty of the Medical Equipment', 'Count']

    # Paso 2: calcular porcentaje
    counts['Percentage'] = 100 * counts['Count'] / counts['Count'].sum()

    # Paso 3: agrupar las especialidades con <5% en "Others"
    others = counts[counts['Percentage'] < 5]['Count'].sum()
    main = counts[counts['Percentage'] >= 5].copy()

    # agregar la categoría Others
    if others > 0:
        main.loc[len(main)] = ['Others', others, 100 * others / counts['Count'].sum()]

    # Paso 4: graficar
    fig_17 = px.pie(
        main,
        names='Specialty of the Medical Equipment',
        values='Count',
        title=f'Distribución de especialidades de clínicas {" - " + estado_filtro if estado_filtro and estado_filtro != "Todos" else ""}',
        color='Specialty of the Medical Equipment',
        hover_data=['Percentage']
    )

    fig_17.update_traces(
        textposition='inside',
        textinfo='percent+label',
        hoverinfo='label+percent+name'
    )

    # === FIGURA 18: Mapa de clínicas por especialidad ===
    fig_18 = px.scatter_mapbox(
        clinicas_filtradas,
        lat='latitud',
        lon='longitud',
        hover_name='Contract',
        hover_data={
            'Specialty of the Medical Equipment': True,
            'Clinic Size': True,
            'Number of Consultations per Month': True
        },
        color='Specialty of the Medical Equipment',
        color_discrete_sequence=px.colors.sequential.Viridis,
        mapbox_style='open-street-map',
        zoom=4,
        height=600,
        title=f'Mapa de Clínicas Backbone por Especialidad{" - " + estado_filtro if estado_filtro and estado_filtro != "Todos" else ""}'
    )

    fig_18.update_layout(
        title_font_size=18,
        margin=dict(l=20, r=20, t=60, b=20),
        mapbox=dict(center=dict(lat=23.6345, lon=-102.5528)),
    )

    fig_18.update_traces(
        marker=dict(size=12),
        selector=dict(mode='markers')
    )

    # === FIGURA 19: Promedio de consultas por especialidad ===
    # Paso 1: calcular promedio de consultas por especialidad
    promedios = (
        clinicas_filtradas
        .groupby('Specialty of the Medical Equipment', as_index=False)
        ['Number of Consultations per Month']
        .mean()
        .sort_values('Number of Consultations per Month', ascending=False)
    )

    # Paso 2: graficar los promedios
    fig_19 = px.bar(
        promedios,
        x='Specialty of the Medical Equipment',
        y='Number of Consultations per Month',
        title=f'Promedio de Consultas por Especialidad{" - " + estado_filtro if estado_filtro and estado_filtro != "Todos" else ""}',
        hover_data={
            'Number of Consultations per Month': ':.2f',
        },
        color='Specialty of the Medical Equipment',
    )

    # Paso 3: personalizar diseño
    fig_19.update_layout(
        xaxis_title='Especialidad',
        yaxis_title='Promedio de Consultas por Mes',
        xaxis_tickangle=-45,
        showlegend=False,
        title_font_size=16,
        xaxis_title_font_size=12,
        yaxis_title_font_size=12
    )

    # === FIGURA 20: Total de consultas por NSE ===
    # Agrupar por NSE y sumar las consultas
    nse_consultas = (
        merged_filtrado
        .groupby('nse_nivel_v2', as_index=False)
        ['Number of Consultations per Month']
        .sum()
        .sort_values('Number of Consultations per Month', ascending=False)
    )

    fig_20 = px.bar(
        nse_consultas,
        x='nse_nivel_v2',
        y='Number of Consultations per Month',
        title=f'Total de Consultas por NSE{" - " + estado_filtro if estado_filtro and estado_filtro != "Todos" else ""}',
        hover_data={
            'Number of Consultations per Month': ':.2f',
        },
        color='nse_nivel_v2',
    )

    fig_20.update_layout(
        xaxis_title='NSE',
        yaxis_title='Total de Consultas por Mes',
        xaxis_tickangle=-45,
        showlegend=False,
        title_font_size=16,
        xaxis_title_font_size=12,
        yaxis_title_font_size=12
    )

    clinicas_filtradas['precio_promedio'] = pd.to_numeric(
        clinicas_filtradas['precio_promedio'], errors='coerce'
    )
    merged_filtrado['precio_promedio'] = pd.to_numeric(
        merged_filtrado['precio_promedio'], errors='coerce'
    )

    fig_22 = px.scatter(
        clinicas_filtradas.dropna(subset=['precio_promedio', 'Number of Consultations per Month']),
        x='precio_promedio',
        y='Number of Consultations per Month',
        color='Clinic Size',
        size='pct_low_income_clean',
        size_max=35,
        hover_name='Contract',
        hover_data={
            'precio_promedio': ':.0f',
            'Number of Consultations per Month': True,
            '% of Patients with Middle-Low to Low Income': ':.1f',
            'nse_nivel_v2': True
        },
        color_discrete_sequence=px.colors.sequential.Viridis,
        title=(
            f'Precio Promedio de la Zona vs Consultas por Mes'
            f'{"  —  " + estado_filtro if estado_filtro and estado_filtro != "Todos" else ""}'
            '<br><sup>Tamaño del punto = % Pacientes con ingresos bajos</sup>'
        )
    )
    fig_22.update_layout(
        xaxis_title='Precio Promedio de la Zona (MXN)',
        yaxis_title='Consultas por Mes',
        legend_title_text='Tamaño de Clínica',
        title_font_size=16,
        xaxis_title_font_size=12,
        yaxis_title_font_size=12
    )

    # === FIGURA 24: BARRAS — PRECIO PROMEDIO MEDIANO POR TAMAÑO DE CLÍNICA ===
    # Responde si las clínicas más grandes tienden a operar en zonas con mayor
    # poder adquisitivo. Se usa la mediana para mitigar el efecto de outliers.

    precio_por_tamano = (
        clinicas_filtradas
        .dropna(subset=['precio_promedio'])
        .groupby('Clinic Size', as_index=False)['precio_promedio']
        .median()
        .sort_values('precio_promedio', ascending=False)
    )

    fig_24 = px.bar(
        precio_por_tamano,
        x='Clinic Size',
        y='precio_promedio',
        color='Clinic Size',
        color_discrete_sequence=px.colors.sequential.Viridis,
        text_auto=',.0f',
        title=(
            f'Precio Promedio Mediano de la Zona por Tamaño de Clínica'
            f'{"  —  " + estado_filtro if estado_filtro and estado_filtro != "Todos" else ""}'
        )
    )
    fig_24.update_layout(
        xaxis_title='Tamaño de Clínica',
        yaxis_title='Precio Promedio Mediano (MXN)',
        title_font_size=16,
        xaxis_title_font_size=12,
        yaxis_title_font_size=12,
        showlegend=False
    )
    fig_24.update_traces(textposition='outside')

    # === FIGURA 27: BOX PLOT — CONSULTAS POR MES POR NSE Y TAMAÑO DE CLÍNICA ===
    # Combina el nivel socioeconómico de la zona con el tamaño de la clínica para
    # ver si el volumen de consultas varía significativamente entre segmentos.
    # Las categorías nse_nivel_v2 provienen de la agrupación AMAI definida al inicio.

    orden_amai = ['E', 'D', 'D+', 'C-', 'C', 'C+', 'AB']

    df_27 = merged_filtrado.dropna(subset=['nse_nivel_v2', 'Number of Consultations per Month']).copy()
    df_27['nse_nivel_v2'] = pd.Categorical(
        df_27['nse_nivel_v2'],
        categories=[n for n in orden_amai if n in df_27['nse_nivel_v2'].unique()],
        ordered=True
    )

    fig_27 = px.box(
        df_27,
        x='nse_nivel_v2',
        y='Number of Consultations per Month',
        color='Clinic Size',
        color_discrete_sequence=px.colors.sequential.Viridis,
        title=(
            f'Consultas por Mes según NSE y Tamaño de Clínica'
            f'{"  —  " + estado_filtro if estado_filtro and estado_filtro != "Todos" else ""}'
        ),
        category_orders={'nse_nivel_v2': orden_amai}
    )
    fig_27.update_layout(
        xaxis_title='Nivel NSE (AMAI)',
        yaxis_title='Consultas por Mes',
        legend_title_text='Tamaño de Clínica',
        title_font_size=16,
        xaxis_title_font_size=12,
        yaxis_title_font_size=12,
        yaxis_showgrid=True,
        yaxis_gridcolor='lightgray',
        yaxis_gridwidth=0.5
    )


       # === DASHBOARDS REORGANIZADOS ===
    
    # Dashboard 1: fig_13, fig_16, fig_17
    # ── Estilos compartidos ──────────────────────────────────
    SECTION_TITLE = {
        'fontFamily': '"Playfair Display", Georgia, serif',
        'fontWeight': '700',
        'fontSize': '22px',
        'color': '#4A5E3A',
        'textAlign': 'left',
        'margin': '0 0 16px 4px',
        'letterSpacing': '-0.3px',
        'borderLeft': '4px solid #F2C12E',
        'paddingLeft': '12px'
    }
    CARD = {
        'background': '#FFFFFF',
        'borderRadius': '10px',
        'padding': '6px',
        'boxShadow': '0 2px 8px rgba(74,94,58,0.10)'
    }
    GRID_WRAP = {
        'display': 'grid',
        'gap': '12px',
        'padding': '0 4px 4px 4px'
    }

    def card(fig, height='420px'):
        return html.Div(
            dcc.Graph(figure=fig, style={'height': height}, config={'responsive': True}),
            style=CARD
        )

    dashboard1 = html.Div([
        html.H2(
            f"Clínicas Backbone{' — ' + estado_filtro if estado_filtro and estado_filtro != 'Todos' else ''}",
            style=SECTION_TITLE
        ),
        html.Div([
            html.Div([card(fig_16, '380px'), card(fig_17, '380px')],
                style={**GRID_WRAP, 'gridTemplateColumns': '1fr 1fr'}),
            card(fig_13, '480px'),
        ], style=GRID_WRAP)
    ])

    # Dashboard 2: fig_3, fig_18, fig_19, fig_20
    dashboard2 = html.Div([
        html.H2(
            f"Población beneficiada{' — ' + estado_filtro if estado_filtro and estado_filtro != 'Todos' else ''}",
            style=SECTION_TITLE
        ),
        html.Div([
            html.Div([card(fig_20, '380px'), card(fig_19, '380px')],
                style={**GRID_WRAP, 'gridTemplateColumns': '1fr 1fr'}),
            card(fig_3, '480px'),
            card(fig_18, '420px'),
        ], style=GRID_WRAP)
    ])

    # Dashboard 3: fig_1, fig_8, fig_9
    dashboard3 = html.Div([
        html.H2(
            f"Perfil general de las clínicas{' — ' + estado_filtro if estado_filtro and estado_filtro != 'Todos' else ''}",
            style=SECTION_TITLE
        ),
        html.Div([
            html.Div([card(fig_1, '400px'), card(fig_8, '400px')],
                style={**GRID_WRAP, 'gridTemplateColumns': '1fr 1fr'}),
            card(fig_9, '420px'),
        ], style=GRID_WRAP)
    ])

    # Dashboard 4: fig_11, fig_12, fig_14, fig_15
    dashboard4 = html.Div([
        html.H2(
            f"Perfil operatorio de las clínicas{' — ' + estado_filtro if estado_filtro and estado_filtro != 'Todos' else ''}",
            style=SECTION_TITLE
        ),
        html.Div([
            html.Div([card(fig_11, '400px'), card(fig_12, '400px')],
                style={**GRID_WRAP, 'gridTemplateColumns': '1fr 1fr'}),
            html.Div([card(fig_14, '400px'), card(fig_15, '400px')],
                style={**GRID_WRAP, 'gridTemplateColumns': '1fr 1fr'}),
        ], style=GRID_WRAP)
    ])

    # Dashboard 5: fig_5, fig_7, fig_6
    dashboard5 = html.Div([
        html.H2(
            f"Perfil de los pacientes{' — ' + estado_filtro if estado_filtro and estado_filtro != 'Todos' else ''}",
            style=SECTION_TITLE
        ),
        html.Div([
            html.Div([card(fig_5, '400px'), card(fig_7, '400px')],
                style={**GRID_WRAP, 'gridTemplateColumns': '1fr 1fr'}),
            card(fig_6, '480px'),
        ], style=GRID_WRAP)
    ])

    dashboard6 = html.Div([
        html.H2(
            f"Análisis del precio promedio{' — ' + estado_filtro if estado_filtro and estado_filtro != 'Todos' else ''}",
            style=SECTION_TITLE
        ),
        html.Div([
            html.Div([card(fig_24, '400px'), card(fig_27, '400px')],
                style={**GRID_WRAP, 'gridTemplateColumns': '1fr 1fr'}),
            card(fig_22, '480px'),
        ], style=GRID_WRAP)
    ])


    return {
        'dashboard1': dashboard1,
        'dashboard2': dashboard2,
        'dashboard3': dashboard3,
        'dashboard4': dashboard4,
        'dashboard5': dashboard5,
        'dashboard6': dashboard6
    }
# ==========================
# Layout principal
# ==========================
# Obtener opciones únicas para los dropdowns
estados = ["Todos"] + sorted(merged['State'].dropna().unique().tolist())
nse_options = ["Todos"] + sorted(merged['nse_nivel_v2'].dropna().unique().tolist())
especialidades = ["Todos"] + sorted(merged['Specialty of the Medical Equipment'].dropna().unique().tolist())

# ── Google Fonts inyectado vía index_string ──────────────────
app.index_string = """<!DOCTYPE html>
<html>
<head>
    {%metas%}
    <title>Backbone · Dashboard</title>
    {%favicon%}
    {%css%}
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=DM+Sans:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            background: #F5F2EB;
            font-family: 'DM Sans', sans-serif;
            color: #1C1C1C;
        }
        /* ── Tabs ── */
        .custom-tabs .tab {
            font-family: 'DM Sans', sans-serif !important;
            font-weight: 500 !important;
            font-size: 13px !important;
            color: #4A5E3A !important;
            background: #EDE8DC !important;
            border: none !important;
            border-bottom: 3px solid transparent !important;
            padding: 10px 18px !important;
            letter-spacing: 0.2px;
            transition: all 0.2s;
        }
        .custom-tabs .tab--selected {
            color: #8B3A1A !important;
            background: #FAF7F0 !important;
            border-bottom: 3px solid #F2C12E !important;
            font-weight: 600 !important;
        }
        .custom-tabs .tab:hover:not(.tab--selected) {
            background: #E3DDD1 !important;
            color: #8B3A1A !important;
        }
        /* ── Dropdowns ── */
        .Select-control { border-color: #C8BFA8 !important; border-radius: 6px !important; }
        .Select-control:hover { border-color: #F2C12E !important; }
        .is-focused .Select-control { border-color: #F2C12E !important; box-shadow: 0 0 0 2px rgba(242,193,46,0.2) !important; }
        /* ── Scrollbar sutil ── */
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: #F5F2EB; }
        ::-webkit-scrollbar-thumb { background: #C8BFA8; border-radius: 3px; }
    </style>
</head>
<body>
    {%app_entry%}
    <footer>{%config%}{%scripts%}{%renderer%}</footer>
</body>
</html>"""

# === Layout principal ===
app.layout = html.Div([

    # ── HEADER ──────────────────────────────────────────────
    html.Div([
        # Franja de acento superior
        html.Div(style={
            'height': '4px',
            'background': 'linear-gradient(90deg, #F2C12E 0%, #8B8B2E 40%, #4A5E3A 100%)',
        }),
        # Contenido del header
        html.Div([
            # Logo + título
            html.Div([
                html.Div(style={
                    'width': '36px', 'height': '36px',
                    'background': '#4A5E3A',
                    'borderRadius': '6px',
                    'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center',
                    'marginRight': '12px', 'flexShrink': '0'
                }),
                html.Div([
                    html.H1("Backbone · Clínicas", style={
                        'fontFamily': '"Playfair Display", Georgia, serif',
                        'fontWeight': '700',
                        'fontSize': '20px',
                        'color': '#1C1C1C',
                        'lineHeight': '1.1'
                    }),
                    html.P("Dashboard de análisis integral 2026", style={
                        'fontSize': '12px',
                        'color': '#7A7060',
                        'marginTop': '2px',
                        'fontWeight': '400'
                    })
                ])
            ], style={'display': 'flex', 'alignItems': 'center', 'flex': '1'}),

            # Filtros
            html.Div([
                # Estado
                html.Div([
                    html.Label("Estado", style={
                        'fontSize': '11px', 'fontWeight': '600',
                        'color': '#7A7060', 'letterSpacing': '0.5px',
                        'textTransform': 'uppercase', 'marginBottom': '3px',
                        'display': 'block'
                    }),
                    dcc.Dropdown(
                        id='estado-filter',
                        options=[{'label': e, 'value': e} for e in estados],
                        value='Todos',
                        clearable=False,
                        style={'width': '180px', 'fontSize': '13px'}
                    )
                ], style={'marginRight': '16px'}),
                # NSE
                html.Div([
                    html.Label("NSE", style={
                        'fontSize': '11px', 'fontWeight': '600',
                        'color': '#7A7060', 'letterSpacing': '0.5px',
                        'textTransform': 'uppercase', 'marginBottom': '3px',
                        'display': 'block'
                    }),
                    dcc.Dropdown(
                        id='nse-filter',
                        options=[{'label': n, 'value': n} for n in nse_options],
                        value='Todos',
                        clearable=False,
                        style={'width': '130px', 'fontSize': '13px'}
                    )
                ], style={'marginRight': '16px'}),
                # Especialidad
                html.Div([
                    html.Label("Especialidad", style={
                        'fontSize': '11px', 'fontWeight': '600',
                        'color': '#7A7060', 'letterSpacing': '0.5px',
                        'textTransform': 'uppercase', 'marginBottom': '3px',
                        'display': 'block'
                    }),
                    dcc.Dropdown(
                        id='especialidad-filter',
                        options=[{'label': esp, 'value': esp} for esp in especialidades],
                        value='Todos',
                        clearable=False,
                        style={'width': '200px', 'fontSize': '13px'}
                    )
                ]),
            ], style={'display': 'flex', 'alignItems': 'flex-end'})
        ], style={
            'display': 'flex',
            'justifyContent': 'space-between',
            'alignItems': 'center',
            'padding': '14px 24px 14px 24px',
        })
    ], style={
        'background': '#FAF7F0',
        'borderBottom': '1px solid #DDD8CC',
        'boxShadow': '0 2px 8px rgba(0,0,0,0.06)',
        'position': 'sticky', 'top': '0', 'zIndex': '1000'
    }),

    # ── TABS ────────────────────────────────────────────────
    html.Div([
        dcc.Tabs(id='tabs', value='dashboard1',
            className='custom-tabs',
            children=[
                dcc.Tab(label='Clínicas Backbone',         value='dashboard1', className='tab'),
                dcc.Tab(label='Población beneficiada',     value='dashboard2', className='tab'),
                dcc.Tab(label='Perfil general',            value='dashboard3', className='tab'),
                dcc.Tab(label='Perfil operatorio',         value='dashboard4', className='tab'),
                dcc.Tab(label='Perfil de pacientes',       value='dashboard5', className='tab'),
                dcc.Tab(label='Precio promedio',           value='dashboard6', className='tab'),
                dcc.Tab(label='Backbone APP',              value='dashboard7', className='tab'),
            ]
        )
    ], style={
        'background': '#EDE8DC',
        'borderBottom': '1px solid #DDD8CC',
        'padding': '0 24px'
    }),

    # ── CONTENIDO ───────────────────────────────────────────
    html.Div(
        id='dashboard-content',
        style={'padding': '20px 24px', 'minHeight': 'calc(100vh - 120px)'}
    )

], style={'background': '#F5F2EB', 'minHeight': '100vh'})

# === Callback para actualizar los dashboards cuando cambian los filtros o la pestaña ===
@app.callback(
    Output('dashboard-content', 'children'),
    [Input('tabs', 'value'),
     Input('estado-filter', 'value'),
     Input('nse-filter', 'value'),
     Input('especialidad-filter', 'value')]
)
def update_dashboard(tab_value, estado_filtro, nse_filtro, especialidad_filtro):
    if tab_value == 'dashboard7':
        # Retorna la app de AGEB
        return create_ageb_layout()
    else:
        # Generar dashboards con los filtros aplicados
        dashboards = create_dashboards(estado_filtro, nse_filtro, especialidad_filtro)
        return dashboards.get(tab_value, dashboards['dashboard1'])

# Registra los callbacks de AGEB DESPUÉS de definir la app
register_ageb_callbacks(app)

# ==========================
# Ejecutar app
# ==========================
if __name__ == '__main__':
    app.run(debug=True)