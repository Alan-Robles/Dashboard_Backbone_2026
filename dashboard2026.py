# dashboard2026.py
import re
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

import dash
from dash import dcc, html, Input, Output

from flask_caching import Cache

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import create_ageb_layout, register_ageb_callbacks

# ==========================
# URLs de datos
# ==========================
url_clinicas = "https://raw.githubusercontent.com/Alan-Robles/Dashboard_Backbone_2026/main/data/clinicas_con_final.csv"
url_nse = "https://raw.githubusercontent.com/Alan-Robles/Dashboard_Backbone_2026/main/data/df_nse.csv"

# ==========================
# Inicialización de Dash
# ==========================
app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server

cache = Cache(app.server, config={
    'CACHE_TYPE': 'SimpleCache',
    'CACHE_DEFAULT_TIMEOUT': 600
})

# ==========================
# Variables globales de datos
# ==========================
_data_loaded = False
merged = None
backbone = None
estados = ["Todos"]
nse_options = ["Todos"]
especialidades = ["Todos"]


def load_data():
    """Carga y preprocesa los datos una sola vez."""
    global _data_loaded, merged, backbone, estados, nse_options, especialidades
    if _data_loaded:
        return

    backbone_raw = pd.read_csv(url_clinicas)
    nse = pd.read_csv(url_nse)

    col_inicio = 'POBTOT'
    col_fin = 'NSE'
    backbone_raw = backbone_raw.drop(columns=backbone_raw.columns[
        backbone_raw.columns.get_loc(col_inicio): backbone_raw.columns.get_loc(col_fin) + 1
    ])

    backbone_raw = backbone_raw.rename(columns={
        'Specialty of the \nMedical Equipment': 'Specialty of the Medical Equipment',
        'Number of \nConsultations per Month': 'Number of Consultations per Month'
    })

    merged_raw = backbone_raw.merge(nse, on='CVEGEO', how='left')
    merged_raw = merged_raw.rename(columns={
        'Specialty of the \nMedical Equipment': 'Specialty of the Medical Equipment',
        'Number of \nConsultations per Month': 'Number of Consultations per Month'
    })

    # Limpiar coordenadas
    def convertir_coordenada(coordenada):
        if pd.isna(coordenada):
            return None
        if isinstance(coordenada, str):
            coordenada = coordenada.replace(',', '.')
            nums = re.findall(r'-?\d+\.?\d*', coordenada)
            if nums:
                return float(nums[0])
            return None
        try:
            return float(coordenada)
        except (ValueError, TypeError):
            return None

    merged_raw['latitud'] = merged_raw['latitud'].apply(convertir_coordenada)
    merged_raw['longitud'] = merged_raw['longitud'].apply(convertir_coordenada)

    # Limpiar columna de edad
    if merged_raw['Average Age of the SME (years)'].dtype == object:
        merged_raw['Average Age of the SME (years)'] = (
            merged_raw['Average Age of the SME (years)']
            .str.replace(',', '.', regex=False)
            .astype(float, errors='ignore')
        )

    # Limpiar % bajo ingreso
    def clean_percentage(value):
        if pd.isna(value):
            return None
        if isinstance(value, str):
            if '%' in value and len(value.split('%')) > 2:
                return float(value.split('%')[0])
            return float(value.replace('%', ''))
        return float(value)

    merged_raw['pct_low_income_clean'] = merged_raw[
        '% of Patients with Middle-Low to Low Income'
    ].apply(clean_percentage)

    merged_raw['precio_promedio'] = pd.to_numeric(merged_raw['precio_promedio'], errors='coerce')

    # Columnas demográficas
    cols_demo = ['pob_60ymas', 'pob_total', 'pob_15a64', 'pob_0a14', 'pob_15a49_f', 'pob_fem']
    for col in cols_demo:
        if col in merged_raw.columns:
            merged_raw[col] = pd.to_numeric(
                merged_raw[col].astype(str).str.replace(',', '', regex=False).str.strip(),
                errors='coerce'
            )

    merged_raw['pob_total'] = merged_raw['pob_total'].replace(0, np.nan)
    merged_raw['pob_fem'] = merged_raw['pob_fem'].replace(0, np.nan)

    if 'pob_60ymas' in merged_raw.columns:
        merged_raw['pct_adultos_mayores'] = (
            merged_raw['pob_60ymas'] / merged_raw['pob_total']
        ).replace([np.inf, -np.inf], np.nan) * 100

    merged_raw['consultas_por_staff'] = (
        merged_raw['Number of Consultations per Month'] / merged_raw['Total Staff']
    )

    merged = merged_raw
    backbone = backbone_raw

    estados = ["Todos"] + sorted(merged['State'].dropna().unique().tolist())
    nse_options = ["Todos"] + sorted(merged['nse_nivel_v2'].dropna().unique().tolist())
    especialidades = ["Todos"] + sorted(
        merged['Specialty of the Medical Equipment'].dropna().unique().tolist()
    )

    _data_loaded = True


# Los datos se cargan en el post_fork hook de gunicorn.conf.py,
# DESPUÉS de que el puerto ya está ligado. Aquí solo garantizamos
# que si se ejecuta localmente (sin gunicorn) también funcione.
import os
if os.environ.get('WERKZEUG_RUN_MAIN') == 'true' or __name__ == '__main__':
    load_data()


# ==========================
# Crear dashboards
# ==========================
@cache.memoize()
def create_dashboards(estado_filtro=None, nse_filtro=None, especialidad_filtro=None):
    load_data()  # garantía adicional

    orden_amai = ['E', 'D', 'D+', 'C-', 'C', 'C+', 'AB']

    if estado_filtro and estado_filtro != "Todos":
        clinicas_filtradas = merged[merged['State'] == estado_filtro].copy()
        merged_filtrado = merged[merged['State'] == estado_filtro].copy()
    else:
        clinicas_filtradas = merged.copy()
        merged_filtrado = merged.copy()

    if nse_filtro and nse_filtro != "Todos":
        merged_filtrado = merged_filtrado[merged_filtrado['nse_nivel_v2'] == nse_filtro].copy()
        clinicas_filtradas = clinicas_filtradas[clinicas_filtradas['nse_nivel_v2'] == nse_filtro].copy()

    if especialidad_filtro and especialidad_filtro != "Todos":
        clinicas_filtradas = clinicas_filtradas[
            clinicas_filtradas['Specialty of the Medical Equipment'] == especialidad_filtro
        ].copy()
        merged_filtrado = merged_filtrado[
            merged_filtrado['Specialty of the Medical Equipment'] == especialidad_filtro
        ].copy()

    sfx = f" — {estado_filtro}" if estado_filtro and estado_filtro != "Todos" else ""

    # --- Helpers de limpieza local ---
    def clean_percentage(value):
        if pd.isna(value):
            return None
        if isinstance(value, str):
            if '%' in value and len(value.split('%')) > 2:
                return float(value.split('%')[0])
            return float(value.replace('%', ''))
        return float(value)

    clinicas_filtradas = clinicas_filtradas.copy()
    clinicas_filtradas['pct_low_income_clean'] = clinicas_filtradas[
        '% of Patients with Middle-Low to Low Income'
    ].apply(clean_percentage)

    # === FIGURA 1: Distribución de tamaños ===
    clinic_counts = clinicas_filtradas['Clinic Size'].value_counts().reset_index()
    clinic_counts.columns = ['Clinic Size', 'Count']
    fig_1 = px.pie(
        clinic_counts, names='Clinic Size', values='Count',
        title=f'Distribución de tamaños de clínicas{sfx}',
        color='Clinic Size',
        color_discrete_map={'Small': 'red', 'Medium': 'blue', 'Micro': 'orange'},
        hover_data=['Count']
    )
    fig_1.update_traces(textposition='inside', textinfo='percent+label', hoverinfo='label+percent')

    # === FIGURA 2: Top 20 municipios por consultas ===
    top20 = (
        clinicas_filtradas
        .groupby(['State', 'Municipality', 'Clinic Size'])['Number of Consultations per Month']
        .mean().reset_index()
        .sort_values('Number of Consultations per Month', ascending=False).head(20)
    )
    top20['estado_mun'] = top20['State'] + ', ' + top20['Municipality']
    fig_2 = px.bar(
        top20, x='estado_mun', y='Number of Consultations per Month', color='Clinic Size',
        color_discrete_sequence=px.colors.sequential.Viridis,
        title=f'Top 20 Promedio de Consultas por Municipio{sfx}',
        hover_data={'Number of Consultations per Month': ':.2f', 'estado_mun': True, 'Clinic Size': True}
    )
    fig_2.update_layout(xaxis_tickangle=-45, xaxis_title='Municipio',
                        yaxis_title='Promedio Consultas/Mes', title_font_size=16)

    # === FIGURA 3: Mapa por tamaño ===
    fig_3 = px.scatter_mapbox(
        clinicas_filtradas, lat='latitud', lon='longitud', hover_name='Contract',
        hover_data={'Clinic Size': True, 'Number of Consultations per Month': ':.2f'},
        color='Clinic Size', mapbox_style='open-street-map', zoom=4, height=600,
        title=f'Mapa de Clínicas por Tamaño{sfx}'
    )
    fig_3.update_traces(marker=dict(size=18))
    fig_3.update_layout(title_font_size=18, margin=dict(l=20, r=20, t=60, b=20),
                        mapbox=dict(center=dict(lat=23.6345, lon=-102.5528)))

    # === FIGURA 5: % pacientes bajo ingreso por tamaño ===
    grouped = clinicas_filtradas.groupby('Clinic Size')['pct_low_income_clean'].mean().reset_index()
    grouped.columns = ['Clinic Size', '% Bajo Ingreso']
    fig_5 = px.bar(
        grouped, x='Clinic Size', y='% Bajo Ingreso', color='Clinic Size',
        color_discrete_sequence=['seagreen'],
        title=f'% Promedio Pacientes Bajo Ingreso por Tipo de Clínica{sfx}'
    )
    fig_5.update_layout(xaxis_title='Tamaño', yaxis_title='%', title_font_size=16, showlegend=False)

    # === FIGURA 6: Mapa % bajo ingreso ===
    fig_6 = px.scatter_mapbox(
        clinicas_filtradas, lat='latitud', lon='longitud', hover_name='Contract',
        color='Clinic Size', size='pct_low_income_clean',
        mapbox_style='open-street-map', zoom=4, height=600,
        title=f'Mapa Clínicas por % Pacientes Bajos Ingresos{sfx}'
    )
    fig_6.update_layout(title_font_size=18, margin=dict(l=20, r=20, t=60, b=20),
                        mapbox=dict(center=dict(lat=23.6345, lon=-102.5528)))

    # === FIGURA 7: Seguro médico ===
    cc = clinicas_filtradas['Attends Patients with Health Insurance'].value_counts().reset_index()
    cc.columns = ['Atiende Seguro Médico', 'Cantidad']
    fig_7 = px.pie(cc, names='Atiende Seguro Médico', values='Cantidad',
                   color_discrete_sequence=px.colors.sequential.Viridis,
                   title=f'Clínicas que Atienden Pacientes con Seguro Médico{sfx}')
    fig_7.update_traces(textinfo='percent+label', pull=[0.05, 0])
    fig_7.update_layout(title_font_size=16)

    # === FIGURA 8: Histograma antigüedad ===
    fig_8 = px.histogram(
        clinicas_filtradas, x='Average Age of the SME (years)', nbins=20,
        title=f'Antigüedad de las Clínicas{sfx}'
    )
    fig_8.update_traces(marker_color='indianred')
    fig_8.update_layout(xaxis_title='Años', yaxis_title='Frecuencia', title_font_size=16)

    # === FIGURA 9: Box antigüedad por especialidad ===
    fig_9 = px.box(
        clinicas_filtradas, x='Specialty of the Medical Equipment',
        y='Average Age of the SME (years)', color='Specialty of the Medical Equipment',
        title=f'Especialidades con Clínicas más Antiguas{sfx}'
    )
    fig_9.update_layout(xaxis_tickangle=-45, xaxis_title='Especialidad',
                        yaxis_title='Años de operación', title_font_size=16, showlegend=False)

    # === FIGURA 11: Personal vs consultas ===
    fig_11 = px.scatter(
        clinicas_filtradas, x='Total Staff', y='Number of Consultations per Month',
        color='Clinic Size', size='Number of Consultations per Month', size_max=40,
        title=f'Personal Médico vs Consultas Mensuales{sfx}'
    )
    fig_11.update_layout(xaxis_title='Personal Total', yaxis_title='Consultas/Mes', title_font_size=16)

    # === FIGURA 12: Consultas por personal por especialidad ===
    clinicas_filtradas['consultas_por_staff'] = (
        clinicas_filtradas['Number of Consultations per Month'] / clinicas_filtradas['Total Staff']
    )
    fig_12 = px.box(
        clinicas_filtradas, x='Specialty of the Medical Equipment', y='consultas_por_staff',
        color='Specialty of the Medical Equipment',
        title=f'Consultas por Personal por Especialidad{sfx}'
    )
    fig_12.update_layout(xaxis_tickangle=-45, xaxis_title='Especialidad',
                         yaxis_title='Consultas/Personal', title_font_size=16,
                         yaxis_showgrid=True, yaxis_gridcolor='lightgray', showlegend=False)

    # === FIGURA 13: Mapa NSE ===
    fig_13 = px.scatter_mapbox(
        clinicas_filtradas, lat='latitud', lon='longitud', hover_name='Contract',
        color='nse_nivel_v2',
        color_discrete_sequence=["#3B528B", "#440154", "#FDE725", "#21918C"],
        mapbox_style='open-street-map', zoom=4, height=600,
        title=f'Mapa de Clínicas por NSE{sfx}'
    )
    fig_13.update_traces(marker=dict(size=18))
    fig_13.update_layout(title_font_size=18, margin=dict(l=20, r=20, t=60, b=20),
                         mapbox=dict(center=dict(lat=23.6345, lon=-102.5528)))

    # === FIGURAS 14-15: Clínicas sobrecargadas ===
    df_op = clinicas_filtradas[['Contract', 'Clinic Size', 'Number of Consultations per Month',
                                 'Total Staff', 'City', 'Specialty of the Medical Equipment']].copy()
    df_op['operational_ratio'] = df_op['Number of Consultations per Month'] / df_op['Total Staff']
    limits = {'Micro': 100, 'Small': 150, 'Medium': 200}
    df_op['overloaded'] = df_op.apply(
        lambda x: x['operational_ratio'] > limits.get(x['Clinic Size'], np.inf), axis=1)
    overloaded = df_op[df_op['overloaded']]

    city_counts = overloaded['City'].value_counts().head(10).reset_index()
    city_counts.columns = ['City', 'Cantidad']
    fig_14 = px.bar(city_counts, x='Cantidad', y='City', orientation='h',
                    title=f'Clínicas Sobrecargadas por Ciudad{sfx}',
                    color='Cantidad', color_continuous_scale='Reds')
    fig_14.update_layout(yaxis=dict(autorange="reversed"))

    spec_counts = overloaded['Specialty of the Medical Equipment'].value_counts().head(10).reset_index()
    spec_counts.columns = ['Especialidad', 'Cantidad']
    fig_15 = px.bar(spec_counts, x='Cantidad', y='Especialidad', orientation='h',
                    title=f'Clínicas Sobrecargadas por Especialidad{sfx}',
                    color='Cantidad', color_continuous_scale='Reds')
    fig_15.update_layout(yaxis=dict(autorange="reversed"))

    # === FIGURA 16: Distribución NSE ===
    nse_counts = clinicas_filtradas['nse_nivel_v2'].value_counts().reset_index()
    nse_counts.columns = ['nse_nivel_v2', 'Cantidad']
    fig_16 = px.pie(nse_counts, names='nse_nivel_v2', values='Cantidad',
                    color_discrete_sequence=px.colors.sequential.Viridis,
                    title=f'Distribución de NSE{sfx}')
    fig_16.update_traces(textinfo='percent+label', pull=[0.05] * len(nse_counts))
    fig_16.update_layout(title_font_size=16)

    # === FIGURA 17: Especialidades (agrupando <5% en Others) ===
    counts = clinicas_filtradas['Specialty of the Medical Equipment'].value_counts().reset_index()
    counts.columns = ['Specialty of the Medical Equipment', 'Count']
    counts['Percentage'] = 100 * counts['Count'] / counts['Count'].sum()
    others = counts[counts['Percentage'] < 5]['Count'].sum()
    main = counts[counts['Percentage'] >= 5].copy()
    if others > 0:
        main.loc[len(main)] = ['Others', others, 100 * others / counts['Count'].sum()]
    fig_17 = px.pie(main, names='Specialty of the Medical Equipment', values='Count',
                    title=f'Distribución de Especialidades{sfx}',
                    color='Specialty of the Medical Equipment', hover_data=['Percentage'])
    fig_17.update_traces(textposition='inside', textinfo='percent+label')

    # === FIGURA 18: Mapa por especialidad ===
    fig_18 = px.scatter_mapbox(
        clinicas_filtradas, lat='latitud', lon='longitud', hover_name='Contract',
        hover_data={'Specialty of the Medical Equipment': True, 'Clinic Size': True,
                    'Number of Consultations per Month': True},
        color='Specialty of the Medical Equipment',
        color_discrete_sequence=px.colors.sequential.Viridis,
        mapbox_style='open-street-map', zoom=4, height=600,
        title=f'Mapa de Clínicas por Especialidad{sfx}'
    )
    fig_18.update_layout(title_font_size=18, margin=dict(l=20, r=20, t=60, b=20),
                         mapbox=dict(center=dict(lat=23.6345, lon=-102.5528)))
    fig_18.update_traces(marker=dict(size=12), selector=dict(mode='markers'))

    # === FIGURA 19: Promedio consultas por especialidad ===
    promedios = (
        clinicas_filtradas
        .groupby('Specialty of the Medical Equipment', as_index=False)
        ['Number of Consultations per Month'].mean()
        .sort_values('Number of Consultations per Month', ascending=False)
    )
    fig_19 = px.bar(promedios, x='Specialty of the Medical Equipment',
                    y='Number of Consultations per Month',
                    title=f'Promedio de Consultas por Especialidad{sfx}',
                    color='Specialty of the Medical Equipment',
                    hover_data={'Number of Consultations per Month': ':.2f'})
    fig_19.update_layout(xaxis_tickangle=-45, showlegend=False, title_font_size=16)

    # === FIGURA 20: Total consultas por NSE ===
    nse_consultas = (
        merged_filtrado.groupby('nse_nivel_v2', as_index=False)
        ['Number of Consultations per Month'].sum()
    )
    nse_consultas['nse_nivel_v2'] = pd.Categorical(
        nse_consultas['nse_nivel_v2'], categories=orden_amai, ordered=True)
    nse_consultas = nse_consultas.sort_values('nse_nivel_v2')
    fig_20 = px.bar(nse_consultas, x='nse_nivel_v2', y='Number of Consultations per Month',
                    color='Number of Consultations per Month', color_continuous_scale='Viridis',
                    title=f'Total de Consultas por NSE{sfx}', text_auto='.2s')
    fig_20.update_layout(xaxis_title='NSE', yaxis_title='Total Consultas/Mes',
                         title_font_size=16, showlegend=False)

    # === FIGURA 22: Precio promedio vs consultas ===
    clinicas_filtradas['precio_promedio'] = pd.to_numeric(
        clinicas_filtradas['precio_promedio'], errors='coerce')
    fig_22 = px.scatter(
        clinicas_filtradas.dropna(subset=['precio_promedio', 'Number of Consultations per Month']),
        x='precio_promedio', y='Number of Consultations per Month', color='Clinic Size',
        size='pct_low_income_clean', size_max=35, hover_name='Contract',
        hover_data={'precio_promedio': ':.0f', 'Number of Consultations per Month': True,
                    'pct_low_income_clean': ':.1f', 'nse_nivel_v2': True},
        color_discrete_sequence=px.colors.sequential.Viridis,
        title=f'Precio Promedio de Zona vs Consultas/Mes{sfx}<br>'
              '<sup>Tamaño = % Pacientes bajos ingresos</sup>'
    )
    fig_22.update_layout(xaxis_title='Precio Promedio (MXN)', yaxis_title='Consultas/Mes',
                         title_font_size=16)

    # === FIGURA 24: Precio mediano por tamaño ===
    precio_por_tamano = (
        clinicas_filtradas.dropna(subset=['precio_promedio'])
        .groupby('Clinic Size', as_index=False)['precio_promedio']
        .median().sort_values('precio_promedio', ascending=False)
    )
    fig_24 = px.bar(precio_por_tamano, x='Clinic Size', y='precio_promedio',
                    color='Clinic Size', color_discrete_sequence=px.colors.sequential.Viridis,
                    text_auto=',.0f', title=f'Precio Mediano de Zona por Tamaño de Clínica{sfx}')
    fig_24.update_layout(xaxis_title='Tamaño', yaxis_title='Precio Mediano (MXN)',
                         title_font_size=16, showlegend=False)
    fig_24.update_traces(textposition='outside')

    # === FIGURA 27: Box consultas por NSE y tamaño ===
    df_27 = merged_filtrado.dropna(subset=['nse_nivel_v2', 'Number of Consultations per Month']).copy()
    df_27['nse_nivel_v2'] = pd.Categorical(
        df_27['nse_nivel_v2'],
        categories=[n for n in orden_amai if n in df_27['nse_nivel_v2'].unique()],
        ordered=True
    )
    fig_27 = px.box(df_27, x='nse_nivel_v2', y='Number of Consultations per Month',
                    color='Clinic Size', color_discrete_sequence=px.colors.sequential.Viridis,
                    title=f'Consultas por NSE y Tamaño de Clínica{sfx}',
                    category_orders={'nse_nivel_v2': orden_amai})
    fig_27.update_layout(xaxis_title='NSE (AMAI)', yaxis_title='Consultas/Mes',
                         title_font_size=16, yaxis_showgrid=True, yaxis_gridcolor='lightgray')

    # ── Estilos compartidos ──────────────────────────────────
    SECTION_TITLE = {
        'fontFamily': '"Playfair Display", Georgia, serif',
        'fontWeight': '700', 'fontSize': '22px', 'color': '#4A5E3A',
        'textAlign': 'left', 'margin': '0 0 16px 4px', 'letterSpacing': '-0.3px',
        'borderLeft': '4px solid #F2C12E', 'paddingLeft': '12px'
    }
    CARD = {
        'background': '#FFFFFF', 'borderRadius': '10px',
        'padding': '6px', 'boxShadow': '0 2px 8px rgba(74,94,58,0.10)'
    }
    GRID_WRAP = {'display': 'grid', 'gap': '12px', 'padding': '0 4px 4px 4px'}

    def card(fig, height='420px'):
        return html.Div(
            dcc.Graph(figure=fig, style={'height': height}, config={'responsive': True}),
            style=CARD
        )

    dashboard1 = html.Div([
        html.H2(f"Clínicas Backbone{sfx}", style=SECTION_TITLE),
        html.Div([
            html.Div([card(fig_16, '380px'), card(fig_17, '380px')],
                     style={**GRID_WRAP, 'gridTemplateColumns': '1fr 1fr'}),
            card(fig_13, '480px'),
        ], style=GRID_WRAP)
    ])

    dashboard2 = html.Div([
        html.H2(f"Población beneficiada{sfx}", style=SECTION_TITLE),
        html.Div([
            html.Div([card(fig_20, '380px'), card(fig_19, '380px')],
                     style={**GRID_WRAP, 'gridTemplateColumns': '1fr 1fr'}),
            card(fig_3, '480px'),
            card(fig_18, '420px'),
        ], style=GRID_WRAP)
    ])

    dashboard3 = html.Div([
        html.H2(f"Perfil general de las clínicas{sfx}", style=SECTION_TITLE),
        html.Div([
            html.Div([card(fig_1, '400px'), card(fig_8, '400px')],
                     style={**GRID_WRAP, 'gridTemplateColumns': '1fr 1fr'}),
            card(fig_9, '420px'),
        ], style=GRID_WRAP)
    ])

    dashboard4 = html.Div([
        html.H2(f"Perfil operatorio de las clínicas{sfx}", style=SECTION_TITLE),
        html.Div([
            html.Div([card(fig_11, '400px'), card(fig_12, '400px')],
                     style={**GRID_WRAP, 'gridTemplateColumns': '1fr 1fr'}),
            html.Div([card(fig_14, '400px'), card(fig_15, '400px')],
                     style={**GRID_WRAP, 'gridTemplateColumns': '1fr 1fr'}),
        ], style=GRID_WRAP)
    ])

    dashboard5 = html.Div([
        html.H2(f"Perfil de los pacientes{sfx}", style=SECTION_TITLE),
        html.Div([
            html.Div([card(fig_5, '400px'), card(fig_7, '400px')],
                     style={**GRID_WRAP, 'gridTemplateColumns': '1fr 1fr'}),
            card(fig_6, '480px'),
        ], style=GRID_WRAP)
    ])

    dashboard6 = html.Div([
        html.H2(f"Análisis del precio promedio{sfx}", style=SECTION_TITLE),
        html.Div([
            html.Div([card(fig_24, '400px'), card(fig_27, '400px')],
                     style={**GRID_WRAP, 'gridTemplateColumns': '1fr 1fr'}),
            card(fig_22, '480px'),
        ], style=GRID_WRAP)
    ])

    return {
        'dashboard1': dashboard1, 'dashboard2': dashboard2,
        'dashboard3': dashboard3, 'dashboard4': dashboard4,
        'dashboard5': dashboard5, 'dashboard6': dashboard6
    }


# ==========================
# Layout principal
# ==========================
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
        body { background: #F5F2EB; font-family: 'DM Sans', sans-serif; color: #1C1C1C; }
        .custom-tabs .tab {
            font-family: 'DM Sans', sans-serif !important; font-weight: 500 !important;
            font-size: 13px !important; color: #4A5E3A !important;
            background: #EDE8DC !important; border: none !important;
            border-bottom: 3px solid transparent !important;
            padding: 10px 18px !important; transition: all 0.2s;
        }
        .custom-tabs .tab--selected {
            color: #8B3A1A !important; background: #FAF7F0 !important;
            border-bottom: 3px solid #F2C12E !important; font-weight: 600 !important;
        }
        .custom-tabs .tab:hover:not(.tab--selected) { background: #E3DDD1 !important; color: #8B3A1A !important; }
        .Select-control { border-color: #C8BFA8 !important; border-radius: 6px !important; }
        .Select-control:hover { border-color: #F2C12E !important; }
        .is-focused .Select-control { border-color: #F2C12E !important; box-shadow: 0 0 0 2px rgba(242,193,46,0.2) !important; }
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

def serve_layout():
    """Genera el layout en cada carga de página, cuando los datos ya están listos."""
    load_data()  # no-op si ya se cargó en post_fork
    return html.Div([
    html.Div([
        html.Div(style={'height': '4px', 'background': 'linear-gradient(90deg, #F2C12E 0%, #8B8B2E 40%, #4A5E3A 100%)'}),
        html.Div([
            html.Div([
                html.Div(style={'width': '36px', 'height': '36px', 'background': '#4A5E3A',
                                'borderRadius': '6px', 'marginRight': '12px', 'flexShrink': '0'}),
                html.Div([
                    html.H1("Backbone · Clínicas", style={
                        'fontFamily': '"Playfair Display", Georgia, serif',
                        'fontWeight': '700', 'fontSize': '20px', 'color': '#1C1C1C', 'lineHeight': '1.1'
                    }),
                    html.P("Dashboard de análisis integral 2026", style={
                        'fontSize': '12px', 'color': '#7A7060', 'marginTop': '2px', 'fontWeight': '400'
                    })
                ])
            ], style={'display': 'flex', 'alignItems': 'center', 'flex': '1'}),
            html.Div([
                html.Div([
                    html.Label("Estado", style={'fontSize': '11px', 'fontWeight': '600',
                               'color': '#7A7060', 'textTransform': 'uppercase',
                               'marginBottom': '3px', 'display': 'block'}),
                    dcc.Dropdown(id='estado-filter',
                                 options=[{'label': e, 'value': e} for e in estados],
                                 value='Todos', clearable=False,
                                 style={'width': '180px', 'fontSize': '13px'})
                ], style={'marginRight': '16px'}),
                html.Div([
                    html.Label("NSE", style={'fontSize': '11px', 'fontWeight': '600',
                               'color': '#7A7060', 'textTransform': 'uppercase',
                               'marginBottom': '3px', 'display': 'block'}),
                    dcc.Dropdown(id='nse-filter',
                                 options=[{'label': n, 'value': n} for n in nse_options],
                                 value='Todos', clearable=False,
                                 style={'width': '130px', 'fontSize': '13px'})
                ], style={'marginRight': '16px'}),
                html.Div([
                    html.Label("Especialidad", style={'fontSize': '11px', 'fontWeight': '600',
                               'color': '#7A7060', 'textTransform': 'uppercase',
                               'marginBottom': '3px', 'display': 'block'}),
                    dcc.Dropdown(id='especialidad-filter',
                                 options=[{'label': esp, 'value': esp} for esp in especialidades],
                                 value='Todos', clearable=False,
                                 style={'width': '200px', 'fontSize': '13px'})
                ]),
            ], style={'display': 'flex', 'alignItems': 'flex-end'})
        ], style={'display': 'flex', 'justifyContent': 'space-between',
                  'alignItems': 'center', 'padding': '14px 24px'})
    ], style={'background': '#FAF7F0', 'borderBottom': '1px solid #DDD8CC',
              'boxShadow': '0 2px 8px rgba(0,0,0,0.06)', 'position': 'sticky', 'top': '0', 'zIndex': '1000'}),

    html.Div([
        dcc.Tabs(id='tabs', value='dashboard1', className='custom-tabs', children=[
            dcc.Tab(label='Clínicas Backbone',     value='dashboard1', className='tab'),
            dcc.Tab(label='Población beneficiada', value='dashboard2', className='tab'),
            dcc.Tab(label='Perfil general',         value='dashboard3', className='tab'),
            dcc.Tab(label='Perfil operatorio',      value='dashboard4', className='tab'),
            dcc.Tab(label='Perfil de pacientes',    value='dashboard5', className='tab'),
            dcc.Tab(label='Precio promedio',        value='dashboard6', className='tab'),
            dcc.Tab(label='Backbone APP',           value='dashboard7', className='tab'),
        ])
    ], style={'background': '#EDE8DC', 'borderBottom': '1px solid #DDD8CC', 'padding': '0 24px'}),

    html.Div(id='dashboard-content',
             style={'padding': '20px 24px', 'minHeight': 'calc(100vh - 120px)'})

    ], style={'background': '#F5F2EB', 'minHeight': '100vh'})


app.layout = serve_layout


# ==========================
# Callbacks
# ==========================

@app.callback(
    Output('dashboard-content', 'children'),
    Input('tabs', 'value'),
    Input('estado-filter', 'value'),
    Input('nse-filter', 'value'),
    Input('especialidad-filter', 'value')
)
def update_dashboard(tab_value, estado_filtro, nse_filtro, especialidad_filtro):
    if tab_value == 'dashboard7':
        return create_ageb_layout()
    dashboards = create_dashboards(estado_filtro, nse_filtro, especialidad_filtro)
    return dashboards.get(tab_value, dashboards['dashboard1'])


register_ageb_callbacks(app)

if __name__ == '__main__':
    app.run(debug=True)
