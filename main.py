# main.py
from dash import html, dcc, Input, Output, callback
import plotly.graph_objects as go
import pandas as pd
import geopandas as gpd
import json
from utils import crear_buffer, buffers_gdf, buffers_coords, NSE_Conjunto_AGEBS, AGEBS_Ponderados

url_ageb = "https://raw.githubusercontent.com/Alan-Robles/Dashboard_Backbone_2026/main/data/AGEBS_NSE_NL.geojson"
url_clinicas = "https://raw.githubusercontent.com/Alan-Robles/Dashboard_Backbone_2026/main/data/clinicas_con_final.csv"

# Cargar datos una sola vez (fuera de las funciones)
gdf = gpd.read_file(url_ageb)
df_full = pd.read_csv(url_clinicas)
df_nl = df_full[df_full["State"]=="Nuevo León"]

def create_ageb_layout():
    """Retorna el layout de la aplicación AGEB"""
    return html.Div(
        className='main', 
        children=[
            html.H1(children='Análisis de Clínicas por AGEB', style={'textAlign': 'center'}),
            html.Div(
                className="Recuadro",
                children=[
                    html.H2("Mapa AGEBS y Clínicas"),
                    dcc.Input(
                        id="Input-Km",
                        placeholder='Distancia en Km...',
                        type='number',
                        value=5  # Valor por defecto
                    ),
                    dcc.Checklist(
                        id='layer-control',
                        options=[
                            {'label': 'Mostrar Clínicas', 'value': 'clinicas'},
                            {'label': 'Mostrar AGEBS', 'value': 'AGEBS'},
                            {'label': 'Mostrar Áreas de Influencia', 'value': 'buffers'}
                        ],
                        value=['clinicas', 'AGEBS'],  # capas activadas al inicio
                        inline=True
                    )
                ],
                style={'margin': '20px', 'padding': '15px', 'border': '1px solid #ddd', 'borderRadius': '5px'}
            ),
            dcc.Store(id="Buffers"),
            dcc.Graph(id='mapa', figure=go.Figure(), style={'height': '600px'})
        ]
    )

def register_ageb_callbacks(app):
    """Registra los callbacks de la aplicación AGEB"""
    
    @app.callback(
        Output("Buffers", "data"),
        Input("Input-Km", "value")
    )
    def update_buffers(km):
        if km is None or km <= 0:
            return None
        
        buffers_creados = buffers_gdf(df_nl, km)
        
        # Convert GeoDataFrame to GeoJSON for storage
        return json.loads(buffers_creados.to_json())

    @app.callback(
        Output('mapa', 'figure'),
        Input('layer-control', 'value'),
        Input('Buffers', 'data')
    )
    def update_layers(selected_layers, buffers_data):
        
        if buffers_data is None:
            # Figura vacía inicial
            fig = go.Figure()
            fig.update_layout(
                height=600,
                margin=dict(r=0, t=0, l=0, b=0),
                map_style="open-street-map",
                showlegend=False
            )
            return fig
        
        # Convert back to GeoDataFrame
        buffers = gpd.GeoDataFrame.from_features(buffers_data)
        
        # Buffers con los NSE
        buffers, AGEBS_within = AGEBS_Ponderados(buffers, gdf)
        
        # Obtener los puntos de cada área de cada buffer para graficarlos
        lat_poly, lon_poly = buffers_coords(buffers) 
        
        # gdf con AGEBS FILTRADOS
        gdf_filtrado = gdf[gdf["CODE"].isin(AGEBS_within["CODE"])]
        
        fig = go.Figure()

        # Capa de AGEBS
        if 'AGEBS' in selected_layers:
            AGEBS_layer = go.Choroplethmap(
                geojson=gdf_filtrado.__geo_interface__,
                locations=gdf_filtrado.index,
                z=gdf_filtrado['NSE_score'],
                text=gdf_filtrado['NSE'],
                hoverinfo="location+z+text",
                colorscale="Viridis",
                colorbar=dict(title="NSE"),
                marker_opacity=0.6,
                marker_line_width=0.5,
                name="AGEBS",
                showlegend=True
            )
            fig.add_trace(AGEBS_layer)

        # Capa de buffers
        if 'buffers' in selected_layers:
            buffers_layer = go.Scattermap(
                lon=lon_poly,
                lat=lat_poly,
                mode='lines',
                fill='toself',
                fillcolor='rgba(0, 100, 255, 0.2)',
                line=dict(color='blue', width=2),
                name="Áreas de Influencia",
                showlegend=True
            )
            fig.add_trace(buffers_layer)
        
        # Capa de clínicas
        if 'clinicas' in selected_layers:
            clinicas_layer = go.Scattermap(
                lat=buffers['lat_central'],
                lon=buffers['lon_central'],
                mode='markers',
                marker=dict(size=12, color='red'),
                name='Clínicas',
                customdata=buffers[['NSE','Clinic Aid Code']],
                hovertemplate=(
                    "<b>AID CODE:</b> %{customdata[1]}<br>" +
                    "<b>NSE Promedio:</b> %{customdata[0]:.2f}<br>" +
                    "Lat: %{lat}<br>" +
                    "Lon: %{lon}<extra></extra>"
                ),
                showlegend=True
            )
            fig.add_trace(clinicas_layer)
        
        # Configuración del mapa
        fig.update_layout(
            height=600,
            margin=dict(r=0, t=0, l=0, b=0),
            map_style="open-street-map",
            map_zoom=10,
            map_center=dict(lat=buffers.lat_central.iloc[0], lon=buffers.lon_central.iloc[0]),
            legend=dict(
                y=0,
                x=1,
                xanchor='right',
                yanchor='bottom'
            )
        )
        
        return fig
