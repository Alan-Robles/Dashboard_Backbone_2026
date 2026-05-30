import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import numpy as np

#Crear Buffer con valores de lat, lon y km
def crear_buffer(lat, lon, distancia_km):
    """
    Método más directo usando un CRS apropiado
    """
    # Crear punto
    punto = Point(lon, lat)
    
    # Crear GeoDataFrame
    gdf = gpd.GeoDataFrame(
        {'nombre': ['area_buffer']},
        geometry=[punto],
        crs="EPSG:4326"
    )
    
    # Usar CRS que preserve distancias (UTM)
    gdf_utm = gdf.to_crs("EPSG:6362")  # UTM
    
    # Crear buffer en metros
    gdf_buffer = gdf_utm.buffer(distancia_km * 1000)
    
    # Crear GeoDataFrame con el buffer
    resultado = gpd.GeoDataFrame(
        {
            'lat_central': [lat],
            'lon_central': [lon],
            'distancia_km': [distancia_km],
            'area_km2': [gdf_buffer.area.iloc[0] / 1000000]
        },
        geometry=gdf_buffer,
        crs=gdf_utm.crs
    ).to_crs("EPSG:4326")  # Volver a WGS84
    
    return resultado

def buffers_gdf(df,km):
    """
    Devuelve un gdf con una lista de buffers para cada lon y lat 
    """
    buffers = pd.DataFrame()
    for i in range(len(df)):
        buffer = crear_buffer(df.latitud.iloc[i],df.longitud.iloc[i],km)
        buffers = pd.concat([buffers,buffer])
    buffers = buffers.reset_index(drop = True) 
    buffers["Clinic Aid Code"] = df.copy().reset_index(drop = True).iloc[:,0]
    return buffers

def buffers_coords(df):
    """
    Devuelve dos listas con los valores de cordenadas de cada polígono con divisiónes "None" para graficar en conjunto
    """
    lat = []
    lon = []
    for i in range(len(df)):
        x, y = df.geometry[i].exterior.xy
        lon_poly = list(x)
        lat_poly = list(y)
        lon.extend(lon_poly)
        lat.extend(lat_poly)
        lon.append(None)
        lat.append(None)
    return lat , lon

def NSE_Conjunto_AGEBS(gdf):
    gdf = gdf.copy()
    gdf["NSE_Ponderado"] = (gdf["POBTOT"]/gdf["POBTOT"].sum())*gdf["NSE_score"]
    resultado = gdf["NSE_Ponderado"].sum()
    return resultado

def AGEBS_Ponderados(buffers,gdf_AGEBS):
    """
    Devuelve un df con AGEBS dentro de los buffers y el df de buffers con valores de NSE
    """
    buffers["NSE"] = pd.Series(dtype='float64')
    gdf = gdf_AGEBS.copy()
    AGEBS_within = pd.DataFrame()
    for i in range(len(buffers)):
        poligonos_dentro = gdf[gdf.within(buffers.geometry.iloc[i])]
        NSE = NSE_Conjunto_AGEBS(poligonos_dentro)
        buffers.loc[i,"NSE"] = NSE
        AGEBS_within = pd.concat([AGEBS_within,poligonos_dentro["CODE"]])
    AGEBS_within = AGEBS_within.drop_duplicates()
    return buffers , AGEBS_within