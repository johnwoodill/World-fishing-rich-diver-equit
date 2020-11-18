import pandas as pd
import geopandas as gpd
from geopandas import GeoDataFrame
from shapely.geometry import Point, Polygon

# WCPFC Statistical Area from 2019 Document
WCPFC_lon = [210, 210, 230, 230, 150, 150, 140, 140, 130, 130, 100, 100, 210]
WCPFC_lat = [65, 0, 0, -65, -65, -55, -55, -30, -30, -10, -10, 70, 65]


# IOTC Statistical Area from Website
# https://www.iotc.org/about-iotc/competence

IOTC_lon = [20, 20, 100, 100, 130, 130, 150, 150, 80, 80, 20]
IOTC_lat = [-45, 30, 30, -10, -10, -30, -30, -55, -55, -45, -45]

# IATTC Statistical Area
IATTC_lon = [210, 210, 290, 300, 255, 255, 210]
IATTC_lat = [50, -50, -50, 0, 21, 50, 50]

# ICCAT Statistical Area
ICCAT_e_lon = [295, 295, 295, 255, 255, 300, 290, 290, 361, 361, 361]
ICCAT_e_lat = [70, 50, 50, 55, 21, 0, -50, -55, -55, 30, 70]

ICCAT_w_lon = [20, 20, 0, 0, 20]
ICCAT_w_lat = [-45, 30, 30, -45, -45]



WCPFC_geom = Polygon(zip(WCPFC_lon, WCPFC_lat))
IOTC_geom = Polygon(zip(IOTC_lon, IOTC_lat))
IATTC_geom = Polygon(zip(IATTC_lon, IATTC_lat))
ICCAT_e_geom = Polygon(zip(ICCAT_e_lon, ICCAT_e_lat))
ICCAT_w_geom = Polygon(zip(ICCAT_w_lon, ICCAT_w_lat))

WCPFC_df = GeoDataFrame(index=["WCPFC"], geometry=[WCPFC_geom])  
IOTC_df = GeoDataFrame(index=["IOTC"], geometry=[IOTC_geom])  
IATTC_df = GeoDataFrame(index=["IATTC"], geometry=[IATTC_geom])  
ICCAT_e_df = GeoDataFrame(index=["ICCAT_East"], geometry=[ICCAT_e_geom])  
ICCAT_w_df = GeoDataFrame(index=["ICCAT_West"], geometry=[ICCAT_w_geom])  


gdf = WCPFC_df.append(IOTC_df)
gdf = gdf.append(IATTC_df)
gdf = gdf.append(ICCAT_e_df)
gdf = gdf.append(ICCAT_w_df)

gdf = gdf.reset_index().rename(columns={'index': 'RFMO'})

gdf.set_crs(epsg=4326, inplace=True)     

gdf.to_file(filename='data/RFMO_shapefile/RFMO_coords.shp', driver="ESRI Shapefile")
