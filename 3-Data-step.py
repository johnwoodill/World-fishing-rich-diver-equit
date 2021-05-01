import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, Polygon, shape, LinearRing
from shapely.ops import nearest_points
from geopy.distance import geodesic
import shapefile
from shapely.geometry import shape, Point
import cartopy.io.shapereader as shpreader
import math
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
ProgressBar().register()

def proc_pivot(x):
    outdat = x.pivot_table(index=['lat_lon', 'lat', 'lon'], columns=['var_name', 'period'], values=['mean', 'max', 'min', 'var', 'skew', 'kurt'])
    outdat.columns = outdat.columns.map('_'.join)
    outdat = outdat.reset_index()
    return outdat


def dist(lat1, lon1, lat2, lon2):
    return np.sqrt( (lat2 - lat1)**2 + (lon2 - lon1)**2)



def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(np.sqrt(a))
    r = 6371  # Radius of earth in kilometers. Use 3956 for miles
    return c * r



def get_coastlines():
    coastlines = gpd.read_file("data/coastline_shp/ne_10m_coastline.shp")
    bdat = [pd.DataFrame({'coast': i, 'lon': coastlines.geometry[i].xy[0], 'lat': coastlines.geometry[i].xy[1]}) for i in range(len(coastlines))]
    return pd.concat(bdat)


        
def get_ports():
    ports = []
    ne_ports = shpreader.natural_earth(resolution='10m',
                                    category='cultural',
                                    name='ports')
    reader = shpreader.Reader(ne_ports)                             
    ports = reader.records()
    port = next(ports)
    port_df = pd.DataFrame()
    for port in ports:
        geom_port = port.geometry
        geom_coords = geom_port.coords[:]
        lon = geom_coords[0][0]
        lat = geom_coords[0][1]
        port_name = port.attributes['name']
        odat = pd.DataFrame({'port': port_name, 'lon': [lon], 'lat': [lat]})
        port_df = pd.concat([port_df, odat])
    return port_df



def port_dist(ndat):
    lon = ndat['lon'].iat[0]
    lat = ndat['lat'].iat[0]
    # print("Port Distance", lon, lat)
    indat = ports
    indat = indat.assign(lon = np.where(indat['lon'] < 0, indat['lon'] + 361, indat['lon']))
    # if lon < -130:
    #     lon = lon + 360
    #     indat = indat.assign(lon = np.where(indat['lon'] < 0, indat['lon'] + 360, indat['lon']))
    indat.loc[:, 'distance'] = indat.apply(lambda row: haversine(lon, lat, row['lon'], row['lat']), axis=1)
    indat = indat.sort_values('distance')
    outdat = pd.DataFrame({'port': [indat['port'].iat[0]], 'port_distance': [indat['distance'].iat[0]]})
    return outdat
    # return (indat['port'].iat[0], indat['distance'].iat[0])




def coast_dist(ndat):
    lon = ndat['lon'].iat[0]
    lat = ndat['lat'].iat[0]
    # print("Coast Dist: ", lon, lat)
    # lat = ndat['lat'].iat[0]
    # lon = ndat['lon'].iat[0]
    indat = coasts
    indat = indat.assign(lon = np.where(indat['lon'] < 0, indat['lon'] + 361, indat['lon']))       
    # if lon < -130:
    #     lon = lon + 360
    #    indat = indat.assign(lon = np.where(indat['lon'] < 0, indat['lon'] + 360, indat['lon']))        
    indat = indat[(indat['lon'] >= (lon - 40)) & (indat['lon'] <= (lon + 40))]
    indat = indat[(indat['lat'] >= (lat - 40)) & (indat['lat'] <= (lat + 40))]
    distance = indat.apply(lambda row: haversine(lon, lat, row['lon'], row['lat']), axis=1)
    indat = indat.assign(distance = distance)
    indat = indat.sort_values('distance')
    distance = indat['distance'].iat[0]
    outdat = pd.DataFrame({'lat': [lat], 'lon': [lon], 'coast_distance': [distance]})
    return outdat



# Assign EEZ 
def eez_check(ndat):
    lon = ndat['lon'].iat[0]
    lat = ndat['lat'].iat[0]
    # print("EEZ:", lon, lat)
    for territory in unique_eez:
        arg = eez_shp[eez_shp.Territory1 == territory].reset_index(drop=True)
        pnts = gpd.GeoDataFrame(geometry=[Point(lon, lat)])
        polys = gpd.GeoSeries({'territory': arg.geometry.values[0]})
        check = pnts.assign(**{key: pnts.within(geom) for key, geom in polys.items()})
        check_ = check.territory.values[0]
        if check_ == True:
            outdat = pd.DataFrame({'lat': [lat], 'lon': [lon], 'eez': [territory]})
            return outdat
    outdat = pd.DataFrame({'lat': [lat], 'lon': [lon], 'eez': [0]})
    return outdat
        
        
        

# Assign MPA indicator for GFW
def mpa_check(ndat):
    lon = ndat['lon'].iat[0]
    lat = ndat['lat'].iat[0]
    # print("MPA:", lon, lat)
    # lon = np.where(lon < 180, lon + 360, lon)   # Shape file 0-360 lon
    for mpa_loc in unique_mpa:
        arg = mpa_shp[mpa_shp.NAME == mpa_loc].reset_index(drop=True)
        pnts = gpd.GeoDataFrame(geometry=[Point(lon, lat)])
        polys = gpd.GeoSeries({'mpa_loc': arg.geometry.values[0]})
        check = pnts.assign(**{key: pnts.within(geom) for key, geom in polys.items()})
        check_ = check.mpa_loc.values[0]
        if check_ == True:
            outdat = pd.DataFrame({'lat': [lat], 'lon': [lon], 'mpa': [mpa_loc]})
            return outdat
    outdat = pd.DataFrame({'lat': [lat], 'lon': [lon], 'mpa': [0]})
    return outdat




# Assign RFMO indicator for GFW
def rfmo_check(ndat):
    lon = ndat['lon'].iat[0]
    lat = ndat['lat'].iat[0]
    # print("MPA:", lon, lat)
    # lon = np.where(lon < 180, lon + 360, lon)   # Shape file 0-360 lon
    for rfmo_loc in unique_rfmo:
        arg = rfmo_shp[rfmo_shp.RFMO == rfmo_loc].reset_index(drop=True)
        pnts = gpd.GeoDataFrame(geometry=[Point(lon, lat)])
        polys = gpd.GeoSeries({'rfmo_loc': arg.geometry.values[0]})
        check = pnts.assign(**{key: pnts.within(geom) for key, geom in polys.items()})
        check_ = check.rfmo_loc.values[0]
        if check_ == True:
            if rfmo_loc == "ICCAT_East" or rfmo_loc == "ICCAT_West":
                rfmo_loc = "ICCAT"
            outdat = pd.DataFrame({'lat': [lat], 'lon': [lon], 'rfmo': [rfmo_loc]})
            return outdat
    outdat = pd.DataFrame({'lat': [lat], 'lon': [lon], 'rfmo': [0]})
    return outdat







# ------------------------------------

# ### Get ports and coastline data
# ports = get_ports()
# coasts = get_coastlines()

# eez_shp = gpd.read_file("data/EEZ/eez_v10.shp")
# unique_eez = eez_shp.Territory1.unique()

# mpa_shp = gpd.read_file("data/mpa_shapefiles/vlmpa.shp")
# unique_mpa = mpa_shp.NAME.unique()

# rfmo_shp = gpd.read_file("data/RFMO_shapefile/RFMO_coords.shp")
# unique_rfmo = rfmo_shp.RFMO.unique()

# # ### Get CMIP Coords with Dask
# cmip_data = pd.read_hdf('data/full_CMIP6_historical.hdf', key='historical')
# cmip_coords = cmip_data.groupby('lat_lon').agg({'lat': 'mean', 'lon': 'mean'}).sort_values(['lon', 'lat']).reset_index()

# # cmip_coords = cmip_coords.head()
# cmip_coords = dd.from_pandas(cmip_coords, npartitions = 50)
# # cmip_coords = cmip_coords.sample(10).reset_index(drop=True)


# # ### Distance to closests port
# port_dat = cmip_coords.groupby('lat_lon').apply(lambda x: port_dist(x)).compute(scheduler='processes')
# port_dat = port_dat.reset_index().drop(columns='level_1')
# port_dat.to_csv('data/port_dat.csv', index = False)




# # ### Distance to coast
# coast_dist_dat = cmip_coords.groupby('lat_lon').apply(lambda x: coast_dist(x)).compute(scheduler='processes')
# coast_dist_dat = coast_dist_dat.reset_index().drop(columns='level_1')
# coast_dist_dat.to_csv('data/coast_dist_dat.csv', index = False)



# # ### EEZ Check
# eez_check_dat = cmip_coords.groupby('lat_lon').apply(lambda x: eez_check(x)).compute(scheduler='processes')
# eez_check_dat = eez_check_dat.reset_index().drop(columns='level_1')
# eez_check_dat.to_csv('data/eez_check_dat.csv', index = False)




# # ### MPA Check
# mpa_check_dat = cmip_coords.groupby('lat_lon').apply(lambda x: mpa_check(x)).compute(scheduler='processes')
# mpa_check_dat = mpa_check_dat.reset_index().drop(columns='level_1')
# mpa_check_dat.to_csv('data/mpa_check_dat.csv', index = False)


# # ### RFMO Check
# rfmo_check_dat = cmip_coords.groupby('lat_lon').apply(lambda x: rfmo_check(x)).compute(scheduler='processes')
# rfmo_check_dat = rfmo_check_dat.reset_index().drop(columns='level_1')
# rfmo_check_dat.to_csv('data/rfmo_check_dat.csv', index = False)


# -----------------------------------------------------------------
print("Binding compiled data")
# Grid all 1d data in the world


# ### Calc whether country has ever fished at grid
feffort_nat_dat = pd.read_csv('data/total_fishing_effort_nation.csv')

unique_flags = feffort_nat_dat['flag_gfw'].unique()

ll_dat = feffort_nat_dat.groupby('lat_lon').agg({'lat': 'mean', 'lon': 'mean'}).sort_values(['lon', 'lat']).reset_index()
for flag_ in unique_flags:
    indat = feffort_nat_dat[['lat_lon', 'lat', 'lon', 'flag_gfw']]
    indat = indat[indat['flag_gfw'] == flag_]
    indat = indat.assign(flag_gfw=1)
    indat = indat.rename(columns={'flag_gfw': f"{flag_}_present"})
    ll_dat = ll_dat.merge(indat, how='left', on=['lat_lon', 'lat', 'lon']).fillna(0)

 
# ### Calc whether vessel_class_gfw has ever fished at grid
feffort_gear_dat = pd.read_csv('data/total_fishing_effort_gear.csv')
unique_vessel_class_gfw = feffort_gear_dat['vessel_class_gfw'].unique()

gg_dat = feffort_gear_dat.groupby('lat_lon').agg({'lat': 'mean', 'lon': 'mean'}).sort_values(['lon', 'lat']).reset_index()
for gear_ in unique_vessel_class_gfw:
    indat = feffort_gear_dat[['lat_lon', 'lat', 'lon', 'vessel_class_gfw']]
    indat = indat[indat['vessel_class_gfw'] == gear_]
    indat = indat.assign(vessel_class_gfw=1)
    indat = indat.rename(columns={'vessel_class_gfw': f"{gear_}_gear"})
    gg_dat = gg_dat.merge(indat, how='left', on=['lat_lon', 'lat', 'lon']).fillna(0)
    


# ### Bind data
feffort_dat = pd.read_csv('data/total_fishing_effort.csv')
shan_divi = pd.read_csv('data/shannon_div_equ.csv')
richness_dat = pd.read_csv('data/total_species_richness.csv')

port_dat = pd.read_csv('data/port_dat.csv')
coast_dist_dat = pd.read_csv('data/coast_dist_dat.csv')
eez_check_dat = pd.read_csv('data/eez_check_dat.csv')
mpa_check_dat = pd.read_csv('data/mpa_check_dat.csv')
rfmo_check_dat = pd.read_csv('data/rfmo_check_dat.csv')


full_dat = feffort_dat.merge(shan_divi, how='left', on=['lat_lon', 'lon', 'lat'])
full_dat = full_dat.merge(richness_dat, how='left', on=['lat_lon', 'lon', 'lat'])
full_dat = full_dat.merge(ll_dat, how='left', on=['lat_lon', 'lon', 'lat'])
full_dat = full_dat.merge(gg_dat, how='left', on=['lat_lon', 'lon', 'lat'])

full_dat = full_dat.assign(lat_lon=full_dat['lat'].astype(str) + "_" + full_dat['lon'].astype(str))


print("Binding CMIP6 data")
# Bind CMIP6 data (add 361 to shift data)
hist_dat = pd.read_hdf('data/full_CMIP6_historical.hdf', key='historical')

ssp126_2015_2030_dat = pd.read_hdf('data/full_CMIP6_ssp126_2015_2030.hdf', key='ssp126_2015_2030')
ssp126_2030_2045_dat = pd.read_hdf('data/full_CMIP6_ssp126_2030_2045.hdf', key='ssp126_2030_2045')
ssp126_2045_2060_dat = pd.read_hdf('data/full_CMIP6_ssp126_2045_2060.hdf', key='ssp126_2045_2060')
ssp126_2060_2075_dat = pd.read_hdf('data/full_CMIP6_ssp126_2060_2075.hdf', key='ssp126_2060_2075')
ssp126_2075_2090_dat = pd.read_hdf('data/full_CMIP6_ssp126_2075_2090.hdf', key='ssp126_2075_2090')

ssp585_2015_2030_dat = pd.read_hdf('data/full_CMIP6_ssp585_2015_2030.hdf', key='ssp585_2015_2030')
ssp585_2030_2045_dat = pd.read_hdf('data/full_CMIP6_ssp585_2030_2045.hdf', key='ssp585_2030_2045')
ssp585_2045_2060_dat = pd.read_hdf('data/full_CMIP6_ssp585_2045_2060.hdf', key='ssp585_2045_2060')
ssp585_2060_2075_dat = pd.read_hdf('data/full_CMIP6_ssp585_2060_2075.hdf', key='ssp585_2060_2075')
ssp585_2075_2090_dat = pd.read_hdf('data/full_CMIP6_ssp585_2075_2090.hdf', key='ssp585_2075_2090')


print("Pivot CMIP6 data for merge")
# ### Pivot data and bind distance measures
hist_dat = proc_pivot(hist_dat)
hist_dat = hist_dat.merge(port_dat, how='left', on=['lat_lon'])
hist_dat = hist_dat.merge(eez_check_dat, how='left', on=['lat_lon', 'lat', 'lon'])
hist_dat = hist_dat.merge(mpa_check_dat, how='left', on=['lat_lon', 'lat', 'lon'])
hist_dat = hist_dat.merge(rfmo_check_dat, how='left', on=['lat_lon', 'lat', 'lon'])


print("Merge CMIP6 and GFW Data")
ssp126_2015_2030_dat = proc_pivot(ssp126_2015_2030_dat)
ssp126_2015_2030_dat = ssp126_2015_2030_dat.merge(port_dat, how='left', on=['lat_lon'])
ssp126_2015_2030_dat = ssp126_2015_2030_dat.merge(eez_check_dat, how='left', on=['lat_lon', 'lat', 'lon'])
ssp126_2015_2030_dat = ssp126_2015_2030_dat.merge(mpa_check_dat, how='left', on=['lat_lon', 'lat', 'lon'])
ssp126_2015_2030_dat = ssp126_2015_2030_dat.merge(rfmo_check_dat, how='left', on=['lat_lon', 'lat', 'lon'])

ssp585_2015_2030_dat = proc_pivot(ssp585_2015_2030_dat)
ssp585_2015_2030_dat = ssp585_2015_2030_dat.merge(port_dat, how='left', on=['lat_lon'])
ssp585_2015_2030_dat = ssp585_2015_2030_dat.merge(eez_check_dat, how='left', on=['lat_lon', 'lat', 'lon'])
ssp585_2015_2030_dat = ssp585_2015_2030_dat.merge(mpa_check_dat, how='left', on=['lat_lon', 'lat', 'lon'])
ssp585_2015_2030_dat = ssp585_2015_2030_dat.merge(rfmo_check_dat, how='left', on=['lat_lon', 'lat', 'lon'])



ssp126_2030_2045_dat = proc_pivot(ssp126_2030_2045_dat)
ssp126_2030_2045_dat = ssp126_2030_2045_dat.merge(port_dat, how='left', on=['lat_lon'])
ssp126_2030_2045_dat = ssp126_2030_2045_dat.merge(eez_check_dat, how='left', on=['lat_lon', 'lat', 'lon'])
ssp126_2030_2045_dat = ssp126_2030_2045_dat.merge(mpa_check_dat, how='left', on=['lat_lon', 'lat', 'lon'])
ssp126_2030_2045_dat = ssp126_2030_2045_dat.merge(rfmo_check_dat, how='left', on=['lat_lon', 'lat', 'lon'])

ssp585_2030_2045_dat = proc_pivot(ssp585_2030_2045_dat)
ssp585_2030_2045_dat = ssp585_2030_2045_dat.merge(port_dat, how='left', on=['lat_lon'])
ssp585_2030_2045_dat = ssp585_2030_2045_dat.merge(eez_check_dat, how='left', on=['lat_lon', 'lat', 'lon'])
ssp585_2030_2045_dat = ssp585_2030_2045_dat.merge(mpa_check_dat, how='left', on=['lat_lon', 'lat', 'lon'])
ssp585_2030_2045_dat = ssp585_2030_2045_dat.merge(rfmo_check_dat, how='left', on=['lat_lon', 'lat', 'lon'])



ssp126_2045_2060_dat = proc_pivot(ssp126_2045_2060_dat)
ssp126_2045_2060_dat = ssp126_2045_2060_dat.merge(port_dat, how='left', on=['lat_lon'])
ssp126_2045_2060_dat = ssp126_2045_2060_dat.merge(eez_check_dat, how='left', on=['lat_lon', 'lat', 'lon'])
ssp126_2045_2060_dat = ssp126_2045_2060_dat.merge(mpa_check_dat, how='left', on=['lat_lon', 'lat', 'lon'])
ssp126_2045_2060_dat = ssp126_2045_2060_dat.merge(rfmo_check_dat, how='left', on=['lat_lon', 'lat', 'lon'])

ssp585_2045_2060_dat = proc_pivot(ssp585_2045_2060_dat)
ssp585_2045_2060_dat = ssp585_2045_2060_dat.merge(port_dat, how='left', on=['lat_lon'])
ssp585_2045_2060_dat = ssp585_2045_2060_dat.merge(eez_check_dat, how='left', on=['lat_lon', 'lat', 'lon'])
ssp585_2045_2060_dat = ssp585_2045_2060_dat.merge(mpa_check_dat, how='left', on=['lat_lon', 'lat', 'lon'])
ssp585_2045_2060_dat = ssp585_2045_2060_dat.merge(rfmo_check_dat, how='left', on=['lat_lon', 'lat', 'lon'])


ssp126_2060_2075_dat = proc_pivot(ssp126_2060_2075_dat)
ssp126_2060_2075_dat = ssp126_2060_2075_dat.merge(port_dat, how='left', on=['lat_lon'])
ssp126_2060_2075_dat = ssp126_2060_2075_dat.merge(eez_check_dat, how='left', on=['lat_lon', 'lat', 'lon'])
ssp126_2060_2075_dat = ssp126_2060_2075_dat.merge(mpa_check_dat, how='left', on=['lat_lon', 'lat', 'lon'])
ssp126_2060_2075_dat = ssp126_2060_2075_dat.merge(rfmo_check_dat, how='left', on=['lat_lon', 'lat', 'lon'])

ssp585_2060_2075_dat = proc_pivot(ssp585_2060_2075_dat)
ssp585_2060_2075_dat = ssp585_2060_2075_dat.merge(port_dat, how='left', on=['lat_lon'])
ssp585_2060_2075_dat = ssp585_2060_2075_dat.merge(eez_check_dat, how='left', on=['lat_lon', 'lat', 'lon'])
ssp585_2060_2075_dat = ssp585_2060_2075_dat.merge(mpa_check_dat, how='left', on=['lat_lon', 'lat', 'lon'])
ssp585_2060_2075_dat = ssp585_2060_2075_dat.merge(rfmo_check_dat, how='left', on=['lat_lon', 'lat', 'lon'])



ssp126_2075_2090_dat = proc_pivot(ssp126_2075_2090_dat)
ssp126_2075_2090_dat = ssp126_2075_2090_dat.merge(port_dat, how='left', on=['lat_lon'])
ssp126_2075_2090_dat = ssp126_2075_2090_dat.merge(eez_check_dat, how='left', on=['lat_lon', 'lat', 'lon'])
ssp126_2075_2090_dat = ssp126_2075_2090_dat.merge(mpa_check_dat, how='left', on=['lat_lon', 'lat', 'lon'])
ssp126_2075_2090_dat = ssp126_2075_2090_dat.merge(rfmo_check_dat, how='left', on=['lat_lon', 'lat', 'lon'])

ssp585_2075_2090_dat = proc_pivot(ssp585_2075_2090_dat)
ssp585_2075_2090_dat = ssp585_2075_2090_dat.merge(port_dat, how='left', on=['lat_lon'])
ssp585_2075_2090_dat = ssp585_2075_2090_dat.merge(eez_check_dat, how='left', on=['lat_lon', 'lat', 'lon'])
ssp585_2075_2090_dat = ssp585_2075_2090_dat.merge(mpa_check_dat, how='left', on=['lat_lon', 'lat', 'lon'])
ssp585_2075_2090_dat = ssp585_2075_2090_dat.merge(rfmo_check_dat, how='left', on=['lat_lon', 'lat', 'lon'])




print("Saving Historical Data")
# ------------------------------------------------------------------------
##### Historical Data
full_dat_hist_dat = hist_dat.merge(full_dat, how='left', on=['lat_lon', 'lat', 'lon'])


# ### Remove inf and na
full_dat_hist_dat = full_dat_hist_dat.replace([np.inf, -np.inf], np.nan)

present_cols = full_dat_hist_dat.filter(like='present').columns
full_dat_hist_dat[present_cols] = full_dat_hist_dat[present_cols].apply(lambda x: x.fillna(0))

gear_cols = full_dat_hist_dat.filter(like='gear').columns
full_dat_hist_dat[gear_cols] = full_dat_hist_dat[gear_cols].apply(lambda x: x.fillna(0))

full_dat_hist_dat.to_csv("data/full_gfw_cmip_dat.csv", index=False)
# ------------------------------------------------------------------------





print("Saving SSP 126 2015-2030 Data")
# ------------------------------------------------------------------------
#### ssp126_2015_2030_dat
full_dat_ssp126_2015_2030_dat = ssp126_2015_2030_dat.merge(full_dat, how='left', on=['lat_lon', 'lat', 'lon'])

# ### Remove inf and na
full_dat_ssp126_2015_2030_dat = full_dat_ssp126_2015_2030_dat.replace([np.inf, -np.inf], np.nan)

present_cols = full_dat_ssp126_2015_2030_dat.filter(like='present').columns
full_dat_ssp126_2015_2030_dat[present_cols] = full_dat_ssp126_2015_2030_dat[present_cols].apply(lambda x: x.fillna(0))

gear_cols = full_dat_ssp126_2015_2030_dat.filter(like='gear').columns
full_dat_ssp126_2015_2030_dat[gear_cols] = full_dat_ssp126_2015_2030_dat[gear_cols].apply(lambda x: x.fillna(0))

# ### NA values are located on land so remove
full_dat_ssp126_2015_2030_dat.to_csv("data/full_dat_ssp126_2015_2030_dat.csv", index=False)

# ------------------------------------------------------------------------



print("Saving SSP 585 2015-2030 Data")
# ------------------------------------------------------------------------
#### ssp585_2015_2030_dat
full_dat_ssp585_2015_2030_dat = ssp585_2015_2030_dat.merge(full_dat, how='left', on=['lat_lon', 'lat', 'lon'])

# ### Remove inf and na
full_dat_ssp585_2015_2030_dat = full_dat_ssp585_2015_2030_dat.replace([np.inf, -np.inf], np.nan)

present_cols = full_dat_ssp585_2015_2030_dat.filter(like='present').columns
full_dat_ssp585_2015_2030_dat[present_cols] = full_dat_ssp585_2015_2030_dat[present_cols].apply(lambda x: x.fillna(0))

gear_cols = full_dat_ssp585_2015_2030_dat.filter(like='gear').columns
full_dat_ssp585_2015_2030_dat[gear_cols] = full_dat_ssp585_2015_2030_dat[gear_cols].apply(lambda x: x.fillna(0))

# ### NA values are located on land so remove
full_dat_ssp585_2015_2030_dat.to_csv("data/full_dat_ssp585_2015_2030_dat.csv", index=False)

# ------------------------------------------------------------------------






print("Saving SSP 126 2030-2045 Data")
# ------------------------------------------------------------------------
#### ssp126_2030_2045_dat
# ### Merge on data
full_dat_ssp126_2030_2045_dat = ssp126_2030_2045_dat.merge(full_dat, how='left', on=['lat_lon', 'lat', 'lon'])

# ### Remove inf and na
full_dat_ssp126_2030_2045_dat = full_dat_ssp126_2030_2045_dat.replace([np.inf, -np.inf], np.nan)

present_cols = full_dat_ssp126_2030_2045_dat.filter(like='present').columns
full_dat_ssp126_2030_2045_dat[present_cols] = full_dat_ssp126_2030_2045_dat[present_cols].apply(lambda x: x.fillna(0))

gear_cols = full_dat_ssp126_2030_2045_dat.filter(like='gear').columns
full_dat_ssp126_2030_2045_dat[gear_cols] = full_dat_ssp126_2030_2045_dat[gear_cols].apply(lambda x: x.fillna(0))

full_dat_ssp126_2030_2045_dat.to_csv("data/full_dat_ssp126_2030_2045_dat.csv", index=False)
# ------------------------------------------------------------------------

print("Saving SSP 585 2030-2045 Data")
# ------------------------------------------------------------------------
#### ssp585_2030_2045_dat
# ### Merge on data
full_dat_ssp585_2030_2045_dat = ssp585_2030_2045_dat.merge(full_dat, how='left', on=['lat_lon', 'lat', 'lon'])

# ### Remove inf and na
full_dat_ssp585_2030_2045_dat = full_dat_ssp585_2030_2045_dat.replace([np.inf, -np.inf], np.nan)

present_cols = full_dat_ssp585_2030_2045_dat.filter(like='present').columns
full_dat_ssp585_2030_2045_dat[present_cols] = full_dat_ssp585_2030_2045_dat[present_cols].apply(lambda x: x.fillna(0))

gear_cols = full_dat_ssp585_2030_2045_dat.filter(like='gear').columns
full_dat_ssp585_2030_2045_dat[gear_cols] = full_dat_ssp585_2030_2045_dat[gear_cols].apply(lambda x: x.fillna(0))

full_dat_ssp585_2030_2045_dat.to_csv("data/full_dat_ssp585_2030_2045_dat.csv", index=False)
# ------------------------------------------------------------------------



print("Saving SSP 126 2045-2060 Data")
# ------------------------------------------------------------------------
#### ssp126_2045_2060_dat
# ### Merge on data
full_dat_ssp126_2045_2060_dat = ssp126_2045_2060_dat.merge(full_dat, how='left', on=['lat_lon', 'lat', 'lon'])

# ### Remove inf and na
full_dat_ssp126_2045_2060_dat = full_dat_ssp126_2045_2060_dat.replace([np.inf, -np.inf], np.nan)

present_cols = full_dat_ssp126_2045_2060_dat.filter(like='present').columns
full_dat_ssp126_2045_2060_dat[present_cols] = full_dat_ssp126_2045_2060_dat[present_cols].apply(lambda x: x.fillna(0))

gear_cols = full_dat_ssp126_2045_2060_dat.filter(like='gear').columns
full_dat_ssp126_2045_2060_dat[gear_cols] = full_dat_ssp126_2045_2060_dat[gear_cols].apply(lambda x: x.fillna(0))

full_dat_ssp126_2045_2060_dat.to_csv("data/full_dat_ssp126_2045_2060_dat.csv", index=False)
# ------------------------------------------------------------------------

print("Saving SSP 585 2045-2060 Data")
# ------------------------------------------------------------------------
#### ssp585_2045_2060_dat
# ### Merge on data
full_dat_ssp585_2045_2060_dat = ssp585_2045_2060_dat.merge(full_dat, how='left', on=['lat_lon', 'lat', 'lon'])

# ### Remove inf and na
full_dat_ssp585_2045_2060_dat = full_dat_ssp585_2045_2060_dat.replace([np.inf, -np.inf], np.nan)

present_cols = full_dat_ssp585_2045_2060_dat.filter(like='present').columns
full_dat_ssp585_2045_2060_dat[present_cols] = full_dat_ssp585_2045_2060_dat[present_cols].apply(lambda x: x.fillna(0))

gear_cols = full_dat_ssp585_2045_2060_dat.filter(like='gear').columns
full_dat_ssp585_2045_2060_dat[gear_cols] = full_dat_ssp585_2045_2060_dat[gear_cols].apply(lambda x: x.fillna(0))

full_dat_ssp585_2045_2060_dat.to_csv("data/full_dat_ssp585_2045_2060_dat.csv", index=False)
# ------------------------------------------------------------------------





print("Saving SSP 126 2060-2075 Data")
# ------------------------------------------------------------------------
#### ssp126_2060_2075_dat
# ### Merge on data
full_dat_ssp126_2060_2075_dat = ssp126_2060_2075_dat.merge(full_dat, how='left', on=['lat_lon', 'lat', 'lon'])

# ### Remove inf and na
full_dat_ssp126_2060_2075_dat = full_dat_ssp126_2060_2075_dat.replace([np.inf, -np.inf], np.nan)

present_cols = full_dat_ssp126_2060_2075_dat.filter(like='present').columns
full_dat_ssp126_2060_2075_dat[present_cols] = full_dat_ssp126_2060_2075_dat[present_cols].apply(lambda x: x.fillna(0))

gear_cols = full_dat_ssp126_2060_2075_dat.filter(like='gear').columns
full_dat_ssp126_2060_2075_dat[gear_cols] = full_dat_ssp126_2060_2075_dat[gear_cols].apply(lambda x: x.fillna(0))

full_dat_ssp126_2060_2075_dat.to_csv("data/full_dat_ssp126_2060_2075_dat.csv", index=False)
# ------------------------------------------------------------------------


print("Saving SSP 585 2060-2075 Data")
# ------------------------------------------------------------------------
#### ssp585_2060_2075_dat
# ### Merge on data
full_dat_ssp585_2060_2075_dat = ssp585_2060_2075_dat.merge(full_dat, how='left', on=['lat_lon', 'lat', 'lon'])

# ### Remove inf and na
full_dat_ssp585_2060_2075_dat = full_dat_ssp585_2060_2075_dat.replace([np.inf, -np.inf], np.nan)

present_cols = full_dat_ssp585_2060_2075_dat.filter(like='present').columns
full_dat_ssp585_2060_2075_dat[present_cols] = full_dat_ssp585_2060_2075_dat[present_cols].apply(lambda x: x.fillna(0))

gear_cols = full_dat_ssp585_2060_2075_dat.filter(like='gear').columns
full_dat_ssp585_2060_2075_dat[gear_cols] = full_dat_ssp585_2060_2075_dat[gear_cols].apply(lambda x: x.fillna(0))

full_dat_ssp585_2060_2075_dat.to_csv("data/full_dat_ssp585_2060_2075_dat.csv", index=False)
# ------------------------------------------------------------------------




print("Saving SSP 126 2075-2090 Data")
# ------------------------------------------------------------------------
#### ssp126_2075_2090_dat
# ### Merge on data
full_dat_ssp126_2075_2090_dat = ssp126_2075_2090_dat.merge(full_dat, how='left', on=['lat_lon', 'lat', 'lon'])

# ### Remove inf and na
full_dat_ssp126_2075_2090_dat = full_dat_ssp126_2075_2090_dat.replace([np.inf, -np.inf], np.nan)

present_cols = full_dat_ssp126_2075_2090_dat.filter(like='present').columns
full_dat_ssp126_2075_2090_dat[present_cols] = full_dat_ssp126_2075_2090_dat[present_cols].apply(lambda x: x.fillna(0))

gear_cols = full_dat_ssp126_2075_2090_dat.filter(like='gear').columns
full_dat_ssp126_2075_2090_dat[gear_cols] = full_dat_ssp126_2075_2090_dat[gear_cols].apply(lambda x: x.fillna(0))

full_dat_ssp126_2075_2090_dat.to_csv("data/full_dat_ssp126_2075_2090_dat.csv", index=False)
# ------------------------------------------------------------------------


print("Saving SSP 585 2075-2090 Data")
# ------------------------------------------------------------------------
#### ssp585_2075_2090_dat
# ### Merge on data
full_dat_ssp585_2075_2090_dat = ssp585_2075_2090_dat.merge(full_dat, how='left', on=['lat_lon', 'lat', 'lon'])

# ### Remove inf and na
full_dat_ssp585_2075_2090_dat = full_dat_ssp585_2075_2090_dat.replace([np.inf, -np.inf], np.nan)

present_cols = full_dat_ssp585_2075_2090_dat.filter(like='present').columns
full_dat_ssp585_2075_2090_dat[present_cols] = full_dat_ssp585_2075_2090_dat[present_cols].apply(lambda x: x.fillna(0))

gear_cols = full_dat_ssp585_2075_2090_dat.filter(like='gear').columns
full_dat_ssp585_2075_2090_dat[gear_cols] = full_dat_ssp585_2075_2090_dat[gear_cols].apply(lambda x: x.fillna(0))

full_dat_ssp585_2075_2090_dat.to_csv("data/full_dat_ssp585_2075_2090_dat.csv", index=False)
# ------------------------------------------------------------------------