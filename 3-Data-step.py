import pandas as pd
import numpy as np
import xarray
import glob as glob
import geopandas as gpd
from shapely.geometry import Point, Polygon, shape, LinearRing
from shapely.ops import nearest_points
from geopy.distance import geodesic
import shapefile
from shapely.geometry import shape, Point
import cartopy.io.shapereader as shpreader
import math

def proc_pivot(x):
    outdat = x.pivot_table(index=['lat_lon', 'lat', 'lon'], columns=['var_name', 'period'], values=['mean', 'var', 'skew', 'kurt'])
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
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r



def get_coastlines():
    coastlines = gpd.read_file("data/coastline_shp/ne_10m_coastline.shp")
    bdat = [pd.DataFrame({'coast': i, 'lon': coastlines.geometry[i].xy[0], 'lat': coastlines.geometry[i].xy[1]}) for i in range(len(coastlines))]
    return pd.concat(bdat)



      
        
        
def get_ports():
    ports = []
    ne_ports = shpreader.natural_earth(resolution = '10m',
                                    category = 'cultural',
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



def port_dist(lon, lat):
    lon = lon.iat[0]
    lat = lat.iat[0]
    indat = ports
    if lon < -130:
        lon = lon + 360
        indat = indat.assign(lon = np.where(indat['lon'] < 0, indat['lon'] + 360, indat['lon']))
    indat.loc[:, 'distance'] = indat.apply(lambda row: haversine(lon, lat, row['lon'], row['lat']), axis=1)
    indat = indat.sort_values('distance')
    return (indat['port'].iat[0], indat['distance'].iat[0])





def coast_dist(ndat):
    lat = ndat['lat'].iat[0]
    lon = ndat['lon'].iat[0]
    indat = coasts
    if lon < -130:
        lon = lon + 360
        indat = indat.assign(lon = np.where(indat['lon'] < 0, indat['lon'] + 360, indat['lon']))
        
    indat = indat[(indat['lon'] >= (lon - 20)) & (indat['lon'] <= (lon + 20))]
    indat = indat[(indat['lat'] >= (lat - 20)) & (indat['lat'] <= (lat + 20))]
    distance = indat.apply(lambda row: haversine(lon, lat, row['lon'], row['lat']), axis=1)
    indat = indat.assign(distance = distance)
    indat = indat.sort_values('distance')
    return (indat['distance'].iat[0])






# Assign EEZ 
def eez_check(lon, lat):
    for territory in unique_eez:
        arg = eez_shp[eez_shp.Territory1 == territory].reset_index(drop=True)
        pnts = gpd.GeoDataFrame(geometry=[Point(lon, lat)])
        polys = gpd.GeoSeries({'territory': arg.geometry})
        check = pnts.assign(**{key: pnts.within(geom) for key, geom in polys.items()})
        check_ = check.territory.values[0]
        if check_ == True:
            return territory
        

# Assign EEZ indicator for GFW
def mpa_check(lon, lat):
    lon = np.where(lon < 180, lon + 360, lon)   # Shape file 0-360 lon
    for mpa_loc in unique_mpa:
        arg = mpa_shp[mpa_shp.NAME == mpa_loc].reset_index(drop=True)
        pnts = gpd.GeoDataFrame(geometry=[Point(lon, lat)])
        polys = gpd.GeoSeries({'mpa_loc': arg.geometry})
        check = pnts.assign(**{key: pnts.within(geom) for key, geom in polys.items()})
        check_ = check.mpa_loc.values[0]
        if check_ == True:
            return mpa_loc



# ------------------------------------

# ### Get ports and coastline data
ports = get_ports()
coasts = get_coastlines()

eez_shp = gpd.read_file("data/EEZ/eez_v10.shp")
unique_eez = eez_shp.Territory1.unique()

mpa_shp = gpd.read_file("data/mpa_shapefiles/vlmpa.shp")
unique_mpa = mpa_shp.NAME.unique()

# ### Calc whether country has ever fished at grid
feffort_nat_dat = pd.read_csv('data/total_fishing_effort_nation.csv')
unique_flags = feffort_nat_dat['flag'].unique()

ll_dat = feffort_nat_dat.groupby('lat_lon').agg({'lat': 'mean', 'lon': 'mean'}).sort_values(['lon', 'lat']).reset_index()
for flag_ in unique_flags:
    indat = feffort_nat_dat[['lat_lon', 'lat', 'lon', 'flag']]
    indat = indat[indat['flag'] == flag_]
    indat = indat.assign(flag = 1)
    indat = indat.rename(columns={'flag': f"{flag_}_present"})
    ll_dat = ll_dat.merge(indat, how='left', on=['lat_lon', 'lat', 'lon']).fillna(0)
    
    
    
# ### Calc whether geartype has ever fished at grid
feffort_gear_dat = pd.read_csv('data/total_fishing_effort_gear.csv')
unique_geartype = feffort_gear_dat['geartype'].unique()

gg_dat = feffort_gear_dat.groupby('lat_lon').agg({'lat': 'mean', 'lon': 'mean'}).sort_values(['lon', 'lat']).reset_index()
for gear_ in unique_geartype:
    indat = feffort_gear_dat[['lat_lon', 'lat', 'lon', 'geartype']]
    indat = indat[indat['geartype'] == gear_]
    indat = indat.assign(geartype = 1)
    indat = indat.rename(columns={'geartype': f"{gear_}_gear"})
    gg_dat = gg_dat.merge(indat, how='left', on=['lat_lon', 'lat', 'lon']).fillna(0)
    


# ### Distance to closests port
port_dat = ll_dat.groupby('lat_lon').apply(lambda x: port_dist(x['lon'], x['lat']))
port_name = [x[0] for x in port_dat]
port_dist_dat = [x[1] for x in port_dat]
port_dat_lat_lon = port_dat.reset_index().iloc[:, 0]

port_dat = pd.DataFrame({'lat_lon': port_dat_lat_lon,
                         'port_name': port_name,
                         'port_dist_dat': port_dist_dat})



port_dat.to_csv('data/port_dat.csv', index = False)
port_dat = pd.read_csv('data/port_dat.csv')

# ### Distance to coast
coast_dist_dat = ll_dat.groupby('lat_lon').apply(lambda x: coast_dist(x))
coast_dist_dat.reset_index().to_csv('data/coast_dist_dat.csv', index = False)
coast_dist_dat = pd.read_csv('data/coast_dist_dat.csv')


# ### EEZ Check
eez_check_dat = ll_dat.groupby('lat_lon').apply(lambda x: eez_check(x['lon'], x['lat']))
eez_check_dat = eez_check_dat.reset_index()
eez_check_dat = eez_check_dat.rename(columns={eez_check_dat.columns[1]: 'eez'})
eez_check_dat.to_csv('data/eez_check_dat.csv', index = False)
eez_check_dat = pd.read_csv('data/eez_check_dat.csv')

# ### MPA Check
mpa_check_dat = ll_dat.groupby('lat_lon').apply(lambda x: mpa_check(x['lon'], x['lat']))
mpa_check_dat = mpa_check_dat.reset_index()
mpa_check_dat = mpa_check_dat.rename(columns={mpa_check_dat.columns[1]: 'mpa'})
mpa_check_dat.to_csv('data/mpa_check_dat.csv', index = False)
mpa_check_dat = pd.read_csv('data/mpa_check_dat.csv')


# ### Bind data
feffort_dat = pd.read_csv('data/total_fishing_effort.csv')
shan_divi = pd.read_csv('data/shannon_div_equ.csv')
richness_dat = pd.read_csv('data/total_species_richness.csv')

full_dat = feffort_dat.merge(shan_divi, how='left', on=['lat_lon', 'lon', 'lat'])
full_dat = full_dat.merge(richness_dat, how='left', on=['lat_lon', 'lon', 'lat'])

full_dat = full_dat.merge(port_dat, how='left', on=['lat_lon'])
full_dat = full_dat.merge(eez_check_dat, how='left', on=['lat_lon'])
full_dat = full_dat.merge(mpa_check_dat, how='left', on=['lat_lon'])
full_dat = full_dat.merge(ll_dat, how='left', on=['lat_lon', 'lon', 'lat'])
full_dat = full_dat.merge(gg_dat, how='left', on=['lat_lon', 'lon', 'lat'])




# Bind CMIP6 data (add 361 to shift data)
full_dat = full_dat.assign(lon = np.where(full_dat['lon'] < 0, full_dat['lon'] + 361, full_dat['lon']))
full_dat = full_dat.assign(lat_lon = full_dat['lat'].astype(str) + "_" + full_dat['lon'].astype(str))


na_free = full_dat.lat_lon.drop_duplicates()
full_dat[~full_dat.index.isin(na_free.index)]


hist_dat = pd.read_hdf('data/full_CMIP6_historical.hdf', key='historical')
ssp126_2015_2030_dat = pd.read_hdf('data/full_CMIP6_ssp126_2015_2030.hdf', key='ssp126_2015_2030')
ssp126_2030_2045_dat = pd.read_hdf('data/full_CMIP6_ssp126_2030_2045.hdf', key='ssp126_2030_2045')
ssp126_2045_2060_dat = pd.read_hdf('data/full_CMIP6_ssp126_2045_2060.hdf', key='ssp126_2045_2060')
ssp126_2060_2075_dat = pd.read_hdf('data/full_CMIP6_ssp126_2060_2075.hdf', key='ssp126_2060_2075')
ssp126_2075_2090_dat = pd.read_hdf('data/full_CMIP6_ssp126_2075_2090.hdf', key='ssp126_2075_2090')

# ### Pivot data
hist_dat = proc_pivot(hist_dat)
ssp126_2015_2030_dat = proc_pivot(ssp126_2015_2030_dat)
ssp126_2030_2045_dat = proc_pivot(ssp126_2030_2045_dat)
ssp126_2045_2060_dat = proc_pivot(ssp126_2045_2060_dat)
ssp126_2060_2075_dat = proc_pivot(ssp126_2060_2075_dat)
ssp126_2075_2090_dat = proc_pivot(ssp126_2075_2090_dat)


##### Historical Data
# ### Merge on data
full_dat_hist_dat = full_dat.merge(hist_dat, how='left', on=['lat_lon', 'lat', 'lon'])

# ### Remove inf and na
full_dat_hist_dat = full_dat_hist_dat.replace([np.inf, -np.inf], np.nan)
len(full_dat_hist_dat.dropna())/len(full_dat_hist_dat)

full_dat_hist_dat.to_csv("data/full_gfw_cmip_dat.csv", index=False)




#### ssp126_2015_2030_dat
# ### Merge on data
full_dat_ssp126_2015_2030_dat = full_dat.merge(ssp126_2015_2030_dat, how='left', on=['lat_lon', 'lat', 'lon'])

# ### Remove inf and na
full_dat_ssp126_2015_2030_dat = full_dat_ssp126_2015_2030_dat.replace([np.inf, -np.inf], np.nan)
len(full_dat_ssp126_2015_2030_dat.dropna())/len(full_dat_ssp126_2015_2030_dat)

full_dat_ssp126_2015_2030_dat.to_csv("data/full_dat_ssp126_2015_2030_dat.csv", index=False)





#### ssp126_2045_2060_dat
# ### Merge on data
full_dat_ssp126_2045_2060_dat = full_dat.merge(ssp126_2045_2060_dat, how='left', on=['lat_lon', 'lat', 'lon'])

# ### Remove inf and na
full_dat_ssp126_2045_2060_dat = full_dat_ssp126_2045_2060_dat.replace([np.inf, -np.inf], np.nan)
len(full_dat_ssp126_2045_2060_dat.dropna())/len(full_dat_ssp126_2045_2060_dat)

full_dat_ssp126_2045_2060_dat.to_csv("data/full_dat_ssp126_2045_2060_dat.csv", index=False)




len(full_dat_ssp126_2045_2060_dat.lat_lon.unique())

