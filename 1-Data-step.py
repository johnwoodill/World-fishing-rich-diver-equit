import pandas as pd
import numpy as np
import xarray
import glob as glob


def shan_div(indat):
    lon_lat = indat['lon_lat'].iat[0]
    lat = indat['lat'].iat[0]
    lon = indat['lon'].iat[0]
    indat = indat.assign(total = sum(indat['obs_count']))
    indat = indat.assign(p = indat['obs_count']/indat['total'])
    indat = indat.assign(ln_p = np.log(indat['p']))
    indat = indat.assign(p_ln_p = indat['p'] * indat['ln_p'])
    H = -1 * (np.sum(indat['p_ln_p']))
    E = H / np.log(len(indat))
    outdat = pd.DataFrame({'lon_lat': [lon_lat], 'lon': [lon], 'lat': [lat], 'H': [H], 'E': [E]})
    return outdat



# Check cmip6 raster file
ds = xarray.open_dataset('data/chlos_Omon_GFDL-ESM4_ssp585_r1i1p1f1_gr_209501-210012.nc')
ds_dat = ds.to_dataframe().reset_index()

# Subtract 0.5 to center graid
ds_dat = ds_dat.assign(lat = ds_dat.lat - 0.5,
                       lon = ds_dat.lon - 0.5)

# Bind GFW data
GFW_DIR = '/data2/GFW_public/fishing_effort_10d/daily_csvs/'
files = glob.glob('/data2/GFW_public/fishing_effort_10d/daily_csvs/*.csv')

list_ = []
for file in files:
    df = pd.read_csv(f"{file}")
    df = df.assign(lat = ( (df['lat_bin']/10) + 0.05),
                   lon = ( (df['lon_bin']/10) + 0.05))
    list_.append(df)

dat = pd.concat(list_, sort=False)

dat.to_csv('data/full_GFW_public_10d.csv')

### Load data
dat = pd.read_csv('data/full_GFW_public_10d.csv', index_col=False)

### Column names
# Index(['date', 'lat_bin', 'lon_bin', 'flag', 'geartype',
#        'vessel_hours', 'fishing_hours', 'mmsi_present', 'lat', 'lon',
#        'lon_lat'],
#       dtype='object')

### Aggregate to 10th degree
dat = dat.assign(lon = round(dat['lon'], 0),
                 lat = round(dat['lat'], 0),
                 year = pd.DatetimeIndex(dat['date']).year)

### Set lon/lat -0 to 0 for aggregation
dat = dat.assign(lon = np.where(dat['lon'] == -0.0, 0.0, dat['lon']))
dat = dat.assign(lat = np.where(dat['lat'] == -0.0, 0.0, dat['lat']))

### Get unique lon/lat grids
dat = dat.assign(lon_lat = dat['lon'].astype(str) + "_" + dat['lat'].astype(str))


### Aggregate to 10th degree lon/lat: 

# Total Fishing Effort
feffort_dat = dat.groupby(['year', 'lon_lat']).agg({'lat': 'mean', 'lon': 'mean', 'fishing_hours': 'sum'}).reset_index()
feffort_dat = dat.groupby(['lon_lat']).agg({'lat': 'mean', 'lon': 'mean', 'fishing_hours': 'mean'}).reset_index()
feffort_dat.to_csv('data/total_fishing_effort.csv', index=False)

# Total Fishing Effort by nation
feffort_nat_dat = dat.groupby(['year', 'lon_lat', 'flag']).agg({'lat': 'mean', 'lon': 'mean', 'fishing_hours': 'sum'}).reset_index()
feffort_nat_dat = dat.groupby(['lon_lat', 'flag']).agg({'lat': 'mean', 'lon': 'mean', 'fishing_hours': 'mean'}).reset_index()
feffort_nat_dat.to_csv('data/total_fishing_effort_nation.csv', index=False)


# Species Richness
richness_dat = dat.groupby(['lon_lat']).agg({'lat': 'mean', 'lon': 'mean', 'flag': 'nunique'}).reset_index()
richness_dat.to_csv('data/total_species_richness.csv', index=False)


# Shannon Diversity Index
sdiv_dat = dat.groupby(['lon_lat', 'flag']).agg({'lat': 'mean', 'lon': 'mean', 'year': 'count'}).reset_index()
sdiv_dat = sdiv_dat.rename(columns={'year': 'obs_count'})

shan_divi = sdiv_dat.groupby('lon_lat').apply(lambda x: shan_div(x))
shan_divi.reset_index(drop=True)
shan_divi.to_csv('data/shannon_div_equ.csv', index=False)




def flag_int(indat, flag_i, flag_j):
    indat = indat[( indat['flag'] == flag_i) | (indat['flag'] == flag_j)]
    return len(indat)
    

# Flag interactions
fint_dat = dat.groupby(['lon_lat', 'flag']).agg({'lat': 'mean', 'lon': 'mean', 'fishing_hours': 'sum'}).reset_index()



unique_flags = fint_dat.flag.unique()

retdat = pd.DataFrame()
for i in unique_flags:
    for j in unique_flags:
        flag_1 = i
        flag_2 = j
        Ai = len(fint_dat[fint_dat['flag'] == flag_1])
        Aj = len(fint_dat[fint_dat['flag'] == flag_2])
        red_dat = fint_dat[(fint_dat['flag'] == flag_1) | (fint_dat['flag'] == flag_2)]
        ret_flag_int = red_dat.groupby('lon_lat').apply(lambda x: flag_int(x, flag_1, flag_2))
        Aij = Ai / sum(ret_flag_int)
        outdat = pd.DataFrame({'flag_1': [flag_1], 'flag_2': [flag_2], 'interaction': [Aij]})
        retdat = pd.concat([retdat, outdat])
        print(f"{flag_1}-{flag_2}")
        
        
        
        
retdat.to_csv('data/flag_interactions.csv', index=False)