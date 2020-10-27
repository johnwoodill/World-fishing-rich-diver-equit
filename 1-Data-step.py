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

# Load vessel characteristics
vessel_dat = pd.read_csv('/data2/GFW_public/fishing_vessels/fishing_vessels.csv')

# Merge on mmsi
dat = dat.merge(vessel_dat, how='left', on='mmsi')

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

dat = dat.sort_values('date').reset_index(drop=True)

dat = dat.dropna()

# Columns
# ------------------------------------------------------------------------------
# Index(['date', 'lat_bin', 'lon_bin', 'mmsi', 'fishing_hours', 'lat', 'lon',
#        'flag', 'geartype', 'length', 'tonnage', 'engine_power', 'active_2012',
#        'active_2013', 'active_2014', 'active_2015', 'active_2016', 'year',
#        'lon_lat'],
#       dtype='object')
# ------------------------------------------------------------------------------

dat.to_csv('data/full_GFW_public_10d.csv')

### Load data
dat = pd.read_csv('data/full_GFW_public_10d.csv', index_col=False)

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
shan_divi = shan_divi.reset_index(drop=True)
shan_divi.to_csv('data/shannon_div_equ.csv', index=False)


def flag_count(indat, flat_i, flag_j):
    indat = indat[( indat['flag'] == flag_i ) | ( indat['flag'] == flag_j )]
    if len(indat) == 2:
        return 1
    else:
        return 0


def flag_int(fint_dat, flag_i, flag_j):
    if flag_i != flag_j:
        Ai = len(fint_dat[fint_dat['flag'] == flag_i])
        red_dat = fint_dat[(fint_dat['flag'] == flag_i) | (fint_dat['flag'] == flag_j)]
        red_dat = red_dat[['lon_lat', 'flag', 'fishing_hours']].pivot(index='lon_lat', columns='flag', values = 'fishing_hours').reset_index()
        red_dat = red_dat.dropna()
        ret_flag_int = red_dat[(red_dat.iloc[:, 1] > 0) & (red_dat.iloc[:, 2] > 0)]
        
        #ret_flag_int = red_dat.groupby('lon_lat').apply(lambda x: flag_count(x, flag_i, flag_j))
        Aij = len(ret_flag_int) / Ai
        return (flag_i, flag_j, Aij)
    else:
        return (flag_i, flag_j, 1)
    
        
    
# Flag interactions
fint_dat = dat.groupby(['lon_lat', 'flag']).agg({'lat': 'mean', 'lon': 'mean', 'fishing_hours': 'sum'}).reset_index()

# Flag interactions
unique_flags = sorted(fint_dat.flag.unique())

unique_flags = ['USA', 'CHN']

res = [flag_int(fint_dat, i, j) for i in unique_flags for j in unique_flags]

flag_1 = [x[0] for x in res]
flag_2 = [x[1] for x in res]
interaction = [x[2] for x in res]
save_dat = pd.DataFrame({'flag_1': flag_1, 'flag_2': flag_2, 'interaction': interaction})
save_dat.to_csv('data/flag_interactions.csv', index=False)


retdat = pd.read_csv('data/flag_interactions.csv')

# Spread data frame to matrix
mat_retdat = retdat.pivot(index='flag_1', columns='flag_2', values='interaction')
mat_retdat.to_csv('data/flag_interactions_matrix.csv')

mat_retdat = np.matrix(mat_retdat)
np.save('data/flag_interactions_matrix.npy', mat_retdat) 



