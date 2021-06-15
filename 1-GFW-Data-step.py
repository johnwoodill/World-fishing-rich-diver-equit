import pandas as pd
import numpy as np
import glob as glob


def shan_div(indat):
    lat_lon = indat['lat_lon'].iat[0]
    lat = indat['lat'].iat[0]
    lon = indat['lon'].iat[0]
    indat = indat.assign(total=sum(indat['obs_count']))
    indat = indat.assign(p=indat['obs_count']/indat['total'])
    indat = indat.assign(ln_p=np.log(indat['p']))
    indat = indat.assign(p_ln_p=indat['p'] * indat['ln_p'])
    H = -1 * (np.sum(indat['p_ln_p']))
    E = H / np.log(len(indat))
    outdat = pd.DataFrame({'lat_lon': [lat_lon], 'lon': [lon],
        'lat': [lat], 'H': [H], 'E': [E]})
    return outdat

print("Binding GFW Data")
# Bind GFW data
files = glob.glob('/data2/GFW_V2/GFW_10d_2012_2020/public/*.csv')

list_ = []
for file_ in files:
    df = pd.read_csv(file_)
    df = df[df['fishing_hours'] > 0]
    df = df.assign(lat=(df['cell_ll_lat'] + 0.05),
                   lon=(df['cell_ll_lon'] + 0.05))
    
    df = df[['date', 'lat', 'lon', 'mmsi', 'hours', 'fishing_hours']]
    list_.append(df)

dat = pd.concat(list_, sort=False)
del list_

# Load vessel characteristics
vessel_dat = pd.read_csv('/data2/GFW_V2/GFW_10d_2012_2020/docs/fishing-vessels-v2.csv')
vessel_dat = vessel_dat.get(['mmsi', 'flag_gfw', 'vessel_class_gfw', 'engine_power_kw_gfw'])

# Merge on mmsi
dat = dat.merge(vessel_dat, how='left', on='mmsi')

# Column names
# 'date', 'cell_ll_lat', 'cell_ll_lon', 'mmsi', 'hours', 'fishing_hours',
#        'lat', 'lon', 'flag_ais', 'flag_registry', 'flag_gfw',
#        'vessel_class_inferred', 'vessel_class_inferred_score',
#        'vessel_class_registry', 'vessel_class_gfw',
#        'self_reported_fishing_vessel', 'length_m_inferred',
#        'length_m_registry', 'length_m_gfw', 'engine_power_kw_inferred',
#        'engine_power_kw_registry', 'engine_power_kw_gfw',
#        'tonnage_gt_inferred', 'tonnage_gt_registry', 'tonnage_gt_gfw',
#        'registries_listed', 'fishing_hours_2012', 'fishing_hours_2013',
#        'fishing_hours_2014', 'fishing_hours_2015', 'fishing_hours_2016',
#        'fishing_hours_2017', 'fishing_hours_2018', 'fishing_hours_2019',
#        'fishing_hours_2020']


# Aggregate to 10th degree
dat = dat.assign(lon=round(dat['lon'], 0),
                 lat=round(dat['lat'], 0),
                 year=pd.DatetimeIndex(dat['date']).year)

# Set lon/lat -0 to 0 for aggregation
dat = dat.assign(lon=np.where(dat['lon'] == -0.0, 0.0, dat['lon']))
dat = dat.assign(lat=np.where(dat['lat'] == -0.0, 0.0, dat['lat']))

# Adjust to anti-meridian
dat = dat.assign(lon=np.where(dat['lon'] < 0, dat['lon'] + 360, dat['lon']))

# Get unique lon/lat grids
dat = dat.assign(lat_lon=dat['lat'].astype(str) + "_" + dat['lon'].astype(str))

dat = dat.sort_values('date').reset_index(drop=True)

dat = dat.dropna()

# Columns
# ------------------------------------------------------------------------------
# Index(['date', 'lat_bin', 'lon_bin', 'mmsi', 'fishing_hours', 'lat', 'lon',
#        'flag', 'vessel_class_gfw', 'length', 'tonnage', 'engine_power',
#        'active_2012', 'active_2013', 'active_2014', 'active_2015',
#        'active_2016', 'year', 'lat_lon'], dtype='object')
# ------------------------------------------------------------------------------

# GFW 2012-2020
dat.to_parquet('data/full_GFW_public_1d.parquet')
dat.to_csv('data/full_GFW_public_1d.csv', index=False)

# Load data
dat = pd.read_parquet('data/full_GFW_public_1d.parquet')

# dat = dat[dat['year'] <= 2016]

dat.head()

len(dat)
len(dat.drop_duplicates(['date', 'lat_lon']))

print("Calculating Total Fishing Effort")
# Total Fishing Effort
feffort_dat = dat.groupby(['year', 'lat_lon']).agg(
    {'lat': 'mean', 'lon': 'mean', 'fishing_hours': 'sum',
     'mmsi': 'count'}).reset_index()

feffort_dat = feffort_dat.groupby(['lat_lon']).agg(
    {'lat': 'mean', 'lon': 'mean', 'fishing_hours': 'mean',
     'mmsi': 'mean'}).reset_index()

feffort_dat = feffort_dat.rename(columns={'mmsi': 'mmsi_count'})

feffort_dat.to_csv('data/total_fishing_effort.csv', index=False)

len(feffort_dat)

print("Calculating Total Fishing Effort by nation")
# Total Fishing Effort by nation
feffort_nat_dat = dat.groupby(['year', 'lat_lon', 'flag_gfw']).agg(
    {'lat': 'mean', 'lon': 'mean', 'fishing_hours': 'sum'}).reset_index()

feffort_nat_dat = feffort_nat_dat.groupby(['lat_lon', 'flag_gfw']).agg(
    {'lat': 'mean', 'lon': 'mean', 'fishing_hours': 'mean'}).reset_index()

feffort_nat_dat.to_csv('data/total_fishing_effort_nation.csv', index=False)

print("Calculating Total Fishing Effort by Geartype")
# Total Fishing Effort by geartype
feffort_gear_nat_dat = dat.groupby(
    ['year', 'lat_lon', 'vessel_class_gfw']).agg(
    {'lat': 'mean', 'lon': 'mean', 'fishing_hours': 'sum'}).reset_index()
    
feffort_gear_nat_dat = feffort_gear_nat_dat.groupby(
    ['lat_lon', 'vessel_class_gfw']).agg(
        {'lat': 'mean', 'lon': 'mean', 'fishing_hours': 'mean'}).reset_index()
    
feffort_gear_nat_dat.to_csv('data/total_fishing_effort_gear.csv', index=False)

print("Calculating Species Richness")
# Species Richness
richness_dat = dat.groupby(['lat_lon']).agg(
    {'lat': 'mean', 'lon': 'mean', 'flag_gfw': 'nunique'}).reset_index()
richness_dat = richness_dat.rename(columns={'flag_gfw': 'richness'})
richness_dat.to_csv('data/total_species_richness.csv', index=False)

print("Calculating Shannon Diversity Index")
# Shannon Diversity Index
sdiv_dat = dat.groupby(['lat_lon', 'flag_gfw']).agg(
    {'lat': 'mean', 'lon': 'mean', 'year': 'count'}).reset_index()

sdiv_dat = sdiv_dat.rename(columns={'year': 'obs_count'})

shan_divi = sdiv_dat.groupby('lat_lon').apply(lambda x: shan_div(x))
shan_divi = shan_divi.reset_index(drop=True)

pdat = shan_divi[['lon', 'lat', 'H']]
pdat = pdat.set_index(['lon', 'lat'])
shan_divi.to_csv('data/shannon_div_equ.csv', index=False)





# ----------------------------------------------------------------------------
# Flag Interactions (For Clustering Analysis)
# def flag_int(fint_dat, flag_i, flag_j):
#     if flag_i != flag_j:
#         Ai = len(fint_dat[fint_dat['flag'] == flag_i])
#         red_dat = fint_dat[(fint_dat['flag'] == flag_i) | (fint_dat['flag'] == flag_j)]
#         red_dat = red_dat[['lat_lon', 'flag', 'fishing_hours']].pivot(index='lat_lon', columns='flag', values = 'fishing_hours').reset_index()
#         red_dat = red_dat.dropna()
#         ret_flag_int = red_dat[(red_dat.iloc[:, 1] > 0) & (red_dat.iloc[:, 2] > 0)]
#         Aij = len(ret_flag_int) / (Ai + len(ret_flag_int))
#         return (flag_i, flag_j, Aij)
#     else:
#         return (flag_i, flag_j, 1)
    
        
    
# # Flag interactions
# fint_dat = dat.groupby(['lat_lon', 'flag']).agg({'lat': 'mean', 'lon': 'mean', 'fishing_hours': 'mean'}).reset_index()


# # Adjust to anti-meridian
# fint_dat = fint_dat.assign(lon = np.where(fint_dat['lon'] < 0, fint_dat['lon'] + 360, fint_dat['lon']))

# # Subset within WCP
# fdat1 = fint_dat[(fint_dat['lon'] <= -150 + 360) & (fint_dat['lon'] >= 100) & (fint_dat['lat'] >= 0)]
# fdat2 = fint_dat[(fint_dat['lon'] <= -130 + 360) & (fint_dat['lon'] >= 140) & (fint_dat['lat'] < 0) & (fint_dat['lat'] >= -55)]
# fdat3 = fint_dat[(fint_dat['lon'] <= -130 + 360) & (fint_dat['lon'] >= 150) & (fint_dat['lat'] <= -55) & (fint_dat['lat'] >= -60)]

# # Bind data
# fint_dat = pd.concat([fdat1, fdat2, fdat3]).reset_index(drop=True)
# fint_dat = fint_dat.drop_duplicates()

# # Get top 20
# top_20 = fint_dat.groupby('flag')['fishing_hours'].sum().reset_index().sort_values('fishing_hours', ascending=False)
# top_20 = top_20.iloc[0:19, :]

# # Flag interactions
# unique_flags = sorted(top_20.flag.unique())

# res = [flag_int(fint_dat, i, j) for i in unique_flags for j in unique_flags]

# flag_1 = [x[0] for x in res]
# flag_2 = [x[1] for x in res]
# interaction = [x[2] for x in res]
# save_dat = pd.DataFrame({'flag_1': flag_1, 'flag_2': flag_2, 'interaction': interaction})
# save_dat.to_csv('data/flag_interactions.csv', index=False)

# retdat = pd.read_csv('data/flag_interactions.csv')

# # Spread data frame to matrix
# mat_retdat = retdat.pivot(index='flag_1', columns='flag_2', values='interaction')
# mat_retdat.to_csv('data/flag_interactions_matrix.csv', index=True)

# retdat.shape

# mat_retdat = np.matrix(mat_retdat)
# np.save('data/flag_interactions_matrix.npy', mat_retdat) 


