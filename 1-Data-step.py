import pandas as pd
import numpy as np

### Load data
dat = pd.read_csv('data/full_GFW_public_100d.csv', index_col=False)

# Index(['date', 'lat_bin', 'lon_bin', 'flag', 'geartype',
#        'vessel_hours', 'fishing_hours', 'mmsi_present', 'lat', 'lon',
#        'lon_lat'],
#       dtype='object')

### Aggregate to 10th degree
dat = dat.assign(lon = round(dat['lon'], 1),
                 lat = round(dat['lat'], 1))

### Get unique lon/lat grids
dat = dat.assign(lon_lat = dat['lon'].astype(str) + "_" + dat['lat'].astype(str))

### Aggregate to 10th degree lon/lat: 

# Species Richness
flag_dat = dat.groupby(['date', 'lon_lat']).agg({'lat': 'mean', 'lon': 'mean', 'flag': 'nunique'}).reset_index()
flag_dat.to_csv('data/daily_species_richness.csv', index=False)


# vessel_hours
vhours_dat = dat.groupby(['date', 'lon_lat']).agg({'lat': 'mean', 'lon': 'mean', 'vessel_hours': 'sum'}).reset_index()




# fishing_hours
fhours_dat = dat.groupby(['date', 'lon_lat']).agg({'lat': 'mean', 'lon': 'mean', 'fishing_hours': 'sum'}).reset_index()



# mmsi_present
mmsi_dat = dat.groupby(['date', 'lon_lat']).agg({'lat': 'mean', 'lon': 'mean', 'mmsi': 'nunique'}).reset_index()

