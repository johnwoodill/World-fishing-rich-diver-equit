import pandas as pd
import numpy as np
import xarray
import glob as glob
from scipy.stats import skew, kurtosis

def proc_dat(file_, min_year, max_year, var_name):
    outdat = pd.DataFrame()
    if isinstance(file_, list):
        for i in file_:
            ### Read nc files and convert to df
            ds = xarray.open_dataset(i)
            indat = ds.to_dataframe().reset_index()
            indat = indat.dropna()
            
            indat = indat.drop(['lat', 'lon', 'time', 'bnds'], axis = 1)
            
            indat = indat.rename(columns={"lat_bnds": "lat", "lon_bnds": "lon", "time_bnds": "time"})
            
            ### Clean up dates
            indat = indat.assign(time = pd.to_datetime(indat['time'], format='%Y-%m-%d %H:00:00'))
            
            indat = indat.assign(year = indat['time'].dt.year,
                                lon = round( indat['lon'], 0),
                                lat = round( indat['lat'], 0))
            
            ### Only keep years between 2000 and 2014
            indat = indat[( indat['year'] >= min_year) & ( indat['year'] <= max_year)]
            
            ### Clean up data
            # indat = indat.iloc[:, [3, 8, 1, 2, 5]]
            indat.columns = indat.columns.str.replace(f"{var_name}", "value")
            indat = indat.assign(var = var_name,
                                lat_lon = indat['lat'].astype(str) + "_" + indat['lon'].astype(str) )
            indat = indat[['time', 'year', 'lat_lon', 'lat', 'lon', 'var', 'value']]
            # indat = indat.iloc[:, [2, 4, 6, 0, 1, 5, 3]]
            outdat = pd.concat([outdat, indat])
            
    else:
        ### Read nc files and convert to df
        ds = xarray.open_dataset(file_)
        indat = ds.to_dataframe().reset_index()
        indat = indat.dropna()
        
        indat = indat.drop(['lat', 'lon', 'time', 'bnds'], axis = 1)
        
        indat = indat.rename(columns={"lat_bnds": "lat", "lon_bnds": "lon", "time_bnds": "time"})
        
        ### Clean up dates
        indat = indat.assign(time = pd.to_datetime(indat['time'], format='%Y-%m-%d %H:00:00'))
        
        indat = indat.assign(year = indat['time'].dt.year,
                            lon = round( indat['lon'], 0),
                            lat = round( indat['lat'], 0))
        
        ### Only keep years between 2000 and 2014
        indat = indat[( indat['year'] >= min_year) & ( indat['year'] <= max_year)]
        
        ### Clean up data
        # indat = indat.iloc[:, [3, 8, 1, 2, 5]]
        indat.columns = indat.columns.str.replace(f"{var_name}", "value")
        indat = indat.assign(var = var_name,
                            lat_lon = indat['lat'].astype(str) + "_" + indat['lon'].astype(str) )
        indat = indat[['time', 'year', 'lat_lon', 'lat', 'lon', 'var', 'value']]
        # indat = indat.iloc[:, [2, 4, 6, 0, 1, 5, 3]]
        outdat = pd.concat([outdat, indat])
        
    ### Assign period
    period = f"{min(outdat['year'])}-{max(outdat['year'])}"
    
    ### Calc mean, var, skew, and kurt
    vmean = outdat.groupby(['lat_lon'])['value'].apply(lambda x: np.mean(x)).reset_index()
    vmax = outdat.groupby(['lat_lon'])['value'].apply(lambda x: np.max(x)).reset_index()
    vmin = outdat.groupby(['lat_lon'])['value'].apply(lambda x: np.min(x)).reset_index()
    vvar = outdat.groupby(['lat_lon'])['value'].apply(lambda x: np.var(x)).reset_index()
    vskew = outdat.groupby(['lat_lon'])['value'].apply(lambda x: skew(x)).reset_index()
    vkurt = outdat.groupby(['lat_lon'])['value'].apply(lambda x: kurtosis(x)).reset_index()
    

    ### Bind data    
    voutdat = vmean.merge(vmax, how='left', on='lat_lon')
    voutdat = voutdat.merge(vmin, how='left', on='lat_lon')
    voutdat = voutdat.merge(vvar, how='left', on='lat_lon')
    voutdat = voutdat.merge(vskew, how='left', on='lat_lon')
    voutdat = voutdat.merge(vkurt, how='left', on='lat_lon')

    ### rename columns
    voutdat.columns = ['lat_lon', 'mean', 'max', 'min', 'var', 'skew', 'kurt']

    ### Get lat/lon, var_name, and assign period
    lat = voutdat['lat_lon'].str.split("_", expand=True)[0].astype(float)
    lon = voutdat['lat_lon'].str.split("_", expand=True)[1].astype(float)

    voutdat = voutdat.assign(var_name = var_name,
                        period = period,
                        lat = lat, 
                        lon = lon)
    
    ### Reorder columns and return
    voutdat = voutdat[['period', 'lat_lon', 'lat', 'lon', 'var_name', 'mean', 'max', 'min', 'var', 'skew', 'kurt']]
    return voutdat
    
    
    
    

# Setup data dir locations
# 
# CMIP6_data/arag 
# CMIP6_data/chl
# CMIP6_data/oxy 
# CMIP6_data/ph 
# CMIP6_data/salinity 
# CMIP6_data/si 
# CMIP6_data/sst 
# CMIP6_data/zoo 
 
# Sub directories
# historical/
# ssp126
# ssp585

print("Processing Historical Data")
# -----------------------------------------------------------------
# ### Historical Data
arag_files = sorted(glob.glob('CMIP6_data/arag/historical/*'))[-2:]
chl_files = sorted(glob.glob('CMIP6_data/chl/historical/*'))[-2:]
oxy_files = sorted(glob.glob('CMIP6_data/oxy/historical/*'))[-2:]
ph_files = sorted(glob.glob('CMIP6_data/pH/historical/*'))[-2:]
sal_files = sorted(glob.glob('CMIP6_data/salinity/historical/*'))[-2:]
si_files = sorted(glob.glob('CMIP6_data/si/historical/*'))[-2:]
sst_files = sorted(glob.glob('CMIP6_data/sst/historical/*'))[-2:]
zoo_files = sorted(glob.glob('CMIP6_data/zoo/historical/*'))[-2:]

# ### Get individual data
hist_arag_data = proc_dat(arag_files, min_year = 2000, max_year = 2014, var_name = "aragos")
hist_chl_data = proc_dat(chl_files, min_year = 2000, max_year = 2014, var_name = "chlos")
hist_oxy_data = proc_dat(oxy_files, min_year = 2000, max_year = 2014, var_name = "o2satos")
hist_ph_data = proc_dat(ph_files, min_year = 2000, max_year = 2014, var_name = "phos")
hist_sal_data = proc_dat(sal_files, min_year = 2000, max_year = 2014, var_name = "sos")
hist_si_data = proc_dat(si_files, min_year = 2000, max_year = 2014, var_name = "sios")
hist_sst_data = proc_dat(sst_files, min_year = 2000, max_year = 2014, var_name = "tos")
hist_zoo_data = proc_dat(zoo_files, min_year = 2000, max_year = 2014, var_name = "zoocos")

# ### Concat data
hist_dat = pd.concat([hist_arag_data, hist_chl_data, hist_oxy_data, hist_ph_data, 
                    hist_sal_data, hist_si_data, hist_sst_data, hist_zoo_data])

# ### Save data
hist_dat = hist_dat.reset_index(drop=False)
hist_dat.to_hdf('data/full_CMIP6_historical.hdf', key='historical')

hist_dat = pd.read_hdf('data/full_CMIP6_historical.hdf', key='historical')

print("Processing 2015-2030")
# -----------------------------------------------------------------
# ### ssp126 Data 2015-2030
arag_files = sorted(glob.glob('CMIP6_data/arag/ssp126/*'))[0]
chl_files = sorted(glob.glob('CMIP6_data/chl/ssp126/*'))[0]
oxy_files = sorted(glob.glob('CMIP6_data/oxy/ssp126/*'))[0]
ph_files = sorted(glob.glob('CMIP6_data/pH/ssp126/*'))[0]
sal_files = sorted(glob.glob('CMIP6_data/salinity/ssp126/*'))[0]
si_files = sorted(glob.glob('CMIP6_data/si/ssp126/*'))[0]
sst_files = sorted(glob.glob('CMIP6_data/sst/ssp126/*'))[0]
zoo_files = sorted(glob.glob('CMIP6_data/zoo/ssp126/*'))[0]

# ### Get individual data
ssp126_arag_data = proc_dat(arag_files, min_year = 2015, max_year = 2030, var_name = "aragos")
ssp126_chl_data = proc_dat(chl_files, min_year = 2015, max_year = 2030, var_name = "chlos")
ssp126_oxy_data = proc_dat(oxy_files, min_year = 2015, max_year = 2030, var_name = "o2satos")
ssp126_ph_data = proc_dat(ph_files, min_year = 2015, max_year = 2030, var_name = "phos")
ssp126_sal_data = proc_dat(sal_files, min_year = 2015, max_year = 2030, var_name = "sos")
ssp126_si_data = proc_dat(si_files, min_year = 2015, max_year = 2030, var_name = "sios")
ssp126_sst_data = proc_dat(sst_files, min_year = 2015, max_year = 2030, var_name = "tos")
ssp126_zoo_data = proc_dat(zoo_files, min_year = 2015, max_year = 2030, var_name = "zoocos")

# ### Concat data
ssp126_dat = pd.concat([ssp126_arag_data, ssp126_chl_data, ssp126_oxy_data, ssp126_ph_data, 
                    ssp126_sal_data, ssp126_si_data, ssp126_sst_data, ssp126_zoo_data])

# ### Save data
ssp126_dat.to_hdf('data/full_CMIP6_ssp126_2015_2030.hdf', key='ssp126_2015_2030')










print("Processing 2030-2045")
# -----------------------------------------------------------------
# ### ssp126 Data 2030-2045
arag_files = sorted(glob.glob('CMIP6_data/arag/ssp126/*'))[0:2]
chl_files = sorted(glob.glob('CMIP6_data/chl/ssp126/*'))[0:2]
oxy_files = sorted(glob.glob('CMIP6_data/oxy/ssp126/*'))[0:2]
ph_files = sorted(glob.glob('CMIP6_data/pH/ssp126/*'))[0:2]
sal_files = sorted(glob.glob('CMIP6_data/salinity/ssp126/*'))[0:2]
si_files = sorted(glob.glob('CMIP6_data/si/ssp126/*'))[0:2]
sst_files = sorted(glob.glob('CMIP6_data/sst/ssp126/*'))[0:2]
zoo_files = sorted(glob.glob('CMIP6_data/zoo/ssp126/*'))[0:2]

# ### Get individual data
ssp126_arag_data = proc_dat(arag_files, min_year = 2030, max_year = 2045, var_name = "aragos")
ssp126_chl_data = proc_dat(chl_files, min_year = 2030, max_year = 2045, var_name = "chlos")
ssp126_oxy_data = proc_dat(oxy_files, min_year = 2030, max_year = 2045, var_name = "o2satos")
ssp126_ph_data = proc_dat(ph_files, min_year = 2030, max_year = 2045, var_name = "phos")
ssp126_sal_data = proc_dat(sal_files, min_year = 2030, max_year = 2045, var_name = "sos")
ssp126_si_data = proc_dat(si_files, min_year = 2030, max_year = 2045, var_name = "sios")
ssp126_sst_data = proc_dat(sst_files, min_year = 2030, max_year = 2045, var_name = "tos")
ssp126_zoo_data = proc_dat(zoo_files, min_year = 2030, max_year = 2045, var_name = "zoocos")

# ### Concat data
ssp126_dat = pd.concat([ssp126_arag_data, ssp126_chl_data, ssp126_oxy_data, ssp126_ph_data, 
                    ssp126_sal_data, ssp126_si_data, ssp126_sst_data, ssp126_zoo_data])

# ### Save data
ssp126_dat.to_hdf('data/full_CMIP6_ssp126_2030_2045.hdf', key='ssp126_2030_2045')





print("Processing 2045-2060")
# -----------------------------------------------------------------
# ### ssp126 Data 2045-2060
arag_files = sorted(glob.glob('CMIP6_data/arag/ssp126/*'))[1:3]
chl_files = sorted(glob.glob('CMIP6_data/chl/ssp126/*'))[1:3]
oxy_files = sorted(glob.glob('CMIP6_data/oxy/ssp126/*'))[1:3]
ph_files = sorted(glob.glob('CMIP6_data/pH/ssp126/*'))[1:3]
sal_files = sorted(glob.glob('CMIP6_data/salinity/ssp126/*'))[1:3]
si_files = sorted(glob.glob('CMIP6_data/si/ssp126/*'))[1:3]
sst_files = sorted(glob.glob('CMIP6_data/sst/ssp126/*'))[1:3]
zoo_files = sorted(glob.glob('CMIP6_data/zoo/ssp126/*'))[1:3]

# ### Get individual data
ssp126_arag_data = proc_dat(arag_files, min_year = 2045, max_year = 2060, var_name = "aragos")
ssp126_chl_data = proc_dat(chl_files, min_year = 2045, max_year = 2060, var_name = "chlos")
ssp126_oxy_data = proc_dat(oxy_files, min_year = 2045, max_year = 2060, var_name = "o2satos")
ssp126_ph_data = proc_dat(ph_files, min_year = 2045, max_year = 2060, var_name = "phos")
ssp126_sal_data = proc_dat(sal_files, min_year = 2045, max_year = 2060, var_name = "sos")
ssp126_si_data = proc_dat(si_files, min_year = 2045, max_year = 2060, var_name = "sios")
ssp126_sst_data = proc_dat(sst_files, min_year = 2045, max_year = 2060, var_name = "tos")
ssp126_zoo_data = proc_dat(zoo_files, min_year = 2045, max_year = 2060, var_name = "zoocos")

# ### Concat data
ssp126_dat = pd.concat([ssp126_arag_data, ssp126_chl_data, ssp126_oxy_data, ssp126_ph_data, 
                    ssp126_sal_data, ssp126_si_data, ssp126_sst_data, ssp126_zoo_data])

# ### Save data
ssp126_dat.to_hdf('data/full_CMIP6_ssp126_2045_2060.hdf', key='ssp126_2045_2060')








print("Processing 2060-2075")
# -----------------------------------------------------------------
# ### ssp126 Data 2060-2075
arag_files = sorted(glob.glob('CMIP6_data/arag/ssp126/*'))[2:4]
chl_files = sorted(glob.glob('CMIP6_data/chl/ssp126/*'))[2:4]
oxy_files = sorted(glob.glob('CMIP6_data/oxy/ssp126/*'))[2:4]
ph_files = sorted(glob.glob('CMIP6_data/pH/ssp126/*'))[2:4]
sal_files = sorted(glob.glob('CMIP6_data/salinity/ssp126/*'))[2:4]
si_files = sorted(glob.glob('CMIP6_data/si/ssp126/*'))[2:4]
sst_files = sorted(glob.glob('CMIP6_data/sst/ssp126/*'))[2:4]
zoo_files = sorted(glob.glob('CMIP6_data/zoo/ssp126/*'))[2:4]

# ### Get individual data
ssp126_arag_data = proc_dat(arag_files, min_year = 2060, max_year = 2075, var_name = "aragos")
ssp126_chl_data = proc_dat(chl_files, min_year = 2060, max_year = 2075, var_name = "chlos")
ssp126_oxy_data = proc_dat(oxy_files, min_year = 2060, max_year = 2075, var_name = "o2satos")
ssp126_ph_data = proc_dat(ph_files, min_year = 2060, max_year = 2075, var_name = "phos")
ssp126_sal_data = proc_dat(sal_files, min_year = 2060, max_year = 2075, var_name = "sos")
ssp126_si_data = proc_dat(si_files, min_year = 2060, max_year = 2075, var_name = "sios")
ssp126_sst_data = proc_dat(sst_files, min_year = 2060, max_year = 2075, var_name = "tos")
ssp126_zoo_data = proc_dat(zoo_files, min_year = 2060, max_year = 2075, var_name = "zoocos")

# ### Concat data
ssp126_dat = pd.concat([ssp126_arag_data, ssp126_chl_data, ssp126_oxy_data, ssp126_ph_data, 
                    ssp126_sal_data, ssp126_si_data, ssp126_sst_data, ssp126_zoo_data])

# ### Save data
ssp126_dat.to_hdf('data/full_CMIP6_ssp126_2060_2075.hdf', key='ssp126_2060_2075')





print("Processing 2075-2090")
# -----------------------------------------------------------------
# ### ssp126 Data 2075-2090
arag_files = sorted(glob.glob('CMIP6_data/arag/ssp126/*'))[3:4]
chl_files = sorted(glob.glob('CMIP6_data/chl/ssp126/*'))[3:4]
oxy_files = sorted(glob.glob('CMIP6_data/oxy/ssp126/*'))[3:4]
ph_files = sorted(glob.glob('CMIP6_data/pH/ssp126/*'))[3:4]
sal_files = sorted(glob.glob('CMIP6_data/salinity/ssp126/*'))[3:4]
si_files = sorted(glob.glob('CMIP6_data/si/ssp126/*'))[3:4]
sst_files = sorted(glob.glob('CMIP6_data/sst/ssp126/*'))[3:4]
zoo_files = sorted(glob.glob('CMIP6_data/zoo/ssp126/*'))[3:4]

# ### Get individual data
ssp126_arag_data = proc_dat(arag_files, min_year = 2075, max_year = 2090, var_name = "aragos")
ssp126_chl_data = proc_dat(chl_files, min_year = 2075, max_year = 2090, var_name = "chlos")
ssp126_oxy_data = proc_dat(oxy_files, min_year = 2075, max_year = 2090, var_name = "o2satos")
ssp126_ph_data = proc_dat(ph_files, min_year = 2075, max_year = 2090, var_name = "phos")
ssp126_sal_data = proc_dat(sal_files, min_year = 2075, max_year = 2090, var_name = "sos")
ssp126_si_data = proc_dat(si_files, min_year = 2075, max_year = 2090, var_name = "sios")
ssp126_sst_data = proc_dat(sst_files, min_year = 2075, max_year = 2090, var_name = "tos")
ssp126_zoo_data = proc_dat(zoo_files, min_year = 2075, max_year = 2090, var_name = "zoocos")

# ### Concat data
ssp126_dat = pd.concat([ssp126_arag_data, ssp126_chl_data, ssp126_oxy_data, ssp126_ph_data, 
                    ssp126_sal_data, ssp126_si_data, ssp126_sst_data, ssp126_zoo_data])

# ### Save data
ssp126_dat.to_hdf('data/full_CMIP6_ssp126_2075_2090.hdf', key='ssp126_2075_2090')


