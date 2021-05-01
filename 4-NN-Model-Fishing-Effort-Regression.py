# from re import escape
# from pygam import LinearGAM, s, f
# from pygam.datasets import wage
import pandas as pd
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
# import geopandas as gpd
# from shapely.geometry import Polygon, Point
import numpy as np
from keras.layers import Input, Dense, Dropout
from keras.models import Model, Sequential
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import cross_val_score, KFold
from keras import optimizers, callbacks
from sklearn.preprocessing import MinMaxScaler



def procEffortReg(dat):
    dat = dat.drop(columns=['richness', 'lat_lon', 'E', 'H', 'port'])

    dat = dat.assign(mmsi_count=dat['mmsi_count'].fillna(0),
                     eez=dat['eez'].fillna(0),
                     mpa=dat['mpa'].fillna(0),
                     rfmo=dat['rfmo'].fillna(0),
                     fishing_hours=dat['fishing_hours'].fillna(0))

    dat['eez'] = np.where(dat['eez'] == 0, 0, 1)
    dat['mpa'] = np.where(dat['mpa'] == 0, 0, 1)
    dat['rfmo'] = np.where(dat['rfmo'] == 0, 0, 1)
   
    X = dat
   
    X = X.dropna().reset_index(drop=True)
   
    y = X['fishing_hours'] / X['mmsi_count']
    y = y.fillna(0)
    # y = np.log(1 + y)

    X = X.drop(columns=['fishing_hours', 'mmsi_count'])

    # ### Predictors that reduce model accuracy
    # X = X[X.columns.drop(list(X.filter(regex='gear')))]
    # X = X[X.columns.drop(list(X.filter(regex='present')))]
    # X = X[X.columns.drop(list(X.filter(regex='skew')))]
    # X = X[X.columns.drop(list(X.filter(regex='kurt')))]
          
    return X, y





def rmse(y_true, y_pred):
	return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))



def adapt_learning_rate(epoch):
    return 0.001 * epoch


es = EarlyStopping(monitor='val_loss',
              min_delta=0,
              patience=3,
              verbose=0, mode='auto')



my_lr_scheduler = callbacks.LearningRateScheduler(adapt_learning_rate)






#%%
# --------------------------------------------------------------------
# NN Model
full_dat = pd.read_csv("data/full_gfw_cmip_dat.csv")

X, y = procEffortReg(full_dat)

X_lon = X['lon']
X_lat = X['lat']

scaler = MinMaxScaler().fit(X)
X_scaled = scaler.transform(X)

ksmod = Sequential()
# ksmod.add(Dropout(0.5, input_shape=(len(X.columns),)))
ksmod.add(Dense(70, activation='relu'))
ksmod.add(Dense(60, activation='relu'))
ksmod.add(Dense(30, activation='relu'))
ksmod.add(Dense(10, activation='relu'))
ksmod.add(Dense(5, activation='relu'))
# ksmod.add(Dropout(0.5, input_shape=(len(X.columns),)))
ksmod.add(Dense(1, activation='relu'))
ksmod.compile(optimizer='adam', loss='mean_squared_error')
ksmod.fit(X_scaled, y.values, epochs=1000,  batch_size=32, validation_split=0.1, shuffle=True, callbacks=[es, my_lr_scheduler])


### Predict train/test set
y_pred_train = ksmod.predict(X_scaled)

### Get regression metrics
r2 = r2_score(y, y_pred_train)
mse_ = mean_squared_error(y, y_pred_train)

print(f"R-squared: {round(r2, 3)*100}% // RMSE: {round(mse_, 3)}")

#%%


print("Processing 2000-2014")
# ------------------
# 2000-2014

y_hist_pred_2015 = ksmod.predict(X_scaled)

# Get residuals
y_residuals = y.ravel() - y_hist_pred_2015.ravel()





print("Processing 2015-2030")
# ------------------
# 2015-2030
full_dat_ssp126_2015_2030_dat = pd.read_csv("data/full_dat_ssp126_2015_2030_dat.csv")
full_dat_ssp126_2015_2030_dat.columns = full_dat_ssp126_2015_2030_dat.columns.str.replace("_2015-2030", "")
X_ssp126_2015, y_ssp126_2015 = procEffortReg(full_dat_ssp126_2015_2030_dat)

full_dat_ssp585_2015_2030_dat = pd.read_csv("data/full_dat_ssp585_2015_2030_dat.csv")
full_dat_ssp585_2015_2030_dat.columns = full_dat_ssp585_2015_2030_dat.columns.str.replace("_2015-2030", "")
X_ssp585_2015, y_ssp585_2015 = procEffortReg(full_dat_ssp585_2015_2030_dat)

X_ssp126_2015_scaled = scaler.transform(X_ssp126_2015)
X_ssp585_2015_scaled = scaler.transform(X_ssp585_2015)

y_ssp126_pred_2015 = ksmod.predict(X_ssp126_2015_scaled)
y_ssp585_pred_2015 = ksmod.predict(X_ssp585_2015_scaled)

ssp126_pred_2015 = pd.DataFrame({'lat': X_ssp126_2015['lat'], 'lon': X_ssp126_2015['lon'], 'y_ssp126_pred_2015': y_ssp126_pred_2015.ravel()})
ssp585_pred_2015 = pd.DataFrame({'lat': X_ssp585_2015['lat'], 'lon': X_ssp585_2015['lon'], 'y_ssp585_pred_2015': y_ssp585_pred_2015.ravel()})
                                 
                                 



print("Processing 2030-2045")
# ------------------
# 2030-2045
full_dat_ssp126_2030_2045_dat = pd.read_csv("data/full_dat_ssp126_2030_2045_dat.csv")
full_dat_ssp126_2030_2045_dat.columns = full_dat_ssp126_2030_2045_dat.columns.str.replace("_2030-2045", "")
X_ssp126_2030, y_ssp126_2030 = procEffortReg(full_dat_ssp126_2030_2045_dat)

full_dat_ssp585_2030_2045_dat = pd.read_csv("data/full_dat_ssp585_2030_2045_dat.csv")
full_dat_ssp585_2030_2045_dat.columns = full_dat_ssp585_2030_2045_dat.columns.str.replace("_2030-2045", "")
X_ssp585_2030, y_ssp585_2030 = procEffortReg(full_dat_ssp585_2030_2045_dat)

X_ssp126_2030_scaled = scaler.transform(X_ssp126_2030)
X_ssp585_2030_scaled = scaler.transform(X_ssp585_2030)

y_ssp126_pred_2030 = ksmod.predict(X_ssp126_2030_scaled)
y_ssp585_pred_2030 = ksmod.predict(X_ssp585_2030_scaled)

ssp126_pred_2030 = pd.DataFrame({'lat': X_ssp126_2030['lat'], 'lon': X_ssp126_2030['lon'], 'y_ssp126_pred_2030': y_ssp126_pred_2030.ravel()})
ssp585_pred_2030 = pd.DataFrame({'lat': X_ssp585_2030['lat'], 'lon': X_ssp585_2030['lon'], 'y_ssp585_pred_2030': y_ssp585_pred_2030.ravel()})





print("Processing 2045-2060")
# ------------------
# 2045-2060
full_dat_ssp126_2045_2060_dat = pd.read_csv("data/full_dat_ssp126_2045_2060_dat.csv")
full_dat_ssp126_2045_2060_dat.columns = full_dat_ssp126_2045_2060_dat.columns.str.replace("_2045-2060", "")
X_ssp126_2045, y_ssp126_2045 = procEffortReg(full_dat_ssp126_2045_2060_dat)

full_dat_ssp585_2045_2060_dat = pd.read_csv("data/full_dat_ssp585_2045_2060_dat.csv")
full_dat_ssp585_2045_2060_dat.columns = full_dat_ssp585_2045_2060_dat.columns.str.replace("_2045-2060", "")
X_ssp585_2045, y_ssp585_2045 = procEffortReg(full_dat_ssp585_2045_2060_dat)

X_ssp126_2045_scaled = scaler.transform(X_ssp126_2045)
X_ssp585_2045_scaled = scaler.transform(X_ssp585_2045)

y_ssp126_pred_2045 = ksmod.predict(X_ssp126_2045_scaled)
y_ssp585_pred_2045 = ksmod.predict(X_ssp585_2045_scaled)

ssp126_pred_2045 = pd.DataFrame({'lat': X_ssp126_2045['lat'], 'lon': X_ssp126_2045['lon'], 'y_ssp126_pred_2045': y_ssp126_pred_2045.ravel()})
ssp585_pred_2045 = pd.DataFrame({'lat': X_ssp585_2045['lat'], 'lon': X_ssp585_2045['lon'], 'y_ssp585_pred_2045': y_ssp585_pred_2045.ravel()})




print("Processing 2060-2075")
# ------------------
# 2060-2075
full_dat_ssp126_2060_2075_dat = pd.read_csv("data/full_dat_ssp126_2060_2075_dat.csv")
full_dat_ssp126_2060_2075_dat.columns = full_dat_ssp126_2060_2075_dat.columns.str.replace("_2060-2075", "")
X_ssp126_2060, y_ssp126_2060 = procEffortReg(full_dat_ssp126_2060_2075_dat)

full_dat_ssp585_2060_2075_dat = pd.read_csv("data/full_dat_ssp585_2060_2075_dat.csv")
full_dat_ssp585_2060_2075_dat.columns = full_dat_ssp585_2060_2075_dat.columns.str.replace("_2060-2075", "")
X_ssp585_2060, y_ssp585_2060 = procEffortReg(full_dat_ssp585_2060_2075_dat)

X_ssp126_2060_scaled = scaler.transform(X_ssp126_2060)
X_ssp585_2060_scaled = scaler.transform(X_ssp585_2060)

y_ssp126_pred_2060 = ksmod.predict(X_ssp126_2060_scaled)
y_ssp585_pred_2060 = ksmod.predict(X_ssp585_2060_scaled)

ssp126_pred_2060 = pd.DataFrame({'lat': X_ssp126_2060['lat'], 'lon': X_ssp126_2060['lon'], 'y_ssp126_pred_2060': y_ssp126_pred_2060.ravel()})
ssp585_pred_2060 = pd.DataFrame({'lat': X_ssp585_2060['lat'], 'lon': X_ssp585_2060['lon'], 'y_ssp585_pred_2060': y_ssp585_pred_2060.ravel()})




print("Processing 2075-2090")
# ------------------
# 2075-2090
full_dat_ssp126_2075_2090_dat = pd.read_csv("data/full_dat_ssp126_2075_2090_dat.csv")
full_dat_ssp126_2075_2090_dat.columns = full_dat_ssp126_2075_2090_dat.columns.str.replace("_2075-2090", "")
X_ssp126_2075, y_ssp126_2075 = procEffortReg(full_dat_ssp126_2075_2090_dat)

full_dat_ssp585_2075_2090_dat = pd.read_csv("data/full_dat_ssp585_2075_2090_dat.csv")
full_dat_ssp585_2075_2090_dat.columns = full_dat_ssp585_2075_2090_dat.columns.str.replace("_2075-2090", "")
X_ssp585_2075, y_ssp585_2075 = procEffortReg(full_dat_ssp585_2075_2090_dat)

X_ssp126_2075_scaled = scaler.transform(X_ssp126_2075)
X_ssp585_2075_scaled = scaler.transform(X_ssp585_2075)

y_ssp126_pred_2075 = ksmod.predict(X_ssp126_2075_scaled)
y_ssp585_pred_2075 = ksmod.predict(X_ssp585_2075_scaled)

ssp126_pred_2075 = pd.DataFrame({'lat': X_ssp126_2075['lat'], 'lon': X_ssp126_2075['lon'], 'y_ssp126_pred_2075': y_ssp126_pred_2075.ravel()})
ssp585_pred_2075 = pd.DataFrame({'lat': X_ssp585_2075['lat'], 'lon': X_ssp585_2075['lon'], 'y_ssp585_pred_2075': y_ssp585_pred_2075.ravel()})



savedat = pd.DataFrame({'lat': X_lat, 
                        'lon': X_lon,
                        'y_true_historical': y, 
                        'y_pred_historical': y_hist_pred_2015.ravel()})

savedat = savedat.merge(ssp126_pred_2015, on=['lat', 'lon'])
savedat = savedat.merge(ssp585_pred_2015, on=['lat', 'lon'])

savedat = savedat.merge(ssp126_pred_2030, on=['lat', 'lon'])
savedat = savedat.merge(ssp585_pred_2030, on=['lat', 'lon'])

savedat = savedat.merge(ssp126_pred_2045, on=['lat', 'lon'])
savedat = savedat.merge(ssp585_pred_2045, on=['lat', 'lon'])

savedat = savedat.merge(ssp126_pred_2060, on=['lat', 'lon'])
savedat = savedat.merge(ssp585_pred_2060, on=['lat', 'lon'])

savedat = savedat.merge(ssp126_pred_2075, on=['lat', 'lon'])
savedat = savedat.merge(ssp585_pred_2075, on=['lat', 'lon'])


# savedat = savedat.assign(ssp585_pred_2075_diff = savedat['y_ssp585_pred_2075'] - savedat['y_pred_historical'])


# Pandas dataframe
print("Saving: 'data/NN_fishing_effort_regression_model_results.csv'")
savedat.to_csv('data/NN_fishing_effort_regression_model_results.csv', index=False)
# savedat = pd.read_csv('data/NN_fishing_effort_regression_model_results.csv')

print("Saving: 'data/NN_fishing_effort_regression_model_results.nc'")
savedat.melt(id_vars=['lat', 'lon']).set_index(['lat', 'lon', 'variable']).to_xarray().to_netcdf('data/NN_fishing_effort_regression_model_results.nc')

