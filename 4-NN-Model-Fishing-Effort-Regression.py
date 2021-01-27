from re import escape
from pygam import LinearGAM, s, f
from pygam.datasets import wage
import pandas as pd 
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
import geopandas as gpd
from shapely.geometry import Polygon, Point
import numpy as np
from keras.layers import Input, Dense, Dropout
from keras.models import Model, Sequential
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import cross_val_score, KFold
from keras import optimizers, callbacks
from sklearn.preprocessing import StandardScaler



def procEffortReg(dat):
    dat = dat.drop(columns=['richness', 'E', 'H', 'lat_lon'])
    
    dat = dat.assign(lat_x_lon = dat['lat'] * dat['lon'],
                     mmsi = dat['mmsi'].fillna(0),
                     eez = dat['eez'].fillna(0),
                     mpa = dat['mpa'].fillna(0),
                     rfmo = dat['rfmo'].fillna(0),
                     fishing_hours = dat['fishing_hours'].fillna(0))

    dat = dat.drop(columns='port')

    dat['eez'] = np.where(dat['eez'] == 0, 0, 1)
    dat['mpa'] = np.where(dat['mpa'] == 0, 0, 1)
    dat['rfmo'] = np.where(dat['rfmo'] == 0, 0, 1)
    
    X = dat
    
    X = X.dropna().reset_index(drop=True)
    
    y = X['fishing_hours'] / X['mmsi']
    y = y.fillna(0)
    y = np.log(1 + y)

    X = X.drop(columns=['fishing_hours', 'mmsi'])

    # ### Predictors that reduce model accuracy
    X = X[X.columns.drop(list(X.filter(regex='gear')))]
    X = X[X.columns.drop(list(X.filter(regex='present')))]
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







# --------------------------------------------------------------------
# NN Model
full_dat = pd.read_csv("data/full_gfw_cmip_dat.csv")

X_train, y_train = procEffortReg(full_dat)

len(X_train.lat.unique())
X_train['y']

X_train_lon = X_train['lon']
X_train_lat = X_train['lat']

scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)

ksmod = Sequential()
ksmod.add(Dropout(0.25, input_shape=(len(X_train.columns),)))
ksmod.add(Dense(60, activation='relu'))
ksmod.add(Dense(30, activation='relu'))
ksmod.add(Dense(10, activation='relu'))
ksmod.add(Dense(5, activation='relu'))
ksmod.add(Dense(1, activation='relu'))
ksmod.compile(optimizer='adam', loss='mean_squared_error')
ksmod.fit(X_train, y_train.values, epochs=1000,  batch_size=100, validation_split=0.1, shuffle=True, callbacks=[es, my_lr_scheduler])






print("Processing 2000-2014")
# ------------------
# 2000-2014

y_hist_pred_2015 = ksmod.predict(X_train)







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




print("Processing 2060-207")
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






sum(full_dat_ssp126_2000_2014_dat['mean_tos'])
sum(full_dat_ssp126_2015_2030_dat['mean_tos'])
sum(full_dat_ssp126_2030_2045_dat['mean_tos'])
sum(full_dat_ssp126_2045_2060_dat['mean_tos'])
sum(full_dat_ssp126_2060_2075_dat['mean_tos'])
sum(full_dat_ssp126_2075_2090_dat['mean_tos'])



savedat = pd.DataFrame({'lat': X_train_lat, 
                        'lon': X_train_lon,
                        'y_true_historical': y_train, 
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

savedat.to_csv('data/NN_fishing_effort_regression_model_results.csv', index=False)
