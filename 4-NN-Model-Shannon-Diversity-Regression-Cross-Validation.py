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
from keras import backend as K
import random
import gc

def build_cv_grids(delta):
    # Getting range of lat/lon 
    # (set start of lon to delta to deal with border issues)
    xmin, ymin, xmax, ymax = (delta, -90, 361, 91)

    # Build out range of lat/lon
    lats = list(range(int(np.floor(ymin)), int(np.ceil(ymax)), delta))
    lons = list(range(int(np.floor(xmin)), int(np.ceil(xmax)), delta))
    lats.reverse()
    lats
    lons
        
    # Get count grids to alternative across lon
    even_grids = [0, 1] * int(len(lats))
    odd_grids = [1, 0] * int(len(lats))

    # Iterate through each and build polygon grids
    polygons = []
    grids = []
    for i in range(0, len(lons)):
        x = lons[i]
        if i % 2 == 0:
            grids.extend(even_grids[0:len(lats)])
        else:
            grids.extend(odd_grids[0:len(lats)])
        for y in lats:        
            polygons.append( Polygon([(x, y), (x - delta, y), (x - delta, y + delta), (x, y + delta)]) )
            
    grid = gpd.GeoDataFrame({'geometry':polygons, 'grid': np.linspace(0, len(polygons), len(polygons)).astype(int), 'cv_grid': grids})
    return grid



def check_grid(lat, lon, grid_data):
    # If on the edge, don't center
    if lon == 0:
        lon = lon + 0.01
        lat = lat - 0.01
    else: 
        lon = lon - 0.01
        lat = lat - 0.01
    
    # Convert lon/lat to Point()
    p1 = Point(lon, lat)

    # Loop through each grid and check if in polygon
    for i in range(0, len(grid_data)):
        # Get polygon
        poly = grid_data.loc[i, 'geometry']
        
        # Check if in poly
        check = p1.within(poly)
        
        ### Get Variables if true
        if check == True:
            # print("true")
            n_grid = grid_data.loc[i, 'grid']
            cv_n_grid = grid_data.loc[i, 'cv_grid']
            return (n_grid, cv_n_grid)
    return (999, 999)




def procEffortReg(dat):
    dat = dat.drop(columns=['richness', 'E'])
    
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
    
    y = X['H']
    y = y.fillna(0)
    y = np.log(1 + y)

    X = X.drop(columns=['H', 'mmsi'])

    # ### Predictors that reduce model accuracy
    # X = X[X.columns.drop(list(X.filter(regex='gear')))]
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
# 5-Fold Cross-validation

full_dat = pd.read_csv("data/full_gfw_cmip_dat.csv")

X, y = procEffortReg(full_dat)

kf = KFold(n_splits=5, shuffle=True)
kf.get_n_splits(X)

outdat = pd.DataFrame()
cv = 0
for train_index, test_index in kf.split(X):
    cv = cv + 1

    ### Get train/test splits
    X_train, X_test = X[X.index.isin(train_index)], X[X.index.isin(test_index)]
    y_train, y_test = y[y.index.isin(train_index)], y[y.index.isin(test_index)]
        
    X_train = preprocessing.scale(X_train)
    X_test = preprocessing.scale(X_test)

    ksmod = Sequential()
    # ksmod.add(Dense(100, activation='relu'))
    ksmod.add(Dropout(0.50, input_shape=(len(X.columns),)))
    ksmod.add(Dense(70, activation='relu'))
    ksmod.add(Dense(60, activation='relu'))
    ksmod.add(Dense(30, activation='relu'))
    ksmod.add(Dense(10, activation='relu'))
    ksmod.add(Dense(5, activation='relu'))
    ksmod.add(Dropout(0.50, input_shape=(len(X.columns),)))
    ksmod.add(Dense(1, activation='relu'))
    ksmod.compile(optimizer='adam', loss='mean_squared_error')
    ksmod.fit(X_train, y_train.values, epochs=1000,  batch_size=100, validation_split=0.1, shuffle=True, callbacks=[es, my_lr_scheduler])

    ### Predict train/test set
    y_pred_train = ksmod.predict(X_train)    
    y_pred_test = ksmod.predict(X_test)   
        
    ### Get regression metrics
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    
    train_rmse = mean_squared_error(y_train, y_pred_train)
    test_rmse = mean_squared_error(y_test, y_pred_test)
    
    ### Bind data
    indat = pd.DataFrame({'cv': cv, 
                        #   'lat': X_test['lat'], 'lon': X_test['lon'],
                          'y_true': y_test, 'y_pred': y_pred_test.ravel(), 'r2_train': train_r2,
                          'r2_test': test_r2, 'rmse_train': train_rmse, 'rmse_test': test_rmse})
    
    outdat = pd.concat([outdat, indat])
    print(f"[{cv}] Cross-validation complete - Accuracy Score:")
    print("----------------------------------------------------------")
    print(f"r2: {train_r2} (Train R2) ------- {test_r2} (Test R2)")
    print(f"RMSE: {train_rmse} (Train RMSE) ------- {test_rmse} (Test RMSE)")
    print("----------------------------------------------------------")


outdat.groupby('cv').apply(lambda x: print(np.mean(x['rmse_test']), np.mean(x['rmse_train']), np.mean(x['rmse_test']) - np.mean(x['rmse_train'])))

outdat.groupby('cv').apply(lambda x: print(np.mean(x['r2_train']), "---", np.mean(x['r2_test'])))













# --------------------------------
# Bootstrap 5-Degree Block Cross validation data

# cv_dat = pd.read_csv("data/full_gfw_cmip_dat.csv")
# cv_dat = cv_dat.iloc[:, 1:3]
# grid_data = build_cv_grids(5)
# grid_results = cv_dat.apply(lambda x: check_grid(x['lat'], x['lon'], grid_data), axis=1)
# cv_dat = cv_dat.assign(grid = [x[0] for x in grid_results], cv_grid = [x[1] for x in grid_results])


cv_dat = pd.read_csv('./data/full_gfw_cmip_dat_5cvgrids.csv')

full_dat = pd.read_csv("data/full_gfw_cmip_dat.csv")
# full_dat = full_dat.merge(cv_dat, on=['lat', 'lon'])

X, y = procEffortReg(full_dat)

# y = np.log(1 + y)

X = X.merge(cv_dat, on=['lat', 'lon'])

nblocks = round(len(X[X['cv_grid'] == 0].grid.unique()) * 0.75)

outdat = pd.DataFrame()
for cv in range(0, 2000):
    rint = random.randint(0, 1)
    train_blocks = pd.Series(sorted(X[X['cv_grid'] == rint].grid.unique())).sample(nblocks)
    test_blocks = pd.Series(sorted(X[X['cv_grid'] != rint].grid.unique())).sample(nblocks)

    train_index = X[X['grid'].isin(train_blocks)].index
    test_index = X[X['grid'].isin(test_blocks)].index

    X_train, y_train = X[X.index.isin(train_index)].drop(columns={'grid', 'cv_grid', 'lat_lon'}), y[y.index.isin(train_index)]
    X_test, y_test = X[X.index.isin(test_index)].drop(columns={'grid', 'cv_grid', 'lat_lon'}), y[y.index.isin(test_index)]

    X_train = preprocessing.scale(X_train)
    X_test = preprocessing.scale(X_test)

    ksmod = Sequential()
    # ksmod.add(Dense(100, activation='relu'))
    ksmod.add(Dropout(0.10, input_shape=(len(X.columns) - 3,)))
    ksmod.add(Dense(70, activation='relu'))
    ksmod.add(Dense(60, activation='relu'))
    ksmod.add(Dense(30, activation='relu'))
    ksmod.add(Dense(10, activation='relu'))
    ksmod.add(Dense(5, activation='relu'))
    ksmod.add(Dropout(0.10, input_shape=(len(X.columns) - 3,)))
    ksmod.add(Dense(1, activation='relu'))
    ksmod.compile(optimizer='adam', loss='mean_squared_error')
    ksmod.fit(X_train, y_train.values, epochs=1000,  batch_size=100, validation_split=0.1, shuffle=True, callbacks=[es, my_lr_scheduler])


    ### Predict train/test set
    y_pred_train = ksmod.predict(X_train)    
    y_pred_test = ksmod.predict(X_test)   
        
    ### Get regression metrics
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)

    train_rmse = mean_squared_error(y_train, y_pred_train)
    test_rmse = mean_squared_error(y_test, y_pred_test)

    ### Bind data
    indat = pd.DataFrame({'cv': [cv], 
                        'r2_train': [train_r2],
                        'r2_test': [test_r2], 
                        'rmse_train': [train_rmse], 
                        'rmse_test': [test_rmse]})
    
    # indat.to_csv(f"tmp/nn_cv_dat/cv_{cv}.csv", index=False)
    outdat = pd.concat([outdat, indat])
    
    del ksmod
    K.clear_session()
    gc.collect()

    print(f"[{cv}] Cross-validation complete - Accuracy Score:")
    print("----------------------------------------------------------")
    print(f"r2: {round(train_r2, 3)} (Train R2)  {round(test_r2, 3)} (Test R2)")
    print(f"RMSE: {round(train_rmse, 3)} (Train RMSE) {round(test_rmse, 3)} (Test RMSE)")
    print("----------------------------------------------------------")
    
    




print('saving')
    
    
outdat.to_csv('data/NN_ShanDiv_5D_block_cv_dropout05_results.csv', index=False)












# 10-Degree Block Cross validation data

cv_dat = pd.read_csv("data/full_gfw_cmip_dat.csv")
cv_dat = cv_dat.iloc[:, 1:3]

grid_data = build_cv_grids(10)

grid_results = cv_dat.apply(lambda x: check_grid(x['lat'], x['lon'], grid_data), axis=1)

cv_dat = cv_dat.assign(grid = [x[0] for x in grid_results], cv_grid = [x[1] for x in grid_results])

# cv_dat = cv_dat[['lat_lon', 'lat', 'lon', 'grid', 'cv_grid']]


cv_dat = cv_dat[['lat_lon', 'grid', 'cv_grid']]

# full_dat = full_dat.merge(cv_dat, on='lat_lon')

full_dat


X = X.merge(cv_dat, on='lat_lon')
X = X.iloc[:, 1:]

test_index = X[X['cv_grid'] == 0].index
train_index = X[X['cv_grid'] == 1].index

test_index
train_index

X_train, y_train = X[X.index.isin(train_index)], y[y.index.isin(train_index)]
X_test, y_test = X[X.index.isin(test_index)], y[y.index.isin(test_index)]



X_train = preprocessing.scale(X_train)
X_test = preprocessing.scale(X_test)

ksmod = Sequential()
# ksmod.add(Dense(100, activation='relu'))
ksmod.add(Dropout(0.10, input_shape=(len(X.columns),)))
ksmod.add(Dense(60, activation='relu'))
ksmod.add(Dense(30, activation='relu'))
ksmod.add(Dense(10, activation='relu'))
ksmod.add(Dense(5, activation='relu'))
ksmod.add(Dense(1, activation='relu'))
ksmod.compile(optimizer='adam', loss='mean_squared_error')
ksmod.fit(X_train, y_train.values, epochs=1000,  batch_size=100, validation_split=0.1, shuffle=True, callbacks=[es, my_lr_scheduler])
# ksmod.fit(X_train, y_train.values, epochs=1000,  batch_size=100, validation_data=(X_test, y_test.values), callbacks=[es, my_lr_scheduler])

### Predict train/test set
y_pred_train = ksmod.predict(X_train)    
y_pred_test = ksmod.predict(X_test)   
    
### Get regression metrics
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)

train_rmse = mean_squared_error(y_train, y_pred_train)
test_rmse = mean_squared_error(y_test, y_pred_test)

### Bind data
indat = pd.DataFrame({'cv': cv, 
                    #   'lat': X_test['lat'], 'lon': X_test['lon'],
                        'y_true': y_test, 'y_pred': y_pred_test.ravel(), 'r2_train': train_r2,
                        'r2_test': test_r2, 'rmse_train': train_rmse, 'rmse_test': test_rmse})

outdat = pd.concat([outdat, indat])
print(f"[{cv}] Cross-validation complete - Accuracy Score:")
print("----------------------------------------------------------")
print(f"r2: {train_r2} (Train R2) ------- {test_r2} (Test R2)")
print(f"RMSE: {train_rmse} (Train RMSE) ------- {test_rmse} (Test RMSE)")
print("----------------------------------------------------------")









# 20-Degree Block Cross validation data

cv_dat = pd.read_csv("data/full_gfw_cmip_dat.csv")
cv_dat = cv_dat.iloc[:, 1:3]

grid_data = build_cv_grids(20)

grid_results = cv_dat.apply(lambda x: check_grid(x['lat'], x['lon'], grid_data), axis=1)

cv_dat = cv_dat.assign(grid = [x[0] for x in grid_results], cv_grid = [x[1] for x in grid_results])

# cv_dat = cv_dat[['lat_lon', 'lat', 'lon', 'grid', 'cv_grid']]


cv_dat = cv_dat[['lat_lon', 'grid', 'cv_grid']]

# full_dat = full_dat.merge(cv_dat, on='lat_lon')

full_dat


X = X.merge(cv_dat, on='lat_lon')
X = X.iloc[:, 1:]

test_index = X[X['cv_grid'] == 0].index
train_index = X[X['cv_grid'] == 1].index

test_index
train_index

X_train, y_train = X[X.index.isin(train_index)], y[y.index.isin(train_index)]
X_test, y_test = X[X.index.isin(test_index)], y[y.index.isin(test_index)]



X_train = preprocessing.scale(X_train)
X_test = preprocessing.scale(X_test)

ksmod = Sequential()
# ksmod.add(Dense(100, activation='relu'))
ksmod.add(Dropout(0.10, input_shape=(len(X.columns),)))
ksmod.add(Dense(60, activation='relu'))
ksmod.add(Dense(30, activation='relu'))
ksmod.add(Dense(10, activation='relu'))
ksmod.add(Dense(5, activation='relu'))
ksmod.add(Dense(1, activation='relu'))
ksmod.compile(optimizer='adam', loss='mean_squared_error')
ksmod.fit(X_train, y_train.values, epochs=1000,  batch_size=100, validation_split=0.1, shuffle=True, callbacks=[es, my_lr_scheduler])
# ksmod.fit(X_train, y_train.values, epochs=1000,  batch_size=100, validation_data=(X_test, y_test.values), callbacks=[es, my_lr_scheduler])

### Predict train/test set
y_pred_train = ksmod.predict(X_train)    
y_pred_test = ksmod.predict(X_test)   
    
### Get regression metrics
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)

train_rmse = mean_squared_error(y_train, y_pred_train)
test_rmse = mean_squared_error(y_test, y_pred_test)

### Bind data
indat = pd.DataFrame({'cv': cv, 
                    #   'lat': X_test['lat'], 'lon': X_test['lon'],
                        'y_true': y_test, 'y_pred': y_pred_test.ravel(), 'r2_train': train_r2,
                        'r2_test': test_r2, 'rmse_train': train_rmse, 'rmse_test': test_rmse})

outdat = pd.concat([outdat, indat])
print(f"[{cv}] Cross-validation complete - Accuracy Score:")
print("----------------------------------------------------------")
print(f"r2: {train_r2} (Train R2) ------- {test_r2} (Test R2)")
print(f"RMSE: {train_rmse} (Train RMSE) ------- {test_rmse} (Test RMSE)")
print("----------------------------------------------------------")