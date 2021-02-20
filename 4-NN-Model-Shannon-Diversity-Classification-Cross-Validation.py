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
from sklearn.preprocessing import LabelEncoder
import random
import gc
from sklearn.metrics import accuracy_score, roc_auc_score
from keras.utils import np_utils

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




def procShannon(dat):
    dat = dat.drop(columns=['richness', 'lat_lon', 'E', 'eez', 'mpa', 'rfmo', 'mmsi'])
    
    dat = dat.assign(lat_x_lon = dat['lat'] * dat['lon'],
                     H = dat['H'].fillna(0),
                     fishing_hours = dat['fishing_hours'].fillna(0))

    dat = dat.drop(columns='port')

    # dat['eez'] = np.where(dat['eez'] == 0, 0, 1)
    # dat['mpa'] = np.where(dat['mpa'] == 0, 0, 1)
    # dat['rfmo'] = np.where(dat['rfmo'] == 0, 0, 1)
    
    X = dat
    
    X = X.dropna().reset_index(drop=True)
    
    X1 = X[X['H'] == 0]
    X2 = X[X['H'] > 0]

    pd.qcut(X2['H'], q=3)
    
    X2 = X2.assign(H = pd.qcut(X2['H'], q=3, labels = [1, 2, 3]))

    X = pd.concat([X1, X2])
    
    y = X['H']
    y = pd.Series(np.where(y == -0, 0, y))

    ### Check count of Shannon
    check_y = pd.DataFrame({'y': y, 'count': 1})
    check_y.groupby('y').count()
    
    X = X.drop(columns='H')

    # ### Predictors that reduce model accuracy
    # X = X[X.columns.drop(list(X.filter(regex='gear')))]
    X = X[X.columns.drop(list(X.filter(regex='present')))]
    # X = X[X.columns.drop(list(X.filter(regex='skew')))]
    # X = X[X.columns.drop(list(X.filter(regex='kurt')))]
            
    return X, y



def roc_auc_score_multiclass(actual_class, pred_class, average = "macro"):

  #creating a set of all the unique classes using the actual class list
  unique_class = set(actual_class)
  roc_auc_dict = {}
  for per_class in unique_class:
    #creating a list of all the classes except the current class 
    other_class = [x for x in unique_class if x != per_class]

    #marking the current class as 1 and all other classes as 0
    new_actual_class = [0 if x in other_class else 1 for x in actual_class]
    new_pred_class = [0 if x in other_class else 1 for x in pred_class]

    #using the sklearn metrics method to calculate the roc_auc_score
    roc_auc = roc_auc_score(new_actual_class, new_pred_class, average = average)
    roc_auc_dict[per_class] = roc_auc

  return roc_auc_dict





def rmse(y_true, y_pred):
	return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))



def adapt_learning_rate(epoch):
    return 0.001 * epoch


es = EarlyStopping(monitor='val_loss',
              min_delta=0,
              patience=3,
              verbose=0, mode='auto')



my_lr_scheduler = callbacks.LearningRateScheduler(adapt_learning_rate)





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

X, y = procShannon(full_dat)

X = X.merge(cv_dat, on=['lat', 'lon'])

nblocks = round(len(X[X['cv_grid'] == 0].grid.unique()) * 0.75)

outdat = pd.DataFrame()
for dropout_ in [0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
    for cv in range(0, 2000):
        rint = random.randint(0, 1)
        train_blocks = pd.Series(sorted(X[X['cv_grid'] == rint].grid.unique())).sample(nblocks)
        test_blocks = pd.Series(sorted(X[X['cv_grid'] != rint].grid.unique())).sample(nblocks)

        train_index = X[X['grid'].isin(train_blocks)].index
        test_index = X[X['grid'].isin(test_blocks)].index

        X_train, y_train = X[X.index.isin(train_index)].drop(columns={'grid', 'cv_grid'}), y[y.index.isin(train_index)]
        X_test, y_test = X[X.index.isin(test_index)].drop(columns={'grid', 'cv_grid'}), y[y.index.isin(test_index)]

        sc_X = StandardScaler().fit(X_train)
        X_train = sc_X.fit_transform(X_train)
        X_test = sc_X.transform(X_test)

        # Old way remove
        # X_train = preprocessing.scale(X_train)
        # X_test = preprocessing.scale(X_test)

        # encode class values as integers
        encoder = LabelEncoder()
        encoder.fit(y_train)
        encoded_Y = encoder.transform(y_train)
        dummy_y = np_utils.to_categorical(encoded_Y)

        ksmod = Sequential()
        ksmod.add(Dropout(dropout_, input_shape=(len(X.columns) - 2,)))
        ksmod.add(Dense(70, activation='relu'))
        ksmod.add(Dense(60, activation='relu'))
        ksmod.add(Dense(30, activation='relu'))
        ksmod.add(Dense(10, activation='relu'))
        ksmod.add(Dense(5, activation='relu'))
        ksmod.add(Dropout(dropout_, input_shape=(len(X.columns) - 2,)))
        ksmod.add(Dense(4, activation='softmax'))
        ksmod.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        ksmod.fit(X_train, dummy_y, verbose=0, epochs=1000,  batch_size=100, validation_split=0.1, shuffle=True, callbacks=[es, my_lr_scheduler])

        ### Predict train/test set
        y_pred_train = ksmod.predict(X_train)    
        y_pred_test = ksmod.predict(X_test)   
            
        y_pred_train = [np.argmax(x) for x in y_pred_train]
        y_pred_test = [np.argmax(x) for x in y_pred_test]
        
        ### Get regression metrics
        roc_ = roc_auc_score_multiclass(y_test, y_pred_test, average='macro')
        roc_0 = roc_[0]
        roc_1 = roc_[1]
        roc_2 = roc_[2]
        roc_3 = roc_[3]
        test_roc_mean = np.mean([roc_0, roc_1, roc_2, roc_3])
        
        roc_ = roc_auc_score_multiclass(y_train, y_pred_train)
        roc_0 = roc_[0]
        roc_1 = roc_[1]
        roc_2 = roc_[2]
        roc_3 = roc_[3]
        train_roc_mean = np.mean([roc_0, roc_1, roc_2, roc_3])
        
        ### Get accuracy score
        acc_test_score = accuracy_score(y_test, y_pred_test)
        acc_train_score = accuracy_score(y_train, y_pred_train)

        del ksmod
        K.clear_session()
        gc.collect()

        ### Bind data
        indat = pd.DataFrame({'cv': [cv], 'dropout': [dropout_], 'test_score': [acc_test_score], 'train_score': [acc_train_score], 
                            'roc_0': [roc_0], 'roc_1': [roc_1], 'roc_2': [roc_2], 'roc_3': [roc_3], 'test_roc_avg': [test_roc_mean],
                            'train_roc_avg': train_roc_mean})
        outdat = pd.concat([outdat, indat])
        print(f"[{cv}] => Dropout = {dropout_} -- Train Acc. Score: {round(acc_train_score*100, 3)}%  -- Test Acc. Score: {round(acc_test_score*100, 3)}% // Train Macro Mean: {round(train_roc_mean*100, 3)}% -- Test Macro Mean: {round(test_roc_mean*100, 3)}%")
        




outdat.to_csv('data/NN_ShanDiv_5D_block_cv_alldropouts_results.csv', index=False)







# # --------------------------------------------------------------------
# # NN Model
# # 5-Fold Cross-validation

# full_dat = pd.read_csv("data/full_gfw_cmip_dat.csv")

# X, y = procShannon(full_dat)

# kf = KFold(n_splits=5, shuffle=True)
# kf.get_n_splits(X)

# outdat = pd.DataFrame()
# cv = 0
# for train_index, test_index in kf.split(X):
#     cv = cv + 1

#     ### Get train/test splits
#     X_train, X_test = X[X.index.isin(train_index)], X[X.index.isin(test_index)]
#     y_train, y_test = y[y.index.isin(train_index)], y[y.index.isin(test_index)]
        
#     X_train = preprocessing.scale(X_train)
#     X_test = preprocessing.scale(X_test)
    
#     # encode class values as integers
#     encoder = LabelEncoder()
#     encoder.fit(y_train)
#     encoded_Y = encoder.transform(y_train)
#     # convert integers to dummy variables (i.e. one hot encoded)
#     dummy_y = np_utils.to_categorical(encoded_Y)

#     ksmod = Sequential()
#     # ksmod.add(Dense(100, activation='relu'))
#     ksmod.add(Dropout(0.50, input_shape=(len(X.columns),)))
#     ksmod.add(Dense(70, activation='relu'))
#     ksmod.add(Dense(60, activation='relu'))
#     ksmod.add(Dense(30, activation='relu'))
#     ksmod.add(Dense(10, activation='relu'))
#     ksmod.add(Dense(5, activation='relu'))
#     ksmod.add(Dropout(0.50, input_shape=(len(X.columns),)))
#     ksmod.add(Dense(4, activation='softmax'))
#     ksmod.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#     ksmod.fit(X_train, dummy_y, epochs=1000,  batch_size=100, validation_split=0.1, shuffle=True, callbacks=[es, my_lr_scheduler])

#     ### Predict train/test set
#     y_pred_train = ksmod.predict(X_train)    
#     y_pred_test = ksmod.predict(X_test)   
        
#     y_pred_train = [np.argmax(x) for x in y_pred_train]
#     y_pred_test = [np.argmax(x) for x in y_pred_test]
    
#     ### Get regression metrics
#     roc_ = roc_auc_score_multiclass(y_test, y_pred_test)
#     roc_0 = roc_[0]
#     roc_1 = roc_[1]
#     roc_2 = roc_[2]
#     roc_3 = roc_[3]
#     roc_mean = np.mean([roc_0, roc_1, roc_2, roc_3])
    
#     ### Get accuracy score
#     acc_test_score = accuracy_score(y_test, y_pred_test)
#     acc_train_score = accuracy_score(y_train, y_pred_train)
    
#     ### Bind data
#     indat = pd.DataFrame({'cv': cv, 
#                           'y_true': y_test, 'y_pred': y_pred_test, 'test_score': acc_test_score,
#                           'train_score': acc_train_score, 'roc_0': roc_0, 'roc_1': roc_1, 'roc_2': roc_2, 
#                           'roc_3': roc_3, 'roc_avg': roc_mean})
#     outdat = pd.concat([outdat, indat])
#     print(f"[{cv}] Train Accuracy Score: {round(acc_train_score*100, 3)}%  ---- Test Accuracy Score: {round(acc_test_score*100, 3)}% ---- Macro Mean: {round(roc_mean*100, 3)}%")
    





# outdat.groupby('cv').apply(lambda x: print(np.mean(x['rmse_test']), np.mean(x['rmse_train']), np.mean(x['rmse_test']) - np.mean(x['rmse_train'])))

# outdat.groupby('cv').apply(lambda x: print(np.mean(x['r2_train']), "---", np.mean(x['r2_test'])))


























# # 10-Degree Block Cross validation data

# cv_dat = pd.read_csv("data/full_gfw_cmip_dat.csv")
# cv_dat = cv_dat.iloc[:, 1:3]

# grid_data = build_cv_grids(10)

# grid_results = cv_dat.apply(lambda x: check_grid(x['lat'], x['lon'], grid_data), axis=1)

# cv_dat = cv_dat.assign(grid = [x[0] for x in grid_results], cv_grid = [x[1] for x in grid_results])

# # cv_dat = cv_dat[['lat_lon', 'lat', 'lon', 'grid', 'cv_grid']]


# cv_dat = cv_dat[['lat_lon', 'grid', 'cv_grid']]

# # full_dat = full_dat.merge(cv_dat, on='lat_lon')

# full_dat


# X = X.merge(cv_dat, on='lat_lon')
# X = X.iloc[:, 1:]

# test_index = X[X['cv_grid'] == 0].index
# train_index = X[X['cv_grid'] == 1].index

# test_index
# train_index

# X_train, y_train = X[X.index.isin(train_index)], y[y.index.isin(train_index)]
# X_test, y_test = X[X.index.isin(test_index)], y[y.index.isin(test_index)]



# X_train = preprocessing.scale(X_train)
# X_test = preprocessing.scale(X_test)

# ksmod = Sequential()
# # ksmod.add(Dense(100, activation='relu'))
# ksmod.add(Dropout(0.10, input_shape=(len(X.columns),)))
# ksmod.add(Dense(60, activation='relu'))
# ksmod.add(Dense(30, activation='relu'))
# ksmod.add(Dense(10, activation='relu'))
# ksmod.add(Dense(5, activation='relu'))
# ksmod.add(Dense(1, activation='relu'))
# ksmod.compile(optimizer='adam', loss='mean_squared_error')
# ksmod.fit(X_train, y_train.values, epochs=1000,  batch_size=100, validation_split=0.1, shuffle=True, callbacks=[es, my_lr_scheduler])
# # ksmod.fit(X_train, y_train.values, epochs=1000,  batch_size=100, validation_data=(X_test, y_test.values), callbacks=[es, my_lr_scheduler])

# ### Predict train/test set
# y_pred_train = ksmod.predict(X_train)    
# y_pred_test = ksmod.predict(X_test)   
    
# ### Get regression metrics
# train_r2 = r2_score(y_train, y_pred_train)
# test_r2 = r2_score(y_test, y_pred_test)

# train_rmse = mean_squared_error(y_train, y_pred_train)
# test_rmse = mean_squared_error(y_test, y_pred_test)

# ### Bind data
# indat = pd.DataFrame({'cv': cv, 
#                     #   'lat': X_test['lat'], 'lon': X_test['lon'],
#                         'y_true': y_test, 'y_pred': y_pred_test.ravel(), 'r2_train': train_r2,
#                         'r2_test': test_r2, 'rmse_train': train_rmse, 'rmse_test': test_rmse})

# outdat = pd.concat([outdat, indat])
# print(f"[{cv}] Cross-validation complete - Accuracy Score:")
# print("----------------------------------------------------------")
# print(f"r2: {train_r2} (Train R2) ------- {test_r2} (Test R2)")
# print(f"RMSE: {train_rmse} (Train RMSE) ------- {test_rmse} (Test RMSE)")
# print("----------------------------------------------------------")









# # 20-Degree Block Cross validation data

# cv_dat = pd.read_csv("data/full_gfw_cmip_dat.csv")
# cv_dat = cv_dat.iloc[:, 1:3]

# grid_data = build_cv_grids(20)

# grid_results = cv_dat.apply(lambda x: check_grid(x['lat'], x['lon'], grid_data), axis=1)

# cv_dat = cv_dat.assign(grid = [x[0] for x in grid_results], cv_grid = [x[1] for x in grid_results])

# # cv_dat = cv_dat[['lat_lon', 'lat', 'lon', 'grid', 'cv_grid']]


# cv_dat = cv_dat[['lat_lon', 'grid', 'cv_grid']]

# # full_dat = full_dat.merge(cv_dat, on='lat_lon')

# full_dat


# X = X.merge(cv_dat, on='lat_lon')
# X = X.iloc[:, 1:]

# test_index = X[X['cv_grid'] == 0].index
# train_index = X[X['cv_grid'] == 1].index

# test_index
# train_index

# X_train, y_train = X[X.index.isin(train_index)], y[y.index.isin(train_index)]
# X_test, y_test = X[X.index.isin(test_index)], y[y.index.isin(test_index)]



# X_train = preprocessing.scale(X_train)
# X_test = preprocessing.scale(X_test)

# ksmod = Sequential()
# # ksmod.add(Dense(100, activation='relu'))
# ksmod.add(Dropout(0.10, input_shape=(len(X.columns),)))
# ksmod.add(Dense(60, activation='relu'))
# ksmod.add(Dense(30, activation='relu'))
# ksmod.add(Dense(10, activation='relu'))
# ksmod.add(Dense(5, activation='relu'))
# ksmod.add(Dense(1, activation='relu'))
# ksmod.compile(optimizer='adam', loss='mean_squared_error')
# ksmod.fit(X_train, y_train.values, epochs=1000,  batch_size=100, validation_split=0.1, shuffle=True, callbacks=[es, my_lr_scheduler])
# # ksmod.fit(X_train, y_train.values, epochs=1000,  batch_size=100, validation_data=(X_test, y_test.values), callbacks=[es, my_lr_scheduler])

# ### Predict train/test set
# y_pred_train = ksmod.predict(X_train)    
# y_pred_test = ksmod.predict(X_test)   
    
# ### Get regression metrics
# train_r2 = r2_score(y_train, y_pred_train)
# test_r2 = r2_score(y_test, y_pred_test)

# train_rmse = mean_squared_error(y_train, y_pred_train)
# test_rmse = mean_squared_error(y_test, y_pred_test)

# ### Bind data
# indat = pd.DataFrame({'cv': cv, 
#                     #   'lat': X_test['lat'], 'lon': X_test['lon'],
#                         'y_true': y_test, 'y_pred': y_pred_test.ravel(), 'r2_train': train_r2,
#                         'r2_test': test_r2, 'rmse_train': train_rmse, 'rmse_test': test_rmse})

# outdat = pd.concat([outdat, indat])
# print(f"[{cv}] Cross-validation complete - Accuracy Score:")
# print("----------------------------------------------------------")
# print(f"r2: {train_r2} (Train R2) ------- {test_r2} (Test R2)")
# print(f"RMSE: {train_rmse} (Train RMSE) ------- {test_rmse} (Test RMSE)")
# print("----------------------------------------------------------")