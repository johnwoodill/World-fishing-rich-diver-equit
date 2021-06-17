
import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import cross_val_score, KFold

from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.metrics import mean_squared_error
from keras import backend
import keras 
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn import preprocessing
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from numpy.random import seed
import sys
# import tensorflow as tf

# Set seed for replication
seed(123)

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





def procShannon(dat):
    dat = dat.drop(columns=['richness', 'lat_lon', 'E',
                            'rfmo', 'mmsi_count', 'port'])
    
    dat = dat.assign(H = dat['H'].fillna(0))

    dat['eez'] = np.where(dat['eez'] == 0, 0, 1)
    dat['mpa'] = np.where(dat['mpa'] == 0, 0, 1)
    # dat['rfmo'] = np.where(dat['rfmo'] == 0, 0, 1)
    
    X = dat.dropna().reset_index(drop=True)
    
    X1 = X[X['H'] == 0]
    X2 = X[X['H'] > 0]

    pd.qcut(X2['H'], q=3)
    
    X2 = X2.assign(H = pd.qcut(X2['H'], q=3, labels = [1, 2, 3]))

    X = pd.concat([X1, X2])
    
    y = X['H']

    # ### Check count of Shannon
    check_y = pd.DataFrame({'y': y, 'count': 1})
    check_y.groupby('y').count()
    
    X = X.drop(columns='H')

    # ### Predictors that reduce model accuracy
    # X = X[X.columns.drop(list(X.filter(regex='gear')))]
    X = X[X.columns.drop(list(X.filter(regex='present')))]
    # X = X[X.columns.drop(list(X.filter(regex='skew')))]
    # X = X[X.columns.drop(list(X.filter(regex='kurt')))]
            
    return X, y





# -----------------------------------------------
# ### K-Fold Cross-validation
full_dat = pd.read_csv("data/full_gfw_cmip_dat.csv")

X, y = procShannon(full_dat)

kf = KFold(n_splits=10)
kf.get_n_splits(X)

outdat = pd.DataFrame()
cv = 0
for train_index, test_index in kf.split(X):
    cv = cv + 1
    ### Setup Classifier
    # clf = RandomForestClassifier(n_estimators = 1000, min_samples_split = 2, min_samples_leaf = 1,
    #                          max_features = 'auto', max_depth = 10, bootstrap = False)
    
    clf = RandomForestClassifier(n_jobs=-1)
    
    ### Get train/test splits
    X_train, X_test = X[X.index.isin(train_index)], X[X.index.isin(test_index)]
    y_train, y_test = y[y.index.isin(train_index)], y[y.index.isin(test_index)]
    
    ### Fit training Set
    clf.fit(X_train.drop(columns=['lat', 'lon']), y_train)
        
    ### Predict test set
    y_pred = clf.predict(X_test.drop(columns=['lat', 'lon']))    
    
    roc_ = roc_auc_score_multiclass(y_test, y_pred)
    roc_0 = roc_[0]
    roc_1 = roc_[1]
    roc_2 = roc_[2]
    roc_3 = roc_[3]
    roc_mean = np.mean([roc_0, roc_1, roc_2, roc_3])
    
    ### Get accuracy score
    acc_score = accuracy_score(y_test, y_pred)
    
    ### Bind data
    indat = pd.DataFrame({'lat': X_test['lat'], 'lon': X_test['lon'], 'cv': cv, 
                          'y_true': y_test, 'y_pred': y_pred, 'score': acc_score,
                          'roc_0': roc_0, 'roc_1': roc_1, 'roc_2': roc_2, 'roc_3': roc_3,
                          'roc_avg': roc_mean})
    outdat = pd.concat([outdat, indat])
    print(f"[{cv}] Cross-validation complete - Accuracy Score: {round(acc_score, 4)}  -////-  Macro Mean: {roc_mean}")



print(f"ROC Avg Macro: {round(np.mean(outdat.roc_avg)*100, 2)}%")


outdat[['cv', 'y_true', 'y_pred']].groupby('cv').sum()


# sys.exit()

outdat.to_csv('data/rf_shannon_cross_validation_results.csv', index=False)
    
    
# ### Random Forest Classifier
# clf = RandomForestClassifier(n_estimators = 1000, min_samples_split = 2, min_samples_leaf = 1,
#                             max_features = 'auto', max_depth = 10, bootstrap = False)

# clf = RandomForestClassifier()

# np.mean(cross_val_score(clf, X, y, cv=10))

#
# >>> np.mean(cross_val_score(clf, X, y, cv=10))
# 0.7194471922804744


# ### Results from hyper-parameter tuning
# {'n_estimators': 1000, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'auto', 'max_depth': 10, 'bootstrap': False}


print("Processing 2000-2014")
# ------------------
# 2000-2014

X_lat = X['lat']
X_lon = X['lon']

clf = RandomForestClassifier(n_jobs=-1).fit(X.drop(columns=['lat', 'lon']), y)

y_hist_pred_2015 = clf.predict(X.drop(columns=['lat', 'lon']))

# Get residuals
y_residuals = y.ravel() - y_hist_pred_2015.ravel()





print("Processing 2015-2030")
# ------------------
# 2015-2030
full_dat_ssp126_2015_2030_dat = pd.read_csv("data/full_dat_ssp126_2015_2030_dat.csv")
full_dat_ssp126_2015_2030_dat.columns = full_dat_ssp126_2015_2030_dat.columns.str.replace("_2015-2030", "")
X_ssp126_2015, y_ssp126_2015 = procShannon(full_dat_ssp126_2015_2030_dat)

full_dat_ssp370_2015_2030_dat = pd.read_csv("data/full_dat_ssp370_2015_2030_dat.csv")
full_dat_ssp370_2015_2030_dat.columns = full_dat_ssp370_2015_2030_dat.columns.str.replace("_2015-2030", "")
X_ssp370_2015, y_ssp370_2015 = procShannon(full_dat_ssp370_2015_2030_dat)

full_dat_ssp585_2015_2030_dat = pd.read_csv("data/full_dat_ssp585_2015_2030_dat.csv")
full_dat_ssp585_2015_2030_dat.columns = full_dat_ssp585_2015_2030_dat.columns.str.replace("_2015-2030", "")
X_ssp585_2015, y_ssp585_2015 = procShannon(full_dat_ssp585_2015_2030_dat)

y_ssp126_pred_2015 = clf.predict(X_ssp126_2015.drop(columns=['lat', 'lon']))
y_ssp370_pred_2015 = clf.predict(X_ssp370_2015.drop(columns=['lat', 'lon']))
y_ssp585_pred_2015 = clf.predict(X_ssp585_2015.drop(columns=['lat', 'lon']))

ssp126_pred_2015 = pd.DataFrame({'lat': X_ssp126_2015['lat'], 'lon': X_ssp126_2015['lon'], 'y_ssp126_pred_2015': y_ssp126_pred_2015.ravel()})
ssp370_pred_2015 = pd.DataFrame({'lat': X_ssp370_2015['lat'], 'lon': X_ssp370_2015['lon'], 'y_ssp370_pred_2015': y_ssp370_pred_2015.ravel()})
ssp585_pred_2015 = pd.DataFrame({'lat': X_ssp585_2015['lat'], 'lon': X_ssp585_2015['lon'], 'y_ssp585_pred_2015': y_ssp585_pred_2015.ravel()})
                                 
                                 



print("Processing 2030-2045")
# ------------------
# 2030-2045
full_dat_ssp126_2030_2045_dat = pd.read_csv("data/full_dat_ssp126_2030_2045_dat.csv")
full_dat_ssp126_2030_2045_dat.columns = full_dat_ssp126_2030_2045_dat.columns.str.replace("_2030-2045", "")
X_ssp126_2030, y_ssp126_2030 = procShannon(full_dat_ssp126_2030_2045_dat)

full_dat_ssp370_2030_2045_dat = pd.read_csv("data/full_dat_ssp370_2030_2045_dat.csv")
full_dat_ssp370_2030_2045_dat.columns = full_dat_ssp370_2030_2045_dat.columns.str.replace("_2030-2045", "")
X_ssp370_2030, y_ssp370_2030 = procShannon(full_dat_ssp370_2030_2045_dat)

full_dat_ssp585_2030_2045_dat = pd.read_csv("data/full_dat_ssp585_2030_2045_dat.csv")
full_dat_ssp585_2030_2045_dat.columns = full_dat_ssp585_2030_2045_dat.columns.str.replace("_2030-2045", "")
X_ssp585_2030, y_ssp585_2030 = procShannon(full_dat_ssp585_2030_2045_dat)

y_ssp126_pred_2030 = clf.predict(X_ssp126_2030.drop(columns=['lat', 'lon']))
y_ssp370_pred_2030 = clf.predict(X_ssp370_2030.drop(columns=['lat', 'lon']))
y_ssp585_pred_2030 = clf.predict(X_ssp585_2030.drop(columns=['lat', 'lon']))

ssp126_pred_2030 = pd.DataFrame({'lat': X_ssp126_2030['lat'], 'lon': X_ssp126_2030['lon'], 'y_ssp126_pred_2030': y_ssp126_pred_2030.ravel()})
ssp370_pred_2030 = pd.DataFrame({'lat': X_ssp370_2030['lat'], 'lon': X_ssp370_2030['lon'], 'y_ssp370_pred_2030': y_ssp370_pred_2030.ravel()})
ssp585_pred_2030 = pd.DataFrame({'lat': X_ssp585_2030['lat'], 'lon': X_ssp585_2030['lon'], 'y_ssp585_pred_2030': y_ssp585_pred_2030.ravel()})





print("Processing 2045-2060")
# ------------------
# 2045-2060
full_dat_ssp126_2045_2060_dat = pd.read_csv("data/full_dat_ssp126_2045_2060_dat.csv")
full_dat_ssp126_2045_2060_dat.columns = full_dat_ssp126_2045_2060_dat.columns.str.replace("_2045-2060", "")
X_ssp126_2045, y_ssp126_2045 = procShannon(full_dat_ssp126_2045_2060_dat)

full_dat_ssp370_2045_2060_dat = pd.read_csv("data/full_dat_ssp370_2045_2060_dat.csv")
full_dat_ssp370_2045_2060_dat.columns = full_dat_ssp370_2045_2060_dat.columns.str.replace("_2045-2060", "")
X_ssp370_2045, y_ssp370_2045 = procShannon(full_dat_ssp370_2045_2060_dat)

full_dat_ssp585_2045_2060_dat = pd.read_csv("data/full_dat_ssp585_2045_2060_dat.csv")
full_dat_ssp585_2045_2060_dat.columns = full_dat_ssp585_2045_2060_dat.columns.str.replace("_2045-2060", "")
X_ssp585_2045, y_ssp585_2045 = procShannon(full_dat_ssp585_2045_2060_dat)

y_ssp126_pred_2045 = clf.predict(X_ssp126_2045.drop(columns=['lat', 'lon']))
y_ssp370_pred_2045 = clf.predict(X_ssp370_2045.drop(columns=['lat', 'lon']))
y_ssp585_pred_2045 = clf.predict(X_ssp585_2045.drop(columns=['lat', 'lon']))

ssp126_pred_2045 = pd.DataFrame({'lat': X_ssp126_2045['lat'], 'lon': X_ssp126_2045['lon'], 'y_ssp126_pred_2045': y_ssp126_pred_2045.ravel()})
ssp370_pred_2045 = pd.DataFrame({'lat': X_ssp370_2045['lat'], 'lon': X_ssp370_2045['lon'], 'y_ssp370_pred_2045': y_ssp370_pred_2045.ravel()})
ssp585_pred_2045 = pd.DataFrame({'lat': X_ssp585_2045['lat'], 'lon': X_ssp585_2045['lon'], 'y_ssp585_pred_2045': y_ssp585_pred_2045.ravel()})




print("Processing 2060-2075")
# ------------------
# 2060-2075
full_dat_ssp126_2060_2075_dat = pd.read_csv("data/full_dat_ssp126_2060_2075_dat.csv")
full_dat_ssp126_2060_2075_dat.columns = full_dat_ssp126_2060_2075_dat.columns.str.replace("_2060-2075", "")
X_ssp126_2060, y_ssp126_2060 = procShannon(full_dat_ssp126_2060_2075_dat)

full_dat_ssp370_2060_2075_dat = pd.read_csv("data/full_dat_ssp370_2060_2075_dat.csv")
full_dat_ssp370_2060_2075_dat.columns = full_dat_ssp370_2060_2075_dat.columns.str.replace("_2060-2075", "")
X_ssp370_2060, y_ssp370_2060 = procShannon(full_dat_ssp370_2060_2075_dat)

full_dat_ssp585_2060_2075_dat = pd.read_csv("data/full_dat_ssp585_2060_2075_dat.csv")
full_dat_ssp585_2060_2075_dat.columns = full_dat_ssp585_2060_2075_dat.columns.str.replace("_2060-2075", "")
X_ssp585_2060, y_ssp585_2060 = procShannon(full_dat_ssp585_2060_2075_dat)

y_ssp126_pred_2060 = clf.predict(X_ssp126_2060.drop(columns=['lat', 'lon']))
y_ssp370_pred_2060 = clf.predict(X_ssp370_2060.drop(columns=['lat', 'lon']))
y_ssp585_pred_2060 = clf.predict(X_ssp585_2060.drop(columns=['lat', 'lon']))

ssp126_pred_2060 = pd.DataFrame({'lat': X_ssp126_2060['lat'], 'lon': X_ssp126_2060['lon'], 'y_ssp126_pred_2060': y_ssp126_pred_2060.ravel()})
ssp370_pred_2060 = pd.DataFrame({'lat': X_ssp370_2060['lat'], 'lon': X_ssp370_2060['lon'], 'y_ssp370_pred_2060': y_ssp370_pred_2060.ravel()})
ssp585_pred_2060 = pd.DataFrame({'lat': X_ssp585_2060['lat'], 'lon': X_ssp585_2060['lon'], 'y_ssp585_pred_2060': y_ssp585_pred_2060.ravel()})




print("Processing 2075-2090")
# ------------------
# 2075-2090
full_dat_ssp126_2075_2090_dat = pd.read_csv("data/full_dat_ssp126_2075_2090_dat.csv")
full_dat_ssp126_2075_2090_dat.columns = full_dat_ssp126_2075_2090_dat.columns.str.replace("_2075-2090", "")
X_ssp126_2075, y_ssp126_2075 = procShannon(full_dat_ssp126_2075_2090_dat)

full_dat_ssp370_2075_2090_dat = pd.read_csv("data/full_dat_ssp370_2075_2090_dat.csv")
full_dat_ssp370_2075_2090_dat.columns = full_dat_ssp370_2075_2090_dat.columns.str.replace("_2075-2090", "")
X_ssp370_2075, y_ssp370_2075 = procShannon(full_dat_ssp370_2075_2090_dat)

full_dat_ssp585_2075_2090_dat = pd.read_csv("data/full_dat_ssp585_2075_2090_dat.csv")
full_dat_ssp585_2075_2090_dat.columns = full_dat_ssp585_2075_2090_dat.columns.str.replace("_2075-2090", "")
X_ssp585_2075, y_ssp585_2075 = procShannon(full_dat_ssp585_2075_2090_dat)

y_ssp126_pred_2075 = clf.predict(X_ssp126_2075.drop(columns=['lat', 'lon']))
y_ssp370_pred_2075 = clf.predict(X_ssp370_2075.drop(columns=['lat', 'lon']))
y_ssp585_pred_2075 = clf.predict(X_ssp585_2075.drop(columns=['lat', 'lon']))

ssp126_pred_2075 = pd.DataFrame({'lat': X_ssp126_2075['lat'], 'lon': X_ssp126_2075['lon'], 'y_ssp126_pred_2075': y_ssp126_pred_2075.ravel()})
ssp370_pred_2075 = pd.DataFrame({'lat': X_ssp370_2075['lat'], 'lon': X_ssp370_2075['lon'], 'y_ssp370_pred_2075': y_ssp370_pred_2075.ravel()})
ssp585_pred_2075 = pd.DataFrame({'lat': X_ssp585_2075['lat'], 'lon': X_ssp585_2075['lon'], 'y_ssp585_pred_2075': y_ssp585_pred_2075.ravel()})



savedat = pd.DataFrame({'lat': X_lat, 
                        'lon': X_lon,
                        'y_true_historical': y, 
                        'y_pred_historical': y_hist_pred_2015.ravel()})

savedat = savedat.merge(ssp126_pred_2015, on=['lat', 'lon'])
savedat = savedat.merge(ssp370_pred_2015, on=['lat', 'lon'])
savedat = savedat.merge(ssp585_pred_2015, on=['lat', 'lon'])

savedat = savedat.merge(ssp126_pred_2030, on=['lat', 'lon'])
savedat = savedat.merge(ssp370_pred_2030, on=['lat', 'lon'])
savedat = savedat.merge(ssp585_pred_2030, on=['lat', 'lon'])

savedat = savedat.merge(ssp126_pred_2045, on=['lat', 'lon'])
savedat = savedat.merge(ssp370_pred_2045, on=['lat', 'lon'])
savedat = savedat.merge(ssp585_pred_2045, on=['lat', 'lon'])

savedat = savedat.merge(ssp126_pred_2060, on=['lat', 'lon'])
savedat = savedat.merge(ssp370_pred_2060, on=['lat', 'lon'])
savedat = savedat.merge(ssp585_pred_2060, on=['lat', 'lon'])

savedat = savedat.merge(ssp126_pred_2075, on=['lat', 'lon'])
savedat = savedat.merge(ssp370_pred_2075, on=['lat', 'lon'])
savedat = savedat.merge(ssp585_pred_2075, on=['lat', 'lon'])


# savedat = savedat.assign(ssp585_pred_2075_diff = savedat['y_ssp585_pred_2075'] - savedat['y_pred_historical'])

print(np.sum(savedat['y_pred_historical']))
print(np.sum(savedat['y_ssp126_pred_2075']))
print(np.sum(savedat['y_ssp370_pred_2075']))
print(np.sum(savedat['y_ssp585_pred_2075']))

# Pandas dataframe
print("Saving: 'data/RF_shannon_div_classification_model_results.csv'")
savedat.to_csv('data/RF_shannon_div_classification_model_results.csv', index=False)
# savedat = pd.read_csv('data/NN_fishing_effort_regression_model_results.csv')

print("Saving: 'data/RF_shannon_div_classification_model_results.nc'")
savedat.melt(id_vars=['lat', 'lon']).set_index(['lat', 'lon', 'variable']).to_xarray().to_netcdf('data/RF_shannon_div_classification_model_results.nc')

