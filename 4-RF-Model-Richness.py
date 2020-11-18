
import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import cross_val_score

from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.metrics import mean_squared_error
from keras import backend
import keras 
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn import preprocessing
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score


def rmse(y_true, y_pred):
	return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))




def procRichness(dat):
    dat = dat.drop(columns=['fishing_hours', 'H', 'E', 'lat_lon'])
    
    dat = dat.assign(eez = dat['eez'].fillna(0),
                     mpa = dat['mpa'].fillna(0),
                     rfmo = dat['rfmo'].fillna(0),
                     richness = dat['richness'].fillna(0),
                     lat_x_lon = dat['lat'] * dat['lon'])

    dat = dat.drop(columns='port')

    dat['eez'] = np.where(dat['eez'] == 0, 0, 1)
    dat['mpa'] = np.where(dat['mpa'] == 0, 0, 1)
    dat['rfmo'] = np.where(dat['rfmo'] == 0, 0, 1)
    
    X = dat
    
    na_free = X.dropna()
    X[~X.index.isin(na_free.index)]
    
    X = X.dropna().reset_index(drop=True)

    y = X['richness']

    y = np.where(y == 1, 1, y)
    y = np.where(( (y > 1) & (y <= 3) ), 2, y)
    y = np.where(y > 3, 3, y)
    y = pd.Series(y)
    
    # ### Check count of richness
    check_y = pd.DataFrame({'y': y, 'count': 1})
    check_y.groupby('y').count()
    
    X = X.drop(columns='richness')

    # ### Predictors that reduce model accuracy
    X = X[X.columns.drop(list(X.filter(regex='skew')))]
    X = X[X.columns.drop(list(X.filter(regex='kurt')))]
            
    return X, y





# -----------------------------------------------
# ### K-Fold Cross-validation
full_dat = pd.read_csv("data/full_gfw_cmip_dat.csv")

X, y = procRichness(full_dat)

kf = KFold(n_splits=10)
kf.get_n_splits(X)

outdat = pd.DataFrame()
cv = 0
for train_index, test_index in kf.split(X):
    cv = cv + 1
    ### Setup Classifier
    clf = RandomForestClassifier(n_estimators = 1000, min_samples_split = 2, min_samples_leaf = 1,
                             max_features = 'auto', max_depth = 10, bootstrap = False)
    
    ### Get train/test splits
    X_train, X_test = X[X.index.isin(train_index)], X[X.index.isin(test_index)]
    y_train, y_test = y[y.index.isin(train_index)], y[y.index.isin(test_index)]
    
    ### Fit training Set
    clf.fit(X_train, y_train)
    
    ### Predict test set
    y_pred = clf.predict(X_test)    
    
    ### Get accuracy score
    acc_score = accuracy_score(y_test, y_pred)
    
    ### Bind data
    indat = pd.DataFrame({'lat': X_test['lat'], 'lon': X_test['lon'], 'cv': cv, 'y_true': y_test, 'y_pred': y_pred, 'score': acc_score})
    outdat = pd.concat([outdat, indat])
    print(f"[{cv}] Cross-validation complete - Accuracy Score: {round(acc_score, 4)}")

    
    
outdat.to_csv('data/richness_cross_validation_results.csv', index=False)
    
# ### Random Forest Classifier
clf = RandomForestClassifier(n_estimators = 1000, min_samples_split = 2, min_samples_leaf = 1,
                            max_features = 'auto', max_depth = 10, bootstrap = False)
np.mean(cross_val_score(clf, X, y, cv=10))

#
# >>> np.mean(cross_val_score(clf, X, y, cv=10))
# 0.7194471922804744


clf_fit = clf.fit(X_2000, y_2000)
fea_import = pd.DataFrame({'variable': X_2000.columns , 'importance': clf.feature_importances_})

fea_import.sort_values('importance', ascending=False).reset_index(drop=True).iloc[0:20, :]

# ### Results from hyper-parameter tuning
# {'n_estimators': 1000, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'auto', 'max_depth': 10, 'bootstrap': False}


# ------------------
# 2000-2014
full_dat_ssp126_2000_2014_dat = pd.read_csv("data/full_gfw_cmip_dat.csv")
full_dat_ssp126_2000_2014_dat.columns = full_dat_ssp126_2000_2014_dat.columns.str.replace("_2000-2014", "")
X_2000, y_2000 = procRichness(full_dat_ssp126_2000_2014_dat)

clf_fit = clf.fit(X_2000, y_2000)
y_pred_2015 = clf_fit.predict(X_2000)
dat_2000 = pd.DataFrame({'pred_2000_2014': y_pred_2015, 'lat': X_2000['lat'], 'lon': X_2000['lon']})









# ------------------
# 2015-2030
full_dat_ssp126_2015_2030_dat = pd.read_csv("data/full_dat_ssp126_2015_2030_dat.csv")
full_dat_ssp126_2015_2030_dat.columns = full_dat_ssp126_2015_2030_dat.columns.str.replace("_2015-2030", "")
X_2015, y_2015 = procRichness(full_dat_ssp126_2015_2030_dat)

y_pred_2015 = clf_fit.predict(X_2015)
dat_2015 = pd.DataFrame({'pred_2015_2030': y_pred_2015, 'lat': X_2015['lat'], 'lon': X_2015['lon']})




# ------------------
# 2030-2045
full_dat_ssp126_2030_2045_dat = pd.read_csv("data/full_dat_ssp126_2030_2045_dat.csv")
full_dat_ssp126_2030_2045_dat.columns = full_dat_ssp126_2030_2045_dat.columns.str.replace("_2030-2045", "")
X_2030, y_2030 = procRichness(full_dat_ssp126_2030_2045_dat)

y_pred_2030 = clf_fit.predict(X_2030)
dat_2030 = pd.DataFrame({'pred_2030_2045': y_pred_2030, 'lat': X_2030['lat'], 'lon': X_2030['lon']})






# ------------------
# 2045-2060
full_dat_ssp126_2045_2060_dat = pd.read_csv("data/full_dat_ssp126_2045_2060_dat.csv")
full_dat_ssp126_2045_2060_dat.columns = full_dat_ssp126_2045_2060_dat.columns.str.replace("_2045-2060", "")
X_2045, y_2045 = procRichness(full_dat_ssp126_2045_2060_dat)

y_pred_2045 = clf_fit.predict(X_2045)
dat_2045 = pd.DataFrame({'pred_2045_2060': y_pred_2045, 'lat': X_2045['lat'], 'lon': X_2045['lon']})




# ------------------
# 2060-2075
full_dat_ssp126_2060_2075_dat = pd.read_csv("data/full_dat_ssp126_2060_2075_dat.csv")
full_dat_ssp126_2060_2075_dat.columns = full_dat_ssp126_2060_2075_dat.columns.str.replace("_2060-2075", "")
X_2060, y_2060 = procRichness(full_dat_ssp126_2060_2075_dat)

y_pred_2060 = clf_fit.predict(X_2060)
dat_2060 = pd.DataFrame({'pred_2060_2075': y_pred_2060, 'lat': X_2060['lat'], 'lon': X_2060['lon']})




# ------------------
# 2075-2090
full_dat_ssp126_2075_2090_dat = pd.read_csv("data/full_dat_ssp126_2075_2090_dat.csv")
full_dat_ssp126_2075_2090_dat.columns = full_dat_ssp126_2075_2090_dat.columns.str.replace("_2075-2090", "")
X_2075, y_2075 = procRichness(full_dat_ssp126_2075_2090_dat)

y_pred_2075 = clf_fit.predict(X_2075)
dat_2075 = pd.DataFrame({'pred_2075_2090': y_pred_2075, 'lat': X_2075['lat'], 'lon': X_2075['lon']})


sum(full_dat_ssp126_2000_2014_dat['mean_tos'])
sum(full_dat_ssp126_2015_2030_dat['mean_tos'])
sum(full_dat_ssp126_2030_2045_dat['mean_tos'])
sum(full_dat_ssp126_2045_2060_dat['mean_tos'])
sum(full_dat_ssp126_2060_2075_dat['mean_tos'])
sum(full_dat_ssp126_2075_2090_dat['mean_tos'])



savedat = dat_2000.merge(dat_2015, on=['lat', 'lon'], how='left')
savedat = savedat.merge(dat_2030, on=['lat', 'lon'], how='left')
savedat = savedat.merge(dat_2045, on=['lat', 'lon'], how='left')
savedat = savedat.merge(dat_2060, on=['lat', 'lon'], how='left')
savedat = savedat.merge(dat_2075, on=['lat', 'lon'], how='left')


savedat.to_csv('data/rf_richness_model_results.csv', index=False)
