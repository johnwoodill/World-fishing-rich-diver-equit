
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



def rmse(y_true, y_pred):
	return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))


full_dat = pd.read_csv("data/full_gfw_cmip_dat.csv")




def procCMIP(dat):
    dat = dat.assign(fclass = np.where(dat['fishing_hours'] >= 1, 1, 0),
                            eez = dat['eez'].fillna(0),
                            mpa = dat['mpa'].fillna(0))

    dat = dat.drop(columns='port_name')

    dat['eez'] = np.where(dat['eez'] == 0, 0, 1)
    dat['mpa'] = np.where(dat['mpa'] == 0, 0, 1)

    X = dat.iloc[:, 6:len(dat.columns)]
    X2 = dat.loc[:, 'lat':'lon']
    X = pd.concat([X, X2], axis=1)
    X = X.assign(lat_x_lon = X['lat']*X['lon'])
    X = X.dropna().drop_duplicates().reset_index(drop=True)
    
    na_free = X.dropna()
    X[~X.index.isin(na_free.index)]
    y = X['richness']
    
    # ### Predictors that reduce model accuracy
    X = X[X.columns.drop(list(X.filter(regex='skew')))]
    X = X[X.columns.drop(list(X.filter(regex='kurt')))]
    X = X[X.columns.drop(list(X.filter(regex='gear')))]

    # ### Check count of richness
    # check_y = pd.DataFrame({'y': y, 'count': 1})
    # check_y.groupby('y').count()

    # ### Remove richness above 17
    X = X[X['richness'] <= 17]

    # y = dat['fishing_hours']
    y = X['richness']

    X = X.drop(columns='richness')
    
    X = preprocessing.scale(X)
        
    return X, y



# Subset within WCP
# fdat1 = full_dat[(full_dat['lon'] <= -150 + 360) & (full_dat['lon'] >= 100) & (full_dat['lat'] >= 0)]
# fdat2 = full_dat[(full_dat['lon'] <= -130 + 360) & (full_dat['lon'] >= 140) & (full_dat['lat'] < 0) & (full_dat['lat'] >= -55)]
# fdat3 = full_dat[(full_dat['lon'] <= -130 + 360) & (full_dat['lon'] >= 150) & (full_dat['lat'] <= -55) & (full_dat['lat'] >= -60)]

# Bind data
# reg_dat = pd.concat([fdat1, fdat2, fdat3]).reset_index(drop=True)
# reg_dat = reg_dat.drop_duplicates()

full_dat.columns = full_dat.columns.str.replace("_2000-2014", "")

X, y = procCMIP(full_dat)

# ### Random Forest Classifier
clf = RandomForestClassifier()
np.mean(cross_val_score(clf, X, y, cv=10))



clf_fit = clf.fit(X, y)
y_pred = clf_fit.predict(X)
dat_2000 = pd.DataFrame({'pred_2000_2014': y_pred, 'lat': X['lat'], 'lon': X['lon']})






# ------------------
# 2015-2030
full_dat_ssp126_2015_2030_dat = pd.read_csv("data/full_dat_ssp126_2015_2030_dat.csv")
full_dat_ssp126_2015_2030_dat.columns = full_dat_ssp126_2015_2030_dat.columns.str.replace("_2015-2030", "")
X_2015, y_2015 = procCMIP(full_dat_ssp126_2015_2030_dat)

y_pred_2015 = clf_fit.predict(X_2015)
dat_2015 = pd.DataFrame({'pred_2015_2030': y_pred_2015, 'lat': X_2015['lat'], 'lon': X_2015['lon']})





# ------------------
# 2045-2060
full_dat_ssp126_2045_2060_dat = pd.read_csv("data/full_dat_ssp126_2045_2060_dat.csv")
full_dat_ssp126_2045_2060_dat.columns = full_dat_ssp126_2045_2060_dat.columns.str.replace("_2045-2060", "")
X_2045, y_2045 = procCMIP(full_dat_ssp126_2045_2060_dat)

y_pred_2045 = clf_fit.predict(X_2045)

sum(y_pred)
sum(y_pred_2015)
sum(y_pred_2045)

dat_2045 = pd.DataFrame({'pred_2045_2060': y_pred_2045, 'lat': X_2045['lat'], 'lon': X_2045['lon']})



savedat = dat_2000.merge(dat_2015, on=['lat', 'lon'], how='left')
savedat = savedat.merge(dat_2045, on=['lat', 'lon'], how='left')
savedat.to_csv('data/rf_model_results.csv', index=False)
