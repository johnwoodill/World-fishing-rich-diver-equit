from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from pprint import pprint
import numpy as np
import pandas as pd


def procRichness(dat):
    dat = dat.drop(columns=['fishing_hours', 'H', 'E', 'lat_lon'])
    
    dat = dat.assign(eez = dat['eez'].fillna(0),
                     mpa = dat['mpa'].fillna(0),
                     richness = dat['richness'].fillna(0),
                     lat_x_lon = dat['lat'] * dat['lon'])

    dat = dat.drop(columns='port')

    dat['eez'] = np.where(dat['eez'] == 0, 0, 1)
    dat['mpa'] = np.where(dat['mpa'] == 0, 0, 1)
    
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
    # X = X[X.columns.drop(list(X.filter(regex='gear')))]
    
    # X = preprocessing.scale(X)
        
    return X, y




def procEffortReg(dat):
    dat = dat.drop(columns=['richness', 'lat_lon', 'E', 'H', 'eez', 'mpa', 'rfmo'])
    
    dat = dat.assign(lat_x_lon = dat['lat'] * dat['lon'],
                     mmsi = dat['mmsi'].fillna(0),
                     fishing_hours = dat['fishing_hours'].fillna(0))

    dat = dat.drop(columns='port')

    # dat['eez'] = np.where(dat['eez'] == 0, 0, 1)
    # dat['mpa'] = np.where(dat['mpa'] == 0, 0, 1)
    # dat['rfmo'] = np.where(dat['rfmo'] == 0, 0, 1)
    
    X = dat
    
    X = X.dropna().reset_index(drop=True)
    
    y = X['fishing_hours'] / X['mmsi']
    y = y.fillna(0)
    y = np.log(1 + y)

    X = X.drop(columns=['fishing_hours', 'mmsi'])

    # ### Predictors that reduce model accuracy
    # X = X[X.columns.drop(list(X.filter(regex='gear')))]
    X = X[X.columns.drop(list(X.filter(regex='present')))]
    X = X[X.columns.drop(list(X.filter(regex='skew')))]
    X = X[X.columns.drop(list(X.filter(regex='kurt')))]
            
    return X, y






# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]

# Number of features to consider at every split
max_features = ['auto', 'sqrt']

# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 50, num = 10)]
max_depth.append(None)

# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]

# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]

# Method of selecting samples for training each tree
bootstrap = [True, False]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
pprint(random_grid)




# Use the random grid to search for best hyperparameters
# First create the base model to tune
# clf = RandomForestClassifier()

# full_dat_ssp126_2000_2014_dat = pd.read_csv("data/full_gfw_cmip_dat.csv")
# full_dat_ssp126_2000_2014_dat.columns = full_dat_ssp126_2000_2014_dat.columns.str.replace("_2000-2014", "")
# X_2000, y_2000 = procRichness(full_dat_ssp126_2000_2014_dat)

# # Random search of parameters, using 3 fold cross validation, 
# # search across 100 different combinations, and use all available cores

# rf_random = RandomizedSearchCV(estimator = clf, param_distributions = random_grid, n_iter = 100, cv = 10, verbose=2, random_state=42, n_jobs = -1)
# # Fit the random search model

# rf_random.fit(X_2000, y_2000)


# print(rf_random.best_params_)






clf = RandomForestRegressor()

full_dat_ssp126_2000_2014_dat = pd.read_csv("data/full_gfw_cmip_dat.csv")
full_dat_ssp126_2000_2014_dat.columns = full_dat_ssp126_2000_2014_dat.columns.str.replace("_2000-2014", "")
X_2000, y_2000 = procEffortReg(full_dat_ssp126_2000_2014_dat)

# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores

rf_random = RandomizedSearchCV(estimator = clf, 
                               param_distributions = random_grid, n_iter = 100, 
                               cv = 10, verbose=5, random_state=42, n_jobs = -1,
                               scoring = 'neg_mean_squared_error')
# Fit the random search model

rf_random.fit(X_2000, y_2000)


print(rf_random.best_params_)