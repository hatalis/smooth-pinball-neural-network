"""
@author: Kostas Hatalis
"""

import pandas as pd
import numpy as np
from math import pi
from numpy import sqrt
from numpy import cos
from numpy import sin
from sklearn.preprocessing import StandardScaler

def feature_engineering(experiment, scaler = None):
    '''
    Given training/testing dataframes of raw data, engineer features and apply scaling.

    Parameters:
        experiment (dict): dictionary of experiment parameters
            df_train (df): dataframe of raw training data.
            df_test (df): dataframe of raw testing data.
        scale (obj): sklearn scaler.

    Returns:
        experiment (dict): dictionary of experiment parameters.
    '''

    # load in raw train/test dataframes
    df_train = experiment['df_train']
    df_test = experiment['df_test']

    # derive train/test X and y dataframes
    X_train = df_train.drop(['ZONEID','TARGETVAR'], axis=1)
    y_train = df_train['TARGETVAR']
    X_test = df_test.drop(['ZONEID','TARGETVAR'], axis=1)
    y_test = df_test['TARGETVAR']

    # add wind features
    # X_train = wind_features(X_train)
    # X_test = wind_features(X_test)

    # add polynomial features
    # X_train = polynomial_features(X_train)
    # X_test = polynomial_features(X_test)

    # create time features
    X_train = time_features(X_train)
    X_test = time_features(X_test)

    # scale X_train & X_test if needed
    if scaler is not None:
        X_train = feature_scaling(X_train, scaler)
        X_test = feature_scaling(X_test, scaler)

    # save X and y dataframes
    experiment['X_train'] = X_train
    experiment['y_train'] = y_train
    experiment['X_test'] = X_test
    experiment['y_test'] = y_test

    # save data on row/column sizes
    experiment['N_features'] = len(X_test.columns)
    experiment['N_train'] = len(y_train.index)
    experiment['N_test'] = len(y_test.index)

    return experiment

# ============================================================================

def feature_scaling(df, scaler):

    # apply sklearn scaling on features
    data_array = scaler.fit_transform(df)

    # convert back into dataframe
    data_norm = pd.DataFrame(data_array, columns = df.columns, index = df.index)

    return data_norm

# ============================================================================

def time_features(df):

    # extract time/dates
    hour = np.array(df.index.hour)
    day = np.array(df.index.dayofyear)
    # month = np.array(df.index.month)

    # generate time features
    data = {}
    data['hour_cos'] = cos((hour / 24) * 2 * pi)
    data['hour_sin'] = sin((hour / 24) * 2 * pi)
    data['day_cos'] = cos((day / 365) * 2 * pi)
    data['day_sin'] = sin((day / 365) * 2 * pi)
    # data['feature_month_cos'] = cos((month / 12) * 2 * pi)
    # data['feature_month_sin'] = sin((month / 12) * 2 * pi)

    # save features back into df
    features = pd.DataFrame(data,index = df.index)
    df = df.join(features, how='outer')

    return df

# ============================================================================

def polynomial_features(X):

    X_temp = np.array(X)
    X_temp = np.column_stack([X_temp, X_temp ** 2, X_temp ** 3])
    X_temp = StandardScaler().fit_transform(X_temp)
    X = pd.DataFrame(X_temp,index = X.index)

    return X

# ============================================================================

def wind_features(X):

    U10 = np.array(X.U10)
    U100 = np.array(X.U100)
    V10 = np.array(X.V10)
    V100 = np.array(X.V100)

    # wind speed features
    ws10 = sqrt(U10 ** 2 + V10 ** 2)
    ws100 = sqrt(U100 ** 2 + V100 ** 2)

    # wind direction features
    wd10 = (180 / pi) * np.arctan(U10 / V10)
    wd100 = (180 / pi) * np.arctan(U100 / V100)
    wd_diff = wd100-wd10

    # wind energy features
    we10 = 0.5 * ws10 ** 3
    we100 = 0.5 * ws100 ** 3
    we_ratio = we100/we10

    X_temp = np.array(X)
    X_temp = np.column_stack([ws10, ws100, wd10, wd100, we10, we100, wd_diff, we_ratio])
    X = pd.DataFrame(X_temp,index = X.index)

    return X