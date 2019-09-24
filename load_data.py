"""
@author: Kostas Hatalis
"""

import pandas as pd

def load_data(experiment, task = 1, zone = 1, training_months = 0):
    '''
    Load in data for specific task+zone. Limit number of training months if requested.

    Parameters:
        experiment (dict): dictionary with training and testing dataframes.
        task (int): specific task to create training/testing sets for.
        zone (int): specific wind farm to load in.
        training_months (int): number of months to limit training data. If 0, use all data.

    Returns:
        experiment (dict): dictionary with training and testing dataframes.
    '''

    # load in the data (based on zone)
    dateparse = lambda dates: pd.datetime.strptime(dates, '%Y%m%d %H:%M')
    df = pd.read_csv('data\zone' + str(zone) + '.csv', parse_dates=['TIMESTAMP'], index_col=1, date_parser=dateparse)

    # fill all missing values with linear interpolation
    df = df.interpolate()

    # prepare training df (based on task)
    df_base = df[df.index.year.isin([2012])]
    if task > 1:
        df_extra = df[df.index.month.isin(list(range(1,task))) & df.index.year.isin([2013])]
        df_train = df_base.append(df_extra, ignore_index=False)
    else:
        df_train = df_base

    # preprare testing df (based on task)
    df_test = df[df.index.month.isin([task]) & df.index.year.isin([2013])]

    # limit number of training months if requested
    if training_months > 0:
        df_train = df_train.last(str(training_months)+'M')

    # save new dataframes to experiment dictionary
    experiment['df_train'] = df_train
    experiment['df_test'] = df_test
    experiment['N_tau'] = len(experiment['tau'])  # num of quantiles
    experiment['N_PI'] = int(experiment['N_tau'] / 2)  # num of prediction intervals

    return experiment