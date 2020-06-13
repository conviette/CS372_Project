import csv, json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.sandbox.regression.predstd import wls_prediction_std

STAT_NAME = 'GDP_GROWTH'

def main():

    with open('..\\data\\president_scores.json') as f:
        emph = json.load(f)

    ind_df = pd.read_csv('..\\data\\Economic_indicators.csv')


    years = dict(map(lambda x:(int(x[:4]), x), emph.keys()))
    print(years)
    results = ind_df[ind_df.YEAR.isin(years)].reset_index(drop=True)
    features = pd.DataFrame.from_dict(emph, orient='index').reset_index(drop=True).astype(float)

    print(features.head())

    #run linear regression for bias score vs statistics
    X = sm.add_constant(features)
    Y = results[STAT_NAME].astype(float)
    model = sm.OLS(Y, X).fit()
    print('p value: ', model.f_pvalue)
    print('r squared: ', model.rsquared)
    print(model.summary())
    with open('..\\results\\{}_results.txt'.format(STAT_NAME), 'w') as f:
        f.write('Total number of data points: {}\n'.format(len(years)))
        f.write('p value: {}\n'.format(model.f_pvalue))
        f.write('r squared: {}\n'.format(model.rsquared))
        f.write(str(model.params))



if __name__ == '__main__':
    main()