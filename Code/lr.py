import csv, json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.sandbox.regression.predstd import wls_prediction_std

STAT_NAME = 'GDP_GROWTH' ##or 'EXPORT_INDEX' or 'UNEMPLOYMENT'
CORPUS_NAME = 'sotus' ##or 'inaugural' or 'oral'

def main():

    with open('..\\data\\{}_scores.json'.format(CORPUS_NAME)) as f:
        emph = json.load(f)

    ind_df = pd.read_csv('..\\data\\Economic_indicators.csv')


    years = dict(map(lambda x:(int(x[:4]), x), emph.keys()))
    results = ind_df[ind_df.YEAR.isin(years)].reset_index(drop=True)
    emph = dict((x, emph[x]) for x in emph.keys() if int(x) in list(map(int, results.YEAR)))
    print(len(emph))
    features = pd.DataFrame.from_dict(emph, orient='index').reset_index(drop=True).astype(float)

    print(features.head())

    #run linear regression for bias score vs statistics
    X = sm.add_constant(features['money'])
    Y = results[STAT_NAME].astype(float)
    model = sm.OLS(Y, X).fit()
    print('p value: ', model.f_pvalue)
    print('r squared: ', model.rsquared)
    print(model.summary())
    with open('..\\results\\{}_{}_results.txt'.format(STAT_NAME, CORPUS_NAME), 'w') as f:
        f.write('Total number of data points: {}\n'.format(len(years)))
        f.write('p value: {}\n'.format(model.f_pvalue))
        f.write('r squared: {}\n'.format(model.rsquared))
        f.write(str(model.params))



if __name__ == '__main__':
    main()
