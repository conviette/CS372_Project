
import csv, json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.sandbox.regression.predstd import wls_prediction_std

STAT_NAME = 'UNEMPLOYMENT' ##'GDP_GROWTH' or 'EXPORT_INDEX' or 'UNEMPLOYMENT'
CORPUS_NAME = 'sotus' ##'sotus' or 'inaugural' or 'oral'

def main():

    with open('..\\data\\{}_train_scores.json'.format(CORPUS_NAME)) as f:
        emph = json.load(f)

    ind_df = pd.read_csv('..\\data\\Economic_indicators.csv')


    years = dict(map(lambda x:(int(x[:4]), x), emph.keys()))
    results = ind_df[ind_df.YEAR.isin(years)].reset_index(drop=True)
    emph = dict((x, emph[x]) for x in emph.keys() if int(x) in list(map(int, results.YEAR)))
    print(len(emph))
    features = pd.DataFrame.from_dict(emph, orient='index').reset_index(drop=True).astype(float)

    print(features.head())

    #run linear regression for bias score vs statistics
    pdict = dict()
    for feature in features.columns:
        X = sm.add_constant(features[feature])
        Y = results[STAT_NAME].astype(float)
        model = sm.OLS(Y, X).fit()
        if model.f_pvalue != np.nan:
            pdict[feature] = model.f_pvalue
    print(model.summary())
    features = sorted(list(pdict.keys()), key=lambda x:pdict[x])[:100]
    for key in features:
        print(key, pdict[key])




if __name__ == '__main__':
    main()
