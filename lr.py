import csv, json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.sandbox.regression.predstd import wls_prediction_std

STAT_NAME = 'GDP_GROWTH'

def main():

    with open('president_scores.json') as f:
        emph = json.load(f)

    ind_df = pd.read_csv('data\\Economic_indicators.csv')


    years = dict(map(lambda x:(int(x[:4]), x), emph.keys()))
    results = ind_df[ind_df.YEAR.isin(years)].reset_index(drop=True)
    results['emph'] = results.YEAR.map(lambda x:emph[years[x]])

    #run linear regression for bias score vs statistics
    X = sm.add_constant(results.emph.astype(float))
    Y = results[STAT_NAME].astype(float)
    model = sm.OLS(Y, X).fit()
    print('p value: ', model.f_pvalue)
    print('r squared: ', model.rsquared)
    print(model.params)

    with open('{}_results.txt'.format(STAT_NAME), 'w') as f:
        f.write('Total number of data points: {}\n'.format(len(years)))
        f.write('p value: {}\n'.format(model.f_pvalue))
        f.write('r squared: {}\n'.format(model.rsquared))
        f.write(str(model.params))


    #plot result
    plt.rcParams["figure.figsize"] = (20,10)
    xmin = min(results.emph)
    xmax = max(results.emph)
    margin = (xmax-xmin)/10
    xmin -= margin
    xmax += margin
    x = np.linspace(xmin, xmax)
    conf = model.conf_int(0.05) #plotting confidence range
    plt.fill_between(x, conf[0][0]+conf[0][1]*x, conf[1][0]+conf[0][1]*x, color=(0.75, 0.75, 0.75, 1))
    plt.fill_between(x, conf[0][0]+conf[1][1]*x, conf[1][0]+conf[1][1]*x, color=(0.75, 0.75, 0.75, 1))
    plt.fill_between(x, conf[0][0]+conf[0][1]*x, conf[1][0]+conf[1][1]*x, color=(0.75, 0.75, 0.75, 1))
    plt.scatter(results.emph, results[STAT_NAME], color=(0, 0.5, 0.5, 1)) #plot datapoints
    plt.plot(x, model.params[0]+model.params[1]*x, 'r-') #plot model
    plt.grid()
    plt.xlabel('Emphasis in Speech', fontsize=30)
    plt.ylabel(STAT_NAME, fontsize=30)
    for (ind, row) in results.iterrows():
        plt.annotate(years[row['YEAR']], (row['emph'], row[STAT_NAME]))
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.xlim(xmin, xmax)
    plt.show()
    #plt.savefig('Plots\\{}_{}_results.png'.format(lang, statname))

if __name__ == '__main__':
    main()
