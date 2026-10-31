"""Selected original research functions; no data or trained parameters included.

"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import rankdata

def rank_z(x):
    r = rankdata(np.asarray(x, float))
    return (r - r.mean()) / r.std()


def zscore(x):
    x = np.asarray(x, float)
    return (x - x.mean()) / x.std()


def within(v, groups):
    v = pd.Series(np.asarray(v, float), index=groups.index)
    return v - v.groupby(groups).transform('mean')


def state_model(y, x, ctrls, data, cov='HC1'):
    d = data.dropna(subset=[y, x, 'state'] + ctrls)
    d = d[d.state.isin(d.state.value_counts()[lambda v: v >= 5].index)]
    Y = within(rank_z(d[y]), d.state)
    X = pd.DataFrame({'x': within(rank_z(d[x]), d.state)})
    for c in ctrls:
        X[c] = within(zscore(d[c]), d.state)
    fit = sm.OLS(Y, sm.add_constant(X)).fit(cov_type=cov)
    return fit, len(d)
