"""Selected original research functions; no data or trained parameters included.
Numerical post-processing from replay_analysis.py. No language-model calls,
judgment prompts, agent configurations or archived incident records are included.
"""
import numpy as np
import pandas as pd
from scipy import stats

def cluster_ols(y, X, groups, names):
    """OLS with cluster-robust (CR1) standard errors; P values on t(G-1)."""
    y = np.asarray(y, float); X = np.asarray(X, float); groups = np.asarray(groups)
    n, k = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    b = XtX_inv @ X.T @ y
    e = y - X @ b
    G = np.unique(groups); meat = np.zeros((k, k))
    for g in G:
        idx = groups == g; s = X[idx].T @ e[idx]; meat += np.outer(s, s)
    ng = len(G)
    V = (ng / (ng - 1)) * ((n - 1) / (n - k)) * XtX_inv @ meat @ XtX_inv
    se = np.sqrt(np.diag(V)); t = b / se; p = 2 * stats.t.sf(np.abs(t), df=ng - 1)
    return pd.DataFrame({'b': b, 'se': se, 't': t, 'p': p, 'n': n, 'clusters': ng}, index=names)


def fit(df, y, xs, cluster='zip'):
    X = np.column_stack([np.ones(len(df))] + [df[x].to_numpy(float) for x in xs])
    return cluster_ols(df[y], X, df[cluster], ['const'] + list(xs))


def _pub(r, outcome, control='first'):
    d = r[r.public == 1].dropna(subset=[f'{outcome}_shown', f'{outcome}_{control}']).copy()
    d['delta'] = d[f'{outcome}_shown'] - d[f'{outcome}_{control}']
    d['has_history'] = (d.fam > 1) | (d.k_prior > 0)
    d['vis_x_swap'] = d.vis_real * d.swap
    return d
