"""Original Study 2a model block wrapped as a callable; file I/O removed."""
import pandas as pd
import statsmodels.formula.api as smf

def zscore(s):
    s = pd.to_numeric(s, errors='coerce')
    return (s - s.mean()) / s.std(ddof=0)


def fit_felt_ownership(survey):
    """Fit the original model to an authorized precomputed survey frame.

    This wrapper supplies the original notebook globals as explicit inputs.
    It is not called by the public result-preview notebook.
    """
    OWNERSHIP = ['place_attachment', 'city_belonging', 'neighbourhood_belonging']
    PREDICTORS_B = ['perceived_aesthetics', 'crime_safety', 'access_protection', 'traffic_calm', 'walking_access']
    COVARIATES_B = ['age', 'sex', 'education', 'income']


    def ownership_frame(columns):
        d = survey.dropna(subset=columns + OWNERSHIP + COVARIATES_B + ['city_key']).copy()
        for c in columns + OWNERSHIP + COVARIATES_B:
            d[c + '_z'] = zscore(d[c])
        d['felt_ownership'] = zscore(d[[c + '_z' for c in OWNERSHIP]].mean(axis=1))
        return d


    frame_b = ownership_frame(PREDICTORS_B)
    formula = 'felt_ownership ~ ' + ' + '.join(p + '_z' for p in PREDICTORS_B) + ' + ' \
              + ' + '.join(c + '_z' for c in COVARIATES_B) + ' + C(city_key)'
    fit_b = smf.ols(formula, frame_b).fit(cov_type='cluster', cov_kwds={'groups': frame_b.city_key})
    ci = fit_b.conf_int()
    felt = pd.DataFrame({'predictor': PREDICTORS_B,
                         'beta': [fit_b.params[p + '_z'] for p in PREDICTORS_B],
                         'se': [fit_b.bse[p + '_z'] for p in PREDICTORS_B],
                         'ci_low': [ci.loc[p + '_z', 0] for p in PREDICTORS_B],
                         'ci_high': [ci.loc[p + '_z', 1] for p in PREDICTORS_B],
                         'p': [fit_b.pvalues[p + '_z'] for p in PREDICTORS_B], 'n': int(fit_b.nobs)})
    return felt
