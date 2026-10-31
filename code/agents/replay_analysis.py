"""
Study 3 replay analysis: the archived incident stream judged twice by a language model
(with the resident's history at the place, and as a first visit), under actual and flipped
aesthetic scores, at invitation strength 1 (the model as run) and 0 (the calibrated value).

Used by notebooks/04_study3_generative_model_archived_runs.ipynb. numpy, scipy and pandas only.

Units and terms
  incident        one noticed problem (the analysis unit)
  judged share    the language model's answer to "out of 100 residents like this one, how many
                  would ...", stored on 0-1; reported in points (x 100)
  history effect  judged share with the resident's history minus the same incident judged as a
                  first visit (delta)
  vis_real        the place's actual aesthetic score (0-1), also under flipped scores
  arm             'observed' = actual aesthetic scores; 'swap' = flipped within each ZIP code
  fam             visit count at the place, counting the current visit; k_prior = problems the
                  resident had noticed there on earlier visits
Outcomes
  civic_action  mean of signup (volunteer for a clean-up) and gather (join residents doing
                something positive for the area): the Study 2a civic-action counterpart
  dosomething   do something about it today: the informal-social-control counterpart
  report        stop and file a 311 report now: the Study 1 reporting counterpart
  steward       deal with it oneself (three fixable problem types): the stewardship counterpart
"""
import json
import numpy as np
import pandas as pd
from scipy import stats

OUTCOMES = ['civic_action', 'dosomething', 'report', 'steward', 'signup', 'gather']
LABEL = {'civic_action': 'Civic action (volunteer for a clean-up; join residents)',
         'dosomething': 'Informal social control (do something about it today)',
         'report': 'Reporting (file a 311 report now)',
         'steward': 'Dealing with it oneself (three fixable problems)',
         'signup': '  volunteer for a clean-up in the area',
         'gather': '  join residents doing something positive for the area'}
COUNTERPART = {'civic_action': 'Study 2a civic action', 'dosomething': 'Study 2a informal social control',
               'report': 'Study 1 reporting (311)', 'steward': 'Study 1 stewardship subset',
               'signup': 'Study 1 volunteering; Study 2a item 10', 'gather': 'Study 1 civic organization; Study 2a item 6'}
SCORING = {'observed': 'actual', 'swap': 'flipped'}


# ---------------------------------------------------------------- estimation
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


# ---------------------------------------------------------------- data
def load_results(path):
    r = pd.read_csv(path, low_memory=False)
    r['swap'] = (r.arm == 'swap').astype(int)
    r['public'] = (r.kind == 'public').astype(int)
    for c in ['shown', 'first', 'unspec']:
        if f'signup_{c}' in r and f'gather_{c}' in r:
            r[f'civic_action_{c}'] = r[[f'signup_{c}', f'gather_{c}']].mean(axis=1)
    if 'k_signup_shown' in r:
        r['k_civic_action_shown'] = r['k_signup_shown'] + '|' + r['k_gather_shown']
    if 'civic_action_shown_sd' not in r and 'signup_shown_sd' in r:
        r['civic_action_shown_sd'] = r[['signup_shown_sd', 'gather_shown_sd']].mean(axis=1)
    return r


def _pub(r, outcome, control='first'):
    d = r[r.public == 1].dropna(subset=[f'{outcome}_shown', f'{outcome}_{control}']).copy()
    d['delta'] = d[f'{outcome}_shown'] - d[f'{outcome}_{control}']
    d['has_history'] = (d.fam > 1) | (d.k_prior > 0)
    d['vis_x_swap'] = d.vis_real * d.swap
    return d


# ---------------------------------------------------------------- blocks
def link1(r):
    """Inviting places draw residents back: visits at a noticed problem on the place's score, by scoring."""
    rows, curves = [], {}
    for arm in ['observed', 'swap']:
        g = r[r.arm == arm].dropna(subset=['fam', 'vis_real'])
        m = fit(g, 'fam', ['vis_real']).loc['vis_real']
        rows.append(dict(scoring=SCORING[arm], b=m.b, se=m.se, p=m.p, n=int(m.n)))
        q = g.assign(bin=pd.qcut(g.vis_real, 5, labels=False, duplicates='drop')).groupby('bin')
        curves[SCORING[arm]] = [dict(x=float(s.vis_real.mean()), y=float(s.fam.mean()), se=float(s.fam.std(ddof=1) / np.sqrt(len(s))))
                                for _, s in q]
    return pd.DataFrame(rows), curves


def probes(r):
    o = r.dropna(subset=['ownership']).copy()
    by_fam = o.groupby('fam').ownership.agg(['mean', 'size']).reset_index()
    by_fam_k = o.groupby(['fam', 'k_prior']).ownership.agg(['mean', 'size']).reset_index()
    m = fit(o, 'ownership', ['fam', 'k_prior'])
    by_arm = {SCORING[a]: fit(o[o.arm == a], 'ownership', ['vis_real']).loc['vis_real'] for a in ['observed', 'swap']}
    out = dict(by_fam=by_fam, by_fam_k=by_fam_k, fam_b=m.loc['fam', 'b'], fam_p=m.loc['fam', 'p'],
               k_b=m.loc['k_prior', 'b'], k_p=m.loc['k_prior', 'p'],
               vis={k: dict(b=v.b, se=v.se, p=v.p) for k, v in by_arm.items()})
    if 'responsibility' in o and o.responsibility.notna().any():
        out['resp_by_fam'] = o.groupby('fam').responsibility.mean().reset_index()
        out['own_resp_corr'] = float(o.ownership.corr(o.responsibility))
    ob = r[r.arm == 'observed'].dropna(subset=['ownership'])
    out['by_day'] = ob.groupby('day').ownership.mean().reset_index()
    return out


def history_effects(r, control='first'):
    rows, curves = [], {}
    for oc in OUTCOMES:
        d = _pub(r, oc, control)
        h = d[d.has_history]
        m = fit(d, 'delta', ['fam', 'k_prior'])
        hi = d[d.fam >= 6]
        rows.append(dict(outcome=oc, label=LABEL[oc], counterpart=COUNTERPART[oc], n=len(d),
                         control_mean=100 * d[f'{oc}_{control}'].mean(), delta_with_history=100 * h.delta.mean(),
                         delta_sd=100 * h.delta.std(ddof=1), share_delta_positive=(h.delta > 0).mean(),
                         per_visit=100 * m.loc['fam', 'b'], per_visit_p=m.loc['fam', 'p'],
                         per_prior_problem=100 * m.loc['k_prior', 'b'], per_prior_problem_p=m.loc['k_prior', 'p'],
                         visits6plus_control=100 * hi[f'{oc}_{control}'].mean(), visits6plus_with_history=100 * hi[f'{oc}_shown'].mean(),
                         n_visits6plus=len(hi)))
        c = d.groupby('fam').agg(delta=('delta', 'mean'), se=('delta', lambda s: s.std(ddof=1) / np.sqrt(len(s))), n=('delta', 'size'))
        curves[oc] = [dict(fam=int(f), delta=100 * v.delta, se=100 * (0 if np.isnan(v.se) else v.se), n=int(v.n)) for f, v in c.iterrows()]
    return pd.DataFrame(rows), curves


def chain_test(r, control='first', cluster='zip'):
    """Effect of the place's aesthetic score on each judgment and on the returning effect (with history minus
    as a first visit), by scoring; and the change in the returning-effect slope when the scores are flipped
    (coefficient on score x flipped in a model of the returning effect)."""
    slopes, diffs = [], []
    for oc in OUTCOMES:
        d = _pub(r, oc, control)
        cl = 'zip' if cluster == 'zip' else f'k_{oc}_shown' if f'k_{oc}_shown' in d else 'zip'
        for arm in ['observed', 'swap']:
            g = d[d.arm == arm]
            for cond, col in [('first visit', f'{oc}_{control}'), ('with history', f'{oc}_shown'), ('returning effect', 'delta')]:
                m = fit(g, col, ['vis_real'], cluster=cl).loc['vis_real']
                slopes.append(dict(outcome=oc, scoring=SCORING[arm], judgment=cond, b=100 * m.b, se=100 * m.se, p=m.p, n=int(m.n)))
        m = fit(d, 'delta', ['vis_real', 'swap', 'vis_x_swap'], cluster=cl)
        eff = {s['scoring']: s for s in slopes if s['outcome'] == oc and s['judgment'] == 'returning effect'}
        row = dict(outcome=oc, label=LABEL[oc], n=len(d), slope_actual=100 * m.loc['vis_real', 'b'], slope_actual_p=eff['actual']['p'],
                   slope_flipped=eff['flipped']['b'], slope_flipped_p=eff['flipped']['p'],
                   change_when_flipped=100 * m.loc['vis_x_swap', 'b'], change_se=100 * m.loc['vis_x_swap', 'se'], change_p=m.loc['vis_x_swap', 'p'])
        s42 = d[d.seed == 42]
        if s42.swap.nunique() == 2:
            m42 = fit(s42, 'delta', ['vis_real', 'swap', 'vis_x_swap'], cluster=cl)
            row.update(change_when_flipped_seed42=100 * m42.loc['vis_x_swap', 'b'], change_p_seed42=m42.loc['vis_x_swap', 'p'],
                       slope_actual_seed42=100 * m42.loc['vis_real', 'b'])
            for arm in ['observed', 'swap']:
                mm = fit(s42[s42.arm == arm], f'{oc}_shown', ['vis_real'], cluster=cl).loc['vis_real']
                row[f'with_history_{SCORING[arm]}_seed42'] = 100 * mm.b; row[f'with_history_{SCORING[arm]}_seed42_p'] = mm.p
        diffs.append(row)
    return pd.DataFrame(slopes), pd.DataFrame(diffs)


def mediation_by_visits(r, outcome='civic_action'):
    """Under actual scores, the score's association with the history effect before and after
    adjusting for the visit count and for the number of problems noticed before."""
    d = _pub(r, outcome); d = d[d.arm == 'observed']
    rows = []
    for adj in ([], ['fam'], ['k_prior'], ['fam', 'k_prior']):
        m = fit(d, 'delta', ['vis_real'] + adj).loc['vis_real']
        rows.append(dict(outcome=outcome, adjusted_for=' + '.join(adj) or 'none', b=100 * m.b, se=100 * m.se, p=m.p, n=int(m.n)))
    return pd.DataFrame(rows)


def within_zip(r, outcome):
    """Slopes with score and outcome demeaned by ZIP code and scoring (places within the same area)."""
    d = _pub(r, outcome)
    rows = []
    for arm in ['observed', 'swap']:
        g = d[d.arm == arm].copy()
        for col in [f'{outcome}_shown', f'{outcome}_first']:
            g[col + '_dm'] = g[col] - g.groupby('zip')[col].transform('mean')
        g['vis_dm'] = g.vis_real - g.groupby('zip').vis_real.transform('mean')
        for cond, col in [('first visit', f'{outcome}_first_dm'), ('with history', f'{outcome}_shown_dm')]:
            m = fit(g, col, ['vis_dm']).loc['vis_dm']
            rows.append(dict(outcome=outcome, scoring=SCORING[arm], judgment=cond, b=100 * m.b, se=100 * m.se, p=m.p))
    return pd.DataFrame(rows)


def per_seed(r, outcome='civic_action'):
    d = _pub(r, outcome); d = d[d.arm == 'observed']
    rows = []
    for s, g in d.groupby('seed'):
        m = fit(g, f'{outcome}_shown', ['vis_real']).loc['vis_real']
        rows.append(dict(outcome=outcome, seed=int(s), b=100 * m.b, se=100 * m.se, p=m.p, n=int(m.n)))
    return pd.DataFrame(rows)


def private_problems(r):
    rows = []
    for oc in ['civic_action', 'dosomething', 'report']:
        d = r[r.public == 0].dropna(subset=[f'{oc}_shown', f'{oc}_first']).copy()
        d['delta'] = d[f'{oc}_shown'] - d[f'{oc}_first']; d['vis_x_swap'] = d.vis_real * d.swap
        h = d[(d.fam > 1) | (d.k_prior > 0)]
        m = fit(d, 'delta', ['vis_real', 'swap', 'vis_x_swap'])
        rows.append(dict(outcome=oc, n=len(d), control_mean=100 * d[f'{oc}_first'].mean(), delta_with_history=100 * h.delta.mean(),
                         change_when_flipped=100 * m.loc['vis_x_swap', 'b'], change_p=m.loc['vis_x_swap', 'p']))
    return pd.DataFrame(rows)


def within_prompt_sd(r):
    return {oc: 100 * r[f'{oc}_shown_sd'].mean() for oc in OUTCOMES if f'{oc}_shown_sd' in r and r[f'{oc}_shown_sd'].notna().any()}


def record_hidden(r):
    """The no-line variant of the reporting prompt, kept for the record."""
    g = r.dropna(subset=['report_hidden', 'report_first'])
    out = dict(first_minus_hidden=100 * (g.report_first - g.report_hidden).mean(), n=len(g))
    if 'propensity' in r:
        o = r.dropna(subset=['propensity', 'report_shown'])
        out['corr_with_original_rule_propensity'] = float(o.report_shown.corr(o.propensity))
    return out


# ---------------------------------------------------------------- everything for the notebook
def analyse(r1, r0=None):
    """r1: results at invitation strength 1; r0: at strength 0. Returns tables and a JSON-ready summary."""
    T = {}
    T['link1'], link1_curves = link1(r1)
    P = probes(r1)
    T['S20B_history_effects'], hist_curves = history_effects(r1)
    T['S20C_slopes'], T['S20C_change'] = chain_test(r1)
    T['S20C_mediation'] = pd.concat([mediation_by_visits(r1, 'civic_action'), mediation_by_visits(r1, 'dosomething')])
    T['S21B_within_zip'] = pd.concat([within_zip(r1, 'civic_action'), within_zip(r1, 'dosomething')])
    T['S21B_per_seed'] = pd.concat([per_seed(r1, 'civic_action'), per_seed(r1, 'dosomething')])
    T['S21B_private'] = private_problems(r1)
    unspec_h, _ = history_effects(r1, control='unspec'); _, unspec_c = chain_test(r1, control='unspec')
    T['S21B_unspec_control'] = unspec_h[['outcome', 'delta_with_history']].merge(unspec_c[['outcome', 'change_when_flipped', 'change_p']], on='outcome')
    _, prompt_c = chain_test(r1, cluster='prompt')
    T['S21B_prompt_cluster'] = prompt_c[['outcome', 'change_when_flipped', 'change_se', 'change_p']]
    S = dict(n=dict(total=int(len(r1)), public=int(r1.public.sum()), arm_seed={f'{a}_{s}': int(n) for (a, s), n in r1.groupby(['arm', 'seed']).size().items()}),
             link1=dict(estimates=T['link1'].to_dict('records'), curves=link1_curves),
             probes=dict(by_fam=P['by_fam'].to_dict('records'), by_fam_k=P['by_fam_k'].to_dict('records'), by_day=P['by_day'].to_dict('records'),
                         fam_b=P['fam_b'], fam_p=P['fam_p'], k_b=P['k_b'], k_p=P['k_p'], vis=P['vis'],
                         resp_by_fam=P.get('resp_by_fam', pd.DataFrame()).to_dict('records'), own_resp_corr=P.get('own_resp_corr')),
             history=dict(table=T['S20B_history_effects'].to_dict('records'), curves=hist_curves),
             chain=dict(slopes=T['S20C_slopes'].to_dict('records'), change=T['S20C_change'].to_dict('records'), mediation=T['S20C_mediation'].to_dict('records')),
             within_prompt_sd=within_prompt_sd(r1), record=record_hidden(r1))
    if r0 is not None:
        T['S21A_link1'], curves0 = link1(r0); P0 = probes(r0)
        T['S21A_history_effects'], _ = history_effects(r0)
        T['S21A_slopes'], T['S21A_change'] = chain_test(r0)
        S['strength0'] = dict(n=dict(total=int(len(r0)), public=int(r0.public.sum()), arm_seed={f'{a}_{s}': int(n) for (a, s), n in r0.groupby(['arm', 'seed']).size().items()}),
                              link1=dict(estimates=T['S21A_link1'].to_dict('records'), curves=curves0),
                              probes=dict(by_fam=P0['by_fam'].to_dict('records'), vis=P0['vis']),
                              history=T['S21A_history_effects'].to_dict('records'),
                              chain=dict(slopes=T['S21A_slopes'].to_dict('records'), change=T['S21A_change'].to_dict('records')))
    return T, S


def dump_json(S, path):
    def conv(o):
        if isinstance(o, (np.floating, np.integer)): return o.item()
        if isinstance(o, np.ndarray): return o.tolist()
        raise TypeError(type(o))
    json.dump(S, open(path, 'w'), indent=1, default=conv)


def signed(b, digits=2):
    return ('+' if b >= 0 else '−') + f'{abs(b):.{digits}f}'


def pfmt(p):
    return '< .001' if p < .001 else f'{p:.3f}'.lstrip('0') if p >= .01 else f'{p:.3f}'.lstrip('0')
