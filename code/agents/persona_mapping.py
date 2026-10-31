"""Original survey-item transformation AST; records and file I/O excluded."""
import numpy as np
import pandas as pd

def build_persona_pool(G):

    def _01v(v, scale=None):
        v = pd.to_numeric(v, errors='coerce')
        if scale is not None:
            v = (v - scale[0]) / (scale[1] - scale[0])
        return v.clip(0, 1)

    def uni(col_wp, col_hwb, scale_hwb=(1, 5)):
        hv = _01v(G.get(col_hwb), scale_hwb) if col_hwb and col_hwb in G else pd.Series(np.nan, index=G.index)
        if col_wp and col_wp in G:
            wv = pd.to_numeric(G[col_wp], errors='coerce')
            wv = wv.mask(wv.isin([1, 2]), (wv == 1).astype(float)).clip(0, 1)
        else:
            wv = pd.Series(np.nan, index=G.index)
        return hv.fillna(wv)
    civ = uni(None, 'HWB10')
    pride = uni(None, 'HWB22')
    sat = uni('WP83', 'HWB9')
    learn = uni('WP65', 'HWB2')
    safe = uni('WP113', 'HWB23')
    purp = uni(None, 'HWB1')
    sh = pd.to_numeric(G.get('M19'), errors='coerce').clip(0, 12)
    GPOOL = pd.DataFrame(dict(zip=G['ZIPCODE'].astype(str).str.zfill(5), respondent_id=np.arange(len(G)), prosocial=(2.5 + 2.5 * pd.concat([civ, pride, sat], axis=1).mean(axis=1)).clip(1, 5), attention=(0.3 + 0.5 * pd.concat([learn, safe], axis=1).mean(axis=1)).clip(0.05, 0.95), efficacy=(0.35 + 0.4 * pd.concat([civ, purp, sat], axis=1).mean(axis=1)).clip(0.05, 0.95), busyness=(0.65 - 0.04 * sh.fillna(3)).clip(0.05, 0.95), attach_prior=pd.concat([sat, pride], axis=1).mean(axis=1).fillna(0.5))).dropna(subset=['prosocial'])
    return GPOOL
