#!/usr/bin/env python3
"""
Study 3 replay: re-decide reporting on the ARCHIVED incident stream with an LLM judgment that is
(a) scene-blind, (b) free of any injected ownership number, and (c) a concrete time trade-off.

Design (2 x 2 on the same events):
  aesthetic arm  : observed | swap      (archived; the visit stream, hence where incidents happen, differs)
  place history  : shown    | first     (same incident; the history line carries the actual visit count and prior noticed
                                         problems, or the structurally matched first-visit control "first visit, no problems
                                         noticed before". A no-line 'hidden' variant is kept for the record only: the battery
                                         placebo showed that merely adding a neutral line shifts the estimate by ~+.04.)

Outcomes (default elicitation: "out of 100 residents like this one, how many would ..."; --elicit logprob = Yes/No
          answered with one token, scored as P(Yes) from the token probabilities — continuous even when every argmax is Yes):
  Outcome set aligned with the paper's own measures (decision of 13 Sep: the study is exploratory; everything is reported in the SM):
  signup      volunteering        sign up for a 2-h clean-up in the area      Study 1 volunteering share; Study 2a "How often did you volunteer for your city?"
  gather      civic organisation  join residents getting together to do something positive for the area   Study 1 civic-organisation density; Study 2a item 6
  civic_action = mean(signup, gather)                                         counterpart of the Study 2a civic-action composite (focal outcome)
  dosomething informal social control  do something about it today          Study 2a informal social control (Sampson et al. 1997), the other focal outcome
  report      311 reporting        stop and file a 311 report now (~5 min)   Study 1 public-space 311 requests (caveated in the paper)
  steward     stewardship          deal with it themselves (~15 min), 3 fixable types   Study 1 stewardship subset; Peck et al. (2021)
  (T3 "chip in $10" was dropped at calibration: public < private, i.e. it did not track the problem.)
Manipulation check: ownership probe (history shown only; never fed back into any decision)

Model: gpt-4o-mini, temperature 0 (same as the archived audit). --model <name> for a robustness re-judgment
(separate --out folder; caches are per model). Prompts are bucketed, so the whole run is a few thousand calls.

Pre-run rule (fixed before any history-shown answer is looked at):
  --stage hidden runs ONLY the history-hidden prompts and prints their spread across disposition x problem.
  The task is accepted if that spread is not degenerate (not ~all 0 or ~all 100). Only the cost/time framing may be
  adjusted at this stage, and every version is kept. The hidden condition contains no manipulation, so this check
  cannot bias the history effect.

Usage:
  export OPENAI_API_KEY=...
  python replay_judgment.py --incidents data/study3/incidents_default.csv.gz --out replay_out --dry-run
  python replay_judgment.py --incidents data/study3/incidents_default.csv.gz --out replay_out --stage hidden
  python replay_judgment.py --incidents data/study3/incidents_default.csv.gz --out replay_out
"""
import argparse, hashlib, json, os, sys, time, re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd, numpy as np

DEFAULT_MODEL = 'gpt-4o-mini'
TEMP = 0
MAX_WORKERS = 16
K_CAP = 5
STEWARD_TYPES = ('blocked_sidewalk', 'fallen_tree', 'illegal_dumping')
TYPES_ALL = ['pothole', 'streetlight_outage', 'signal_malfunction', 'blocked_sidewalk', 'fallen_tree', 'illegal_dumping',
             'loud_music', 'private_noise', 'package_theft', 'property_dispute']

# ---------- phrasing: disposition (tertiles of the archived values) and problem; nothing about the scene/place/ZIP ----------
def eff_phrase(e):
    if e < 0.64: return "generally doubt that reporting problems to the city makes much difference"
    if e < 0.76: return "think that reporting problems to the city sometimes helps"
    return "believe that reporting problems to the city makes a real difference"
def busy_phrase(b):
    if b < 0.40: return "Today they are not in any hurry."
    if b < 0.56: return "Today they are somewhat pressed for time."
    return "Today they are running late for something they cannot easily put off."
PROBLEM = {
    'pothole': 'a deep pothole in the street', 'streetlight_outage': 'a streetlight that is out',
    'signal_malfunction': 'a traffic signal that is not working', 'blocked_sidewalk': 'a sidewalk blocked by debris',
    'fallen_tree': 'a fallen tree across the path', 'illegal_dumping': 'trash and old furniture dumped by the curb',
    'loud_music': 'loud music coming from a party at a house nearby', 'private_noise': 'persistent noise coming from a private residence',
    'package_theft': 'someone taking a package from a neighbour\'s doorstep', 'property_dispute': 'two neighbours arguing over a property line'}
STEWARD_ACTION = {'blocked_sidewalk': 'clearing the debris off the sidewalk', 'fallen_tree': 'dragging the branches out of the way',
                  'illegal_dumping': 'bagging the trash and moving the furniture to the kerb for collection'}

def ordinal(n):
    return f"{n}{'th' if 11 <= n % 100 <= 13 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"
def visit_phrase(fam):
    # archived `fam` counts the current visit (familiarity is incremented before the incident is drawn): fam=1 is the first visit
    return "This is the resident's first visit to this place." if fam <= 1 else f"This is the resident's {ordinal(fam)} visit to this place."
def prior_phrase(kprior):
    if kprior <= 0: return "They have not noticed any problems here before."
    if kprior >= K_CAP: return f"On earlier visits here they noticed at least {K_CAP} other problems."
    return f"On earlier visits here they noticed {kprior} other problem{'s' if kprior != 1 else ''}."

ELICIT = 'freq'   # set from --elicit; 'freq' = "out of 100 residents like this one, how many…"; 'logprob' = Yes/No with token probabilities

def ask_line(action_phrase):
    if ELICIT == 'logprob':
        return f"Would this resident {action_phrase}? Answer with exactly one word: Yes or No."
    return (f"Out of 100 residents like this one, in exactly this situation, how many would {action_phrase}? "
            'Answer with JSON only: {"n_of_100": <integer between 0 and 100>}.')

TASK_LINE = {
    'report':   "Reporting it to the city's 311 service takes about five minutes right there: opening the app, taking a photo, "
                "marking the location and describing the problem. They can stop and file the report now, or keep walking and leave it.",
    'dosomething': None,
    'steward':  None,   # filled per type
    'signup':   "A neighbourhood group is organising a two-hour clean-up and repair session in this area next Saturday morning; "
                "a sign-up sheet is posted nearby.",
    'gather':   "Some residents are getting together this week to do something positive for this area; they have not settled on what yet, "
                "and anyone who joins helps decide. Joining would take about an hour this week.",
}
TASK_ASK = {
    'report': "stop and file the report now",
    'dosomething': "do something about it today, such as reporting it to the city, telling a neighbour or the building, or dealing with it themselves",
    'steward': "stop and take care of it themselves",
    'signup': "sign up today",
    'gather': "join them",
}

UNSPEC_LINE = "Whether the resident has visited this place before is not specified."

def prompt_task(task, typ, eff_b, busy_b, fam=None, kprior=None, unspec=False):
    """fam=None -> no history line (the original 'hidden' variant, kept for the record);
       unspec=True -> alternative control: a same-form line saying the history is not specified;
       fam=1, kprior=0 -> structurally matched first-visit control; otherwise the actual history."""
    lines = ["You are simulating one resident of a large U.S. city.", "",
             f"The resident: they {eff_b}. {busy_b}", "",
             f"Walking through the neighbourhood, they notice {PROBLEM[typ]}."]
    line = TASK_LINE[task]
    if task == 'steward':
        line = (f"Dealing with it themselves would take about fifteen minutes: {STEWARD_ACTION[typ]}, perhaps with help from a passer-by "
                "or a neighbour. They can stop and take care of it now, or keep walking and leave it.")
    if line: lines.append(line)
    if unspec: lines.append(UNSPEC_LINE)
    elif fam is not None: lines.append(f"{visit_phrase(fam)} {prior_phrase(kprior)}")
    lines += ["", ask_line(TASK_ASK[task])]
    return "\n".join(lines)

def prompt_report(typ, eff_b, busy_b, fam=None, kprior=None): return prompt_task('report', typ, eff_b, busy_b, fam, kprior)
def prompt_steward(typ, eff_b, busy_b, fam=None, kprior=None): return prompt_task('steward', typ, eff_b, busy_b, fam, kprior)

def prompt_ownership(fam, kprior):
    return "\n".join(["You are simulating one resident of a large U.S. city.", "",
                      f"{visit_phrase(fam)} {prior_phrase(kprior)}", "",
                      "How much does this place feel like the resident's own place? "
                      'Answer with JSON only: {"ownership_0_100": <integer between 0 and 100>}.'])

def prompt_responsibility(fam, kprior):
    return "\n".join(["You are simulating one resident of a large U.S. city.", "",
                      f"{visit_phrase(fam)} {prior_phrase(kprior)}", "",
                      "How responsible would the resident feel for what happens at this place? "
                      'Answer with JSON only: {"responsibility_0_100": <integer between 0 and 100>}.'])

# ---------- LLM plumbing ----------
def md5(s): return hashlib.md5(s.encode()).hexdigest()

def parse_json(txt, key):
    m = re.search(r'\{.*\}', txt, re.S)
    if not m: return None
    try: v = json.loads(m.group(0)).get(key)
    except Exception: return None
    try: return min(max(float(v), 0.0), 100.0) / 100.0
    except Exception: return None

def make_client():
    from openai import OpenAI
    key = os.environ.get('OPENAI_API_KEY')
    if not key: sys.exit('OPENAI_API_KEY not set')
    return OpenAI(api_key=key)

def p_yes_from_logprobs(choice):
    """P(Yes) from the first-token distribution: mass on Yes-like tokens / (Yes-like + No-like). None if neither appears."""
    import math
    try: tops = choice.logprobs.content[0].top_logprobs
    except Exception: return None
    py = pn = 0.0
    for t in tops:
        tok = t.token.strip().lower()
        if tok.startswith('yes'): py += math.exp(t.logprob)
        elif tok.startswith('no'): pn += math.exp(t.logprob)
    return py / (py + pn) if (py + pn) > 0 else None

def call(client, prompt, key, model, tries=6):
    last = ''
    for t in range(tries):
        try:
            if key == 'yes_no':
                r = client.chat.completions.create(model=model, temperature=TEMP, max_tokens=1, logprobs=True, top_logprobs=20,
                                                   messages=[{'role': 'user', 'content': prompt}])
                txt = r.choices[0].message.content or ''
                return {'value': p_yes_from_logprobs(r.choices[0]), 'raw': txt, 'usage_in': r.usage.prompt_tokens, 'usage_out': r.usage.completion_tokens}
            r = client.chat.completions.create(model=model, temperature=TEMP, max_tokens=40,
                                               messages=[{'role': 'user', 'content': prompt}])
            txt = r.choices[0].message.content or ''
            return {'value': parse_json(txt, key), 'raw': txt, 'usage_in': r.usage.prompt_tokens, 'usage_out': r.usage.completion_tokens}
        except Exception as e:
            last = str(e); time.sleep(min(2 ** t, 30))
    return {'value': None, 'raw': f'ERROR: {last}', 'usage_in': 0, 'usage_out': 0}

def load_cache(p):
    c = {}
    if p.exists():
        for line in open(p):
            try: r = json.loads(line); c[r.pop('k')] = r
            except Exception: pass
    return c

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--incidents', required=True)
    ap.add_argument('--out', default='replay_out')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--stage', choices=['hidden', 'all'], default='all', help="'hidden' = calibration stage: history-hidden prompts only")
    ap.add_argument('--no-steward', action='store_true', help='skip the exploratory stewardship probe')
    ap.add_argument('--zips', default=None); ap.add_argument('--seeds', default=None)
    ap.add_argument('--vp', type=float, default=1.0, help='archived invitation strength to replay (1.0 = focal, in incidents_default; 0.0 = incidents_calibrated, a built-in falsification test: no invitation, no chain)')
    ap.add_argument('--real-from', default=None, help='incidents file holding the observed-arm vp=1.0 rows that define each place\'s real score (default: --incidents)')
    ap.add_argument('--limit', type=int, default=None, help='cap the number of API calls')
    ap.add_argument('--model', default=DEFAULT_MODEL)
    ap.add_argument('--samples', type=int, default=3, help='independent calls per unique prompt; the value used is their mean (within-prompt SD is recorded)')
    ap.add_argument('--elicit', choices=['freq', 'logprob'], default='freq', help='freq: out-of-100 estimate (default); logprob: Yes/No token probabilities (pre-specified fallback)')
    a = ap.parse_args()
    global ELICIT; ELICIT = a.elicit
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    inc = pd.read_csv(a.incidents)
    f = inc[(inc.noticed) & (inc.engine == 'primary') & (inc.vp == a.vp) & (inc.arm.isin(['observed', 'swap']))].copy()
    if a.zips: f = f[f.zip.astype(str).isin(a.zips.split(','))]
    if a.seeds: f = f[f.seed.isin([int(s) for s in a.seeds.split(',')])]
    f = f.sort_values(['arm', 'seed', 'zip', 'agent_id', 'poi_id', 'day']).reset_index(drop=True)
    src = pd.read_csv(a.real_from) if a.real_from else inc
    real = src[(src.arm == 'observed') & (src.vp == 1.0)].groupby(['zip', 'poi_id']).vis_used.first().rename('vis_real')   # the place's real score (same at any vp)
    if real.empty: sys.exit('no observed vp=1.0 rows to define real scores; pass --real-from incidents_default.csv.gz')
    f = f.merge(real, left_on=['zip', 'poi_id'], right_index=True, how='left')
    f = f.rename(columns={'responsibility': 'orig_responsibility'})   # archived LLM audit value from the old model

    # history = visit count + prior NOTICED incidents by this resident at this place on earlier days (never prior reports)
    g = f.groupby(['arm', 'seed', 'zip', 'agent_id', 'poi_id']); f['k_prior'] = g.cumcount()
    same_day = f.duplicated(['arm', 'seed', 'zip', 'agent_id', 'poi_id', 'day'], keep='first')
    f.loc[same_day, 'k_prior'] = f.loc[same_day].groupby(['arm', 'seed', 'zip', 'agent_id', 'poi_id', 'day'])['k_prior'].transform('min')
    f['k_prior'] = f['k_prior'].clip(upper=K_CAP).astype(int); f['fam_c'] = f['fam'].clip(1, 12).astype(int)
    f['eff_b'] = f['efficacy'].map(eff_phrase); f['busy_b'] = f['busyness'].map(busy_phrase)
    f['steward_eligible'] = f['type'].isin(STEWARD_TYPES) & (not a.no_steward)

    P = {}
    fams = {'report': TYPES_ALL, 'dosomething': TYPES_ALL, 'steward': list(STEWARD_TYPES), 'signup': TYPES_ALL, 'gather': TYPES_ALL}
    if a.no_steward: fams.pop('steward')
    for fam_name, types_ok in fams.items():
        el = f['type'].isin(types_ok)
        P[f'{fam_name}_shown'] = [prompt_task(fam_name, t, e, b, fm, kp) if ok else '' for t, e, b, fm, kp, ok in zip(f.type, f.eff_b, f.busy_b, f.fam_c, f.k_prior, el)]
        P[f'{fam_name}_first'] = [prompt_task(fam_name, t, e, b, 1, 0) if ok else '' for t, e, b, ok in zip(f.type, f.eff_b, f.busy_b, el)]
        P[f'{fam_name}_unspec'] = [prompt_task(fam_name, t, e, b, unspec=True) if ok else '' for t, e, b, ok in zip(f.type, f.eff_b, f.busy_b, el)]
    P['report_hidden'] = [prompt_task('report', t, e, b) for t, e, b in zip(f.type, f.eff_b, f.busy_b)]   # no history line at all (record only)
    P['ownership'] = [prompt_ownership(fm, kp) for fm, kp in zip(f.fam_c, f.k_prior)]
    P['ownership_first'] = [prompt_ownership(1, 0)] * len(f)
    P['responsibility'] = [prompt_responsibility(fm, kp) for fm, kp in zip(f.fam_c, f.k_prior)]
    P['responsibility_first'] = [prompt_responsibility(1, 0)] * len(f)
    for c, v in P.items(): f['p_' + c] = v; f['k_' + c] = [md5(x) if x else '' for x in v]
    k_act = 'yes_no' if a.elicit == 'logprob' else 'n_of_100'
    KEY = {c: ('ownership_0_100' if c.startswith('ownership') else 'responsibility_0_100' if c.startswith('responsibility') else k_act) for c in P}
    stage_cols = [c for c in P if c.endswith('_hidden') or c.endswith('_first') or c.endswith('_unspec')] if a.stage == 'hidden' else list(P)
    uniq = {c: f.loc[f['k_' + c] != '', ['k_' + c, 'p_' + c]].drop_duplicates('k_' + c) for c in P}
    n_calls = sum(len(uniq[c]) for c in stage_cols) * a.samples
    print(f'incidents: {len(f):,} | arm×seed {f.groupby(["arm","seed"]).size().to_dict()} | steward-eligible {int(f.steward_eligible.sum()):,}')
    print('unique prompts: ' + ', '.join(f'{c}={len(u):,}' for c, u in uniq.items()) + f'  -> this stage ({a.stage}), {a.samples} samples per prompt: {n_calls:,} calls, ≈ ${n_calls*190*0.15/1e6 + n_calls*12*0.60/1e6:.2f} at gpt-4o-mini prices')
    if a.dry_run:
        for c in ['report_shown', 'report_first', 'report_unspec', 'report_hidden', 'dosomething_shown', 'steward_shown', 'signup_shown', 'gather_shown', 'ownership', 'responsibility']:
            if c in P: ex = next(x for x in P[c] if x); print(f'\n--- {c} ---\n{ex}')
        f.drop(columns=['p_' + c for c in P]).to_csv(out / 'replay_frame_dryrun.csv', index=False)
        pd.concat([u.rename(columns={'k_' + c: 'key', 'p_' + c: 'prompt'}).assign(kind=c) for c, u in uniq.items()]).to_csv(out / 'prompts_dryrun.csv', index=False)
        return

    client = make_client()
    safe = re.sub(r'[^A-Za-z0-9._-]', '_', a.model); cache_p = out / f'llm_cache_{safe}_{a.elicit}.jsonl'; cache = load_cache(cache_p)
    jobs = [(f'{k}#{i}', p, KEY[c], c) for c in stage_cols for k, p in zip(uniq[c]['k_' + c], uniq[c]['p_' + c]) for i in range(a.samples) if f'{k}#{i}' not in cache]
    if a.limit: jobs = jobs[:a.limit]
    print(f'cached {len(cache):,}, to call {len(jobs):,}  (model {a.model})')
    tin = tout = done = 0; t0 = time.time()
    with ThreadPoolExecutor(MAX_WORKERS) as ex, open(cache_p, 'a') as fh:
        futs = {ex.submit(call, client, p, key, a.model): (k, c) for k, p, key, c in jobs}
        for fu in as_completed(futs):
            k, c = futs[fu]; r = fu.result(); r['kind'] = c; cache[k] = r
            fh.write(json.dumps({'k': k, **r}) + '\n'); fh.flush()
            tin += r['usage_in']; tout += r['usage_out']; done += 1
            if done % 200 == 0: print(f'  {done:,}/{len(jobs):,}  {time.time()-t0:.0f}s', flush=True)
    print(f'calls done: {done:,}  tokens in/out {tin:,}/{tout:,}')

    def agg(k):
        vals = [cache[f'{k}#{i}']['value'] for i in range(a.samples) if f'{k}#{i}' in cache and cache[f'{k}#{i}']['value'] is not None]
        return (float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan, len(vals)) if vals else (np.nan, np.nan, 0)
    AG = {k: agg(k) for c in P for k in uniq[c]['k_' + c]}
    for c in P:
        f[c] = f['k_' + c].map(lambda k: AG[k][0] if k else np.nan)
        f[c + '_sd'] = f['k_' + c].map(lambda k: AG[k][1] if k else np.nan)
    print('unparsed / missing per column:', {c: int(f.loc[f["k_" + c] != "", c].isna().sum()) for c in P})
    print('within-prompt SD (mean over unique prompts, points):', {c: round(100 * float(np.nanmean([AG[k][1] for k in uniq[c]['k_' + c]])), 1) for c in P})

    if a.stage == 'hidden':   # calibration report: spread of the hidden-condition judgments (no manipulation present)
        print('\n=== calibration (no manipulation present): control-condition judgments by problem x disposition ===')
        for c in [x for x in P if x.endswith('_first') or x.endswith('_hidden')]:
            if c.startswith('ownership'): continue
            h = f.loc[f['k_' + c] != ''].drop_duplicates('k_' + c)[c].dropna()
            wsd = np.nanmean([AG[k][1] for k in uniq[c]['k_' + c]]) if a.samples > 1 else float('nan')
            print(f'{c:18s} unique {len(h):3d}: min {h.min():.2f}, 10% {h.quantile(.1):.2f}, median {h.median():.2f}, 90% {h.quantile(.9):.2f}, max {h.max():.2f}; SD across prompts {h.std():.3f}; mean within-prompt SD {wsd:.3f}')
        print(f.pivot_table(index='type', columns=['busy_b'], values='report_first', aggfunc='mean').round(2).to_string())
        # pre-specified degeneracy rule, applied mechanically to the primary control prompts
        h = f.loc[f['k_report_first'] != ''].drop_duplicates('k_report_first').report_first.dropna()
        degenerate = (h.std() < .10) or (h.quantile(.1) <= .05) or (h.quantile(.9) >= .95)
        print('\nCALIBRATION VERDICT:', 'DEGENERATE (stop; next step per RUN_ME is --elicit logprob)' if degenerate else 'PASS (not degenerate; proceed to the full run)')
        if degenerate: sys.exit(2)
        print('\nDegenerate = nearly all answers at the same value (SD ≈ 0) or all ≥ .95 / ≤ .05. If not degenerate, run the full stage.\nPre-specified order: --elicit freq first; if degenerate, --elicit logprob; if both degenerate, report that as the finding (no further tuning).')
        return

    keep = ['arm', 'engine', 'vp', 'seed', 'zip', 'agent_id', 'poi_id', 'day', 'kind', 'type', 'fam', 'fam_c', 'k_prior',
            'efficacy', 'busyness', 'eff_b', 'busy_b', 'vis_used', 'vis_real', 'mine', 'salience', 'attention',
            'orig_responsibility', 'propensity', 'report_intent', 'steward_eligible'] + list(P) + [c + '_sd' for c in P] + ['k_' + c for c in P]
    f[keep].to_csv(out / 'replay_results.csv', index=False)
    for c, u in uniq.items():
        u = u.rename(columns={'k_' + c: 'key', 'p_' + c: 'prompt'}).copy()
        u['value'] = u.key.map(lambda k: AG[k][0]); u['sd'] = u.key.map(lambda k: AG[k][1]); u['n_samples'] = u.key.map(lambda k: AG[k][2])
        u['raw_samples'] = u.key.map(lambda k: ' | '.join(str(cache[f'{k}#{i}']['raw']) for i in range(a.samples) if f'{k}#{i}' in cache))
        u.to_csv(out / f'decision_table_{c}.csv', index=False)
    json.dump({'model': a.model, 'elicitation': a.elicit, 'temperature': TEMP, 'samples_per_prompt': a.samples, 'vp': a.vp, 'prompt_version': 'v3 (13 Sep: neighbourhood wording, ownership probe without responsibility, responsibility probe, unspecified-history control)', 'n_incidents': int(len(f)), 'unique_prompts': {c: int(len(u)) for c, u in uniq.items()},
               'tokens_in_this_run': tin, 'tokens_out_this_run': tout, 'steward_types': list(STEWARD_TYPES) if not a.no_steward else [], 'control': 'first-visit, structurally matched (fam=1, k=0)',
               'arm_seed': {f'{x}_{s}': int(n) for (x, s), n in f.groupby(['arm', 'seed']).size().items()}}, open(out / 'replay_manifest.json', 'w'), indent=2)
    print('written:', out / 'replay_results.csv')

if __name__ == '__main__':
    main()
