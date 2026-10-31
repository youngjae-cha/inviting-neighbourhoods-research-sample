"""Original archived simulation functions and manual settings.

Function bodies and settings are retained, including the older audit/fallback behavior.
This archival layer differs from the later scene-blind replay in replay_judgment.py.
Institutional data loading and automatic grid launch have been removed.
Importing this module performs no simulation, API call, data read or file write.
"""
import os,json,math,hashlib,re,time
from pathlib import Path
from dataclasses import dataclass,field
from typing import Dict,List,Any
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
SMOKE=False

CFG=dict(
  N_AGENTS=6 if SMOKE else 30, N_DAYS=3 if SMOKE else 21, N_POI_PER_DAY=3,
  N_ZIPS=3 if SMOKE else None,                 # None = all 53
  SEEDS=[42] if SMOKE else [42,43,44], SEEDS_SECONDARY=[42],
  ARMS=['observed','swap'], ENGINES=['primary','legacy','attach0'],
  VISUAL_PULL=1.0, VISUAL_PULL_GRID=[0.5,1.5],  # exploratory, 6-ZIP subset
  NOTICE_VARIANT='base', NOTICE_GRID=['lenient','strict'],
  INCIDENT_PROB_PER_VISIT=0.18, MINENESS_ALPHA=0.22, MINENESS_DECAY=0.985,
  PUBLIC_REPORT_BASE_INTERCEPT=-2.65, W_PUBLIC_PROBLEM=1.20, W_RESPONSIBILITY=1.25,
  W_MINE=1.35, W_FAM=0.10, W_EFFICACY=0.25, W_BUSY=-0.30, PRIVATE_STRICTNESS=1.75,
  PUBLIC_PROBLEM_PUBLIC=0.88, PUBLIC_PROBLEM_PRIVATE=0.08,
  DECISION_MODEL='gpt-4o-mini', MAX_WORKERS=24, RESUME=True,
  PERSONA_SOURCE='gallup' if not SMOKE else 'syn',
  RUN_LLM_AUDIT=not SMOKE)

PUB=['fallen_tree','streetlight_outage','signal_malfunction','blocked_sidewalk','illegal_dumping','pothole']

PRV=['package_theft','private_noise','property_dispute','loud_music']

NOTICE_PARAMS={'base':(.10,.45,.25,.15,.30),'lenient':(.20,.45,.25,.15,.20),'strict':(.05,.40,.20,.10,.40)}

def extract_json(s):
    m=re.search(r'\{.*\}',s,re.S); return json.loads(m.group(0))

@dataclass
class Agent:
    id:int; persona:Dict[str,Any]; home_poi_id:int
    familiarity:Dict[int,int]=field(default_factory=dict)
    mineness:Dict[int,float]=field(default_factory=dict)

def make_agents(zc,poi_df,n,rng):
    if CFG['PERSONA_SOURCE']=='gallup':
        pool=GPOOL[GPOOL['zip']==zc]
        assert len(pool)>0, f'no Gallup pool for {zc}'
        pick=pool.sample(n,replace=len(pool)<n,random_state=int(rng.integers(1e9)))
        return [Agent(i,{'trait_prosocial':float(np.clip(p.prosocial,1,5)),'baseline_attention':float(p.attention),
                'baseline_efficacy':float(p.efficacy),'baseline_busyness':float(p.busyness),
                'attach_prior':float(p.attach_prior),'respondent_id':str(p.respondent_id)},int(rng.choice(poi_df.poi_id.values)))
                for i,(_,p) in enumerate(pick.iterrows())]
    return [Agent(i,{'trait_prosocial':float(rng.uniform(2.5,5)),'baseline_attention':float(rng.uniform(.45,.8)),
            'baseline_efficacy':float(rng.uniform(.35,.75)),'baseline_busyness':float(rng.uniform(.25,.75)),'attach_prior':.5},
            int(rng.choice(poi_df.poi_id.values))) for i in range(n)]

def day_state(a,rng):
    p=a.persona
    return dict(attention=float(np.clip(rng.normal(p['baseline_attention'],.12),.05,.95)),
                efficacy=float(np.clip(rng.normal(p['baseline_efficacy'],.10),.05,.95)),
                busyness=float(np.clip(rng.normal(p['baseline_busyness'],.16),.05,.95)))

def choose_pois(a,poi_df,vp,rng):
    home=poi_df[poi_df.poi_id==a.home_poi_id].iloc[0]
    dlat=(poi_df.LATITUDE-home.LATITUDE)*111.0; dlon=(poi_df.LONGITUDE-home.LONGITUDE)*111.0*np.cos(np.deg2rad(home.LATITUDE))
    dist=np.sqrt(dlat**2+dlon**2).values
    fam=np.array([a.familiarity.get(int(p),0) for p in poi_df.poi_id]); mine=np.array([a.mineness.get(int(p),0.) for p in poi_df.poi_id])
    novelty=np.exp(-.25*fam); habit=1-np.exp(-.35*fam); drelief=np.exp(-.45*dist)
    pull=0.6+vp*poi_df['vis_used'].values
    sc=np.maximum((0.35*novelty+0.25*habit*(0.5+mine)+0.25*drelief+0.15*rng.random(len(poi_df)))*pull,1e-9)
    idx=rng.choice(len(poi_df),size=min(CFG['N_POI_PER_DAY'],len(poi_df)),replace=False,p=sc/sc.sum())
    return poi_df.iloc[idx]

def p_notice(att,mine,sal,busy,variant='base'):
    b,wa,wm,ws,wb=NOTICE_PARAMS[variant]
    return float(np.clip(b+wa*att+wm*mine+ws*sal-wb*busy,.02,.95))

def audit_prompt(kind_text,fam,mine):
    # BLIND: no scene imagery/text, no place descriptors, no persona identity
    return ("You audit whether a resident would feel a civic duty to act on an incident.\n"
            "Return JSON only: {\"responsibility\": 0-1} — a public duty judgment, not a mood.\n\n"
            f"Incident: {kind_text}\nThe resident has visited this place {fam} times before.\n"
            f"Accumulated sense that this place is 'theirs': {mine:.2f} (0-1).")

def _audit(job):
    k,prompt,kind=job
    for at in range(4):
        try:
            r=client.chat.completions.create(model=CFG['DECISION_MODEL'],messages=[{'role':'user','content':prompt}],temperature=0,max_tokens=60)
            v=float(np.clip(float(extract_json(r.choices[0].message.content).get('responsibility',.5)),0,1))
            return k,{'responsibility':v,'status':'fetched'}
        except Exception as e:
            if at==3: return k,{'responsibility':.65 if kind=='public' else .25,'status':repr(e)[:60]}
            time.sleep(2*(at+1))

def report_prop(pub_problem,resp,mine,fam,eff,busy,private):
    s=(CFG['PUBLIC_REPORT_BASE_INTERCEPT']+CFG['W_PUBLIC_PROBLEM']*pub_problem+CFG['W_RESPONSIBILITY']*resp
       +CFG['W_MINE']*mine+CFG['W_FAM']*fam+CFG['W_EFFICACY']*eff+CFG['W_BUSY']*busy-CFG['PRIVATE_STRICTNESS']*private)
    return float(1/(1+np.exp(-np.clip(s,-30,30))))

def run_one(zc,seed,arm,engine,vp=None,notice_variant=None):
    vp=CFG['VISUAL_PULL'] if vp is None else vp; nv=CFG['NOTICE_VARIANT'] if notice_variant is None else notice_variant  # 0.0 falsy 버그 수정
    tag=f"{zc}_s{seed}_{arm}_{engine}_vp{vp}_{nv}_{CFG['PERSONA_SOURCE']}"
    rd=RUNS/tag; rd.mkdir(parents=True,exist_ok=True); fs=rd/'summary.json'
    if CFG['RESUME'] and fs.exists(): return json.loads(fs.read_text())
    rng=np.random.default_rng(seed*100003+int(zc)+hash((arm,engine,vp,nv))%9973)
    poi_df=poi_vis[poi_vis['zip']==zc].copy().reset_index(drop=True)
    v=pd.to_numeric(poi_df.visual_invitation_actual,errors='coerce').fillna(.5)
    poi_df['vis_used']=(v.min()+v.max()-v) if arm=='swap' else v
    agents=make_agents(zc,poi_df,CFG['N_AGENTS'],rng)
    alpha=0.0 if engine=='attach0' else CFG['MINENESS_ALPHA']
    pend=[]
    for day in range(1,CFG['N_DAYS']+1):
        for a in agents:
            st=day_state(a,rng)
            for _,poi in choose_pois(a,poi_df,vp,rng).iterrows():
                pid=int(poi.poi_id); fam0=a.familiarity.get(pid,0); vis=float(poi.vis_used)
                a.familiarity[pid]=fam0+1
                mult=(0.35+vis) if engine=='legacy' else 0.85
                base_prior=.15*a.persona.get('attach_prior',.5)
                a.mineness[pid]=float(np.clip(a.mineness.get(pid,base_prior)*CFG['MINENESS_DECAY']+alpha*mult*(1-np.exp(-.35*(fam0+1))),0,1))
                if rng.random()<CFG['INCIDENT_PROB_PER_VISIT']:
                    kind='public' if rng.random()<.5 else 'private'
                    typ=str(rng.choice(PUB if kind=='public' else PRV)); text=f"{typ.replace('_',' ')} nearby"
                    mine=a.mineness.get(pid,0); fam=a.familiarity.get(pid,0); sal=float(np.clip(rng.normal(.5,.15),.05,.95))
                    pn=p_notice(st['attention'],mine,sal,st['busyness'],nv)
                    noticed=bool(rng.random()<pn)
                    k=hashlib.md5(f"v61|{text}|{min(fam,10)}|{round(mine,1)}".encode()).hexdigest()
                    pend.append(dict(k=k,zip=zc,agent_id=a.id,day=day,poi_id=pid,kind=kind,type=typ,text=text,
                        vis_used=vis,mine=mine,fam=fam,salience=sal,p_notice=pn,noticed=noticed,**st))
    todo={p['k']:(p['k'],audit_prompt(p['text'],min(p['fam'],10),p['mine']),p['kind']) for p in pend if p['noticed'] and p['k'] not in _cache}
    if todo and client is not None:
        with ThreadPoolExecutor(CFG['MAX_WORKERS']) as ex:
            for k,rec in ex.map(_audit,todo.values()):
                _cache[k]=rec; open(CACHE_F,'a').write(json.dumps({'k':k,**rec})+'\n')
    rng2=np.random.default_rng(seed*7919+int(zc)+hash((arm,engine))%7919)
    rows=[]
    for p in pend:
        resp=(_cache.get(p['k'],{}).get('responsibility', .65 if p['kind']=='public' else .25)) if p['noticed'] else np.nan
        pubp=CFG['PUBLIC_PROBLEM_PUBLIC'] if p['kind']=='public' else CFG['PUBLIC_PROBLEM_PRIVATE']
        prop=report_prop(pubp,resp,p['mine'],min(p['fam'],10)/10,p['efficacy'],p['busyness'],1.0 if p['kind']=='private' else 0.0) if p['noticed'] else np.nan
        intent=bool(p['noticed'] and rng2.random()<prop)
        rows.append({**{kk:p[kk] for kk in ['zip','agent_id','day','poi_id','kind','type','vis_used','mine','fam','salience','p_notice','noticed','attention','efficacy','busyness']},
                     'seed':seed,'arm':arm,'engine':engine,'vp':vp,'notice_variant':nv,'responsibility':resp,'propensity':prop,'report_intent':intent})
    ac=pd.DataFrame(rows); ac.to_csv(rd/'actions.csv',index=False)
    fin=dict(tag=tag,zip=zc,seed=seed,arm=arm,engine=engine,vp=vp,nv=nv,n=len(ac),
        notice_rate=float(ac.noticed.mean()) if len(ac) else np.nan,
        pub=float(ac.loc[ac.kind=='public','report_intent'].mean()) if len(ac) else np.nan,
        prv=float(ac.loc[ac.kind=='private','report_intent'].mean()) if len(ac) else np.nan)
    fs.write_text(json.dumps(fin)); return fin


# Public operational setup only; the original computational functions above are unchanged.
poi_vis=None
GPOOL=None
OUT=None
RUNS=None
CACHE_F=None
_cache={}
client=None

def configure_inputs(place_frame, persona_pool, output_dir, *, judgment_client=None, audit_cache=None):
    """Bind authorized inputs explicitly. No provider client is created here.

    The original runner retains its documented fallback scores if audit values are
    absent. Its formula-based report output is not the later replay outcome.
    """
    global poi_vis,GPOOL,OUT,RUNS,CACHE_F,_cache,client
    required_places={'zip','poi_id','LATITUDE','LONGITUDE','visual_invitation_actual'}
    required_personas={'zip','respondent_id','prosocial','attention','efficacy','busyness','attach_prior'}
    if not required_places.issubset(place_frame) or not required_personas.issubset(persona_pool):
        raise ValueError('Input columns do not satisfy the original runner schema.')
    poi_vis=place_frame.copy();GPOOL=persona_pool.copy()
    poi_vis['zip']=poi_vis['zip'].astype(str).str.zfill(5)
    GPOOL['zip']=GPOOL['zip'].astype(str).str.zfill(5)
    OUT=Path(output_dir);RUNS=OUT/'runs';RUNS.mkdir(parents=True,exist_ok=True)
    CACHE_F=Path(audit_cache) if audit_cache else OUT/'audit_cache.jsonl'
    _cache={}
    if CACHE_F.is_file():
        for line in CACHE_F.read_text().splitlines():
            record=json.loads(line);_cache[record.pop('k')]=record
    client=judgment_client
