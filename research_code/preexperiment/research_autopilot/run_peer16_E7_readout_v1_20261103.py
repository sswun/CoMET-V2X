"""Fit/cal-only readout controls on the current author heading geometry."""
import argparse,json,time,traceback,inspect
import numpy as np
from peer16_v1_20261103 import ROOT,BASE,atomic,ev,load,now,register,record
from run_peer16_cached_methods_v1_20261103 import materials,observation
import run_peer15_author_recipe_v1_20261103 as author
import run_crosscheckpoint_temporal_matched39d_v1_20261029 as matched
from object_history_selector_core_v1_20261021 import FEATURE_NAMES
OUT=BASE/'E7_readout'
REMOVED=(2,3,4,10,11,12,13,14,15,18,19,20,21)
def prepare():
    register('E7_readout',__file__,
        'Compare full/no-position39D at3fixed head seeds on identical current geometry/schedule; IoU50 and dual50/70 utility controls at seed2023; frozen-head protected threshold sensitivity. No val parameter/head choice.',
        dict(GPU_forwards=0,CPU_threads=2,updates='8 heads×2000; same frozen batch draws',wall_h='2-10',disk_MiB='<220 compact predictions/heads; no shared-pool copy'),
        input_contract=ev(BASE/'E2_comet_delta_v2/contract.json'),seeds=[2023,2024,2025],
        removed_features=[FEATURE_NAMES[i] for i in REMOVED],
        geometry='Current heading+author cal8 q=.15; refit both full/no-position on same geometry so this is feature attribution, not comparison of differently trained geometry heads',
        threshold_grid=[.3,.35,.4,.45,.5,.55,.6,.65,.7],threshold_semantics='Fixed geometry/association/head; only protected readout membership changes; report cal8+val21, never choose on val',
        pending='Intersection-location disjointness audited from original metadata; mid-fusion+sameU remains a distinct open input/feature adapter task, not covered by head controls')
def geometry(m):
    obs=observation(m);n=np.asarray([len(z) for _,z in obs]);z=np.asarray([v.mean(0) if len(v) else np.zeros(2) for _,v in obs]);d=author.kf(m,n,z,.15,.6)[0];return author.heading.build_heading(m,d)
def targets(m,b,f,threshold):
    source=inspect.getsource(matched.geometry_material).replace('def geometry_material(','def build(')
    # This only parameterizes utility/quality matching; unrelated IoUs remain continuous.
    source=source.replace(' .7)', ' THRESHOLD)').replace('>= .7)', '>= THRESHOLD)')
    ns=dict(matched.__dict__,THRESHOLD=threshold);exec(compile(source,__file__+'::utility_target','exec'),ns);return ns['build'](m,b,f)
def location_audit():
    data=ROOT/'artifacts/data/v2x_seq_spd_main_domain_shared_v1/V2X-Seq-SPD';result={}
    for side in ('vehicle-side','infrastructure-side','cooperative'):
        info=load(data/side/'data_info.json');keys=sorted(set(k for x in info for k in x if 'loc' in k.lower() or 'intersection' in k.lower()));result[side]=dict(fields=keys,values={k:sorted(set(str(x.get(k)) for x in info)) for k in keys})
    atomic(OUT/'intersection_metadata_audit.json',dict(original_metadata=result,status='schema_audit_only_not_disjoint_training_complete',no_new_official_split=True))
def run():
    start=time.monotonic();record('E7_readout',dict(status='running_current_geometry_fitcal_readout_controls'))
    try:
        location_audit();data,oldhead,q=materials('author');prepared={}
        for role in ('fit24','calibration8'):
            m,_=data[role];b,f=geometry(m);prepared[role]=targets(m,b,f,.7)
        with np.load(oldhead/'training_schedule.npz') as z:batches=z['batches'].copy();weights=z['weights'].copy()
        assert len(weights)==len(prepared['fit24']['features']) and batches.max()<len(weights)
        arms=[(kind,seed) for kind in ('full39','no_position') for seed in (2023,2024,2025)]+[('utility50',2023),('utility_dual',2023)]
        for kind,seed in arms:
            fit=dict(prepared['fit24']);cal=dict(prepared['calibration8'])
            if kind=='no_position':
                fit=dict(fit,features=fit['features'].copy());cal=dict(cal,features=cal['features'].copy());fit['features'][:,REMOVED]=0;cal['features'][:,REMOVED]=0
            if kind in ('utility50','utility_dual'):
                for role,mat in [('fit24',fit),('calibration8',cal)]:
                    m,_=data[role];b,f=geometry(m);other=targets(m,b,f,.5);mat['utility_y']=other['utility_y'] if kind=='utility50' else .5*(mat['utility_y'].astype(float)+other['utility_y'])
            head=OUT/f'{kind}_seed{seed}'
            source=inspect.getsource(matched.train_head).replace('torch.manual_seed(readout.MODEL_SEED)','torch.manual_seed(SEED)').replace('"seed": readout.MODEL_SEED','"seed": SEED')
            ns=dict(matched.__dict__,SEED=seed);exec(compile(source,__file__+'::fixed_seed_training','exec'),ns);ns['train_head'](fit,oldhead/'training_schedule.npz',head)
            raw=matched.score(cal,head);matched.source_map(cal,raw,head);atomic(head/'identity.json',dict(kind=kind,seed=seed,removed_features=list(REMOVED) if kind=='no_position' else [],fit_role='fit24',mapping_role='calibration8',updates=2000,original_schedule=ev(oldhead/'training_schedule.npz')))
            allpan=[]
            for role in ('calibration8','mechanism_holdout14','val21'):
                m,_=data[role];b,f=geometry(m)
                if kind=='no_position':f[:,REMOVED]=0
                score=author.refit.score_with_head(m,f,head);name=f'{kind}_seed{seed}';pp,p=author.refit.fixed.base.evaluate(m,{name:b},{name:score});allpan.extend([dict(x,role=role) for x in pp])
                np.savez_compressed(head/(role+'_predictions.npz'),**p,condition_sequence=m['sequence'],condition_wait_ms=m['wait_ms'])
            atomic(head/'panels.json',allpan);atomic(OUT/'status.json',dict(created_at=now(),completed_head=str(head.name),wall_s=time.monotonic()-start))
        head=OUT/'full39_seed2023';panels=[]
        for role in ('calibration8','val21'):
            m,_=data[role];b,f=geometry(m);score=author.refit.score_with_head(m,f,head)
            for tau in (.3,.35,.4,.45,.5,.55,.6,.65,.7):
                mm=dict(m,protected=((m['source']==0)&(m['raw_scores']>=tau)).astype(np.int8));name='tau_'+str(tau);pp,_=author.refit.fixed.base.evaluate(mm,{name:b},{name:score});panels.extend([dict(p,role=role,tau=tau) for p in pp]);atomic(OUT/'threshold_panels.json',panels)
        atomic(OUT/'terminal.json',dict(created_at=now(),status='complete_light_checked',wall_s=time.monotonic()-start,GPU_forwards=0,heads=len(arms),no_val_selection=True,disjoint_intersection_midfusion_and_independent_pending=True));record('E7_readout',dict(status='complete_light_checked',terminal=ev(OUT/'terminal.json')))
    except Exception as e:
        atomic(OUT/'terminal.json',dict(created_at=now(),status='failed_retained',error=repr(e),traceback=traceback.format_exc()));record('E7_readout',dict(status='failed_retained',terminal=ev(OUT/'terminal.json')));raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run']);globals()[p.parse_args().mode]()
