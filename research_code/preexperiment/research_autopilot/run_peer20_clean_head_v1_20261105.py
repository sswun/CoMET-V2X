"""R6e one decontaminated H1 head per same domain/seed; no successor search.

Same architecture/Adam/.003/2000 updates/384 rows per step. Exact-zero fit
conditions receive zero sampling mass; nonzero candidates preserve relative
original weights. Geometry/q/prefix frozen. Cal8 official five-domain AP70
chooses clean vs original globally, tie original, before reading formal roles.
"""
import argparse,time,traceback
import numpy as np
from peer20_population_core_v1_20261105 import *
from review_peer16_E4_E7_AP_v1_20261103 import check,geometry

OUT=BASE/'R6e';SEEDS=(2023,2024,2025)

def prepare():
    for d in h1.recipe.DOMAINS:
        for mode in ('fit','review'):
            register('R6e_'+d+('_independent' if mode=='review' else ''),__file__,__doc__,dict(GPU_forwards=0,
                updates=6000 if mode=='fit' else 0,CPU_threads=1,disk_MiB='<100/three heads scores and one shared schedule'),
                domain=d,seeds=list(SEEDS),R6a=ev(BASE/'R6a_raw_labels_independent_v2/terminal.json'),
                old_calibration=ev(REC/d/'calibration.json'),architecture='39x32ReLUx1',
                choice='Same frozen H1 q/tau and cal8 official full4wait AP70; clean vs old global5domain mean; tie old. Populations never choose.')
    register('R6e_global_choice',__file__,'Cal8 five-domain clean-vs-original choice only; produce new recipe identity, never overwrite peer19 registry or relabel its predictions.',
        dict(GPU_forwards=0,updates=0,CPU_threads=1,disk_MiB='<1'),domains=list(h1.recipe.DOMAINS),tie='original',
        formal_consumers='If clean selected, new registered table/asset consumers must use clean predictions; old root v3 remains old identity.')

def train(m,path,schedule,seed):
    import torch
    from torch import nn
    from torch.nn import functional as F
    torch.set_num_threads(1);torch.manual_seed(seed);path.mkdir(parents=True,exist_ok=False)
    w=schedule['weights'];batches=schedule['batches'];f=m['features'].astype(np.float64)
    mean=np.sum(f*w[:,None],axis=0);std=np.maximum(np.sqrt(np.sum((f-mean)**2*w[:,None],axis=0)),1e-6)
    matrix=torch.from_numpy(((f-mean)/std).astype(np.float32));target=torch.from_numpy(m['utility_y'].astype(np.float32))
    model=nn.Sequential(nn.Linear(39,32),nn.ReLU(),nn.Linear(32,1));optimizer=torch.optim.Adam(model.parameters(),lr=.003);trace=[]
    for i,ids in enumerate(batches):
        index=torch.from_numpy(ids.astype(np.int64));loss=F.binary_cross_entropy_with_logits(model(matrix[index]).squeeze(-1),target[index])
        assert torch.isfinite(loss)
        optimizer.zero_grad();loss.backward();optimizer.step()
        if (i+1)%250==0:trace.append(dict(update=i+1,loss=float(loss.detach())))
    torch.save(model.state_dict(),path/'direct_u.pth')
    # Scorer only needs normalization here; training indices have one shared authority.
    np.savez_compressed(path/'training_schedule.npz',mean=mean,std=std)
    atomic(path/'training_summary.json',dict(seed=seed,updates=len(batches),batch_size=int(batches.shape[1]),trace=trace,
        schedule=ev(path.parent/'clean_schedule.npz'),normalization_only=True,model=ev(path/'direct_u.pth')))

def execute(domain,review=False):
    name='R6e_'+domain+('_independent' if review else '');dest=BASE/name;src=OUT/domain
    try:
        if review:wait(src/'terminal.json')
        wait(BASE/'R6a_raw_labels_independent_v2/terminal.json');record(name,dict(status='running_clean_head_independent' if review else 'running_fit24_only_clean_head',resource_owner=True))
        dataset,oldhead,_=data(domain);cal=load(REC/domain/'calibration.json');kind=load(REC/'method_registry.json')['selected_kind'];q=cal['q'];tau=cal['choices'][kind]['tau'];h1.recipe.OUT=REC
        panels=[]
        if not review:
            src.mkdir(parents=True,exist_ok=False);m,rows=dataset['fit24'];pops,_=annotations(rows);d=h1.recipe.hard_state(m,q)['delta'];b,f=h1.recipe.geometry(m,d)
            fit=h1.recipe.matched.geometry_material(m,b,f)
            candidate_keep=np.repeat(~pops['zero'],np.diff(m['condition_offsets']))
            with np.load(oldhead/'training_schedule.npz') as z:weights=z['weights'].copy();shape=z['batches'].shape
            weights[~candidate_keep]=0;weights/=weights.sum();rng=np.random.RandomState(20261026)
            batches=rng.choice(len(weights),size=shape,replace=True,p=weights).astype(np.int32)
            assert candidate_keep[batches].all()
            np.savez_compressed(src/'clean_schedule.npz',weights=weights,batches=batches,candidate_keep=candidate_keep)
            atomic(src/'fit_exclusion.json',dict(conditions=len(rows),excluded_conditions=int(pops['zero'].sum()),candidates=len(candidate_keep),
                excluded_candidates=int((~candidate_keep).sum()),original_schedule=ev(oldhead/'training_schedule.npz'),shared_new_schedule=ev(src/'clean_schedule.npz')))
            m,_=dataset['calibration8'];d=h1.recipe.hard_state(m,q)['delta'];b,f=h1.recipe.geometry(m,d);cm=h1.recipe.matched.geometry_material(m,b,f)
            for seed in SEEDS:
                head=src/('seed'+str(seed));train(fit,head,dict(weights=weights,batches=batches),seed)
                h1.recipe.matched.source_map(cm,h1.recipe.matched.score(cm,head),head)
                for label,path in [('clean',head),('original',h1.recipe.head(domain,kind,seed))]:
                    namep=label+'_seed'+str(seed);pp,p=h1.recipe.evaluate(m,b,f,path,namep,tau)
                    panels.extend(dict(r,role='calibration8',variant=label,seed=seed) for r in pp)
                    np.savez_compressed(src/('calibration8_'+namep+'_predictions.npz'),**p)
            atomic(src/'calibration_panels.json',panels);atomic(src/'calibration_ready.json',dict(created_at=now(),q=q,tau=tau,cal_only=True,
                means={v:float(np.mean([r['ap70'] for r in panels if r['variant']==v])) for v in ('clean','original')}))
            wait(OUT/'global_choice/terminal.json')
            for role in ('val21','mechanism_holdout14'):
                m,_=dataset[role];d=h1.recipe.hard_state(m,q)['delta'];b,f=h1.recipe.geometry(m,d)
                for seed in SEEDS:
                    label='clean_seed'+str(seed);pp,p=h1.recipe.evaluate(m,b,f,src/('seed'+str(seed)),label,tau)
                    panels.extend(dict(r,role=role,variant='clean',seed=seed) for r in pp)
                    np.savez_compressed(src/(role+'_'+label+'_predictions.npz'),**p)
                atomic(src/'panels.json',panels)
            atomic(src/'terminal.json',dict(created_at=now(),status='complete_light_checked',choice=ev(OUT/'global_choice/decision.json'),independent_pending=True))
        else:
            gold=load(src/'panels.json');checks=[]
            fm,fr=dataset['fit24'];pops,_=annotations(fr);keep=np.repeat(~pops['zero'],np.diff(fm['condition_offsets']))
            with np.load(src/'clean_schedule.npz') as z:
                assert np.array_equal(keep,z['candidate_keep']) and keep[z['batches']].all() and np.all(z['weights'][~keep]==0)
            for role in ROLES:
                m,_=dataset[role];b=geometry(m,h1.recipe.hard_state(m,q)['delta'])
                for seed in SEEDS:
                    for variant in (('clean','original') if role=='calibration8' else ('clean',)):
                        label=variant+'_seed'+str(seed)
                        with np.load(src/(role+'_'+label+'_predictions.npz')) as p:rr=check(m,b,p,label,[r for r in gold if r['role']==role])
                        assert all(abs(r['error'+str(t)])<1e-8 and r['TP_mismatch'+str(t)]==0 for r in rr for t in (50,70));checks.extend(dict(r,role=role,variant=variant,seed=seed) for r in rr)
                        atomic(dest/'AP.json',checks)
            atomic(dest/'terminal.json',dict(created_at=now(),status='complete_independently_reviewed',AP=ev(dest/'AP.json'),fit_zero_exclusion_verified=True,
                scope='Independent polygon/AP; shared neural training and source-map. Global choice acceptance separately requires all5 independent cal means.'))
        term=dest/'terminal.json' if review else src/'terminal.json'
        record(name,dict(status=load(term)['status'],terminal=ev(term),owner=None,resource_owner=False))
    except Exception as e:
        atomic(dest/'terminal.json',dict(created_at=now(),status='failed_retained',error=repr(e),traceback=traceback.format_exc()))
        record(name,dict(status='failed_retained',terminal=ev(dest/'terminal.json'),owner=None,resource_owner=False));raise

def choose():
    name='R6e_global_choice';dest=OUT/'global_choice'
    try:
        rows=[]
        for d in h1.recipe.DOMAINS:
            path=OUT/d/'calibration_ready.json'
            while not path.exists():time.sleep(30)
            rows.append(dict(domain=d,**load(path)))
        means={k:float(np.mean([r['means'][k] for r in rows])) for k in ('clean','original')};selected='clean' if means['clean']>means['original'] else 'original'
        atomic(dest/'decision.json',dict(created_at=now(),means=means,selected=selected,domains=rows,
            independent_pending=True,cal8_official_only=True,old_registry_unchanged=True,formal_read_for_selection=False,
            required_consumers='New clean-head table and asset consumers if selected; original tables retain exact identity.'))
        atomic(dest/'terminal.json',dict(created_at=now(),status='complete_light_checked',decision=ev(dest/'decision.json')))
        record(name,dict(status='complete_light_checked',terminal=ev(dest/'terminal.json'),owner=None,resource_owner=False))
    except Exception as e:
        atomic(dest/'terminal.json',dict(created_at=now(),status='failed_retained',error=repr(e),traceback=traceback.format_exc()));record(name,dict(status='failed_retained',terminal=ev(dest/'terminal.json')));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','run','review','choose'));p.add_argument('--domain',choices=h1.recipe.DOMAINS);a=p.parse_args()
    prepare() if a.mode=='prepare' else choose() if a.mode=='choose' else execute(a.domain,a.mode=='review')
