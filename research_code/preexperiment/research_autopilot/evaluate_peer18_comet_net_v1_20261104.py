"""Ready-first own-domain and author-zero-shot formal evaluation, CPU-only.

Training targets never imported in inference. Source maps and protected prefix
are frozen N0 domain assets, not refit or chosen from formal outcomes.
"""
import argparse,time,traceback
from collections import OrderedDict
import numpy as np
from peer18_v1_20261104 import *
import run_peer18_comet_net_v1_20261104 as engine
OUT=BASE/'CoMET_Net'
TRAIN=OUT/'training_recovery_v2'

def prepare(seed):
    for mode,label in (('run','evaluation'),('review','independent')):
        register('net_'+label+'_seed'+str(seed),__file__,
            'Ready-first54 fixed own-domain/author-zero-shot jobs. No formal selection. '
            'Independent polygon AP uses shared saved filter states; training/recurrence independence not claimed.',
            dict(detector_forwards=0,updates=0,CPU_threads=1,jobs=54,disk_MiB='<300 compact scores/states; no GT duplication'),
            producer=ev(OUT/'contract.json'),mode=mode,seed=seed,
            transfer_scope='Learned filter and residual U transfer; target frozen N0 U retained.')

def frame_inputs(m,i,motion,seed,f,logit):
    torch=engine.torch;final=engine.final
    a,b=map(int,m['condition_offsets'][i:i+2]);ri=a+np.flatnonzero(m['source'][a:b]==1);ei=a+np.flatnonzero(m['source'][a:b]==0)
    fixed=a+np.flatnonzero((m['source'][a:b]==0)&(m['protected'][a:b]==1));xx,yy=engine.match(m['cv_boxes'][ri],m['cv_boxes'][fixed])
    hard=m['cv_boxes'][fixed[yy],:2].astype(float)-m['cv_boxes'][ri[xx],:2]
    desc=np.empty((len(ri),len(ei),6),np.float32)
    if len(ri) and len(ei):
        desc[:,:,0]=m['raw_scores'][ri,None];desc[:,:,1]=m['raw_scores'][None,ei]
        desc[:,:,2:4]=(m['cv_boxes'][ri,None,3:5]-m['cv_boxes'][None,ei,3:5])/5
        desc[:,:,4]=np.cos(m['cv_boxes'][ri,None,6]-m['cv_boxes'][None,ei,6]);desc[:,:,5]=int(m['wait_ms'][i])/1000
    lookup={int(v):j for j,v in enumerate(ei)}
    hdesc=np.stack([desc[x,lookup[int(fixed[y])]] for x,y in zip(xx,yy)]) if len(xx) else np.empty((0,6),np.float32)
    return dict(remote_xy=engine.tensor(m['cv_boxes'][ri,:2]),ego_xy=engine.tensor(m['cv_boxes'][ei,:2]),
        pair_descriptor=engine.tensor(desc),hard_residual=torch.as_tensor(hard,dtype=torch.float64),hard_descriptor=engine.tensor(hdesc),
        motion=engine.tensor(motion),feature=engine.tensor(f[a:b]),logit=engine.tensor(logit[a:b]))

def infer(m,rows,domain,trained,variant,seed,path):
    torch=engine.torch;final=engine.final;head=final.head(domain,'no_position',seed)
    obs=engine.observation(m);q=.15 if domain=='author' else .05
    n=np.array([len(z) for _,z in obs]);z=np.array([x.mean(0) if len(x) else np.zeros(2) for _,x in obs])
    baseline=final.old.author.kf(m,n,z,q,.6)[0];_,f0=final.old.author.heading.build_heading(m,baseline);f0[:,engine.REMOVED]=0
    original=final.old.matched.readout.score_direct_u(dict(m,features=f0),head)
    logit0=np.log(np.clip(original,1e-7,1-1e-7)/(1-np.clip(original,1e-7,1-1e-7)))
    with np.load(head/'training_schedule.npz') as p:f0=np.clip((f0-p['mean'])/p['std'],-10,10)
    # q is the target N0 domain process noise, learned dimensionless modules transfer.
    model=engine.net.CoMETNet(variant,q);model.load_state_dict(torch.load(path,map_location='cpu'));model.eval()
    dyn=engine.dynamics(rows);states={};last={};delta=[];extra=[];residual=np.zeros(len(m['source']))
    with torch.no_grad():
        for i,r in enumerate(rows):
            key=str(m['sequence'][i]),int(m['wait_ms'][i]);stamp=int(m['receiver_timestamp_us'][i]);v,w,ok=dyn[str(m['sequence'][i]),stamp]
            prev,t,age=last.get(key,(v,stamp-100000,0));mot=[np.linalg.norm(v)/20,np.linalg.norm(v-prev)/10,abs(w),(stamp-t)/100000,age/20]
            frame=frame_inputs(m,i,mot,seed,f0,logit0);state,ex,_=model(frame,states.get(key,model.init('cpu')))
            states[key]=state;last[key]=(v,stamp,0 if len(frame['hard_residual']) else age+1)
            delta.append(state[0].numpy().copy());extra.append(ex.numpy())
            a,b=map(int,m['condition_offsets'][i:i+2]);residual[a:b]=(model.logits(frame['feature'],ex,frame['logit'])-frame['logit']).numpy()
    d=np.asarray(delta);boxes,features=final.old.author.heading.build_heading(m,d);features[:,engine.REMOVED]=0
    raw=final.old.matched.readout.score_direct_u(dict(m,features=features),head)
    raw=np.clip(raw,1e-7,1-1e-7);logit=np.log(raw/(1-raw))+residual
    direct=1/(1+np.exp(-np.clip(logit,-50,50)));maps=load(head/'source_maps.json')
    score=final.old.matched.readout.apply_source_map(direct,m['source'],maps['maps']['direct_to_utility'])
    return boxes,score,d,np.asarray(extra)

def run(seed):
    engine.imports();root=OUT/'formal'/('seed'+str(seed));root.mkdir(parents=True,exist_ok=False)
    record('net_evaluation_seed'+str(seed),dict(status='ready_first_formal',resource_owner=True));start=time.monotonic();done={};cache=OrderedDict()
    jobs=[(trained,target,v) for v in engine.VARIANTS for trained in engine.DOMAINS for target in (engine.DOMAINS if trained=='author' else (trained,))]
    while len(done)<len(jobs):
        assert time.monotonic()-start<48*3600
        for trained,target,v in jobs:
            key=trained+'__to__'+target+'__'+v
            if key in done:continue
            src=TRAIN/('seed'+str(seed))/trained/v/'terminal.json'
            lane=TRAIN/('seed'+str(seed))/'terminal.json'
            if not src.exists() and not lane.exists():continue
            dest=root/key;dest.mkdir(exist_ok=False)
            try:
                assert src.exists(), 'Training lane ended before this job; retain missing result'
                t=load(src);assert t['status']=='complete_light_checked';path=src.parent/'selected.pth';assert ev(path)==t['weights']
                if target not in cache:
                    data,_,q=engine.materials(target);cache[target]={r:data[r] for r in ('val21','mechanism_holdout14')}
                    if len(cache)>2:cache.popitem(last=False)
                panels=[]
                for role,(m,rows) in cache[target].items():
                    boxes,score,d,ex=infer(m,rows,target,trained,v,seed,path);name=v+'_seed'+str(seed)
                    tau=load(engine.final.OUT/target/'calibration.json')['selected_tau']
                    pp,p=engine.final.old.author.refit.fixed.base.evaluate(engine.final.protected(m,tau),{name:boxes},{name:score})
                    panels.extend(dict(r,role=role,target=target,training_detector=trained,variant=v,seed=seed,tau=tau,
                        transfer='zero_shot_filter_and_residual_readout; target frozen N0 U' if target!=trained else 'own_training') for r in pp)
                    np.savez_compressed(dest/(role+'_predictions.npz'),**p)
                    np.savez_compressed(dest/(role+'_state.npz'),delta=d,posterior_evidence=ex)
                    atomic(dest/'panels.json',panels)
                atomic(dest/'terminal.json',dict(created_at=now(),status='complete_light_checked',source=ev(src),
                    inference_script=ev(__file__),independent_matching_pending=True,GPU_forwards=0,
                    baseline=ev(engine.final.OUT/target/'identity.json'),formal_selection=False))
            except Exception as e:
                atomic(dest/'terminal.json',dict(created_at=now(),status='failed_retained',error=repr(e),traceback=traceback.format_exc()))
            done[key]=ev(dest/'terminal.json');atomic(root/'status.json',dict(created_at=now(),completed=done))
        if len(done)<len(jobs):time.sleep(30)
    status='complete_light_checked' if all(load(ROOT/p['path'])['status']=='complete_light_checked' for p in done.values()) else 'failed_retained'
    atomic(root/'terminal.json',dict(created_at=now(),status=status,completed=done,GPU_forwards=0))
    record('net_evaluation_seed'+str(seed),dict(status=status,terminal=ev(root/'terminal.json'),owner=None,resource_owner=False))

def review(seed):
    engine.imports();from_review=__import__('review_peer16_E4_E7_AP_v1_20261103',fromlist=['check','geometry'])
    root=OUT/'independent'/('seed'+str(seed));root.mkdir(parents=True,exist_ok=False);done={};start=time.monotonic();cache=OrderedDict()
    record('net_independent_seed'+str(seed),dict(status='ready_first_independent_AP',resource_owner=True))
    jobs=[(trained,target,v) for v in engine.VARIANTS for trained in engine.DOMAINS for target in (engine.DOMAINS if trained=='author' else (trained,))]
    while len(done)<len(jobs):
        assert time.monotonic()-start<50*3600
        for trained,target,v in jobs:
            key=trained+'__to__'+target+'__'+v;src=OUT/'formal'/('seed'+str(seed))/key
            if key in done or not (src/'terminal.json').exists():continue
            dest=root/key;dest.mkdir(exist_ok=False)
            try:
                assert load(src/'terminal.json')['status']=='complete_light_checked'
                if target not in cache:
                    data,_,_=engine.materials(target);cache[target]={r:data[r] for r in ('val21','mechanism_holdout14')}
                    if len(cache)>2:cache.popitem(last=False)
                data=cache[target];checks=[]
                for role in ('val21','mechanism_holdout14'):
                    m,_=data[role]
                    with np.load(src/(role+'_state.npz')) as z:boxes=from_review.geometry(m,z['delta'])
                    gold=[r for r in load(src/'panels.json') if r['role']==role]
                    with np.load(src/(role+'_predictions.npz')) as p:checks.extend(dict(r,role=role) for r in from_review.check(m,boxes,p,v+'_seed'+str(seed),gold))
                exact=all(abs(r['error'+str(th)])<1e-8 and r['TP_mismatch'+str(th)]==0 for r in checks for th in (50,70))
                atomic(dest/'AP.json',checks);atomic(dest/'terminal.json',dict(created_at=now(),status='complete_independently_reviewed' if exact else 'completed_discrepancy_retained',
                    scope='Independent heading reconstruction/polygon matching/TP/VOC. Shared learned weights and saved filter states; not independent training or full recurrence implementation.'))
            except Exception as e:
                atomic(dest/'terminal.json',dict(created_at=now(),status='failed_retained',error=repr(e),traceback=traceback.format_exc()))
            done[key]=ev(dest/'terminal.json');atomic(root/'status.json',dict(created_at=now(),completed=done))
        if len(done)<len(jobs):time.sleep(30)
    status='complete_independently_reviewed' if all(load(ROOT/p['path'])['status']=='complete_independently_reviewed' for p in done.values()) else 'failed_retained'
    atomic(root/'terminal.json',dict(created_at=now(),status=status,completed=done))
    record('net_independent_seed'+str(seed),dict(status=status,terminal=ev(root/'terminal.json'),owner=None,resource_owner=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run','review']);p.add_argument('--seed',type=int,required=True);a=p.parse_args();globals()[a.mode](a.seed)
