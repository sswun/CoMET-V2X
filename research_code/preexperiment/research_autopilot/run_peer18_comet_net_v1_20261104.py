"""Pack causal box evidence, qualify N0, then one fixed-budget neural round.

The small residual readout uses frozen N0 39D features plus posterior evidence;
N0 utility labels/GT pair IDs are fixed fit/cal training targets. Original U is
frozen; all variants retain it. Formal evaluation reconstructs own geometry.
"""
import argparse,os,time,traceback,json
from pathlib import Path
import numpy as np
from peer18_v1_20261104 import *
OUT=BASE/'CoMET_Net'
VARIANTS=('N2','N3','N4','N5','B1','B2')
SEEDS=(2023,2024,2025)

def imports(gpu=None):
    global torch,net,final,materials,observation,DOMAINS,REMOVED,dynamics,match
    import run_peer17_final_recipe_v1_20261104 as final
    from run_peer16_cached_methods_v1_20261103 import materials,observation,DOMAINS
    from run_peer16_E7_readout_v1_20261103 import REMOVED
    from run_peer16_E1_timing_audit_v1_20261103 import dynamics,match
    # Legacy CPU geometry modules hide CUDA at import. Restore explicitly,
    # before the first CUDA initialization, for this new GPU-only trainer.
    os.environ['CUDA_VISIBLE_DEVICES']='' if gpu is None else str(gpu)
    os.environ['OMP_NUM_THREADS']='2' if gpu is not None else '1'
    import torch
    import peer18_comet_net_v1_20261104 as net
    torch.set_num_threads(2 if gpu is not None else 1)

def prepare():
    register('CoMET_Net',__file__,
        'One preregistered N2/N3/N4/N5/B1/B2 round, no variants after results. '
        'Retain KF structure and compare GRU only as bounded controls. '
        'No upgrade unless author>=.40 CI>0,5domain mean>=.25/none<-.20,hold same sign,AP50>=-.20,3seed SD<=.25.',
        dict(detector_forwards=0,updates='600 per variant/training-domain/seed; cal-only checkpoint selection at100/300/600',
             train_domains=5,seeds=3,variants=6,CPU_threads=2,parameters='<5000 including frozen1313 U',
             disk_MiB='<500 derived compact causal training tensors; no raw points/full GT copy'),
        model=ev(ROOT/'preexperiment/research_autopilot/peer18_comet_net_v1_20261104.py'),
        baseline=ev(P17/'final_recipe/contract.json'),
        training='fit24 only; random sequence windows32 with up-to32-frame causal burn-in, no shuffling time within windows. '
            'Loss centre Huber(alpha1)+utility BCE+innovation NLL(beta.01). N5 filter NLL only; readout BCE uses detached filter evidence.',
        model_selection='cal8 fixed windows: minimum frozen training loss among checkpoints100/300/600, no val selection',
        readout='Frozen original no-position39D U; M4 residual reads normalized N0 features plus[P,nu,S,sumlambda,noanchor]. '
            'Fixed N0 GT utility labels for BCE and fixed nearest GT pair IDs for centre Huber; no formal GT in model inputs.',
        equivalence='Disabled modules torch.float64 recurrence vs current scalar N0, float32 heading geometry and formal AP<.01 before learned evaluation.',
        semantics='M1 all ego/remote box pairs+balanced Sinkhorn dustbin, no2.5m or score gates; M2 sigma=.05+softplus; '
            'M3 diagonal motion-Q; original scalar ego-frame no-transport N0 retained. Normalized residual/relative descriptors, no location or sequence IDs.',
        zero_shot='author model applied to other four detectors as well as each own-domain training, separately reported',
        compute='3090 GPU1 primary author/domain fits; GPU0/4090 seed replication only; independent CPU evaluators. '+
            'No feature/backbone K4 training and no unbounded blackbox direction.')

def tensor(a):return torch.as_tensor(np.asarray(a),dtype=torch.float32)
def pack_domain(domain):
    data,_,q=materials(domain);dest=OUT/'cache'/domain;dest.mkdir(parents=True,exist_ok=False)
    for role in ('fit24','calibration8'):
        m,rows=data[role];obs=observation(m);dy=dynamics(rows)
        n=np.array([len(z) for _,z in obs]);z=np.array([a.mean(0) if len(a) else np.zeros(2) for _,a in obs])
        delta=final.old.author.kf(m,n,z,q,.6)[0];boxes,f=final.old.author.heading.build_heading(m,delta);f[:,REMOVED]=0
        target=final.old.matched.geometry_material(m,boxes,f)
        logits={};norm={}
        for seed in SEEDS:
            head=final.head(domain,'no_position',seed)
            direct=final.old.matched.readout.score_direct_u(dict(m,features=f),head)
            logits[seed]=np.log(np.clip(direct,1e-7,1-1e-7)/(1-np.clip(direct,1e-7,1-1e-7)))
            with np.load(head/'training_schedule.npz') as p:norm[seed]=(p['mean'],p['std'])
        frames=[];groups={};last={};target_rows=[]
        for i,r in enumerate(rows):
            a,b=map(int,m['condition_offsets'][i:i+2]);ga,gb=map(int,m['gt_offsets'][i:i+2])
            ri=a+np.flatnonzero(m['source'][a:b]==1);ei=a+np.flatnonzero(m['source'][a:b]==0)
            # Reproduce original protected anchor matching exactly for N0/M2.
            fixed=a+np.flatnonzero((m['source'][a:b]==0)&(m['protected'][a:b]==1))
            xx,yy=match(m['cv_boxes'][ri],m['cv_boxes'][fixed]);hard=m['cv_boxes'][fixed[yy],:2].astype(float)-m['cv_boxes'][ri[xx],:2]
            size=m['cv_boxes'][ri,None,3:5]-m['cv_boxes'][None,ei,3:5]
            desc=np.empty((len(ri),len(ei),6),np.float32)
            if len(ri) and len(ei):
                desc[:,:,0]=m['raw_scores'][ri,None];desc[:,:,1]=m['raw_scores'][None,ei]
                desc[:,:,2:4]=size/5;desc[:,:,4]=np.cos(m['cv_boxes'][ri,None,6]-m['cv_boxes'][None,ei,6])
                desc[:,:,5]=int(m['wait_ms'][i])/1000
            ei_lookup={int(v):j for j,v in enumerate(ei)}
            hdesc=np.stack([desc[x,ei_lookup[int(fixed[y])]] for x,y in zip(xx,yy)]) if len(xx) else np.empty((0,6),np.float32)
            key=str(m['sequence'][i]),int(m['wait_ms'][i]);v,w,ok=dy[str(m['sequence'][i]),int(m['receiver_timestamp_us'][i])]
            prev,stamp,age=last.get(key,(v,int(m['receiver_timestamp_us'][i])-100000,0))
            motion=[np.linalg.norm(v)/20,np.linalg.norm(v-prev)/10,abs(w),
                (int(m['receiver_timestamp_us'][i])-stamp)/100000,age/20]
            frame=dict(remote_xy=tensor(m['cv_boxes'][ri,:2]),ego_xy=tensor(m['cv_boxes'][ei,:2]),
                pair_descriptor=tensor(desc),hard_residual=torch.as_tensor(hard,dtype=torch.float64),
                hard_descriptor=tensor(hdesc),motion=tensor(motion),
                feature={s:tensor(np.clip((f[a:b]-norm[s][0])/norm[s][1],-10,10)) for s in SEEDS},
                logit={s:tensor(logits[s][a:b]) for s in SEEDS})
            groups.setdefault(key,[]).append(i);frames.append(frame)
            last[key]=(v,int(m['receiver_timestamp_us'][i]),0 if len(hard) else age+1)
            # Targets never enter frame()/model forward; fit/cal only.
            ids,jj=match(boxes[ri],m['gt_boxes'][ga:gb],3.)
            base=final.old.author.heading.heading_boxes(m['stale_boxes'][ri[ids]],m['cv_boxes'][ri[ids]])
            residual=m['gt_boxes'][ga:gb][jj,:2]-base[:,:2]
            target_rows.append(dict(centre=tensor(residual),utility=tensor(target['utility_y'][a:b])))
        path=dest/(role+'_causal_inputs.pt');torch.save(dict(frames=frames,groups=list(groups.values()),q=q),path)
        tp=dest/(role+'_loss_targets.pt');torch.save(target_rows,tp)
        atomic(dest/(role+'_identity.json'),dict(inputs=ev(path),loss_targets=ev(tp),
            original_role=role,cal_waits=sorted(map(int,np.unique(m['wait_ms']))),conditions=len(frames),
            references=ev(final.OUT/domain/'identity.json'),no_formal_GT_cached=True,
            storage='Derived box residual descriptors, normalized readout features and fit/cal-only targets; no raw points or full GT duplicates.'))
    atomic(dest/'terminal.json',dict(created_at=now(),status='complete_light_checked',q=q,domain=domain))

def cache():
    imports();start=time.monotonic();record('net_cache',dict(status='packing_causal_inputs',resource_owner=True))
    try:
        for domain in DOMAINS:pack_domain(domain)
        atomic(OUT/'cache/terminal.json',dict(created_at=now(),status='complete_light_checked',wall_s=time.monotonic()-start))
        record('net_cache',dict(status='complete_light_checked',terminal=ev(OUT/'cache/terminal.json'),owner=None))
    except Exception as e:
        atomic(OUT/'cache/terminal.json',dict(created_at=now(),status='failed_retained',error=repr(e),traceback=traceback.format_exc()));raise

def to_device(frame,device,seed):
    return {k:(v[seed].to(device) if isinstance(v,dict) else v.to(device)) for k,v in frame.items()}

def equivalence():
    imports();out=OUT/'N0_equivalence';out.mkdir(exist_ok=False);checks=[]
    for domain in DOMAINS:
        start=time.monotonic();src=OUT/'cache'/domain/'terminal.json'
        while not src.exists():
            assert time.monotonic()-start<12*3600;time.sleep(30)
        assert load(src)['status']=='complete_light_checked'
        data,_,q=materials(domain);m,_=data['calibration8'];obs=observation(m)
        n=np.array([len(z) for _,z in obs]);z=np.array([x.mean(0) if len(x) else np.zeros(2) for _,x in obs]);gold=final.old.author.kf(m,n,z,q,.6)[0]
        pack=torch.load(src.parent/'calibration8_causal_inputs.pt',map_location='cpu');model=net.CoMETNet('N0',q);delta=np.zeros_like(gold)
        with torch.no_grad():
            for group in pack['groups']:
                state=model.init('cpu')
                for i in group:
                    state,extra,nll=model(pack['frames'][i],state);delta[i]=state[0].numpy()
        err=float(np.max(np.abs(delta-gold),initial=0));assert err<1e-12
        b,f=final.old.author.heading.build_heading(m,delta);gb,_=final.old.author.heading.build_heading(m,gold)
        assert np.array_equal(b,gb);f[:,REMOVED]=0;seed=2023;score=final.old.author.refit.score_with_head(m,f,final.head(domain,'no_position',seed))
        tau=load(final.OUT/domain/'calibration.json')['selected_tau'];pp,p=final.old.author.refit.fixed.base.evaluate(final.protected(m,tau),{'N0':b},{'N0':score})
        old=load(final.OUT/domain/'calibration_panels.json');errors=[]
        for r in pp:
            ref=next(x for x in old if x['seed']==seed and x['tau']==tau and x['wait_ms']==r['wait_ms'])
            errors.extend(abs(r[k]-ref[k]) for k in ('ap50','ap70'))
        assert max(errors)<.01
        checks.append(dict(domain=domain,torch_state_max_abs=err,heading_float32_exact=True,AP_max_error=max(errors),
            parameters_including_frozen_U=sum(p.numel() for p in model.parameters())+1313,
            inference_uses_only_causal_inputs=True,labels_absent_from_forward=True))
        atomic(out/'checks.json',checks)
    atomic(out/'terminal.json',dict(created_at=now(),status='complete_light_checked',checks=checks))
    record('net_equivalence',dict(status='complete_light_checked',terminal=ev(out/'terminal.json'),owner=None))

def wait_ready(path,limit=16*3600):
    start=time.monotonic()
    while not path.exists():assert time.monotonic()-start<limit;time.sleep(30)
    assert load(path)['status']=='complete_light_checked',path

def loss(model,frames,targets,group,seed,device,supervised=True,initial_state=None):
    state=model.init(device) if initial_state is None else initial_state;losses=[]
    for i in group:
        frame=to_device(frames[i],device,seed);state,extra,nll=model(frame,state)
        if not supervised:losses.append(nll.float());continue
        target=targets[i];centre=target['centre'].to(device)
        geometric=torch.nn.functional.smooth_l1_loss(state[0].float().expand_as(centre),centre) if len(centre) else state[0].sum()*0
        ev=extra.detach() if model.variant=='N5' else extra
        logits=model.logits(frame['feature'],ev,frame['logit'])
        utility=torch.nn.functional.binary_cross_entropy_with_logits(logits,target['utility'].to(device)) if len(logits) else geometric*0
        losses.append((utility+.01*nll.float()) if model.variant=='N5' else utility+geometric+.01*nll.float())
    return torch.stack(losses).mean()

def train(seed,gpu):
    imports(gpu);device='cuda:0';assert torch.cuda.is_available();wait_ready(OUT/'N0_equivalence/terminal.json')
    lane=OUT/('seed'+str(seed));lane.mkdir(exist_ok=False);start=time.monotonic()
    record('net_seed'+str(seed),dict(status='training_bounded_round',resource_owner=True,device='GPU'+str(gpu)))
    try:
        for domain in DOMAINS:
            src=OUT/'cache'/domain;wait_ready(src/'terminal.json')
            fit=torch.load(src/'fit24_causal_inputs.pt',map_location='cpu');ft=torch.load(src/'fit24_loss_targets.pt',map_location='cpu')
            cal=torch.load(src/'calibration8_causal_inputs.pt',map_location='cpu');ct=torch.load(src/'calibration8_loss_targets.pt',map_location='cpu')
            for variant in VARIANTS:
                assert time.monotonic()-start<48*3600
                dest=lane/domain/variant;dest.mkdir(parents=True,exist_ok=False)
                torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);rng=np.random.default_rng(seed)
                model=net.CoMETNet(variant,fit['q']).to(device)
                assert sum(p.numel() for p in model.parameters())+1313<=5000
                optimizer=torch.optim.Adam(model.parameters(),lr=.001);trace=[];best=None
                # First32 of each cal sequence/window, fixed independently of AP.
                cg=[g[:32] for g in cal['groups']];selection=[]
                for step in range(1,601):
                    g=fit['groups'][int(rng.integers(len(fit['groups'])))];end=int(rng.integers(1,len(g)+1));begin=max(0,end-32)
                    # Burn preceding context without gradients, never future inputs.
                    state=model.init(device)
                    with torch.no_grad():
                        for i in g[max(0,begin-32):begin]:state,_,_=model(to_device(fit['frames'][i],device,seed),state)
                    value=loss(model,fit['frames'],ft,g[begin:end],seed,device,initial_state=net.detach(state))
                    assert torch.isfinite(value)
                    optimizer.zero_grad();value.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
                    if step%50==0:
                        trace.append(dict(step=step,loss=float(value.detach().cpu())))
                        atomic(dest/'progress.json',dict(trace=trace,elapsed_s=time.monotonic()-start))
                    if step in (100,300,600):
                        with torch.no_grad():criterion=float(np.mean([float(loss(model,cal['frames'],ct,g,seed,device).cpu()) for g in cg]))
                        selection.append(dict(step=step,cal_loss=criterion))
                        if best is None or criterion<best:
                            best=criterion;torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},dest/'selected.pth')
                atomic(dest/'terminal.json',dict(created_at=now(),status='complete_light_checked',seed=seed,domain=domain,variant=variant,
                    weights=ev(dest/'selected.pth'),parameters=sum(p.numel() for p in model.parameters())+1313,updates=600,
                    selection=selection,criterion='minimum fixed cal-window loss, not val AP',
                    truncated_BPTT='32frame gradient windows plus up-to32 preceding causal burn-in frames; full-sequence causal inference.',
                    network_forwards_not_detector=True,GPU_forwards=0,formal_AP_pending=True))
        atomic(lane/'terminal.json',dict(created_at=now(),status='complete_light_checked',seed=seed,updates=18000,wall_s=time.monotonic()-start))
        record('net_seed'+str(seed),dict(status='complete_light_checked',terminal=ev(lane/'terminal.json'),owner=None))
    except Exception as e:
        atomic(lane/'terminal.json',dict(created_at=now(),status='failed_retained',error=repr(e),traceback=traceback.format_exc()))
        record('net_seed'+str(seed),dict(status='failed_retained',terminal=ev(lane/'terminal.json'),owner=None));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','cache','equivalence','train']);p.add_argument('--seed',type=int,default=2023);p.add_argument('--gpu',type=int,default=1);a=p.parse_args()
    train(a.seed,a.gpu) if a.mode=='train' else globals()[a.mode]()
