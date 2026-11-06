"""H1 all-ego hard anchors; fit/cal-only final recipe and declared diagnostics.

q is selected on cal8 with frozen3heads/oldtau, from the original four q values.
Fit three geometry-matched no-position heads at that q using the exact old batch
schedule. Per-domain/kind tau uses cal8/three heads. A separate coordinator
chooses frozen vs refit globally on five-domain cal8 AP70 (tie frozen), before
reading formal outcomes. Diagnostic ablations retain original q/tau/heads.
No learning/association successor is created beyond peer19's explicit controls.
"""
import argparse,inspect,time,traceback
import numpy as np
from peer19_v1_20261104 import *
import run_peer17_final_recipe_v1_20261104 as prior
from run_peer16_cached_methods_v1_20261103 import materials,DOMAINS
from run_peer16_E1_timing_audit_v1_20261103 import match
import run_crosscheckpoint_temporal_matched39d_v1_20261029 as matched
OUT=BASE/'recipe_v3'
SEEDS=prior.SEEDS
Q=(.025,.05,.1,.15)
MODES=('U0','U1','U2','U3','H1','H2','H3','U5','U6','N2')
ROLES=('calibration8','val21','mechanism_holdout14')

def observations(m,threshold=None,gate=2.5):
    n=[];z=[]
    for aa,bb in zip(m['condition_offsets'][:-1],m['condition_offsets'][1:]):
        a,b=int(aa),int(bb);ri=a+np.flatnonzero(m['source'][a:b]==1)
        keep=m['source'][a:b]==0
        if threshold is not None:keep &= m['raw_scores'][a:b]>=threshold
        ei=a+np.flatnonzero(keep);ii,jj=match(m['cv_boxes'][ri],m['cv_boxes'][ei],gate)
        residual=m['cv_boxes'][ei[jj],:2].astype(float)-m['cv_boxes'][ri[ii],:2]
        n.append(len(ii));z.append(residual.mean(0) if len(ii) else np.zeros(2))
    return np.asarray(n),np.asarray(z)

def hard_state(m,q,threshold=None,gate=2.5):
    n,z=observations(m,threshold,gate)
    delta,var,age,_=prior.old.author.kf(m,n,z,q,.6)
    return dict(delta=delta,variance=var,step=age,anchors=n,observation=z)

def diagnostic_state(m,q,mode):
    state={};delta=[];mass=[]
    for a0,b0,seq,w in zip(m['condition_offsets'][:-1],m['condition_offsets'][1:],m['sequence'],m['wait_ms']):
        a,b=int(a0),int(b0);ri=a+np.flatnonzero(m['source'][a:b]==1)
        ei=a+np.flatnonzero(m['source'][a:b]==0);fixed=a+np.flatnonzero((m['source'][a:b]==0)&(m['protected'][a:b]==1))
        key=str(seq),int(w);x,p=state.get(key,(np.zeros(2),np.ones(2)));pp=p+q*q;z=None;precision=0.
        if mode in ('N0','U3','H1','H2','H3'):
            target=fixed if mode in ('N0','U3') else ei;remote=m['cv_boxes'][ri].astype(float).copy()
            if mode in ('U3','H2','H3'):remote[:,:2]+=x
            ii,jj=match(remote,m['cv_boxes'][target],1.5 if mode=='H3' else 2.5)
            if len(ii):
                z=(m['cv_boxes'][target[jj],:2].astype(float)-m['cv_boxes'][ri[ii],:2]).mean(0);precision=len(ii)/.36
        else:
            target=fixed if mode=='U2' else ei
            if len(ri) and len(target):
                res=m['cv_boxes'][target,:2].astype(float)[None,:,:]-m['cv_boxes'][ri,:2][:,None,:]
                innovation=res-(np.zeros(2) if mode=='U1' else x);sigma={'U5':.5,'U6':2.}.get(mode,1.)
                nr,ne=len(ri),len(target);total=nr+ne;lp=np.zeros((nr+1,ne+1));lp[:nr,:ne]=-(innovation**2).sum(-1)/(2*sigma*sigma)
                mu=np.ones(nr+1)/total;nu=np.ones(ne+1)/total;mu[-1]=ne/total;nu[-1]=nr/total
                for _ in range(8):
                    lp+=np.log(mu)[:,None]-np.logaddexp.reduce(lp,axis=1,keepdims=True)
                    lp+=np.log(nu)[None,:]-np.logaddexp.reduce(lp,axis=0,keepdims=True)
                weight=np.exp(lp[:nr,:ne])*total/.36;precision=float(weight.sum())
                if precision>1e-12:z=np.sum(weight[:,:,None]*res,axis=(0,1))/precision
        if z is not None:
            gain=pp/(pp+1/precision);x=x+gain*(z-x);p=(1-gain)*pp
        else:p=pp
        state[key]=(x.copy(),p);delta.append(x.copy());mass.append(precision)
    return np.asarray(delta),np.asarray(mass)

def geometry(m,delta):
    b,f=prior.old.author.heading.build_heading(m,delta);f[:,prior.REMOVED]=0
    return b,f
def head(domain,kind,seed):return prior.head(domain,'no_position',seed) if kind=='frozen' else OUT/domain/('refit_seed'+str(seed))
def score(m,f,path):return prior.old.author.refit.score_with_head(m,f,path)
def evaluate(m,b,f,path,name,tau):return prior.old.author.refit.fixed.base.evaluate(prior.protected(m,tau),{name:b},{name:score(m,f,path)})

def prepare(domain):
    register('recipe_'+domain,__file__,__doc__,dict(GPU_forwards=0,updates=6000,CPU_threads=1,
        disk_MiB='<650 compact outputs/selected states and3heads; shared GT/points untouched',wall_h='bounded24h'),
        parent_recipe=ev(P17/'final_recipe/contract.json'),q_grid=list(Q),tau_grid=list(prior.GRID),
        diagnostic=ev(ROOT/'preexperiment/research_autopilot/main_chat_n2_ablation_diag_20261104.py'),
        source_loader=ev(ROOT/'preexperiment/research_autopilot/run_peer16_cached_methods_v1_20261103.py'),
        seeds=list(SEEDS),domain=domain,selection='q frozen-threehead cal8 first; geometry-matched fit24 heads then cal8 tau per-kind; global five-domain cal8 kind decision, exact tie frozen',
        anchor='ALL source==0; Hungarian+mutual NN <=2.5m; no prediction gating. Readout protected prefix remains separate.',
        sensitivity='All,.1,.2,.3,.4,.5,original-high score and1.5,2,2.5,3m; report only, never select')

def run(domain):
    name='recipe_'+domain;begin=time.monotonic();dest=OUT/domain;dest.mkdir(parents=True,exist_ok=False)
    record(name,dict(status='running_H1_fitcal_recipe',resource_owner=True))
    try:
        data,oldhead,oldq=materials(domain);oldtau=load(prior.OUT/domain/'calibration.json')['selected_tau'];m,_=data['calibration8']
        state=hard_state(m,oldq);n,z=state['anchors'],state['observation'];calq=[]
        for q in Q:
            d=prior.old.author.kf(m,n,z,q,.6)[0];b,f=geometry(m,d)
            for s in SEEDS:
                label='q'+str(q)+'_seed'+str(s);pp,p=evaluate(m,b,f,head(domain,'frozen',s),label,oldtau)
                calq.extend(dict(r,q=q,seed=s,role='calibration8') for r in pp)
                np.savez_compressed(dest/(label+'_cal_predictions.npz'),**p)
            atomic(dest/'q_calibration_panels.json',calq)
        qs=[dict(q=q,mean_AP70=float(np.mean([r['ap70'] for r in calq if r['q']==q]))) for q in Q]
        chosen=max(qs,key=lambda r:(r['mean_AP70'],-r['q']))['q']
        atomic(dest/'q_calibration.json',dict(created_at=now(),selected_q=chosen,grid=qs,selection_role='calibration8',before_formal=True))
        fitcal={}
        for role in ('fit24','calibration8'):
            m,_=data[role];st=hard_state(m,chosen);np.savez_compressed(dest/(role+'_state.npz'),**st)
            b,f=geometry(m,st['delta']);fitcal[role]=matched.geometry_material(m,b,f)
        for s in SEEDS:
            source=inspect.getsource(matched.train_head).replace('torch.manual_seed(readout.MODEL_SEED)','torch.manual_seed(SEED)').replace('"seed": readout.MODEL_SEED','"seed": SEED').replace('torch.set_num_threads(2)','torch.set_num_threads(1)')
            ns=dict(matched.__dict__,SEED=s);exec(compile(source,__file__+'::fixed_seed_H1_readout','exec'),ns)
            path=head(domain,'refit',s);ns['train_head'](fitcal['fit24'],oldhead/'training_schedule.npz',path)
            matched.source_map(fitcal['calibration8'],matched.score(fitcal['calibration8'],path),path)
        del fitcal
        m,_=data['calibration8'];st=hard_state(m,chosen);b,f=geometry(m,st['delta']);cal=[];choices={}
        for kind in ('frozen','refit'):
            for s in SEEDS:
                raw=score(m,f,head(domain,kind,s));allpred={}
                for tau in prior.GRID:
                    label=kind+'_tau'+str(round(tau*100))+'_seed'+str(s)
                    pp,p=prior.old.author.refit.fixed.base.evaluate(prior.protected(m,tau),{label:b},{label:raw})
                    cal.extend(dict(r,kind=kind,seed=s,tau=tau,role='calibration8',q=chosen) for r in pp);allpred.update(p)
                np.savez_compressed(dest/(kind+'_cal_seed'+str(s)+'_predictions.npz'),**allpred)
                atomic(dest/'calibration_panels.json',cal)
            grid=[dict(tau=t,mean_AP70=float(np.mean([r['ap70'] for r in cal if r['kind']==kind and r['tau']==t]))) for t in prior.GRID]
            choices[kind]=max(grid,key=lambda r:(r['mean_AP70'],-r['tau']))
        atomic(dest/'calibration.json',dict(created_at=now(),q=chosen,choices=choices,cal_only=True,seeds=list(SEEDS),global_kind_pending=True))
        panels=[]
        for role in ('val21','mechanism_holdout14'):
            m,rows=data[role];st=hard_state(m,chosen);np.savez_compressed(dest/(role+'_state.npz'),**st)
            atomic(dest/(role+'_state_ready.json'),dict(created_at=now(),status='complete_light_checked',q=chosen,state=ev(dest/(role+'_state.npz')),
                keys=[(str(r['sequence']),int(r['receiver_timestamp_us']),int(r['wait_ms'])) for r in rows]))
            b,f=geometry(m,st['delta'])
            for kind in ('frozen','refit'):
                for s in SEEDS:
                    label='H1_'+kind+'_seed'+str(s);pp,p=evaluate(m,b,f,head(domain,kind,s),label,choices[kind]['tau'])
                    panels.extend(dict(r,role=role,kind=kind,seed=s,q=chosen,tau=choices[kind]['tau']) for r in pp)
                    np.savez_compressed(dest/(role+'_'+kind+'_seed'+str(s)+'_predictions.npz'),**p);atomic(dest/'panels.json',panels)
        atomic(dest/'primary_terminal.json',dict(created_at=now(),status='complete_light_checked',calibration=ev(dest/'calibration.json'),panels=ev(dest/'panels.json'),no_formal_selection=True))
        record(name,dict(status='running_declared_ablation_controls',resource_owner=True))
        ablations=[];diagchecks=[]
        for role in ROLES:
            m,rows=data[role]
            for mode in ('N0',*MODES):
                if mode!='N2':
                    delta,mass=diagnostic_state(m,oldq,mode)
                    np.savez_compressed(dest/(role+'_'+mode+'_diag_state.npz'),delta=delta,mass=mass)
                for s in SEEDS:
                    if mode=='N2':
                        import evaluate_peer18_comet_net_v1_20261104 as neural
                        if role=='calibration8':
                            neural.engine.imports();wp=P18/'CoMET_Net/training_recovery_v2'/('seed'+str(s))/domain/'N2/selected.pth'
                            _,_,delta,_=neural.infer(m,rows,domain,domain,'N2',s,wp)
                        else:
                            with np.load(P18/'CoMET_Net/formal'/('seed'+str(s))/(domain+'__to__'+domain+'__N2')/(role+'_state.npz')) as p:delta=p['delta'].copy()
                        np.savez_compressed(dest/(role+'_N2_seed'+str(s)+'_diag_state.npz'),delta=delta)
                    b,f=geometry(m,delta);label=mode+'_seed'+str(s);pp,p=evaluate(m,b,f,head(domain,'frozen',s),label,oldtau)
                    ablations.extend(dict(r,role=role,seed=s,variant=mode,q=oldq,tau=oldtau) for r in pp)
                    np.savez_compressed(dest/(role+'_'+mode+'_seed'+str(s)+'_diag_predictions.npz'),**p)
                    if s==2023 and mode!='N2':
                        dp=P18/'main_chat_n2_diag'/(domain+('_stage2' if mode in ('H1','H2','H3','U5','U6') else '')+'.json')
                        ref=load(dp)['roles'][role]['methods'][mode]['panels']
                        err=max(abs(r['ap'+str(th)]-next(x for x in ref if x['wait_ms']==r['wait_ms'])['ap'+str(th)]) for r in pp for th in (50,70))
                        assert err<1e-8,(domain,role,mode,err);diagchecks.append(dict(role=role,mode=mode,max_AP_error=err))
                    atomic(dest/'ablation_panels.json',ablations);atomic(dest/'progress.json',dict(created_at=now(),phase='declared_ablation',role=role,variant=mode,seed=s,panels=len(ablations)))
        atomic(dest/'peer_diagnostic_replay.json',dict(status='passed',checks=diagchecks,original_diagnostic_not_independent=True))
        sensitivity=[];m,_=data['val21']
        high=float(np.min(m['raw_scores'][m['protected']==1]))
        specs=[('score_all',None,2.5)]+[('score_'+str(t),t,2.5) for t in (.1,.2,.3,.4,.5)]+[('score_original',high,2.5)]+[('gate_'+str(g),None,g) for g in (1.5,2.,2.5,3.)]
        for label,threshold,gate in specs:
            ss=hard_state(m,chosen,threshold,gate);b,f=geometry(m,ss['delta'])
            np.savez_compressed(dest/(label+'_sensitivity_state.npz'),**ss)
            for kind in ('frozen','refit'):
                for s in SEEDS:
                    tag=label+'_'+kind+'_seed'+str(s);pp,p=evaluate(m,b,f,head(domain,kind,s),tag,choices[kind]['tau'])
                    sensitivity.extend(dict(r,threshold=threshold,gate=gate,kind=kind,seed=s) for r in pp)
                    np.savez_compressed(dest/(tag+'_sensitivity_predictions.npz'),**p)
            atomic(dest/'sensitivity_panels.json',sensitivity)
        atomic(dest/'identity.json',dict(domain=domain,q=chosen,anchor='source==0, Hungarian+mutual NN<=2.5m',prediction_gating=False,
            r_variance='.36/n',heading=True,no_position=True,inference_GT=False,readout_prefix_separate=True,
            heads={k:{str(s):dict(weights=ev(head(domain,k,s)/'direct_u.pth'),maps=ev(head(domain,k,s)/'source_maps.json')) for s in SEEDS} for k in ('frozen','refit')}))
        atomic(dest/'terminal.json',dict(created_at=now(),status='complete_light_checked',primary=ev(dest/'primary_terminal.json'),
            identity=ev(dest/'identity.json'),peer_diagnostic=ev(dest/'peer_diagnostic_replay.json'),independent_pending=True,
            GPU_forwards=0,updates=6000,wall_s=time.monotonic()-begin,new_bytes=sum(p.stat().st_size for p in dest.rglob('*') if p.is_file())))
        record(name,dict(status='complete_light_checked',terminal=ev(dest/'terminal.json'),resource_owner=False,owner=None))
    except Exception as e:
        atomic(dest/'terminal.json',dict(created_at=now(),status='failed_retained',error=repr(e),traceback=traceback.format_exc()))
        record(name,dict(status='failed_retained',terminal=ev(dest/'terminal.json'),resource_owner=False,owner=None));raise

def choose():
    name='recipe_global_choice';record(name,dict(status='waiting_five_domain_calibration',resource_owner=False,owner=None))
    try:
        # calibration.json is an immutable decision payload, not a terminal.
        begin=time.monotonic()
        while not all((OUT/d/'calibration.json').exists() for d in DOMAINS):
            for d in DOMAINS:
                p=OUT/d/'terminal.json'
                if p.exists():assert load(p)['status']=='complete_light_checked',(d,load(p))
            assert time.monotonic()-begin<24*3600;time.sleep(30)
        rows={d:load(OUT/d/'calibration.json') for d in DOMAINS}
        value={k:float(np.mean([r['choices'][k]['mean_AP70'] for r in rows.values()])) for k in ('frozen','refit')}
        selected='refit' if value['refit']>value['frozen'] else 'frozen'
        atomic(OUT/'method_registry.json',dict(created_at=now(),status='complete_light_checked',method='CoMET_all_ego_H1',
            selected_kind=selected,cal8_five_domain_AP70=value,domain_calibration={d:ev(OUT/d/'calibration.json') for d in DOMAINS},
            exact_tie='frozen',selection_roles=['calibration8'],N2_in_main_method=False,old_N0_and_N2_records_preserved=True,no_formal_selection=True))
        atomic(BASE/name/'terminal.json',dict(created_at=now(),status='complete_light_checked',registry=ev(OUT/'method_registry.json')))
        record(name,dict(status='complete_light_checked',terminal=ev(BASE/name/'terminal.json'),resource_owner=False,owner=None))
    except Exception as e:
        atomic(BASE/name/'terminal.json',dict(created_at=now(),status='failed_retained',error=repr(e),traceback=traceback.format_exc()))
        record(name,dict(status='failed_retained',terminal=ev(BASE/name/'terminal.json'),resource_owner=False,owner=None));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run','choose']);p.add_argument('--domain',choices=DOMAINS)
    a=p.parse_args();choose() if a.mode=='choose' else globals()[a.mode](a.domain)
