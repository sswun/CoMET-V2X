"""R2 exact-main H1 transfer: OPV full2170/allCAV and real V2 boundary.

Use selected author q/tau/no-position three heads WITHOUT target fitting.
Original target protocols/GT/ROI/history retained. No detector forward, no
delta/learning/association successor. Old own-head controls not relabeled.
"""
import argparse,inspect,time,traceback
from collections import defaultdict
import numpy as np
from peer19_v1_20261104 import *
import run_peer19_H1_assets_v1_20261104 as h1
OUT=BASE/'H1_boundaries'

def prepare():
    import run_peer15_opv_full_read_v1_20261103 as opv
    import run_peer16_E10_v2_relative_clock_v1_20261103 as v2
    for job in ('opv','v2','opv_independent','v2_independent'):
        register('H1_boundary_'+job,__file__,__doc__,dict(GPU_forwards=0,updates=0,CPU_threads=1,wall_h='bounded34h',disk_MiB='<450 compact output states/boxes per producer; independent checks only'),
            opv_input=ev(opv.INPUT/'terminal.json'),v2_input=ev(v2.INPUT/'contract_recovery_v6.json'),
            main_recipe=ev(BASE/'recipe_global_choice_interface_recovery_v2/contract.json'),
            noise='OPV SPD_structured/iid02 plus receiver-only50/100/150ms; all5fixed families0/400ms; no test tuning',
            exact_main_transfer='Same author head weights/source maps, no-position,q,tau as H1 chosen on SPD cal8. Original target own-head results remain separately labeled.')

def material_opv(rows,arrays,detections,noise,wait,clean=False):
    import run_peer15_opv_full_read_v1_20261103 as old
    m,b,f,d,meta,n=old.material(rows,arrays,detections,noise,wait,clean)
    st=h1.recipe.hard_state(m,SELECTED_Q);d=np.zeros_like(st['delta']) if clean else st['delta']
    b,f=h1.recipe.geometry(m,d);m=h1.recipe.prior.protected(m,TAU)
    return m,b,f,d,meta,st['anchors']

def opv():
    global HEADS,SELECTED_Q,TAU
    kind,SELECTED_Q,TAU,HEADS=h1.context()
    import run_peer15_opv_full_read_v1_20261103 as old
    from run_peer16_E9_opv_pose_time_v1_20261103 import pose_time_arrays
    dest=OUT/'opv';dest.mkdir(parents=True,exist_ok=False)
    # Reuse one materialization per family/wait; three identical head policies
    # share geometry, no duplicate detector data or target-training outputs.
    code=inspect.getsource(old.run)
    code=code.replace("record('opv_full_read'","record('H1_boundary_opv'")
    code=code.replace("rows=collect();assert len(rows)==2170","rows=load(INPUT/'records.json');assert len(rows)==2170")
    code=code.replace("for family in ('SPD_structured','iid_02m_02deg'):\n            noise=noises(rows,arrays,family)","for family in FAMILIES:\n            if family.startswith('ego_pose_time_'):\n                noisy_arrays,audit=pose_time_arrays(rows,arrays,int(family[-5:-2])/1000)\n                atomic(OUT/(family+'_pose_audit.json'),audit)\n                noise={(i,s):np.eye(4) for i,r in enumerate(rows) for s in range(1,len(r['sensors']))}\n            else:noisy_arrays=arrays;noise=noises(rows,arrays,family)")
    code=code.replace('m,b,f,d,meta,n=material(rows,arrays,detections,noise,wait)','m,b,f,d,meta,n=material(rows,noisy_arrays,detections,noise,wait)')
    begin="methods={METHODS[0]:(m,m['stale_boxes'],m['stale_features']),METHODS[1]:(m,b,f),METHODS[2]:(cm,cb,cf)}"
    code=code.replace(begin,"methods={}\n                for seed in HEADS:\n                    sf=m['stale_features'].copy();sf[:,REMOVED]=0\n                    methods['stale_seed'+str(seed)]=(m,m['stale_boxes'],sf)\n                    methods['H1_seed'+str(seed)]=(m,b,f)\n                    methods['oracle_seed'+str(seed)]=(cm,cb,cf)")
    code=code.replace("prior.HEAD) for k,(mm,bb,ff) in methods.items()","HEADS[int(k[-4:])]) for k,(mm,bb,ff) in methods.items()")
    code=code.replace("assert np.array_equal(scores[METHODS[1]],refit.score_with_head(m,f,prior.HEAD))","assert all(np.array_equal(scores['H1_seed'+str(s)],refit.score_with_head(m,f,HEADS[s])) for s in HEADS)")
    families=('SPD_structured','iid_02m_02deg','ego_pose_time_050ms','ego_pose_time_100ms','ego_pose_time_150ms')
    labels=tuple(k+str(s) for s in HEADS for k in ('stale_seed','H1_seed','oracle_seed'))
    # old.run needs a contract at its output root; refer to immutable registration.
    (dest/'contract.json').symlink_to(BASE/'H1_boundary_opv/contract.json')
    ns=dict(old.__dict__,OUT=dest,HEADS=HEADS,REMOVED=h1.recipe.prior.REMOVED,FAMILIES=families,METHODS=labels,material=material_opv,pose_time_arrays=pose_time_arrays,record=record,__file__=__file__)
    exec(compile(code,__file__+'::exact_main_H1_full_OPV','exec'),ns);ns['run']()
    atomic(dest/'main_transfer_identity.json',dict(kind=kind,q=SELECTED_Q,tau=TAU,heads={str(s):ev(p/'direct_u.pth') for s,p in HEADS.items()},no_target_fit=True,original_protocol_preserved=True))

def v2():
    import run_peer16_E10_v2_relative_clock_v1_20261103 as old
    kind,q,tau,heads=h1.context();dest=OUT/'v2';dest.mkdir(parents=True,exist_ok=False);panels=[]
    try:
        c=load(old.INPUT/'contract_recovery_v6.json')
        for role in ('mechanism_holdout14','val21'):
            ds=old.fac.PackedDataset(ROOT/c['datasets']['val' if role=='val21' else 'train']);m,rows=old.transfer.material(c,role,ds)
            st=h1.recipe.hard_state(m,q);b,f=h1.recipe.geometry(m,st['delta']);sf=m['stale_features'].copy();sf[:,h1.recipe.prior.REMOVED]=0
            mm=h1.recipe.prior.protected(m,tau);boxes={};scores={}
            for s,head in heads.items():
                boxes['H1_seed'+str(s)]=b;boxes['stale_seed'+str(s)]=m['stale_boxes']
                scores['H1_seed'+str(s)]=h1.recipe.score(m,f,head);scores['stale_seed'+str(s)]=h1.recipe.score(m,sf,head)
            old.fac.METHODS=tuple(boxes);pp,p=old.fac.evaluate_v2(ds,mm,boxes,scores)
            _,replay=old.fac.evaluate_v2(ds,mm,boxes,scores);assert all(np.array_equal(v,replay[k]) for k,v in p.items())
            panels.extend(dict(r,role=role) for r in pp);atomic(dest/'panels.json',panels);np.savez_compressed(dest/(role+'_predictions.npz'),**p);np.savez_compressed(dest/(role+'_state.npz'),**st)
        atomic(dest/'main_transfer_identity.json',dict(kind=kind,q=q,tau=tau,heads={str(s):ev(p/'direct_u.pth') for s,p in heads.items()},no_V2_fit=True,original_V2_BEV_protocol=True,postgeometry_ROI=False,SPD_head_transferred=True))
        atomic(dest/'terminal.json',dict(status='complete_light_checked',created_at=now(),panels=ev(dest/'panels.json'),independent_pending=True))
        record('H1_boundary_v2',dict(status='complete_light_checked',terminal=ev(dest/'terminal.json'),resource_owner=False,owner=None))
    except Exception as e:
        atomic(dest/'terminal.json',dict(status='failed_retained',created_at=now(),error=repr(e),traceback=traceback.format_exc()));record('H1_boundary_v2',dict(status='failed_retained',terminal=ev(dest/'terminal.json'),resource_owner=False,owner=None));raise

def opv_independent():
    import review_peer15_opv_full_v1_20261103 as old
    dest=OUT/'opv_independent';dest.mkdir(parents=True,exist_ok=False);wait(OUT/'opv/terminal.json',34)
    families=('SPD_structured','iid_02m_02deg','ego_pose_time_050ms','ego_pose_time_100ms','ego_pose_time_150ms')
    labels=tuple(k+str(s) for s in h1.recipe.SEEDS for k in ('stale_seed','H1_seed','oracle_seed'))
    code=inspect.getsource(old.run).replace("'opv_full_independent'","'H1_boundary_opv_independent'")
    code=code.replace("('SPD_structured','iid_02m_02deg')",repr(families)).replace("('stale_same_U','CoMET_heading_U','oracle_clean_heading_U')",repr(labels))
    ns=dict(old.__dict__,OUT=dest,PRODUCER=OUT/'opv',record=record);exec(compile(code,__file__+'::independent_full_OPV_AP','exec'),ns);ns['run']()

def v2_independent():
    import run_peer16_E10_v2_relative_clock_v1_20261103 as old
    from review_peer10_cached_hwl_independent_v1_20261101 import polygons,overlap_matrix,flags,metric
    from review_crosscheckpoint_current_q005_heading_geometry_peer06_v1_20261030 import heading_boxes
    import bootstrap_peer15_paper_v1_20261103 as boot
    dest=OUT/'v2_independent';dest.mkdir(parents=True,exist_ok=False);wait(OUT/'v2/terminal.json',34);reports=[];boots=[]
    c=load(old.INPUT/'contract_recovery_v6.json');source=inspect.getsource(boot.contrast).replace("'four_wait_mean'","'registered_wait_mean'");ns=dict(boot.__dict__);exec(compile(source,__file__+'::original_V2_two_wait_CI','exec'),ns)
    for role in ('mechanism_holdout14','val21'):
        ds=old.fac.PackedDataset(ROOT/c['datasets']['val' if role=='val21' else 'train']);m,rows=old.transfer.material(c,role,ds)
        with np.load(OUT/'v2'/(role+'_state.npz')) as z:d=z['delta'];b=m['cv_boxes'].copy()
        for i,(a,e) in enumerate(zip(m['condition_offsets'][:-1],m['condition_offsets'][1:])):
            a,e=int(a),int(e);ids=a+np.flatnonzero(m['source'][a:e]==1);b[ids]=heading_boxes(m['stale_boxes'][ids],m['cv_boxes'][ids]);b[ids,:2]+=d[i].astype(np.float32)
        gold=[r for r in load(OUT/'v2/panels.json') if r['role']==role]
        with np.load(OUT/'v2'/(role+'_predictions.npz')) as p:
            for s in h1.recipe.SEEDS:
                for prefix,boxes in [('H1',b),('stale',m['stale_boxes'])]:
                    label=prefix+'_seed'+str(s);stats=defaultdict(lambda:dict(score=[],tp50=[],tp70=[],GT=0))
                    for i,w in enumerate(m['wait_ms']):
                        a,e=map(int,p[label+'__offset'][i:i+2]);ids=p[label+'__candidate'][a:e].astype(int);score=p[label+'__score'][a:e];ga,ge=map(int,m['gt_offsets'][i:i+2]);gt=m['gt_boxes'][ga:ge];ov=overlap_matrix(polygons(boxes[ids]),polygons(gt));st=stats[int(w)];st['score'].extend(score);st['GT']+=len(gt)
                        for t in (50,70):
                            f,n=flags(ov,np.zeros(len(gt),np.int8),'all',t/100);assert np.array_equal(f[0],p[label+'__tp'+str(t)][a:e]);st['tp'+str(t)].extend(f[0])
                    for w,st in stats.items():
                        g=next(r for r in gold if r['method']==label and r['wait_ms']==w);r=dict(role=role,method=label,wait_ms=w)
                        for t in (50,70):
                            v=metric(st['tp'+str(t)],1-np.asarray(st['tp'+str(t)]),st['score'],st['GT']);assert abs(v-g['ap'+str(t)])<1e-8;r['AP'+str(t)]=v
                        reports.append(r)
                for t in (50,70):
                    def pack(name):return boot.make(m['sequence'],m['wait_ms'],np.diff(m['gt_offsets']),p[name+'__offset'],p[name+'__score'],p[name+'__tp'+str(t)])
                    boots.extend(ns['contrast'](dest,role+'_H1_seed'+str(s)+'_minus_stale',pack('H1_seed'+str(s)),pack('stale_seed'+str(s)),t,dict(role=role,exact_main_H1_transfer=True)))
                atomic(dest/'AP.json',reports);atomic(dest/'bootstrap.json',boots)
    atomic(dest/'terminal.json',dict(status='complete_independently_reviewed',created_at=now(),scope='Independent heading/BEV polygons/matching/AP; shared estimator states and readout scores; bootstrap established kernel.'))
    record('H1_boundary_v2_independent',dict(status='complete_independently_reviewed',terminal=ev(dest/'terminal.json'),resource_owner=False,owner=None))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','opv','v2','opv_independent','v2_independent']);globals()[p.parse_args().mode]()
