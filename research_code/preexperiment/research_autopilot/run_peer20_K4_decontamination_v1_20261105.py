"""R7 single TraF decontamination round; only fit24 label loading changes.

Same original K4 ONLINE_K1 input, initial e60 weight, budget and three seeds.
Public inference geometry is never fitted to labels. The label translation is
noncausal interpolation inside each fit sequence, applied only to inf-only
objects at exact-zero frames. No source code/dataset is edited in place.
"""
import argparse,copy,inspect,os,subprocess,sys,time,traceback
import numpy as np
from peer20_v1_20261105 import *

OUT=BASE/'R7';SEEDS=(20261101,20261102,20261103)
TRAIN=ROOT/'preexperiment/research_autopilot/run_peer10_k4_native_finetune_v1_20261101.py'
EVAL=ROOT/'preexperiment/research_autopilot/run_peer10_native_evaluate_v1_20261101.py'
MAIN=AUDIT/'cycle15_peer11_v1/contract.json'

def prepare():
    assert load(BASE/'R6a_raw_labels_independent_v2/terminal.json')['confirmed']
    frame=load(BASE/'R6a_raw_labels/frames.json');filled={};ident=[]
    for seq in sorted({r['sequence'] for r in frame if r['role']=='fit24'}):
        rows=sorted([r for r in frame if r['role']=='fit24' and r['sequence']==seq],key=lambda r:r['timestamp_us'])
        t=np.asarray([r['timestamp_us'] for r in rows],float);x=np.asarray([r['offset'] for r in rows],float);ok=np.any(x!=0,axis=1)
        # Entirely missing sequences cannot be filled; retain official with explicit identity.
        f=np.column_stack([np.interp(t,t[ok],x[ok,k]) for k in (0,1)]) if ok.any() else x.copy()
        for r,v in zip(rows,f):
            if r['zero']:filled[r['frame']]=dict(delta=v.tolist(),sequence=seq,nonzero_support=int(ok.sum()),zero=r['zero'])
    atomic(OUT/'fit24_label_deltas.json',dict(created_at=now(),frames=filled,roles=['fit24'],source=ev(BASE/'R6a_raw_labels/frames.json'),noncausal_label_only=True))
    for seed in SEEDS:
        spec=load(MAIN);spec['K4']['seed']=seed
        spec['R7']=dict(label_deltas=ev(OUT/'fit24_label_deltas.json'),source=ev(MAIN),training_script=ev(TRAIN),evaluation_script=ev(EVAL),
            changes='Only dataset.coop_label_dict: fit24 exact-zero inf-only xy += filled offset. Native pointcloud inputs, model, loss, optimizer and budget unchanged.')
        atomic(OUT/('seed'+str(seed)+'_contract.json'),spec)
    for lane,seeds in [('local',(SEEDS[0],SEEDS[2])),('remote',(SEEDS[1],))]:
        register('R7_'+lane,__file__,__doc__,dict(GPU_updates=2000*len(seeds),microbatches=8000*len(seeds),
            native_forwards=(13264+9744+8)*len(seeds),CPU_threads=2,disk_GiB='<2 per seed; active resume then final and compact predictions',wall_h='finite3seeds; no deadline'),
            seeds=list(seeds),R6a=ev(BASE/'R6a_raw_labels_independent_v2/terminal.json'),
            label_delta=ev(OUT/'fit24_label_deltas.json'),old_train=ev(TRAIN),old_eval=ev(EVAL),
            adoption='Co-labelled AP70 >= original K4+.40 and paired2000 CI lower>0; official >= original K4-.20. Otherwise supplement; no successor.',
            inference_input='Exact original K4 ONLINE_K1 (not relabelled H1); compare same-input original K4 and same-budget PUBLIC. No GT or offset in inference.')
    register('R7_analysis',__file__,'After both lanes finish, full independent saved-box AP/official and co-labelled2000CI vs original K4 and same-budget PUBLIC. No root-table adoption without complete comparisons.',
        dict(GPU_forwards=0,CPU_threads=1,disk_MiB='<50 compact AP/CI'),implementation_state='registered_unimplemented_analysis; cannot claim completed')

def install_label_loader(builder,delta):
    def patched(*args,**kwargs):
        ds=builder(*args,**kwargs);changed=0;unchanged=0
        for frame,spec in delta.items():
            if frame not in ds.coop_label_dict:continue
            objects=copy.deepcopy(ds.coop_label_dict[frame]);original=ds.coop_label_dict[frame]
            for a,b in zip(objects,original):
                assert 'from_side' in a,'label loader removed provenance: requires scope review'
                if a['from_side']=='inf':
                    for j,key in enumerate(('x','y')):a['3d_location'][key]+=spec['delta'][j]
                    changed+=1
                else:assert a==b;unchanged+=1
            ds.coop_label_dict[frame]=objects
        ds.peer20_label_patch=dict(changed_inf_objects=changed,unchanged_other_objects=unchanged,
            fit24_only=True,noncausal_label_only=True,no_inference_patch=True)
        return ds
    return patched

def worker(mode,seed,output,gpu):
    contract=OUT/('seed'+str(seed)+'_contract.json')
    if mode=='train':
        import qualify_traf_native4_cpu_v2_20261010 as qual
        original=qual.build;delta=load(OUT/'fit24_label_deltas.json')['frames']
        def build(root):
            h,b,roots,c,s=original(root)
            return h,install_label_loader(b,delta),roots,c,s
        qual.build=build
        sys.argv=[str(TRAIN),'--workspace',str(ROOT),'--contract',str(contract),'--output',str(output),'--arm','ONLINE_K1','--gpu',str(gpu)]
        ns=dict(__file__=str(TRAIN),__name__='peer20_train_loader_only');exec(compile(TRAIN.read_text(),str(TRAIN),'exec'),ns);ns['main']()
    else:
        sys.argv=[str(EVAL),'--workspace',str(ROOT),'--contract',str(contract),'--output',str(output),
            '--weight',str(OUT/('seed'+str(seed))/'training/final.pth'),'--arms','ONLINE_K1','--gpu',str(gpu)]
        code=EVAL.read_text()
        if mode=='hold':
            import run_peer10_k4_hold14_native_v1_20261101 as old
            for a,b in old.CHANGES:
                assert code.count(a)==1,(a,code.count(a));code=code.replace(a,b)
        ns=dict(__file__=str(EVAL),__name__='peer20_unmodified_official_eval');exec(compile(code,str(EVAL)+'::'+mode,'exec'),ns);ns['main']()

def lane(which,gpu):
    seeds=(SEEDS[0],SEEDS[2]) if which=='local' else (SEEDS[1],);name='R7_'+which;dest=BASE/name;results=[]
    try:
        assert load(BASE/'R6a_raw_labels_independent_v2/terminal.json')['confirmed']
        if ROOT!=REMOTE_ROOT:record(name,dict(status='running_fixed_budget_label_only_lane',resource_owner=True))
        for seed in seeds:
            for mode,folder in [('train','training'),('val','val21'),('hold','mechanism_holdout14')]:
                out=OUT/('seed'+str(seed))/folder;out.parent.mkdir(parents=True,exist_ok=True)
                command=[str(ROOT/'.conda/opencood-linux/bin/python'),'-u',str(Path(__file__)),mode,'--seed',str(seed),'--output',str(out),'--gpu',str(gpu)]
                atomic(dest/'progress.json',dict(created_at=now(),seed=seed,mode=mode,command=command,completed=results))
                with (dest/(str(seed)+'_'+mode+'.log')).open('x') as log:
                    rc=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT).returncode
                assert rc==0,(mode,seed,rc)
                results.append(dict(seed=seed,mode=mode,terminal=ev(out/'terminal.json')))
        atomic(dest/'terminal.json',dict(created_at=now(),status='complete_light_checked',results=results,independent_AP_and_comparison_pending=True))
        if ROOT!=REMOTE_ROOT:record(name,dict(status='complete_light_checked',terminal=ev(dest/'terminal.json'),owner=None,resource_owner=False))
    except Exception as e:
        atomic(dest/'terminal.json',dict(created_at=now(),status='failed_retained',error=repr(e),traceback=traceback.format_exc(),results=results))
        if ROOT!=REMOTE_ROOT:record(name,dict(status='failed_retained',terminal=ev(dest/'terminal.json'),owner=None,resource_owner=False))
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','local','remote','train','val','hold'));p.add_argument('--seed',type=int);p.add_argument('--gpu',type=int,default=1);p.add_argument('--output',type=Path);a=p.parse_args()
    if a.mode=='prepare':prepare()
    elif a.mode in ('local','remote'):lane(a.mode,a.gpu)
    else:worker(a.mode,a.seed,a.output,a.gpu)
