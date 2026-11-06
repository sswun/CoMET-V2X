"""Sole R7c collector -> saved-box independent AP/2000CI -> paired tables.

Wait using stdlib only. Science starts after the existing DAG and coverage
chain release their CPU slots; the existing R6f chain can use the other slot.
Native training/detections are shared, matching/AP/bootstrap independently
recomputed. Official GT and ROI fixed; no result-based recipe/seed selection.
"""
import argparse, fcntl, json, shlex, subprocess, sys, time, traceback
from pathlib import Path
from peer21_v1_20261105 import *

NAME='R7c_statistics_chain';OUT=BASE/'R7c_analysis'
SEEDS={'where':(20261101,20261102,20261103),'coalign':(20261102,20261103,20261104)}

def prepare():
    register(NAME,__file__,__doc__,dict(GPU_forwards=0,updates=0,CPU_threads=1,replicates=2000,
        disk_MiB='<450 new compact official/mask outcomes and draws; final/native predictions sole collection'),
        prerequisites=[ev(BASE/('R7c_'+a)/'contract.json') for a in SEEDS],
        CPU_admission='After finite_resource_DAG_v1 and R6_coverage_completion_v1_chain terminals; maximum one R7c statistics process',
        output_scope='val21 only, all/zero/colabelled/nonzero, clean_H1 vs same-budget PUBLIC/old_ONLINE; native method ROI separately retained in raw',
        comparison='Same seed/GT/common ROI; pair sequence draws before mean over three seeds/four waits; no new gate',
        H1_late_reference='Three frozen H1 heads of the appropriate detector domain; separate paradigm, not substituted for untested native geometry')

def collect():
    # Hash each immutable remote part just once after its terminal appears.
    code='''import pathlib,json,hashlib
p=pathlib.Path(REMOTE_PATH); result={}
for seed in (20261102,20261103,20261104):
 for part in ('training','evaluation','evaluation_profile'):
  q=p/'R7c'/'coalign'/('seed'+str(seed))/part;t=q/'terminal.json'
  if not t.exists():continue
  v=json.loads(t.read_text())
  if v['status']=='failed_retained':raise RuntimeError((str(t),v))
  manifest=q/'sole_collection_manifest.json'
  if not manifest.exists():
   if part=='training':names=['terminal.json','final.pth','train_ledger.jsonl','status.json']
   else:names=['terminal.json','records.json','panels.json','input_identity.json']+[f'wait{w:03d}.npz' for w in (0,100,200,400)]+[x.name for x in q.glob('replay_*.json')]
   values={str((q/n).relative_to(p)):hashlib.sha256((q/n).read_bytes()).hexdigest() for n in names}
   manifest.write_text(json.dumps(values))
  result[str(seed)+'/'+part]=json.loads(manifest.read_text())
print(json.dumps(result))'''.replace('REMOTE_PATH',repr(str(REMOTE_ROOT/BASE.relative_to(ROOT))))
    cmd=[str(REMOTE_ROOT/'.conda/opencood-linux/bin/python'),'-c',code]
    r=subprocess.run(['ssh','-o','ConnectTimeout=8','remote-gpu',shlex.join(cmd)],capture_output=True,text=True,timeout=90)
    assert r.returncode==0,(r.stdout,r.stderr)
    manifest=load(BASE/NAME/'collection.json') if (BASE/NAME/'collection.json').exists() else {}
    for part,files in json.loads(r.stdout).items():
        if part in manifest:assert manifest[part]['SHA256']==files;continue
        for rel,digest in files.items():
            target=BASE/rel;target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists():subprocess.run(['rsync','-a','--ignore-existing','remote-gpu:'+str(REMOTE_ROOT/BASE.relative_to(ROOT)/rel),str(target)],check=True)
            assert ev(target)['sha256']==digest,('sole_remote_conflict',target)
        manifest[part]=dict(collected_at=now(),SHA256=files);atomic(BASE/NAME/'collection.json',manifest)
    lane=BASE/'R7c_coalign/terminal.json'
    if len(manifest)==9 and not lane.exists():
        path=REMOTE_ROOT/lane.relative_to(ROOT)
        exists=subprocess.run(['ssh','remote-gpu',shlex.join(['test','-f',str(path)])]).returncode==0
        if not exists:return
        subprocess.run(['rsync','-a','--ignore-existing','remote-gpu:'+str(path),str(lane)],check=True)
        assert load(lane)['status']=='complete_light_checked'
        record('R7c_coalign',dict(status='complete_light_checked',terminal=ev(lane),owner=None,resource_owner=False))

def sources(arch,seed,method):
    if method=='clean_H1':return BASE/'R7c'/arch/('seed'+str(seed))/'evaluation','COMET_H1'
    if arch=='where':
        import run_peer12_where_confirmation_v1_20261102 as w
        from peer12_cycle06_v1_20261102 import CYCLE as old
        name='public' if method=='PUBLIC' else 'online_k1'
        root=old/('where_'+name+'_v6') if seed==20261101 else w.CYCLE/('where_confirmation_gpu'+('0' if method=='PUBLIC' else '1'))/('seed'+str(seed))
        return root/'val21/evaluation','PUBLIC' if method=='PUBLIC' else 'ONLINE_K1'
    import run_peer13_coalign_seed_confirmation_v1_20261102 as co
    if seed==20261102:
        return co.C15/'coalign_full_v1'/('adapted_evaluation' if method=='PUBLIC' else 'online_adapted_evaluation'),'NO_GRAPH' if method=='PUBLIC' else 'K1_INPUT'
    return co.BASE/('seed'+str(seed))/(('public' if method=='PUBLIC' else 'online')+'_reference/evaluation'),'NO_GRAPH' if method=='PUBLIC' else 'K1_INPUT'

def cases(path,arch,arm):
    import numpy as np
    import peer20_saved_native_v1_20261105 as native
    if arch=='where':yield from native.shard_cases(path,arm);return
    records=load(path/'records.json')
    for w in (0,100,200,400):
        rows=[r for r in records if r['wait_ms']==w]
        with np.load(path/f'wait{w:03d}.npz') as z,np.load(ROOT/rows[0]['GT_shard']) as g:
            assert len({r['GT_shard'] for r in rows})==1
            for r in rows:
                j=r['shard_case'];gt=g[f"c{r['GT_case']}_gt"]
                yield r,z[f'c{j}_{arm}_box3d_lidar'],z[f'c{j}_{arm}_scores'],z[f'c{j}_{arm}_label_preds'],gt[:,:7],gt[:,7]

def stats(arch,seed,method,review):
    import numpy as np
    import peer20_saved_native_v1_20261105 as native
    from peer20_population_core_v1_20261105 import save_stats
    tag=arch+'_'+str(seed)+'_'+method;source,arm=sources(arch,seed,method)
    domain='where2comm' if arch=='where' else 'author'
    m,rows,p,b=native.assemble(domain,'val21',cases(source,arch,arm),'native',review)
    src=OUT/tag;dest=OUT/(tag+'_independent') if review else src;dest.mkdir(parents=True,exist_ok=False)
    path=src/'predictions.npz'
    if review:
        with np.load(path) as z:assert set(z.files)==set(p) and all(np.array_equal(z[k],v) for k,v in p.items())
    else:
        np.savez_compressed(path,**p)
        atomic(src/'identity.json',dict(source=ev(source/'terminal.json'),records=ev(source/'records.json'),
            canonical_GT_checked=True,original_common_ROI=True,method=method,input_arm=arm,
            no_original_training_or_prediction_rerun=True))
    save_stats(dest,m,rows,p,'native','result',review,load(src/'result_statistics.json') if review else None)
    atomic(dest/'terminal.json',dict(created_at=now(),status='complete_independently_reviewed' if review else 'complete_light_checked',
        statistics=ev(dest/'result_statistics.json'),matching_independent=review,AP_CI_independent=review,training_shared=True))

def contrasts():
    import numpy as np
    results=[]
    for arch,seeds in SEEDS.items():
        for method in ('PUBLIC','old_ONLINE'):
            for pop in ('all','zero','colabelled','nonzero'):
                for th in (50,70):
                    ap={};draws={}
                    for meth in ('clean_H1',method):
                        points=[];dd=[]
                        for seed in seeds:
                            src=OUT/(arch+'_'+str(seed)+'_'+meth)
                            assert load(src.with_name(src.name+'_independent')/'terminal.json')['status']=='complete_independently_reviewed'
                            rr=load(src/'result_statistics.json')
                            points.append([next(r['AP'] for r in rr if r['population']==pop and r['threshold']==th and r['wait_ms']==w) for w in (0,100,200,400)])
                            with np.load(src/'result_draws.npz') as z:dd.append(z[pop+'_AP'+str(th)])
                        ap[meth]=np.mean(points,axis=0);draws[meth]=np.mean(dd,axis=0)
                    delta=ap['clean_H1']-ap[method];d=draws['clean_H1']-draws[method]
                    for w,pt,dd in zip((0,100,200,400,'four_wait_mean'),list(delta)+[delta.mean()],list(d)+[d.mean(0)]):
                        results.append(dict(architecture=arch,population=pop,threshold=th,wait_ms=w,
                            comparison='clean_H1_minus_'+method,delta_pp=float(pt),CI95=np.quantile(dd,[.025,.975]).tolist(),replicates=2000,
                            seed_pairing=list(seeds),val21_report_only=True))
    atomic(OUT/'contrasts.json',results)
    atomic(OUT/'terminal.json',dict(created_at=now(),status='complete_independently_reviewed',
        contrasts=ev(OUT/'contrasts.json'),not_complete_CoAlign_system_training=True,no_root_table_mutation=True,
        no_new_adoption_gate=True,no_science_followup=True))

def chain():
    dest=BASE/NAME;lock=(dest/'owner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);completed=[]
    deps=[P20/'finite_resource_DAG_v1/terminal.json',P20/'R6_coverage_completion_v1_chain/terminal.json']
    try:
        while True:
            collect()
            waiting=[str(p.relative_to(ROOT)) for p in deps if not p.exists()]
            waiting += [str((BASE/('R7c_'+a)/'terminal.json').relative_to(ROOT)) for a in SEEDS if not (BASE/('R7c_'+a)/'terminal.json').exists()]
            if not waiting:break
            atomic(dest/'progress.json',dict(created_at=now(),waiting=waiting,dataset_waiters=0,no_model_wakeup=True))
            time.sleep(60)
        for p in deps:assert load(p)['status']!='failed_retained'
        for arch,seeds in SEEDS.items():
            for seed in seeds:
                for method in ('clean_H1','PUBLIC','old_ONLINE'):
                    for mode in ('run','review'):
                        tag=arch+'_'+str(seed)+'_'+method+'_'+mode
                        atomic(dest/'progress.json',dict(created_at=now(),current=tag,completed=completed))
                        command=[sys.executable,'-u',str(Path(__file__).resolve()),mode,'--arch',arch,'--seed',str(seed),'--method',method]
                        with (dest/(tag+'.log')).open('x') as log:rc=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT).returncode
                        assert rc==0,(tag,rc);completed.append(tag)
        contrasts()
        atomic(dest/'terminal.json',dict(created_at=now(),status='complete_independently_reviewed',completed=completed,analysis=ev(OUT/'terminal.json')))
        record(NAME,dict(status='complete_independently_reviewed',terminal=ev(dest/'terminal.json'),owner=None,resource_owner=False))
    except Exception as e:
        atomic(dest/'terminal.json',dict(created_at=now(),status='failed_retained',error=repr(e),traceback=traceback.format_exc(),completed=completed))
        record(NAME,dict(status='failed_retained',terminal=ev(dest/'terminal.json'),owner=None,resource_owner=False));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','chain','run','review'))
    p.add_argument('--arch',choices=tuple(SEEDS));p.add_argument('--seed',type=int);p.add_argument('--method',choices=('clean_H1','PUBLIC','old_ONLINE'))
    a=p.parse_args()
    if a.mode=='prepare':prepare()
    elif a.mode=='chain':chain()
    else:stats(a.arch,a.seed,a.method,a.mode=='review')
