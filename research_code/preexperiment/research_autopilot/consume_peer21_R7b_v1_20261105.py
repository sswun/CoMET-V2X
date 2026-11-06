"""Finite sole collector and R7b independent AP/2000CI, cal-only tier choice.

Baseline outcomes reference existing independently checked R7/R6b, not another
native forward or a copy of their raw predictions. H1 uses final three frozen
heads (2023/24/25), paired to three native seeds by their fixed index; all GT
identities must pass the shared original-current-GT canonical box comparison.
No main-table change; R7c awaits a passed independently confirmed gate.
"""
import argparse, fcntl, json, shlex, subprocess, sys, time, traceback
from pathlib import Path
from peer21_v1_20261105 import *
from run_peer21_clean_H1_native_v1_20261105 import OUT as TRAIN,SEEDS,TIERS

ROLES=('calibration8','val21','mechanism_holdout14')
OUT=BASE/'R7b_analysis'
NAME='R7b_consumer_chain'

def prepare():
    register(NAME,__file__,__doc__,dict(GPU_forwards=0,updates=0,CPU_threads=1,replicates=2000,
        disk_MiB='<300 compact statistics and TP; sole final-weight/prediction collector references raw inputs'),
        native_producer=ev(BASE/'R7b_local/contract.json'),
        independent_scope='Independent polygon matching + independent AP/bootstrap, shared raw detections/training',
        H1_reference='Final H1 recipe frozen three heads 2023/2024/2025, not just preview seed2023',
        max_total_heavy_local_CPU=2,launch_memory_GiB=28,
        cal_choice='cal8 official AP70 three-seed four-wait mean, tie A; no formal results read',
        gate='PUBLIC AP50 and AP70 CI low>0; H1 AP50>=+1 and CI low>0, AP70>=-.30; val21 only')
    for n in ('R7b_collector','R7b_statistics','R7b_independent','R7b_gate'):
        record(n,dict(status='implemented_queued_finite_consumer',implementation=ev(__file__),dispatcher=NAME,owner=None,resource_owner=False))
    record('R7c_conditional',dict(status='waiting_independent_R7b_gate_no_launch',owner=None,resource_owner=False,
        no_cross_architecture_training_before_gate=True))

def remote_manifest():
    p=REMOTE_ROOT/str(TRAIN.relative_to(ROOT))
    code='''import pathlib,json,hashlib
p=pathlib.Path(ROOT_PATH);result={}
for tier in ('A','B'):
 for role in ('training','calibration8','val21','mechanism_holdout14'):
  d=p/tier/'seed20261102'/role;t=d/'terminal.json'
  if not t.exists():continue
  state=json.loads(t.read_text())
  if state['status']=='failed_retained':raise RuntimeError((str(t),state))
  if role=='training':names=['terminal.json','status.json','qualification.json','sampling_identity.json','training.jsonl','final.pth']
  else:
   rows=json.loads((d/'records.json').read_text());names=['terminal.json','records.json','panels.json','status.json']+sorted({r['shard'] for r in rows})
  result[tier+'/'+role]={str((d/n).relative_to(p)):hashlib.sha256((d/n).read_bytes()).hexdigest() for n in names}
print(json.dumps(result))'''.replace('ROOT_PATH',repr(str(p)))
    r=subprocess.run(['ssh','-o','ConnectTimeout=8','remote-gpu',shlex.join([str(REMOTE_ROOT/'.conda/opencood-linux/bin/python'),'-c',code])],capture_output=True,text=True,timeout=40)
    assert r.returncode==0,(r.stdout,r.stderr)
    return json.loads(r.stdout)

def collect_available():
    dest=BASE/'R7b_collector';path=dest/'manifest.json';manifest=load(path) if path.exists() else dict(parts={})
    for tag,files in remote_manifest().items():
        if tag in manifest['parts']:
            assert files==manifest['parts'][tag]['SHA256'];continue
        for rel,digest in files.items():
            target=TRAIN/rel;target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists():
                source='remote-gpu:'+str(REMOTE_ROOT/str(TRAIN.relative_to(ROOT))/rel)
                subprocess.run(['rsync','-a','--ignore-existing',source,str(target)],check=True)
            assert ev(target)['sha256']==digest,('sole collector conflict',target)
        manifest['parts'][tag]=dict(collected_at=now(),SHA256=files)
        atomic(path,manifest)
    if len(manifest['parts'])==8 and not (dest/'terminal.json').exists():
        atomic(dest/'terminal.json',dict(created_at=now(),status='complete_light_checked',manifest=ev(path),
            active_resume_excluded=True,no_delete=True,no_overwrite=True))
        record('R7b_collector',dict(status='complete_light_checked',terminal=ev(dest/'terminal.json'),owner=None,resource_owner=False))

def stats(tier,seed,role,review):
    import numpy as np
    from peer20_population_core_v1_20261105 import save_stats
    import peer20_saved_native_v1_20261105 as native
    tag=tier+'_'+str(seed)+'_'+role;src=OUT/tag;dest=OUT/(tag+'_independent') if review else src
    dest.mkdir(parents=True,exist_ok=True);raw=TRAIN/tier/('seed'+str(seed))/role
    state=load(raw/'terminal.json');assert state['status']=='native_AP_complete_light_checked'
    assert not state['forward_GT_input'] and state['weight_unchanged'] and state['arms']==['COMET_H1']
    m,rows,p,b=native.assemble('author',role,native.shard_cases(raw,'COMET_H1'),'H1_clean_native',review)
    path=src/'predictions.npz'
    if review:
        with np.load(path) as z:assert set(z.files)==set(p) and all(np.array_equal(z[k],v) for k,v in p.items())
    else:
        np.savez_compressed(path,**p)
        atomic(src/'identity.json',dict(raw_terminal=ev(raw/'terminal.json'),records=ev(raw/'records.json'),
            contract=ev(TRAIN/(tier+'_seed'+str(seed)+'_contract.json')),canonical_GT_same_as_H1=True,
            current_common_ROI=True,input='CoMET_H1',all_temporal_keys=True))
    save_stats(dest,m,rows,p,'H1_clean_native','result',review,load(src/'result_statistics.json') if review else None)
    # This is an equality test of complete per-wait GT counts, not merely a
    # comparison of total point estimates across two incompatible paradigms.
    hstats=load(P20/'R6b_author'/(role+'_H1_seed2023_statistics.json'))
    results=load(dest/'result_statistics.json')
    for a,b in zip(results,hstats):
        assert (a['population'],a['threshold'],a['wait_ms'])==(b['population'],b['threshold'],b['wait_ms'])
        if a['wait_ms']!='four_wait_mean':assert a['GT']==b['GT'] and a['conditions']==b['conditions']
    atomic(dest/'terminal.json',dict(created_at=now(),status='complete_independently_reviewed' if review else 'complete_light_checked',
        statistics=ev(dest/'result_statistics.json'),canonical_GT_identity_checked=True,
        polygon_matching_independent=review,AP_bootstrap_independent=review,training_shared=True))

def choose():
    import numpy as np
    means={}
    sources=[]
    for tier in TIERS:
        values=[]
        for seed in SEEDS:
            src=OUT/(tier+'_'+str(seed)+'_calibration8')
            assert load(src.with_name(src.name+'_independent')/'terminal.json')['status']=='complete_independently_reviewed'
            values.append(next(r['AP'] for r in load(src/'result_statistics.json') if r['population']=='all' and r['threshold']==70 and r['wait_ms']=='four_wait_mean'))
            sources.append(ev(src/'result_statistics.json'))
        means[tier]=float(np.mean(values))
    selected='B' if means['B']>means['A'] else 'A'
    atomic(OUT/'calibration_choice.json',dict(created_at=now(),selected=selected,means=means,sources=sources,
        roles_used=['calibration8'],formal_results_not_read=True,exact_tie='A'))

def gate():
    import numpy as np
    tier=load(OUT/'calibration_choice.json')['selected'];panels=[];sources=[]
    for role in ('val21','mechanism_holdout14'):
        for pop in ('all','zero','colabelled','nonzero'):
            for th in (50,70):
                distributions={};points={}
                for method in ('H1_clean_native','clean_K1','original_K4','PUBLIC_finetuned','H1_zero_training'):
                    draws=[];ap=[]
                    for i,seed in enumerate(SEEDS):
                        if method=='H1_clean_native':
                            src=OUT/(tier+'_'+str(seed)+'_'+role);stem='result'
                            assert load(src.with_name(src.name+'_independent')/'terminal.json')['status']=='complete_independently_reviewed'
                        elif method=='H1_zero_training':
                            src=P20/'R6b_author';stem=role+'_H1_seed'+str(2023+i)
                            assert load(P20/'R6b_author_independent/terminal.json')['status']=='complete_independently_reviewed'
                        else:
                            src=P20/('R7_analysis_seed'+str(seed));stem=role+'_'+('clean' if method=='clean_K1' else method)
                            assert load(src.with_name(src.name+'_independent')/'terminal.json')['status']=='complete_independently_reviewed'
                        with np.load(src/(stem+'_draws.npz')) as z:draws.append(z[pop+'_AP'+str(th)])
                        rr=[r for r in load(src/(stem+'_statistics.json')) if r['population']==pop and r['threshold']==th]
                        ap.append([next(r['AP'] for r in rr if r['wait_ms']==w) for w in (0,100,200,400)])
                        if pop=='all' and th==50:sources.append(dict(role=role,method=method,seed=seed,stats=ev(src/(stem+'_statistics.json')),draws=ev(src/(stem+'_draws.npz'))))
                    distributions[method]=np.mean(draws,axis=0);points[method]=np.mean(ap,axis=0)
                for ref in ('clean_K1','original_K4','PUBLIC_finetuned','H1_zero_training'):
                    diff=distributions['H1_clean_native']-distributions[ref];point=points['H1_clean_native']-points[ref]
                    for w,dd,pt in zip((0,100,200,400,'four_wait_mean'),list(diff)+[diff.mean(0)],list(point)+[point.mean()]):
                        panels.append(dict(role=role,population=pop,threshold=th,reference=ref,wait_ms=w,
                            delta_pp=float(pt),CI95=np.quantile(dd,[.025,.975]).tolist(),replicates=2000,
                            sequence_paired=True,seed_mean=True,selected_tier=tier))
    def row(ref,th):return next(r for r in panels if r['role']=='val21' and r['population']=='all' and r['reference']==ref and r['threshold']==th and r['wait_ms']=='four_wait_mean')
    pub50,pub70,h50,h70=row('PUBLIC_finetuned',50),row('PUBLIC_finetuned',70),row('H1_zero_training',50),row('H1_zero_training',70)
    checks=dict(PUBLIC_AP50_CI_low_positive=pub50['CI95'][0]>0,PUBLIC_AP70_CI_low_positive=pub70['CI95'][0]>0,
        H1_AP50_at_least1pp=h50['delta_pp']>=1.,H1_AP50_CI_low_positive=h50['CI95'][0]>0,H1_AP70_at_least_minus030=h70['delta_pp']>=-.30)
    atomic(OUT/'contrasts.json',panels);atomic(OUT/'source_identities.json',sources)
    decision=dict(created_at=now(),selected_tier=tier,checks=checks,qualifies=all(checks.values()),
        hold14_reported_not_gate=True,all_seed_same_sign_not_gate=True,main_method_unchanged='CoMET_H1',
        no_automatic_root_table_mutation=True,disposition='main_conversation_second_tier_eligible' if all(checks.values()) else 'supplement_no_followup',
        H1_reference='final three frozen heads mean; preview single-head value is not substituted',
        contrasts=ev(OUT/'contrasts.json'),sources=ev(OUT/'source_identities.json'))
    atomic(OUT/'decision.json',decision);atomic(OUT/'terminal.json',dict(created_at=now(),status='complete_independently_reviewed',decision=ev(OUT/'decision.json')))
    record('R7b_gate',dict(status='complete_independently_reviewed',terminal=ev(OUT/'terminal.json'),owner=None,resource_owner=False))
    record('R7c_conditional',dict(status='passed_gate_pending_architecture_input_qualification' if decision['qualifies'] else 'not_required_gate_failed',
        gate=ev(OUT/'decision.json'),owner=None,resource_owner=False))

def ready(path):
    if not path.exists():return False
    v=load(path)
    if v.get('status')=='failed_retained':raise RuntimeError((path,v))
    return True

def cpu_slot():
    import peer20_finite_resource_DAG_v1_20261105 as dag
    jobs=dag.load(dag.OUT/'state.json')['jobs']
    active=sum(dag.active(j['unit']) for j in jobs)
    memory=int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))*1024
    return active<2 and memory>=28*1024**3

def chain():
    dest=BASE/NAME;lock=(dest/'owner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    completed=[]
    try:
        # Only stdlib while waiting; no dataset/torch/NumPy resident waiters.
        for role in ROLES:
            for tier in TIERS:
                for seed in SEEDS:
                    raw=TRAIN/tier/('seed'+str(seed))/role
                    while not ready(raw/'terminal.json') or not cpu_slot():
                        collect_available();time.sleep(30)
                    for mode in ('run','review'):
                        cmd=[sys.executable,'-u',str(Path(__file__).resolve()),mode,'--tier',tier,'--seed',str(seed),'--role',role]
                        tag=tier+'_'+str(seed)+'_'+role+'_'+mode
                        atomic(dest/'progress.json',dict(created_at=now(),current=tag,completed=completed))
                        with (dest/(tag+'.log')).open('x') as log:rc=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT).returncode
                        assert rc==0,(tag,rc);completed.append(tag)
                    collect_available()
            if role=='calibration8':choose()
        while not ready(P20/'R7_preregistered_read/terminal.json') or not ready(P20/'R6b_author_independent/terminal.json'):
            collect_available();time.sleep(30)
        gate()
        atomic(dest/'terminal.json',dict(created_at=now(),status='complete_light_checked',completed=completed,decision=ev(OUT/'decision.json')))
        for n in ('R7b_statistics','R7b_independent',NAME):record(n,dict(status='complete_independently_reviewed',owner=None,resource_owner=False,terminal=ev(dest/'terminal.json')))
    except Exception as e:
        atomic(dest/'terminal.json',dict(created_at=now(),status='failed_retained',error=repr(e),traceback=traceback.format_exc(),completed=completed))
        record(NAME,dict(status='failed_retained',terminal=ev(dest/'terminal.json'),owner=None,resource_owner=False));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','chain','run','review'))
    p.add_argument('--tier',choices=TIERS);p.add_argument('--seed',type=int,choices=SEEDS);p.add_argument('--role',choices=ROLES);a=p.parse_args()
    if a.mode=='prepare':prepare()
    elif a.mode=='chain':chain()
    else:stats(a.tier,a.seed,a.role,a.mode=='review')
