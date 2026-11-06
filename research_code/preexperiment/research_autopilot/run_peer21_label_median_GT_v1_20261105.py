"""R6f one extra label-only reference, unchanged predictions and official ROI.

Inf-only GT uses same-frame >=3 coop-label displacement median; otherwise use
official nonzero or interpolated-zero reference. No evaluation-GT input, no
selection. Existing R6b core13 methods, all five domains/three heads/three roles.
"""
import argparse,fcntl,inspect,subprocess,sys,time,traceback
from peer21_v1_20261105 import *

NAME='R6f_chain';DOMAINS=('author','strict60','v2xvit_seed2024','v2xvit_seed2025','where2comm')

def prepare():
    register(NAME,__file__,__doc__,dict(GPU_forwards=0,updates=0,CPU_threads=1,replicates=2000,
        disk_MiB='<150/domain compressed TP/statistics; original inputs/predictions referenced'),
        dependencies=['peer20 finite DAG','peer20 population coverage completion','R7b consumer chain'],
        waiting_policy='No dataset imports while dependencies unresolved; cannot count waiting as execution')
    for d in DOMAINS:
        for review in (False,True):
            register('R6f_'+d+('_independent' if review else ''),__file__,__doc__,dict(GPU_forwards=0,updates=0,
                CPU_threads=1,replicates=2000,disk_MiB='<150 compact TP/statistics; no raw prediction duplication'),
                domain=d,shared_core=ev(P20/('R6b_'+d)/'contract.json'),
                reference='>=3 coop labels => displacement median; otherwise official nonzero or zero-filled; inf-only changes',
                original_ROI_frozen=True,no_cal_or_formal_selection=True)
            record('R6f_'+d+('_independent' if review else ''),dict(status='queued_after_existing_finite_consumers',owner=None,resource_owner=False))

def changed_truth(m,rows,role,unused):
    import numpy as np
    from peer20_population_core_v1_20261105 import annotations,fill,codes
    from run_peer21_offset_grid_v1_20261105 import raw_reference
    pop,off=annotations(rows);filled,_,span,support=fill(rows,off);target=filled.copy()
    refs=raw_reference();used=0
    for i,r in enumerate(rows):
        v=refs.get((role,str(r['sequence']),str(r['current_vehicle_frame'])))
        if v is not None:target[i]=v['median'];used+=1
    c,identity=codes(m,rows,role);gt=m['gt_boxes'].copy();count=0
    for i,r in enumerate(rows):
        ga,gb=map(int,m['gt_offsets'][i:i+2]);ids=ga+np.flatnonzero(c[ga:gb]==2)
        delta=target[i]-off[i];gt[ids,:2]+=delta.astype(np.float32)
        if np.any(delta!=0):count+=len(ids)
    assert np.array_equal(gt[c!=2],m['gt_boxes'][c!=2])
    return gt,dict(reference='same-frame median if>=3 coop targets; otherwise official or zero-filled',
        median_reference_conditions=used,changed_inf_GT_occurrences=count,unchanged_other_sources=True,
        original_official_GT_not_modified=True,noncausal_label_only=True,source_mapping=identity)

def run(domain,review):
    import run_peer20_auxiliary_GT_v1_20261105 as prior
    name='R6f_'+domain+('_independent' if review else '')
    # Frozen R6d matching/statistics unchanged. Only the one new target tier
    # and output namespace differ; producer references remain original R6b.
    code=inspect.getsource(prior.run).replace("name='R6d_'+domain", "name='R6f_'+domain")
    code=code.replace("dest=BASE/name;src=BASE/('R6d_'+domain);source=BASE/('R6b_'+domain)",
        "dest=P21/name;src=P21/('R6f_'+domain);source=BASE/('R6b_'+domain)")
    def mapped(n,v):record(n,v)
    ns=dict(prior.__dict__,P21=BASE,TIERS=(('label_median',None),),changed_truth=changed_truth,record=mapped)
    exec(compile(code,__file__+'::one_auxiliary_tier','exec'),ns);ns['run'](domain,review)

def chain():
    dest=BASE/NAME;lock=(dest/'owner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    deps=[P20/'finite_resource_DAG_v1/terminal.json',BASE/'R7b_consumer_slot_v2/terminal.json']
    # Post-DAG coverage uses CPU too. Wait for its real output, not mere audit.
    import complete_peer20_population_coverage_v1_20261105 as coverage
    deps.append(coverage.OUT/'terminal.json')
    while not all(p.exists() for p in deps):
        atomic(dest/'progress.json',dict(created_at=now(),waiting=[str(p.relative_to(ROOT)) for p in deps if not p.exists()],dataset_waiters=0));time.sleep(45)
    completed=[]
    try:
        for d in DOMAINS:
            for mode in ('run','review'):
                cmd=[sys.executable,'-u',str(Path(__file__).resolve()),mode,'--domain',d]
                with (dest/(d+'_'+mode+'.log')).open('x') as log:rc=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT).returncode
                assert rc==0,(d,mode,rc);completed.append([d,mode])
                atomic(dest/'progress.json',dict(created_at=now(),completed=completed))
        atomic(dest/'terminal.json',dict(created_at=now(),status='complete_independently_reviewed',completed=completed,root_table_unchanged=True))
        record(NAME,dict(status='complete_independently_reviewed',terminal=ev(dest/'terminal.json'),owner=None,resource_owner=False))
    except Exception as e:
        atomic(dest/'terminal.json',dict(created_at=now(),status='failed_retained',error=repr(e),traceback=traceback.format_exc(),completed=completed));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','chain','run','review'));p.add_argument('--domain',choices=DOMAINS);a=p.parse_args()
    prepare() if a.mode=='prepare' else chain() if a.mode=='chain' else run(a.domain,a.mode=='review')
