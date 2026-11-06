"""Second-tier auxiliary-GT sensitivity, fixed original ROI, shared predictions.

R7b A, same-budget PUBLIC, and three frozen zero-training H1 heads. Both
all_gaps and label_median, two formal roles, four populations, independent
polygon matching/AP/2000 whole-sequence CI. No selection and no physical truth.
"""
import argparse,subprocess,sys,traceback
from peer22_v1_20261106 import *
SEEDS=(20261101,20261102,20261103)
OUT=BASE/'R6h'
def prepare():
    register('R6h',__file__,__doc__,dict(updates=0,GPU_forwards=0,CPU_threads=1,replicates=2000,
        disk_MiB='<250 compact changed-TP/statistics; native and late predictions shared'),
        references=['R6d all_gaps','R6f label_median'],
        true_data_dependencies=[ev(P20/'R6a_raw_labels_independent_v2/terminal.json'),ev(P21/'R7b_gate_postcheck/terminal.json')],
        R6f_resource_wait_waived='Only raw-label target constructor required, not completed R6f statistical tables',
        budget2000=True,larger_budget='Append only if R7d selected>2000; never replace official GT')
def cases(method,seed,role,review):
    import numpy as np
    import peer20_saved_native_v1_20261105 as native
    from peer20_population_core_v1_20261105 import data,REC,h1
    if method!='H1':
        if method=='A':path=P21/'R7b/A'/('seed'+str(seed))/role;arm='COMET_H1'
        else:
            from consume_peer20_R7_v1_20261105 import VAL,HOLD
            path=native.AUDIT/(VAL if role=='val21' else HOLD)[SEEDS.index(seed)][1];arm='PUBLIC'
        m,rows,p,b=native.assemble('author',role,native.shard_cases(path,arm),method,review)
        return m,rows,p,b,dict(raw_terminal=ev(path/'terminal.json'),records=ev(path/'records.json'),arm=arm)
    m,rows=data('author')[0][role];headseed=2023+SEEDS.index(seed);label='H1_seed'+str(headseed)
    source=P20/'R6b_author';path=source/(role+'_'+label+'_predictions.npz')
    with np.load(path) as z:p={method+k[len(label):]:z[k] for k in z.files}
    with np.load(source/(role+'_H1_state.npz')) as z:d=z['delta']
    if review:
        from review_peer16_E4_E7_AP_v1_20261103 import geometry
        b=geometry(m,d)
    else:b=h1.recipe.geometry(m,d)[0]
    return m,rows,p,b,dict(shared_predictions=ev(path),state=ev(source/(role+'_H1_state.npz')),headseed=headseed)
def work(review):
    import numpy as np
    import run_peer20_auxiliary_GT_v1_20261105 as aux
    import run_peer21_label_median_GT_v1_20261105 as median
    from peer20_population_core_v1_20261105 import save_stats
    src=OUT;dest=BASE/('R6h_independent' if review else 'R6h');done=[]
    dest.mkdir(parents=True,exist_ok=True)
    for role in ('val21','mechanism_holdout14'):
        for method in ('A','PUBLIC','H1'):
            for seed in SEEDS:
                m,rows,p,b,identity=cases(method,seed,role,review)
                for tier,fn in [('all_gaps',aux.changed_truth),('label_median',median.changed_truth)]:
                    gt,gid=fn(m,rows,role,float('inf'));tag=role+'_'+method+'_'+str(seed)+'_'+tier
                    # Explicitly preserve original rank for equal native scores.
                    pp=p
                    if review:
                        pp=dict(p);pp[method+'__score']=np.concatenate([-np.arange(int(hi-lo),dtype=float) for lo,hi in zip(p[method+'__offset'][:-1],p[method+'__offset'][1:])])
                    hit=aux.rematch(m,pp,method,b,gt,review);path=src/(tag+'_TP.npz')
                    if review:
                        with np.load(path) as z:assert set(z.files)==set(hit) and all(np.array_equal(z[k],v) for k,v in hit.items())
                    else:
                        np.savez_compressed(path,**hit);atomic(src/(tag+'_identity.json'),dict(predictions=identity,GT=gid,
                            original_ROI_frozen=True,only_inf_GT_translated=True,not_physical_truth=True))
                    save_stats(dest,m,rows,dict(p,**hit),method,tag,review,load(src/(tag+'_statistics.json')) if review else None)
                    done.append(tag);atomic(dest/'progress.json',dict(created_at=now(),completed=len(done),last=tag))
    atomic(dest/'terminal.json',dict(created_at=now(),status='complete_independently_reviewed' if review else 'complete_light_checked',
        combinations=len(done),AP_CI_rows=40*len(done),original_GT_and_predictions_shared=True,no_selection=True))
def chain():
    dest=BASE/'R6h_chain';done=[]
    try:
        for mode in ('run','review'):
            with (dest/(mode+'.log')).open('x') as log:rc=subprocess.run([sys.executable,'-u',__file__,mode],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT).returncode
            assert rc==0,(mode,rc);done.append(mode)
        atomic(dest/'terminal.json',dict(created_at=now(),status='complete_independently_reviewed',analysis=ev(BASE/'R6h_independent/terminal.json')))
        record('R6h',dict(status='complete_independently_reviewed',owner=None,resource_owner=False,terminal=ev(dest/'terminal.json')))
    except Exception as exc:
        atomic(dest/'terminal.json',dict(created_at=now(),status='failed_retained',error=repr(exc),traceback=traceback.format_exc(),completed=done));raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','chain','run','review'));a=p.parse_args()
    prepare() if a.mode=='prepare' else chain() if a.mode=='chain' else work(a.mode=='review')
