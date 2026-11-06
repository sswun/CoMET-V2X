"""Independent statistics for the frozen single neural/adaptive round.

Reuses accepted saved TP/GT and the separately implemented empirical AP kernel;
does not call production AP/bootstrap/panel functions. No fitting, forward or
method selection. Retain all variants and own-domain held-out negative results.
"""
import argparse,time,traceback
import numpy as np
from peer18_v1_20261104 import *
import review_peer18_final_native_statistics_v1_20261104 as independent
from review_peer18_main_module_statistics_v1_20261104 import pack
import run_peer17_final_recipe_v1_20261104 as final
from run_peer16_cached_methods_v1_20261103 import materials,DOMAINS

def name_for(seed):return 'candidate_statistics_independent_'+str(seed)
def source(seed,domain,variant):
    return (BASE/'adaptive_statistics' if seed==0 else BASE/('net_statistics_seed'+str(seed)))/(domain+'__'+variant)

def prepare(seed):
    register(name_for(seed),__file__,__doc__,dict(GPU_forwards=0,updates=0,CPU_threads=1,
        panels=600,replicates=2000,wall_h='bounded24h',disk_MiB='<40 independent contrasts; shared predictions/GT'),
        source=ev(BASE/('adaptive_statistics' if seed==0 else 'net_statistics_seed'+str(seed))/'terminal.json'),
        kernel=ev(independent.__file__),seed=seed,
        decision='No new variant. Independent full2000draw/CI arithmetic feeds the separately implemented gate reader.')

def run(seed):
    out=BASE/name_for(seed);begin=time.monotonic();done=[]
    record(name_for(seed),dict(status='running_independent_candidate_statistics',resource_owner=True))
    try:
        assert load(out/'contract.json')['script']==ev(__file__)
        independent.synthetic()
        seeds=final.SEEDS if seed==0 else (seed,)
        variants=('N1','N1b') if seed==0 else ('N2','N3','N4','N5','B1','B2')
        for domain in DOMAINS:
            data,_,_=materials(domain)
            for role in ('val21','mechanism_holdout14'):
                m,_=data[role];seqs=sorted(set(m['sequence'].astype(str)))
                draws=np.random.Generator(np.random.PCG64(20261103)).integers(0,len(seqs),(2000,len(seqs)))
                counts=np.asarray([np.bincount(r,minlength=len(seqs)) for r in draws],np.int16)
                dest=out/domain;dest.mkdir(exist_ok=True)
                for s in seeds:
                    with np.load(final.OUT/domain/f'{role}_seed{s}_predictions.npz') as base:
                        for th in (50,70):
                            b=pack(m,base,'CoMET_seed'+str(s),th);reference={}
                            for w in (0,100,200,400):
                                x=independent.panel(b,w,seqs)
                                reference[w]=(x[3],independent.empirical(*x,counts),float(independent.empirical(*x,np.ones((1,len(seqs)),np.int16))[0]))
                            for v in variants:
                                src=source(seed,domain,v)
                                assert load(src/'terminal.json')['status'] in ('complete_light_checked','complete_independently_reviewed')
                                root=(BASE/('N1_N1b_strict_cal4' if domain=='strict60' else 'N1_N1b_v2')/domain if seed==0 else
                                      BASE/'CoMET_Net/formal'/('seed'+str(s))/(domain+'__to__'+domain+'__'+v))
                                path=root/(role+'_'+v+'_predictions.npz' if seed==0 else role+'_predictions.npz')
                                label=role+'_seed'+str(s)+'_'+v+'_minus_N0';distribution={};points=[];errors=[]
                                accepted=load(src/'bootstrap.json')
                                with np.load(path) as pred,np.load(src/(label+'_AP'+str(th)+'.npz')) as gold:
                                    assert np.array_equal(seqs,gold['sequence'].astype(str)) and np.array_equal(counts,gold['cluster_counts'])
                                    a=pack(m,pred,v+'_seed'+str(s),th)
                                    for w in (0,100,200,400):
                                        x=independent.panel(a,w,seqs);ng,bd,bp=reference[w];assert np.array_equal(x[3],ng)
                                        dd=100*(independent.empirical(*x,counts)-bd);distribution['wait'+str(w)]=dd
                                        pt=100*(float(independent.empirical(*x,np.ones((1,len(seqs)),np.int16))[0])-bp);points.append(pt)
                                        err=float(np.max(abs(dd-gold['wait'+str(w)])));errors.append(err)
                                        assert err<1e-8,(domain,label,th,w,err)
                                distribution['four_wait_mean']=np.mean(list(distribution.values()),axis=0)
                                np.savez_compressed(dest/(label+'_AP'+str(th)+'.npz'),**distribution)
                                for w,pt in zip((0,100,200,400,'four_wait_mean'),points+[float(np.mean(points))]):
                                    arr=distribution['four_wait_mean' if isinstance(w,str) else 'wait'+str(w)];ci=np.percentile(arr,[2.5,97.5])
                                    expected=next(r for r in accepted if r['comparison']==label and r['threshold']==th and r['wait_ms']==w)
                                    err=max(abs(pt-expected['delta_pp']),abs(ci[0]-expected['CI95_low']),abs(ci[1]-expected['CI95_high']))
                                    assert err<1e-8,(domain,label,th,w,err)
                                    done.append(dict(domain=domain,role=role,seed=s,variant=v,threshold=th,wait_ms=w,
                                        delta_pp=pt,CI95=ci.tolist(),max_draw_error=max(errors),summary_error=float(err),status='passed'))
                                atomic(out/'panels.json',done)
                                atomic(out/'progress.json',dict(created_at=now(),panels=len(done),domain=domain,role=role,variant=v,seed=s,threshold=th))
            del data
            assert time.monotonic()-begin<24*3600
        assert len(done)==600,len(done)
        atomic(out/'terminal.json',dict(created_at=now(),status='complete_independently_reviewed',panels=ev(out/'panels.json'),
            panel_count=len(done),independent_statistics_implementation=True,saved_matching_shared=True,wall_s=time.monotonic()-begin))
        record(name_for(seed),dict(status='complete_independently_reviewed',terminal=ev(out/'terminal.json'),resource_owner=False,owner=None))
    except Exception as e:
        atomic(out/'terminal.json',dict(created_at=now(),status='failed_retained',error=repr(e),traceback=traceback.format_exc(),panels=len(done)))
        record(name_for(seed),dict(status='failed_retained',terminal=ev(out/'terminal.json'),resource_owner=False,owner=None));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run']);p.add_argument('--seed',type=int,choices=(0,2023,2024,2025),required=True)
    a=p.parse_args();globals()[a.mode](a.seed)
