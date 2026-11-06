"""Independent pooled-AP and sequence-bootstrap arithmetic for four final/native contrasts.

Saved matches/GT are shared and independently geometry-reviewed upstream.
No production bootstrap/AP/panel/counts function is called. Integer sequence
multiplicities weight each sorted detection; AP integrates precision envelope
only at true-positive recall increments. This is not independent detection,
matching, training, or a whole-paper statistics closure.
"""
import argparse,time,traceback
import numpy as np
from peer18_v1_20261104 import *
import run_peer17_final_recipe_v1_20261104 as final
from run_peer16_cached_methods_v1_20261103 import materials
from object_history_selector_core_v1_20261021 import common_mask
OUT=BASE/'final_native_statistics_independent'
SOURCES=dict(strict60=BASE/'final_native_contrasts_recovery_v2/strict60',
    v2xvit_seed2024=BASE/'final_native_contrasts/v2xvit_seed2024',
    v2xvit_seed2025=BASE/'final_native_contrasts_recovery_v2/v2xvit_seed2025',
    where2comm=BASE/'where_actual_native_independent')

def empirical(score,tp,cluster,ng,weights):
    # Preserve the officially accepted argsort tie convention, not stable sort.
    order=np.argsort(-np.asarray(score,dtype=np.float64));t=np.asarray(tp,dtype=np.int64)[order]
    group=np.asarray(cluster,dtype=np.int64)[order];out=[]
    for first in range(0,len(weights),4):
        mult=np.asarray(weights[first:first+4],dtype=np.int64);mass=mult[:,group]
        hits=mass*t[None,:];hit_prefix=np.cumsum(hits,axis=1);rank=np.cumsum(mass,axis=1)
        precision=hit_prefix/np.maximum(rank,1)
        envelope=np.maximum.accumulate(precision[:,::-1],axis=1)[:,::-1]
        # Recall increment equals weighted TP / total resampled GT. No
        # production recall padding/differencing or VOC helper is reused.
        denom=mult@np.asarray(ng,dtype=np.int64)
        out.extend(np.sum(hits*envelope,axis=1)/np.maximum(denom,1))
    return np.asarray(out)

def panel(d,w,seqs):
    ids=np.flatnonzero(np.asarray(d['wait'])==w);index={s:i for i,s in enumerate(seqs)}
    scores=[];hits=[];clusters=[];ng=np.zeros(len(seqs),np.int64)
    for i in ids:
        lo,hi=map(int,d['offset'][i:i+2]);s=str(d['sequence'][i]);j=index[s]
        scores.extend(d['score'][lo:hi]);hits.extend(d['tp'][lo:hi]);clusters.extend([j]*(hi-lo));ng[j]+=int(d['gt'][i])
    return np.asarray(scores),np.asarray(hits),np.asarray(clusters),ng

def synthetic():
    # A higher-ranked FP plus lower-ranked TP: AP=.5; doubling the TP's
    # sequence doubles its GT too, giving 2/3; no detections gives zero.
    x=empirical(np.array([.9,.8]),np.array([0,1]),np.array([0,1]),np.array([0,1]),np.array([[1,1],[1,2]]))
    assert np.allclose(x,[.5,2/3],rtol=0,atol=1e-15)
    assert empirical(np.array([]),np.array([]),np.array([],int),np.array([1]),np.array([[1]]))[0]==0
    return dict(status='passed',analytical_examples=3,values=x.tolist())

def prepare():
    synthetic()
    register('final_native_statistics_independent',__file__,__doc__,dict(GPU_forwards=0,updates=0,CPU_threads=1,
        replicates=2000,panels=120,wall_h='finite24h',disk_MiB='<20 draws/review, shared predictions and material'),
        sources={d:ev(p/'terminal.json') for d,p in SOURCES.items()},
        statistics_independence='Independent AP integration, cluster assembly, resampling histogram and percentile calculation; same saved TP/GT/heads, same accepted score tie convention.',
        scope='Only final3head minus actual native for four domains, not all paper differences; no main.tex edits.')

def run():
    begin=time.monotonic();assert load(OUT/'contract.json')['script']==ev(__file__)
    record('final_native_statistics_independent',dict(status='running_independent_AP_bootstrap_kernel',resource_owner=True))
    done=[]
    try:
        atomic(OUT/'synthetic.json',synthetic())
        for domain,src in SOURCES.items():
            m,_=materials(domain)[0]['val21'];seqs=sorted(set(m['sequence'].astype(str)))
            rng=np.random.Generator(np.random.PCG64(20261103));draw=rng.integers(0,len(seqs),(2000,len(seqs)))
            counts=np.asarray([np.bincount(row,minlength=len(seqs)) for row in draw],np.int16)
            for th in (50,70):
                if domain!='where2comm':
                    with np.load(src/f'native_AP{th}_compact.npz') as z:native={k:z[k].copy() for k in z.files}
                else:
                    native=dict(sequence=[],wait=[],gt=[],offset=[0],score=[],tp=[])
                    for w in (0,100,200,400):
                        with np.load(src/'where'/f'common_PUBLIC_AP{th}_wait{w}.npz') as z:
                            for k in ('sequence','wait','gt','score','tp'):native[k].extend(z[k])
                            native['offset'].extend((z['offset'][1:]+native['offset'][-1]).tolist())
                    native={k:np.asarray(v) for k,v in native.items()}
                native_distributions={};native_points={};gt_by_wait={}
                for w in (0,100,200,400):
                    x=panel(native,w,seqs);native_distributions[w]=empirical(*x,counts);native_points[w]=empirical(*x,np.ones((1,len(seqs)),np.int16))[0];gt_by_wait[w]=x[3]
                for seed in final.SEEDS:
                    label=f'{domain}_final_seed{seed}_minus_actual_native'
                    with np.load(src/(label+f'_AP{th}.npz')) as z:
                        expected={k:z[k].copy() for k in z.files}
                    assert np.array_equal(counts,expected['cluster_counts']) and np.array_equal(seqs,expected['sequence'].astype(str))
                    with np.load(final.OUT/domain/f'val21_seed{seed}_predictions.npz') as z:
                        name=f'CoMET_seed{seed}'
                        a=dict(sequence=m['sequence'],wait=m['wait_ms'],
                            gt=np.asarray([int(common_mask(m['gt_boxes'][int(lo):int(hi)]).sum()) for lo,hi in zip(m['gt_offsets'][:-1],m['gt_offsets'][1:])]),
                            offset=z[name+'__offset'],score=z[name+'__score'],tp=z[name+'__tp'+str(th)])
                        errors=[];dist=[];points=[]
                        for w in (0,100,200,400):
                            x=panel(a,w,seqs);assert np.array_equal(x[3],gt_by_wait[w])
                            delta=100*(empirical(*x,counts)-native_distributions[w]);dist.append(delta)
                            point=100*(empirical(*x,np.ones((1,len(seqs)),np.int16))[0]-native_points[w]);points.append(float(point))
                            error=float(np.max(abs(delta-expected['wait'+str(w)])));errors.append(error)
                            assert error<1e-8,('bootstrap_draw',domain,th,seed,w,error)
                        avg=np.mean(dist,axis=0);accepted=load(src/'panels.json')
                        for w,p,v in zip((0,100,200,400,'four_wait_mean'),points+[float(np.mean(points))],dist+[avg]):
                            r=next(r for r in accepted if r['comparison']==label and r['threshold']==th and r['wait_ms']==w)
                            interval=np.percentile(v,[2.5,97.5]);err=max(abs(p-r['delta_pp']),abs(interval[0]-r['CI95_low']),abs(interval[1]-r['CI95_high']))
                            assert err<1e-8,(domain,th,seed,w,err)
                            done.append(dict(domain=domain,head_seed=seed,threshold=th,wait_ms=w,point=p,CI95=interval.tolist(),maximum_draw_error=max(errors),maximum_summary_error=float(err),status='passed'))
                    atomic(OUT/'panels.json',done);atomic(OUT/'progress.json',dict(created_at=now(),panels=len(done),domain=domain,seed=seed,threshold=th,wall_s=time.monotonic()-begin))
            assert time.monotonic()-begin<24*3600
        assert len(done)==120
        atomic(OUT/'terminal.json',dict(created_at=now(),status='complete_independently_reviewed',panels=ev(OUT/'panels.json'),
            independent_statistics_implementation=True,saved_matching_shared=True,whole_paper_statistics_closed=False,wall_s=time.monotonic()-begin))
        record('final_native_statistics_independent',dict(status='complete_independently_reviewed',terminal=ev(OUT/'terminal.json'),resource_owner=False,owner=None))
    except Exception as e:
        atomic(OUT/'terminal.json',dict(created_at=now(),status='failed_retained',error=repr(e),traceback=traceback.format_exc(),completed_panels=len(done)))
        record('final_native_statistics_independent',dict(status='failed_retained',terminal=ev(OUT/'terminal.json'),resource_owner=False,owner=None));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run']);globals()[p.parse_args().mode]()
