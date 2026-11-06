"""Independent provenance, source attribution, polygon matching and AP.

Fixed FP TP contributions are not additive AP contributions. No method choice.
"""
import argparse
import time
import traceback
from collections import Counter,defaultdict
import numpy as np
from peer16_v1_20261103 import ROOT,BASE,atomic,ev,load,now,register,record
import run_peer15_author_recipe_v1_20261103 as author
import review_peer07_timing_2x2_source_split_ap_v2_20261031 as source
from review_traf_v2xseq_adapted_fixed60_raw_20261006 import polygons_hwl,voc_ap
OUT=BASE/'E8_source_independent'
PROD=BASE/'E8_source_contribution'

def prepare():
    register('E8_source_independent',__file__,
        'Rejoin original label from_side via independently reviewed KD-tree '
        'provenance, then independently rebuild polygon IoU, one-to-one '
        'matches, fixed-FP source counts and ignore-other-source pooled VOC '
        'AP for every current-author E8 panel. Compare all producer values, '
        'retain discrepancies. No new inference, labels, ROI or source fitting.',
        dict(GPU_forwards=0,updates=0,CPU_threads=1,wall_h='1-6',disk_MiB='<5'),
        producer=ev(PROD/'terminal.json'),
        independent_provenance=ev(source.__file__),
        scope='Full E8 provenance/geometry/matching/point-statistic review; '
              'not an independent bootstrap of every paper difference.')

def matrix(b,g):
    p=polygons_hwl(b);q=polygons_hwl(g)
    out=np.zeros((len(p),len(q)),np.float32)
    for i,left in enumerate(p):
        for j,right in enumerate(q):
            inter=left.intersection(right).area
            if inter:out[i,j]=inter/left.union(right).area
    return out

def match(M,th):
    result=np.full(M.shape[0],-1,np.int64);unused=list(range(M.shape[1]))
    for i in range(M.shape[0]):
        if not unused:break
        j=int(np.argmax(M[i,unused]));best=unused[j]
        if M[i,best]>=th:result[i]=best;unused.pop(j)
    return result

def run():
    start=time.monotonic();record('E8_source_independent',dict(status='running_independent_source_geometry_AP'))
    try:
        m,rows,oldpred=author.materials('val21')
        conditions=[None]*(max(map(int,m['dataset_index']))+1)
        for index,r in zip(m['dataset_index'],rows):conditions[int(index)]=r
        codes,join=source.provenance(m,conditions,'val21',ROOT/'artifacts/data/v2x_seq_spd_main_domain_shared_v1/V2X-Seq-SPD/cooperative/label')
        assert join['unmatched']==join['ambiguous']==0
        with np.load(author.BASE/'author_recipe_v2/primary_geometry.npz') as z:current=z['current_boxes'].copy()
        with np.load(author.BASE/'author_recipe_v2/primary_predictions.npz') as z:newpred={k:z[k].copy() for k in z.files}
        cases={'current_heading_calq_matchedU':(current,newpred),'v1_CV_q015_matchedU':(m['kf_boxes'],newpred),'stale__matched_head':(m['stale_boxes'],oldpred)}
        goldAP={(r['method'],r['wait_ms'],r['threshold'],r['source']):r for r in load(PROD/'source_AP.json')}
        goldFP={(r['method'],r['wait_ms'],r['threshold'],r['FP_budget']):r for r in load(PROD/'fixed_FP_TP.json')}
        goldC={r['wait_ms']:r for r in load(PROD/'GT_census.json')['panels']}
        census=[];ap=[];budget=[];mismatches=[]
        for wait in (0,100,200,400):
            indices=np.flatnonzero(m['wait_ms']==wait);counts=Counter()
            for i in indices:
                ga,gb=map(int,m['gt_offsets'][i:i+2]);keep=author.common_mask(m['gt_boxes'][ga:gb]);counts.update(map(int,codes[ga:gb][keep]))
            N=sum(counts.values());assert N==31123
            countrow={('veh','coop','inf')[k]:v for k,v in sorted(counts.items())}
            assert countrow==goldC[wait]['by_original_from_side']
            census.append(dict(wait_ms=wait,GT=N,counts=countrow,inf_fraction=counts[2]/N))
            for name,(boxes,pred) in cases.items():
                vectors={th:dict(score=[],tp=[],side=[],groups=defaultdict(lambda:dict(score=[],tp=[],fp=[],GT=0))) for th in (50,70)}
                for i in indices:
                    lo,hi=map(int,pred[name+'__offset'][i:i+2]);ids=pred[name+'__candidate'][lo:hi].astype(int);ss=pred[name+'__score'][lo:hi]
                    ga,gb=map(int,m['gt_offsets'][i:i+2]);gg=m['gt_boxes'][ga:gb];cc=codes[ga:gb];keep=author.common_mask(gg);gg,cc=gg[keep],cc[keep]
                    M=matrix(boxes[ids],gg)
                    for th in (50,70):
                        v=vectors[th];assigned=match(M,th/100);tt=(assigned>=0).astype(np.int8)
                        assert np.array_equal(tt,pred[name+'__tp'+str(th)][lo:hi]), ('accepted_TP',name,wait,th,int(i))
                        side=np.full(len(ids),-1);side[tt==1]=cc[assigned[tt==1]]
                        v['score'].extend(ss);v['tp'].extend(tt);v['side'].extend(side)
                        for group,mask in (('veh_coop',cc!=2),('inf',cc==2)):
                            target=match(M[:,mask],th/100);tp=(target>=0).astype(np.int8)
                            ignore=(tp==0)&(M[:,~mask].max(1,initial=0)>=th/100)
                            fp=((tp==0)&~ignore).astype(np.int8);q=v['groups'][group]
                            q['score'].extend(ss);q['tp'].extend(tp);q['fp'].extend(fp);q['GT']+=int(mask.sum())
                for th,v in vectors.items():
                    order=np.argsort(-np.asarray(v['score']),kind='stable');tt=np.asarray(v['tp'])[order];side=np.asarray(v['side'])[order];cum=np.cumsum(1-tt)
                    for rate in (.01,.05,.1,.2,.5):
                        limit=int(rate*N);sel=cum<=limit;count={s:int(((side==k)&sel).sum()) for k,s in enumerate(('veh','coop','inf'))}
                        gold=goldFP[name,wait,th,limit];exact=count==gold['TP_by_source'] and int(tt[sel].sum())==gold['TP_total'] and int((1-tt[sel]).sum())==gold['FP_used']
                        budget.append(dict(method=name,wait_ms=wait,threshold=th,FP_budget=limit,TP_by_source=count,producer_exact=exact))
                        if not exact:mismatches.append(budget[-1])
                    for group,q in v['groups'].items():
                        value=voc_ap(q['tp'],q['fp'],q['score'],q['GT'],True);gold=goldAP[name,wait,th,group]
                        row=dict(method=name,wait_ms=wait,threshold=th,source=group,AP=100*value['ap'],GT=q['GT'],TP=value['tp'],FP=value['fp'])
                        row['producer_exact']=abs(row['AP']-gold['AP'])<1e-8 and all(row[k]==gold[k] for k in ('GT','TP','FP'))
                        ap.append(row)
                        if not row['producer_exact']:mismatches.append(row)
                atomic(OUT/'AP.json',ap);atomic(OUT/'fixed_FP.json',budget);atomic(OUT/'status.json',dict(created_at=now(),wait_ms=wait,method=name))
            atomic(OUT/'census.json',dict(join=join,panels=census))
        status='complete_independently_reviewed' if not mismatches else 'completed_discrepancy_retained'
        atomic(OUT/'terminal.json',dict(created_at=now(),status=status,
            AP=ev(OUT/'AP.json'),fixed_FP=ev(OUT/'fixed_FP.json'),census=ev(OUT/'census.json'),
            mismatches=mismatches,GPU_forwards=0,wall_s=time.monotonic()-start,
            independent_bootstrap_not_claimed=True))
        record('E8_source_independent',dict(status=status,terminal=ev(OUT/'terminal.json')))
    except Exception as exc:
        atomic(OUT/'terminal.json',dict(created_at=now(),status='failed_retained',error=repr(exc),traceback=traceback.format_exc()))
        record('E8_source_independent',dict(status='failed_retained',terminal=ev(OUT/'terminal.json')));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run']);globals()[p.parse_args().mode]()
