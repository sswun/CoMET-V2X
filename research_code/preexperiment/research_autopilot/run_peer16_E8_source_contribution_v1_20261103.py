"""Current-author source census and fixed-FP TP decomposition, zero forwards."""
import argparse,time,traceback
from collections import defaultdict,Counter
import numpy as np
from peer16_v1_20261103 import ROOT,BASE,atomic,ev,load,now,register,record
import run_peer15_author_recipe_v1_20261103 as author
from run_peer07_spd_gt_source_split_ap_v1_20261031 import map_provenance,outcome,metric,iou32,greedy
OUT=BASE/'E8_source_contribution'
def prepare():
    register('E8_source_contribution',__file__,
        'Audit actual infra-exclusive denominator and decompose globally ranked TP at identical prespecified FP budgets into original veh/coop/inf tags. All targets unchanged; source-ignore AP is supplementary, not arithmetic AP attribution or physical truth.',
        dict(GPU_forwards=0,CPU_threads=1,updates=0,wall_h='1-5',disk_MiB='<5 compact source receipts; original predictions/boxes/GT referenced'),
        current=ev(author.BASE/'author_recipe_v2/terminal.json'),review=ev(author.BASE/'author_recipe_independent_bulk_v2/terminal.json'),
        FP_rates_per_GT=[.01,.05,.1,.2,.5],thresholds=[.5,.7],
        methods=['current_heading_calq_matchedU','v1_CV_q015_matchedU','stale__matched_head'],role='val21',
        remaining='All other main-table method source contrasts and independent statistical bootstrap remain separate open scope')
def run():
    start=time.monotonic();record('E8_source_contribution',dict(status='running_original_from_side_census_and_fixed_FP_decomposition'))
    try:
        m,rows,oldpred=author.materials('val21');allrows=[None]*(max(map(int,m['dataset_index']))+1)
        for i,r in zip(m['dataset_index'],rows):allrows[int(i)]=r
        codes,join=map_provenance(m,allrows,'val21',ROOT/'artifacts/data/v2x_seq_spd_main_domain_shared_v1/V2X-Seq-SPD/cooperative/label')
        assert join['unmatched']==join['ambiguous']==0,join
        with np.load(author.BASE/'author_recipe_v2/primary_geometry.npz') as z:current=z['current_boxes'].copy()
        with np.load(author.BASE/'author_recipe_v2/primary_predictions.npz') as z:newpred={k:z[k].copy() for k in z.files}
        cases={'current_heading_calq_matchedU':(current,newpred),'v1_CV_q015_matchedU':(m['kf_boxes'],newpred),'stale__matched_head':(m['stale_boxes'],oldpred)}
        source_panels=[];budget=[];census=[]
        for w in (0,100,200,400):
            cond=np.flatnonzero(m['wait_ms']==w);gtcounts=Counter()
            for i in cond:
                ga,gb=map(int,m['gt_offsets'][i:i+2]);use=author.common_mask(m['gt_boxes'][ga:gb]);gtcounts.update(map(int,codes[ga:gb][use]))
            N=sum(gtcounts.values());assert N==31123
            census.append(dict(wait_ms=w,GT=N,by_original_from_side={('veh','coop','inf')[k]:v for k,v in sorted(gtcounts.items())},infra_exclusive_fraction=gtcounts[2]/N))
            for name,(boxes,pred) in cases.items():
                for th in (50,70):
                    scores=[];tp=[];sides=[];group=defaultdict(lambda:dict(score=[],tp=[],fp=[],GT=0))
                    for i in cond:
                        a,b=map(int,pred[name+'__offset'][i:i+2]);ids=pred[name+'__candidate'][a:b].astype(int);bb=boxes[ids];ss=pred[name+'__score'][a:b]
                        ga,gb=map(int,m['gt_offsets'][i:i+2]);gg=m['gt_boxes'][ga:gb];cc=codes[ga:gb];keep=author.common_mask(gg);gg,cc=gg[keep],cc[keep]
                        matched=greedy(iou32(bb,gg),np.arange(len(bb)),th/100);tt=(matched>=0).astype(np.int8);assert np.array_equal(tt,pred[name+'__tp'+str(th)][a:b]),('accepted_TP',name,w,th,int(i))
                        side=np.full(len(bb),-1,np.int8);side[tt==1]=cc[matched[tt==1]];scores.extend(ss);tp.extend(tt);sides.extend(side)
                        for source in ('veh_coop','inf'):
                            target=cc!=2 if source=='veh_coop' else cc==2;t,f,ignore=outcome(bb,gg[target],gg[~target],th/100);v=group[source];v['score'].extend(ss);v['tp'].extend(t);v['fp'].extend(f);v['GT']+=int(target.sum())
                    rank=np.argsort(-np.asarray(scores),kind='stable');tt=np.asarray(tp)[rank];sc=np.asarray(sides)[rank];fp=np.cumsum(1-tt)
                    for rate in (.01,.05,.1,.2,.5):
                        limit=int(rate*N);select=fp<=limit;counts={s:int(((sc==k)&select).sum()) for k,s in enumerate(('veh','coop','inf'))};assert sum(counts.values())==int(tt[select].sum())
                        budget.append(dict(method=name,wait_ms=w,threshold=th,FP_budget=limit,FP_rate_per_GT=rate,TP_by_source=counts,TP_total=int(tt[select].sum()),FP_used=int((1-tt[select]).sum()),GT=N))
                    for source,v in group.items():
                        value=metric(np.asarray(v['tp']),np.asarray(v['fp']),np.asarray(v['score']),v['GT']);source_panels.append(dict(method=name,wait_ms=w,threshold=th,source=source,AP=100*value['ap'],GT=v['GT'],TP=value['tp'],FP=value['fp'],ignore_other_source=True))
                    atomic(OUT/'status.json',dict(created_at=now(),wait_ms=w,method=name,threshold=th,wall_s=time.monotonic()-start))
            atomic(OUT/'source_AP.json',source_panels);atomic(OUT/'fixed_FP_TP.json',budget);atomic(OUT/'GT_census.json',dict(mapping=join,panels=census))
        atomic(OUT/'terminal.json',dict(created_at=now(),status='complete_light_checked',GT_census=ev(OUT/'GT_census.json'),fixed_FP=ev(OUT/'fixed_FP_TP.json'),source_AP=ev(OUT/'source_AP.json'),GPU_forwards=0,wall_s=time.monotonic()-start,independent_pending=True));record('E8_source_contribution',dict(status='complete_light_checked',terminal=ev(OUT/'terminal.json')))
    except Exception as e:
        atomic(OUT/'terminal.json',dict(created_at=now(),status='failed_retained',error=repr(e),traceback=traceback.format_exc()));record('E8_source_contribution',dict(status='failed_retained',terminal=ev(OUT/'terminal.json')));raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run']);globals()[p.parse_args().mode]()
