"""R6g official discrete values/saturation and five-detector label references.

Reuses independently confirmed raw-label projections, not detector-label
matching or the main-session preview. Independent order-statistic reductions
check every published numeric row. A bounded search is an inference from the
value structure, not a claim to possess the author's generation algorithm.
"""
import argparse,csv,json,math,statistics,traceback
from collections import defaultdict
import numpy as np
from peer21_v1_20261105 import *

OUT=BASE/'R6g';DOMAINS=('author','strict60','v2xvit_seed2024','v2xvit_seed2025','where2comm')
ROLES=('calibration8','val21','mechanism_holdout14')
TABLE=ROOT/'docs/paper_draft_v0_20261103/final_tables_v3/aux_colabelled/offset_grid'
REC=P19/'recipe_v3_interface_recovery_v2'

def raw_reference(review=False):
    groups=defaultdict(list)
    with np.load(P20/'R6a_raw_labels/direct_residuals.npz') as z:
        for ident,v in zip(z['identity'],z['residual_and_offset']):
            r,s,f,k,*_=str(ident).split('|')
            if k=='coop':groups[r,s,f].append(v[:2])
    result={}
    for k,v in groups.items():
        if len(v)<3:continue
        median=[statistics.median([float(x[j]) for x in v]) for j in (0,1)] if review else np.median(v,axis=0).tolist()
        result[k]=dict(median=median,n=len(v))
    return result

def prepare():
    assert load(P20/'R6a_raw_labels_independent_v2/terminal.json')['confirmed']
    assert not (OUT/'inputs.json').exists()
    OUT.mkdir(parents=True,exist_ok=True)
    frame=load(P20/'R6a_raw_labels/frames.json');lookup={(r['sequence'],r['timestamp_us']):r for r in frame}
    from run_peer16_cached_methods_v1_20261103 import cross_specs
    sc,_=cross_specs();oldc=load(AUDIT/'contract_v1.json');inputs=[]
    for domain in DOMAINS:
        for role in ROLES:
            source=REC/domain/(role+'_state.npz')
            if role!='calibration8':keys=[tuple(k) for k in load(REC/domain/(role+'_state_ready.json'))['keys']]
            elif domain=='author':
                p=ROOT/'artifacts/research_autopilot/status_audit_20261021_1200/object_history_selector_v1/train_dataset/conditions.jsonl'
                rows=[json.loads(l) for l in p.read_text().splitlines() if l.strip()]
                rows=sorted([r for r in rows if r['role']==role],key=lambda r:(str(r['sequence']),int(r['wait_ms']),int(r['receiver_timestamp_us'])))
                keys=[(str(r['sequence']),int(r['receiver_timestamp_us']),int(r['wait_ms'])) for r in rows]
            else:
                p=ROOT/oldc['inputs'][role+'_material']['path'] if domain=='strict60' else ROOT/sc['domains'][domain]['roles'][role]['population']
                with np.load(p) as z:keys=[(str(s),int(t),int(w)) for s,t,w in zip(z['sequence'],z['receiver_timestamp_us'],z['wait_ms'])]
            with np.load(source) as z:state={k:z[k].copy() for k in ('delta','observation','anchors')}
            assert len(keys)==len(set(keys))==len(state['delta'])
            rows=[lookup[str(s),int(t)] for s,t,w in keys]
            target=OUT/(domain+'_'+role+'_keyed.npz')
            np.savez_compressed(target,**state,sequence=np.array([k[0] for k in keys]),timestamp_us=np.array([k[1] for k in keys]),
                wait_ms=np.array([k[2] for k in keys]),frame=np.array([r['frame'] for r in rows]))
            inputs.append(dict(domain=domain,role=role,keyed=ev(target),state=ev(source),key_source=ev(REC/domain/(role+'_state_ready.json')) if role!='calibration8' else ev(p)))
    atomic(OUT/'inputs.json',dict(created_at=now(),keyed=inputs,raw_projection=ev(P20/'R6a_raw_labels/direct_residuals.npz'),
        frames=ev(P20/'R6a_raw_labels/frames.json'),raw_independent=ev(P20/'R6a_raw_labels_independent_v2/terminal.json')))
    for name in ('R6g','R6g_independent'):
        register(name,__file__,__doc__,dict(GPU_forwards=0,updates=0,CPU_threads=1,disk_MiB='<20 keyed states and source-linked CSVs',replicates=2000),
            inputs=ev(OUT/'inputs.json'),claim_limits='Label-derived reference only, not physical truth; no documented offset-generation algorithm',
            independence='Existing independent projection algebra reused; medians/RMSE/CDF and all source CSV rows independently reduced')

def compute(review=False):
    reference=raw_reference(review);frames=load(P20/'R6a_raw_labels/frames.json');frameby={r['frame']:r for r in frames}
    rawoff=np.asarray([r['offset'] for r in frames]);grid=[sorted(set(float(x) for x in rawoff[:,j])) for j in (0,1)]
    fractions=[]
    for role in ('fit24',*ROLES):
        off=np.asarray([r['offset'] for r in frames if r['role']==role]);nz=np.any(off!=0,axis=1)
        sat=np.any(np.isclose(off,-1.1,atol=1e-8,rtol=0)|np.isclose(off,.66,atol=1e-8,rtol=0),axis=1)
        fractions.append(dict(role=role,frames=len(off),nonzero=int(nz.sum()),saturated=int((nz&sat).sum()),
            fraction_of_nonzero=float((nz&sat).sum()/nz.sum())))
    bound=[];normrows=[]
    for r in frames:
        k=r['role'],r['sequence'],r['frame']
        if k not in reference or r['zero']:continue
        normrows.append((r,np.asarray(reference[k]['median'])))
    for axis in (0,1):
        for value in grid[axis]:
            values=[float(m[axis]) for r,m in normrows if abs(r['offset'][axis]-value)<1e-8]
            if not values:continue
            med=statistics.median(values) if review else float(np.median(values))
            bound.append(dict(axis='xy'[axis],official_value=value,n=len(values),displacement_median=med,
                p10=float(np.quantile(values,.1)),p90=float(np.quantile(values,.9)),
                fraction_beyond=float(sum(v<value if value<0 else v>value for v in values)/len(values))))
    off=np.asarray([r['offset'] for r,m in normrows]);med=np.asarray([m for r,m in normrows])
    common=dict(frames=len(med),official_vs_median_medianabs=float(np.median(np.linalg.norm(off-med,axis=1))),
        corr_x=float(np.corrcoef(off[:,0],med[:,0])[0,1]),corr_y=float(np.corrcoef(off[:,1],med[:,1])[0,1]),
        label_norm_gt15=float(np.mean(np.linalg.norm(med,axis=1)>1.5)),official_norm_gt15=float(np.mean(np.linalg.norm(off,axis=1)>1.5)))
    panels=[]
    for item in load(OUT/'inputs.json')['keyed']:
        with np.load(ROOT/item['keyed']['path']) as z:
            role=item['role'];refs=[];official=[];valid=[]
            for f in z['frame']:
                r=frameby[str(f)];v=reference.get((role,r['sequence'],str(f)))
                refs.append(v['median'] if v else [0.,0.]);official.append(r['offset']);valid.append(v is not None and not r['zero'])
            target=np.asarray(refs);off=np.asarray(official);valid=np.asarray(valid);sat=np.any(np.isclose(off,-1.1,atol=1e-8,rtol=0)|np.isclose(off,.66,atol=1e-8,rtol=0),axis=1)
            for w in sorted(set(map(int,z['wait_ms']))):
                for group,mask in [('all',valid),('saturated',valid&sat),('unsaturated',valid&~sat),
                    ('y_low',valid&np.isclose(off[:,1],-1.1,atol=1e-8,rtol=0))]:
                    use=mask&(z['wait_ms']==w);n=int(use.sum())
                    if not n:continue
                    for method,estimate in [('H1',z['delta']),('single_H1_anchor_mean',z['observation']),('OFFICIAL',off),('PUBLIC',np.zeros_like(off))]:
                        error=np.linalg.norm(estimate[use]-target[use],axis=1)
                        rmse=math.sqrt(math.fsum(float(e)**2 for e in error)/n) if review else float(np.sqrt(np.mean(error**2)))
                        panels.append(dict(domain=item['domain'],role=role,wait_ms=w,population=group,method=method,n=n,
                            median_error=statistics.median(error.tolist()) if review else float(np.median(error)),RMSE=rmse,
                            estimate_y_median=statistics.median(estimate[use,1].tolist()),reference_y_median=statistics.median(target[use,1].tolist())))
    return dict(grid=grid,saturation=fractions,boundary=bound,common=common,panels=panels,
        limits='Shared label-derived reference, not physical truth. Generation search algorithm undocumented; only value-grid/saturation verified.')

def run(review=False):
    name='R6g_independent' if review else 'R6g';dest=BASE/name
    try:
        result=compute(review)
        if review:
            gold=load(OUT/'results.json')
            def equal(a,b):
                if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
                if isinstance(a,list):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
                if isinstance(a,float):return abs(a-b)<1e-9
                return a==b
            assert equal(result,gold),'independent reduction mismatch'
            for k in ('saturation','boundary','panels'):
                with (TABLE/(k+'.csv')).open() as f:rows=list(csv.DictReader(f))
                assert len(rows)==len(result[k])
                for a,b in zip(rows,result[k]):
                    for key,v in b.items():
                        if isinstance(v,(float,int)):assert abs(float(a[key])-v)<1e-9
                        else:assert a[key]==str(v)
            atomic(dest/'review.json',dict(created_at=now(),source=ev(OUT/'results.json'),all_CSV_rows_source_verified=True,
                separate_order_statistics=True,independent_projection_reused=load(OUT/'inputs.json')['raw_independent']))
        else:
            TABLE.mkdir(parents=True,exist_ok=True);atomic(OUT/'results.json',result)
            for k in ('saturation','boundary','panels'):
                with (TABLE/(k+'.csv')).open('x',newline='') as f:
                    w=csv.DictWriter(f,fieldnames=list(result[k][0]));w.writeheader();w.writerows(result[k])
            atomic(TABLE/'sources.json',dict(inputs=ev(OUT/'inputs.json'),results=ev(OUT/'results.json'),independent_pending=True))
        atomic(dest/'terminal.json',dict(created_at=now(),status='complete_independently_reviewed' if review else 'complete_light_checked',
            results=ev(OUT/'results.json'),not_physical_truth=True,all26=[len(g) for g in result['grid']]))
        record(name,dict(status='complete_independently_reviewed' if review else 'complete_light_checked',terminal=ev(dest/'terminal.json'),owner=None,resource_owner=False))
    except Exception as e:
        atomic(dest/'terminal.json',dict(created_at=now(),status='failed_retained',error=repr(e),traceback=traceback.format_exc()))
        record(name,dict(status='failed_retained',terminal=ev(dest/'terminal.json'),owner=None,resource_owner=False));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','run','review'));a=p.parse_args()
    prepare() if a.mode=='prepare' else run(a.mode=='review')
