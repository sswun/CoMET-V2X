"""Actual completed three-seed graph factors and scoped source/ceiling tables."""
import csv,inspect,json
from collections import defaultdict
import numpy as np
from peer14_cycle12_v1_20261103 import ROOT,AUDIT,CYCLE,C09,C06,C18,C15,atomic,ev,load,now,register,record
import analyze_peer14_cycle09_factors_v1_20261103 as old
import run_peer14_coalign_graph_ceilings_v1_20261103 as ceiling

def table(name,rows):
    assert rows
    with (CYCLE/name).open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)

def grouped(rows,keys,value):
    groups=defaultdict(list)
    for r in rows:groups[tuple(r[k] for k in keys)].append(r[value])
    return [dict(zip(keys,k),mean=float(np.mean(v)),sample_SD=float(np.std(v,ddof=1)) if len(v)>1 else None,n=len(v),values=v) for k,v in groups.items()]

def source_rows():
    rows=[];missing=[]
    for lane in ('gpu0','gpu1','remote'):
        seed=old.composed.plans(lane)[0]
        for train in ('public','online'):
            for arm in old.composed.KINDS:
                p=C06/'composed_consumers_v1/sources/native_source_aux'/((old.composed.BASE/lane/train/arm/'evaluation').relative_to(AUDIT))
                if not (p/'terminal.json').exists():missing.append(str(p.relative_to(ROOT)));continue
                assert load(p/'terminal.json')['status']=='complete_light_checked'
                for r in load(p/'panels.json'):rows.append(dict(seed=seed,lane=lane,training=train,**r))
    return rows,missing

def run():
    register('factor_analysis',__file__,'Update only complete four-wait fixed-weight graph factors, actual ordered GT/conditions and weights checked. Accept old remote full independent only from its own terminal. Source/physical/dose tables accept actual producer terminals; partial ceilings report n rather than fill missing. No microvariant/threshold selection.',dict(GPU_forwards=0,training_updates=0,CPU_threads=1,new_disk_MiB='<15 tables/figure data'))
    s=inspect.getsource(old.run).split("    review=CYCLE/")[0]
    s=s[s.index('    rows=[];pairs=[];audits=[];missing=[]'):]
    ns=dict(old.__dict__,CYCLE=CYCLE,table=table);exec(compile(s,__file__+'::exact_completed_factors','exec'),ns)
    rows,pairs,audits,missing,summary=(ns[k] for k in ('rows','pairs','audits','missing','summary'))
    assert not missing and all(r['complete_three_seed'] for r in summary)
    cross=[]
    for seed in sorted({r['seed'] for r in rows}):
        for th in (50,70):
            def v(t,a):
                rr=[r for r in rows if (r['seed'],r['threshold'],r['training'],r['arm'])==(seed,th,t,a)];assert len(rr)==4;return np.mean([r['AP'] for r in rr])
            for a,b in [('K1_GRAPH','PUBLIC_GRAPH'),('K1_GRAPH','K1_GRAPH'),('K2_GRAPH','K2_GRAPH'),('ICP_GRAPH_REFERENCE','ICP_GRAPH_REFERENCE')]:cross.append(dict(seed=seed,threshold=th,contrast='ONLINE_'+a+' minus PUBLIC_'+b,delta_pp=float(v('online',a)-v('public',b))))
    table('CoAlign_training_input_graph_paired_factors.csv',cross)
    source,source_missing=source_rows();table('CoAlign_complete_graph_source_all.csv',source)
    sourcepairs=[]
    for seed in sorted({r['seed'] for r in source}):
        for train in ('public','online'):
            for protocol in ('official','interpolated','no_interpolation'):
                for roi in ('native','common'):
                    for group in ('all','veh','coop','inf','veh_coop'):
                        for th in (.5,.7):
                            for x,y in [('K2_GRAPH','K1_GRAPH'),('ICP_GRAPH_REFERENCE','K1_GRAPH')]:
                                def vv(a):return sorted([r for r in source if (r['seed'],r['training'],r['protocol'],r['roi'],r['source_group'],r['threshold'],r['arm'])==(seed,train,protocol,roi,group,th,a)],key=lambda r:r['wait_ms'])
                                a,b=vv(x),vv(y)
                                if len(a)!=4 or len(b)!=4:continue
                                assert [r['wait_ms'] for r in a]==[r['wait_ms'] for r in b]==[0,100,200,400]
                                sourcepairs.append(dict(seed=seed,training=train,protocol=protocol,roi=roi,source_group=group,threshold=th,contrast=x+' minus '+y,delta_pp=float(np.mean([r['ap'] for r in a])-np.mean([r['ap'] for r in b]))))
    table('CoAlign_complete_graph_source_paired.csv',sourcepairs)
    sourcemeans=grouped(sourcepairs,['training','protocol','roi','source_group','threshold','contrast'],'delta_pp');table('CoAlign_complete_graph_source_paired_summary.csv',sourcemeans)
    upper=[];upper_missing=[]
    for lane in ('gpu0','gpu1','remote'):
        seed=old.composed.plans(lane)[0]
        for train in ('public','online'):
            for arm in ceiling.KINDS:
                p=ceiling.BASE/lane/train/arm/'evaluation'
                if not (p/'terminal.json').exists():upper_missing.append(dict(lane=lane,training=train,arm=arm));continue
                assert load(p/'terminal.json')['status']=='native_AP_complete_light_checked'
                base=old.composed.BASE/lane/train/'K1_GRAPH/evaluation';assert old.identity(p)==old.identity(base)
                inp=load(p/'input_identity.json');assert inp['fusion_weight']==ev(old.composed.plans(lane)[2][train]) and inp['stage1_weight']==ev(old.composed.plans(lane)[1])
                for th in (50,70):
                    aa=old.values(p,arm,th);bb=old.values(base,'K1_GRAPH',th)
                    upper.append(dict(seed=seed,lane=lane,training=train,arm=arm,threshold=th,mean_AP=float(np.mean(aa)),K1_graph_AP=float(np.mean(bb)),reference_minus_K1_graph_pp=float(np.mean(aa)-np.mean(bb)),annotation_reference_only=True))
    if upper:table('CoAlign_full_graph_completed_annotation_ceilings.csv',upper)
    remote=C18/'coalign_full_independent_v1/remote/terminal.json';rv=load(remote);assert rv['status']=='complete_independently_reviewed' and all(rv['checks'].values())
    atomic(CYCLE/'scientific_factor_synthesis.json',dict(created_at=now(),status='complete_light_checked',graph_pair_summary=summary,training_input_graph_summary=grouped(cross,['threshold','contrast'],'delta_pp'),identity_audit=audits,missing_composed=missing,source_pair_summary=sourcemeans,source_missing_at_snapshot=source_missing,annotation_ceiling_completed=upper,annotation_ceiling_missing=upper_missing,old_CoAlign_remote_independent=ev(remote),old_CoAlign_remote_independent_panels=len(load(ROOT/rv['panel_checks']['path'])),new_composed_independent_scope='Own per-producer terminals only; snapshot review manifest separately, not all4320 accepted',OPV_full_async_independent=ev(C09/'opv2v_async_heading_U_independent_v2/producer/terminal.json'),OPV_plot_reference=ev(C09/'OPV2V_async_controlled_factor.pdf'),no_OPV_global_flip=True,GPU_forwards=0,scope='Completed full stage1/author graph/fusion, fixed same-seed weights and actual GT arrays. Shared TraF GT/own CoAlign prediction ROI; K1native q005 translation not late heading+U. Sources/ceilings partial explicit, not physical truth.'))
    record('factor_analysis',dict(status='complete_light_checked',terminal=ev(CYCLE/'scientific_factor_synthesis.json')))
    print(json.dumps(dict(graph_summary=summary,source_completed_producers=len(source)//240,source_missing=len(source_missing),ceiling_completed=len(upper)//2,old_remote_panels=len(load(ROOT/rv['panel_checks']['path']))),indent=2))

if __name__=='__main__':run()
