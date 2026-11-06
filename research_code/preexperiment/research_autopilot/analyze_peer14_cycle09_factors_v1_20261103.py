"""Actual completed graph factors and independently accepted controlled figure."""
import csv,json
from collections import defaultdict
import numpy as np
from peer14_cycle09_v1_20261103 import ROOT,AUDIT,CYCLE,C06,C18,C15,atomic,ev,load,now,register,record
from analyze_peer14_cycle06_factors_v1_20261103 import identity,values
import run_peer14_coalign_composed_graph_v1_20261103 as composed
import run_peer14_opv2v_async_heading_U_v1_20261103 as opv

def table(name,rows):
    assert rows
    with (CYCLE/name).open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)

def run():
    register('factor_analysis',__file__,'Partial completed full graph composition snapshot: require each four-wait terminal and exact ordered conditions/GT/weight identity before paired subtraction. Report actual paired n, never fill missing seed or mix no-graph/full-system. OPV full heading+U factorial signs and sources plotted only after independent288 AP/ranking pass; failed global flip retained.',dict(GPU_forwards=0,CPU_threads=1,new_disk_MiB='<10 CSV/JSON/PDF/PNG'))
    rows=[];pairs=[];audits=[];missing=[]
    for lane in ('gpu0','gpu1','remote'):
        seed=composed.plans(lane)[0]
        for train in ('public','online'):
            ref=C15/'coalign_full_v1'/('adapted_evaluation' if train=='public' else 'online_adapted_evaluation') if seed==20261102 else C18/'coalign_confirmation_v1'/('seed'+str(seed))/(train+'_reference/evaluation')
            graph=ref if seed==20261102 else C18/'coalign_confirmation_v1'/('seed'+str(seed))/(train+'_graph/evaluation')
            loaded={};base_ids=None
            for arm,path in [('NO_GRAPH',ref),('K1_INPUT',ref),('PUBLIC_GRAPH',graph)]+[(k,composed.BASE/lane/train/k/'evaluation') for k in composed.KINDS]:
                if not (path/'terminal.json').exists():missing.append(dict(lane=lane,training=train,arm=arm));continue
                term=load(path/'terminal.json');assert term['status']=='native_AP_complete_light_checked'
                data=identity(path)
                if base_ids is None:base_ids=data
                assert data==base_ids,(lane,train,arm,'condition/GT mismatch')
                inp=load(path/'input_identity.json');assert inp['fusion_weight']==ev(composed.plans(lane)[2][train])
                isgraph=arm not in ('NO_GRAPH','K1_INPUT')
                if isgraph:
                    assert term['graph_calls']>0 and inp['stage1_weight']==ev(composed.plans(lane)[1]),(lane,arm,'not matched complete graph')
                actual='COALIGN_FULL' if arm=='PUBLIC_GRAPH' else arm
                loaded[arm]={th:values(path,actual,th) for th in (50,70)}
                audits.append(dict(lane=lane,training=train,arm=arm,terminal=ev(path/'terminal.json'),GT_and_conditions_exact=True,fusion_weight=inp['fusion_weight'],stage1_graph_fusion=isgraph))
                for th in (50,70):
                    for wait,v in zip((0,100,200,400),loaded[arm][th]):rows.append(dict(seed=seed,lane=lane,training=train,arm=arm,wait_ms=wait,threshold=th,AP=v,stage1_graph_fusion=isgraph))
            for x,y in [('K1_GRAPH','K1_INPUT'),('K1_GRAPH','PUBLIC_GRAPH'),('K2_GRAPH','K1_GRAPH'),('ICP_GRAPH_REFERENCE','K1_GRAPH')]:
                if x in loaded and y in loaded:
                    for th in (50,70):pairs.append(dict(seed=seed,training=train,contrast=x+' minus '+y,threshold=th,delta_pp=float(np.mean(loaded[x][th])-np.mean(loaded[y][th])),x_AP=float(np.mean(loaded[x][th])),y_AP=float(np.mean(loaded[y][th]))))
    summary=[];groups=defaultdict(list)
    for r in pairs:groups[r['training'],r['contrast'],r['threshold']].append(r['delta_pp'])
    for key,v in groups.items():summary.append(dict(training=key[0],contrast=key[1],threshold=key[2],n_completed_paired_seeds=len(v),mean=float(np.mean(v)),sample_SD=float(np.std(v,ddof=1)) if len(v)>1 else None,values=v,complete_three_seed=len(v)==3))
    table('CoAlign_completed_composed_graph_all_wait_factors.csv',rows);table('CoAlign_completed_composed_graph_paired_factors.csv',pairs)
    review=CYCLE/'opv2v_async_heading_U_independent_v2/producer/terminal.json';assert load(review)['status']=='complete_independently_reviewed'
    opvrows=load(opv.BASE/'producer/panels.json');table('OPV2V_async_heading_U_all288.csv',opvrows)
    opvpairs=[]
    for r in opvrows:
        if r['method']!='K1_heading_U':continue
        old=[p for p in opvrows if p['method']=='stale_same_U' and all(p[k]==r[k] for k in ('noise','split','protocol','source_group','threshold','wait_ms'))];assert len(old)==1
        opvpairs.append({**{k:r[k] for k in ('noise','split','protocol','source_group','threshold','wait_ms')},'K1_AP':r['ap'],'stale_AP':old[0]['ap'],'delta_pp':r['ap']-old[0]['ap']})
    table('OPV2V_async_heading_U_paired_figure_data.csv',opvpairs)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(9,3.4),sharey=True)
    for ax,family in zip(axes,('SPD_structured','iid_02m_02deg')):
        for protocol,style in [('P1_simulation','-'),('P2_inherited','--')]:
            for source,marker in [('all','o'),('remote_exclusive','s')]:
                rr=sorted([r for r in opvpairs if r['noise']==family and r['split']=='test' and r['protocol']==protocol and r['source_group']==source and r['threshold']==.7],key=lambda r:r['wait_ms']);assert len(rr)==2
                ax.plot([r['wait_ms'] for r in rr],[r['delta_pp'] for r in rr],style+marker,label=protocol+'/'+source)
        ax.axhline(0,color='black',lw=.7);ax.set_title(family);ax.set_xticks([0,400]);ax.set_xlabel('Mocked source age (ms)');ax.grid(alpha=.2)
    axes[0].set_ylabel('K1 heading+U minus stale same-U (AP70 pp)');axes[1].legend(fontsize=6.5)
    fig.suptitle('Bounded OPV2V controlled transfer: preregistered P1/P2 flip NOT established',fontsize=10);fig.tight_layout();fig.savefig(CYCLE/'OPV2V_async_controlled_factor.pdf');fig.savefig(CYCLE/'OPV2V_async_controlled_factor.png',dpi=160);plt.close(fig)
    atomic(CYCLE/'scientific_factor_synthesis.json',dict(created_at=now(),status='complete_light_checked',graph_pair_summary=summary,missing_at_snapshot=missing,identity_audit=audits,source_independent_pending=True,OPV_independent=ev(review),OPV_preregistered=ev(opv.BASE/'producer/preregistered_read.json'),OPV_figure_data=ev(CYCLE/'OPV2V_async_heading_U_paired_figure_data.csv'),OPV_plot=ev(CYCLE/'OPV2V_async_controlled_factor.pdf'),scope='Within fixed-weight/GT CoAlign partial paired completed factors only. OPV exact full heading+U pipeline but bounded mock-timeline/same-head transfer, no OPV full benchmark or general P1/P2 flip claim.',GPU_forwards=0))
    record('factor_analysis',dict(status='complete_light_checked',terminal=ev(CYCLE/'scientific_factor_synthesis.json')))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':run()
