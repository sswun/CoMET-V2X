"""Fair paired CoAlign factors and scoped static dose figures, zero forwards."""
import csv,json,math
from collections import defaultdict
import numpy as np
from peer14_cycle03_v1_20261103 import ROOT,AUDIT,CYCLE,C00,C18,C15,atomic,ev,load,now,register,record
from close_peer14_cycle03_v1_20261103 import table

def run():
    register('factor_analysis',__file__,'Compare within-CoAlign paired PUBLIC/ONLINE budgets and own-weight graph/plugin contrasts; do not equate fixed-input references to full graph systems. Figures show same-population interventions only, every sign retained.',dict(GPU_forwards=0,CPU_threads=1,new_disk_MiB='<10 tables/PDF/PNG',updates=0))
    p=CYCLE/'CoAlign_all_completed_weight_input_factors.csv'
    rows=list(csv.DictReader(p.open()));groups=defaultdict(list)
    for r in rows:groups[r['producer'],r['arm']].append(float(r['AP70']))
    def value(path,arm):
        v=groups[str(path.relative_to(AUDIT)),arm];assert len(v)==4,(path,arm,len(v));return sum(v)/4
    factors=[];identities=[]
    for seed in (20261102,20261103,20261104):
        paths={}
        for train in ('public','online'):
            if seed==20261102:
                graph=ref=C15/'coalign_full_v1'/('adapted_evaluation' if train=='public' else 'online_adapted_evaluation')
                upper=C00/'coalign_upper_parent_recovery_v2'/train/('first_'+train+'_official')/'official/evaluation'
                slow=C00.parent/'cycle21_peer14_v1_20261102/coalign_reference_completion_v1'/train/'slow/evaluation'
            else:
                root=C18/'coalign_confirmation_v1'/('seed'+str(seed));graph=root/(train+'_graph/evaluation');ref=root/(train+'_reference/evaluation')
                u=C00/'coalign_upper_parent_recovery_v2'/train/('seed'+str(seed)+'_'+train);upper=u/'official/evaluation';slow=u/'slow/evaluation'
            paths[train]=dict(graph=graph,ref=ref,upper=upper,slow=slow)
        # Check exact ordered conditions/GT identifiers before subtracting AP.
        first=None
        for train,ps in paths.items():
            for name,path in ps.items():
                records=load(path/'records.json');ids=[(r['sequence'],r['current_vehicle_frame'],r['wait_ms'],r['global_index'],r['GT_shard'],r['GT_case']) for r in records]
                if first is None:first=ids
                assert first==ids,(seed,train,name,'actual population mismatch')
                identities.append(dict(seed=seed,training=train,factor=name,terminal=ev(path/'terminal.json'),records=ev(path/'records.json')))
        public=paths['public'];online=paths['online'];pub=value(public['ref'],'NO_GRAPH');on=value(online['ref'],'K1_INPUT')
        factors.append(dict(seed=seed,PUBLIC_PUBLIC=pub,ONLINE_K1=on,fair_paired_delta_pp=on-pub,
            PUBLIC_K1_plugin_delta_pp=value(public['ref'],'K1_INPUT')-pub,
            PUBLIC_FULL_graph_delta_pp=value(public['graph'],'COALIGN_FULL')-pub,
            ONLINE_FULL_graph_delta_pp=value(online['graph'],'COALIGN_FULL')-value(online['ref'],'NO_GRAPH'),
            ONLINE_slow_gap_pp=max(value(online['slow'],a) for a in ('OFFICIAL_SEQMEAN_REFERENCE','OFFICIAL_MA5_REFERENCE'))-on,
            ONLINE_full_official_gap_pp=value(online['upper'],'OFFICIAL_REFERENCE')-on,
            scope='AP70, four waits own ROI/shared TraF GT; first seed id denotes existing unaltered original run'))
    table('CoAlign_three_seed_fair_graph_plugin_upper_factors.csv',factors)
    summary={}
    for field in factors[0]:
        if field in ('seed','scope'):continue
        v=[r[field] for r in factors];summary[field]=dict(mean=float(np.mean(v)),sample_SD=float(np.std(v,ddof=1)),n=3,values=v)
    atomic(CYCLE/'CoAlign_three_seed_factor_synthesis.json',dict(created_at=now(),identities=identities,summary=summary,scope='No common/source inference from these shared-TraF-GT own prediction ROI factors; source/common panels separate. Reference inputs nondeployable; graph rows only actual FULL arms.',GPU_forwards=0))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    curves=list(csv.DictReader((CYCLE/'SPD_dose_mean_curve_figure_data.csv').open()))
    fig,axes=plt.subplots(1,2,figsize=(9,3.5),sharex=True)
    for ax,role in zip(axes,('val21','mechanism_holdout14')):
        for method in ('K1_exact','K2_infra','K3_K1_expanded'):
            rr=sorted([r for r in curves if (r['family'],r['role'],r['method'],r['source_group'],r['threshold'])==('current',role,method,'all','70')],key=lambda r:float(r['alpha']))
            assert len(rr)==5
            ax.plot([float(r['alpha']) for r in rr],[float(r['AP_mean']) for r in rr],marker='o',label=method)
        ax.set_title(role);ax.set_xlabel('Annotation translation fraction alpha');ax.set_ylabel('AP70 (%)');ax.grid(alpha=.25);ax.legend(fontsize=8)
    fig.suptitle('SPD: fixed saved predictions, same population within each panel')
    fig.tight_layout();fig.savefig(CYCLE/'SPD_annotation_dose_fixed_predictions.pdf');fig.savefig(CYCLE/'SPD_annotation_dose_fixed_predictions.png',dpi=160);plt.close(fig)
    atomic(CYCLE/'figure_identity.json',dict(created_at=now(),script=ev(__file__),source=ev(CYCLE/'SPD_dose_mean_curve_figure_data.csv'),PDF=ev(CYCLE/'SPD_annotation_dose_fixed_predictions.pdf'),interpretation='Label-derived intervention; sampled peaks and crossings descriptive only, not physical truth/method selection; independent dose review pending'))
    record('factor_analysis',dict(status='complete_light_checked',terminal=ev(CYCLE/'CoAlign_three_seed_factor_synthesis.json')))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':run()
