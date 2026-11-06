"""Eight fixed-weight input factors and OFFICIAL-fit control, zero forwards."""
import csv,json
from collections import defaultdict
import numpy as np
from peer14_cycle06_v1_20261103 import ROOT,AUDIT,CYCLE,C03,C00,C18,C15,atomic,ev,load,now,register,record
from analyze_peer14_cycle03_factors_v2_20261103 import GT_identity,CACHE

def table(name,rows):
    with (CYCLE/name).open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)

def values(path,arm,th):
    rows=load(path/'panels.json');assert len(rows)==4
    return [100*r['metrics'][arm]['global_score_sort']['ap'+str(th)] for r in rows]

def identity(path):
    return [(r['sequence'],r['current_vehicle_frame'],r['wait_ms'],r['global_index'],GT_identity(r['GT_shard']),r['GT_case']) for r in load(path/'records.json')]

def main():
    register('fixed_input_factor_analysis',__file__,
        'Eight completed fixed CoAlign fusion weights: compare exact existing K2/ICP inputs to reused same-weight q005 K1 input and PUBLIC. Check actual ordered population, GT bytes, weight identity first; no full-system claim for no-graph rows. Also report new same-budget OFFICIAL-fit factor to expose annotation-reference ceiling and train/input matching without using it as deployable method.',
        dict(GPU_forwards=0,CPU_threads=1,new_disk_MiB='<5 CSV/JSON/figure data; no predictions/GT copies'))
    c=load(C03/'coalign_physical_inputs_recovery_v2_contract.json');rows=[];paired=[];identities=[]
    for train in ('public','online'):
        assert load(C03/'coalign_physical_inputs_v2'/train/'terminal.json')['status']=='complete_light_checked'
        for i,p in enumerate(c['plans'][train]):
            new=C03/'coalign_physical_inputs_v2'/train/p['name']/'evaluation'
            if train=='public':
                old=C15/'coalign_full_v1/author_evaluation' if i==0 else C15/'coalign_full_v1/adapted_evaluation' if i==1 else C18/'coalign_confirmation_v1'/('seed'+str(20261101+i))/'public_reference/evaluation'
                public_arm,k1_arm='NO_GRAPH','K1_INPUT'
                label='author' if i==0 else 'seed'+str(20261101+i)
            else:
                old=C15/'coalign_full_v1/online_adapted_evaluation' if i==0 else C18/'coalign_confirmation_v1'/('seed'+str(20261102+i))/'online_reference/evaluation' if i<3 else C00/'coalign_upper_parent_recovery_v2/official_trained/factor/evaluation'
                public_arm,k1_arm=('PUBLIC_INPUT','ONLINE_K1_INPUT') if i==3 else ('NO_GRAPH','K1_INPUT')
                label='official_fit' if i==3 else 'seed'+str(20261102+i)
            assert identity(new)==identity(old),(train,i,'population/GT mismatch')
            ni=load(new/'input_identity.json');oi=load(old/'input_identity.json');assert ni['fusion_weight']==oi['fusion_weight']
            identities.append(dict(training=train,weight_label=label,new=ev(new/'terminal.json'),old=ev(old/'terminal.json'),same_weight=ni['fusion_weight'],condition_GT_exact=True))
            for th in (50,70):
                v={a:values(new,a,th) for a in ('K2_SE2_INPUT','RAW_ICP_REFERENCE')};v.update(PUBLIC=values(old,public_arm,th),K1=values(old,k1_arm,th))
                paired.append(dict(training=train if label!='official_fit' else 'official',weight_label=label,threshold=th,PUBLIC=sum(v['PUBLIC'])/4,K1=sum(v['K1'])/4,K2=sum(v['K2_SE2_INPUT'])/4,ICP_REFERENCE=sum(v['RAW_ICP_REFERENCE'])/4,K2_minus_K1_pp=np.mean(v['K2_SE2_INPUT'])-np.mean(v['K1']),ICP_minus_K1_pp=np.mean(v['RAW_ICP_REFERENCE'])-np.mean(v['K1']),scope='fusion only; identical own readout/shared TraF GT; ICP nondeployable; K1 q005 native input not full heading+U late fusion'))
                for j,wait in enumerate((0,100,200,400)):
                    for arm,vs in v.items():rows.append(dict(training=train if label!='official_fit' else 'official',weight_label=label,threshold=th,wait_ms=wait,input_arm=arm,AP=vs[j],complete_CoAlign=False,ICP_nondeployable=arm=='RAW_ICP_REFERENCE'))
    table('CoAlign_eight_weight_input_all_wait_factors.csv',rows);table('CoAlign_eight_weight_input_mean_contrasts.csv',paired)
    summaries=[]
    for train in ('public','online'):
        for th in (50,70):
            selected=[r for r in paired if r['training']==train and r['weight_label'].startswith('seed') and r['threshold']==th];assert len(selected)==3
            for field in ('K2_minus_K1_pp','ICP_minus_K1_pp'):
                v=[r[field] for r in selected];summaries.append(dict(training=train,threshold=th,contrast=field,mean=float(np.mean(v)),sample_SD=float(np.std(v,ddof=1)),n=3,values=v))
    # New official-fit row is a same-budget, annotation-assisted control, not a fair deployable candidate.
    p=C00/'coalign_upper_parent_recovery_v2/official_trained/factor/evaluation';official=[]
    for arm in load(p/'terminal.json')['arms']:
        official.append(dict(training='OFFICIAL_REFERENCE',input=arm,AP50=float(np.mean(values(p,arm,50))),AP70=float(np.mean(values(p,arm,70))),annotation_assisted_training=True,deployable_input=arm!='OFFICIAL_REFERENCE'))
    table('CoAlign_same_budget_OFFICIAL_fit_control.csv',official)
    atomic(CYCLE/'factor_population_GT_identity_audit.json',dict(created_at=now(),identities=identities,GT_sources=CACHE,no_GPU_rerun=True))
    summary=dict(created_at=now(),status='complete_light_checked',fixed_input_three_seed=summaries,official_fit_control=official,all8weights_complete=True,identities=ev(CYCLE/'factor_population_GT_identity_audit.json'),full_independent_pending=True,scope='No stage1/graph in these8weights; complete-system composition is separate new GPU chain. Not across-backbone absolute ranking or new estimator variant.',GPU_forwards=0)
    atomic(CYCLE/'scientific_factor_synthesis.json',summary);record('fixed_input_factor_analysis',dict(status='complete_light_checked',terminal=ev(CYCLE/'scientific_factor_synthesis.json')))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
