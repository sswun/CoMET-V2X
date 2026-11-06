"""Scoped completed native physical-reference and dose tables, zero forwards."""
import csv,time,json
from collections import defaultdict
from peer14_cycle12_v1_20261103 import ROOT,AUDIT,CYCLE,C09,C06,C00,C21,atomic,ev,load,now,register,record

def table(path,rows):
    assert rows
    with path.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)

def means(rows):
    groups=defaultdict(list)
    for r in rows:
        if r['source_group'] not in ('all','veh_coop','inf'):continue
        key=tuple(r[k] for k in ('family','producer','protocol','roi','arm','source_group','threshold'))
        groups[key].append(r)
    result=[]
    for key,rr in groups.items():
        waits=sorted(r['wait_ms'] for r in rr);assert len(waits)==len(set(waits))
        result.append(dict(zip(('family','producer','protocol','roi','arm','source_group','threshold'),key),AP=sum(r['ap'] for r in rr)/len(rr),n_waits=len(rr),waits=json.dumps(waits),four_wait_complete=waits==[0,100,200,400]))
    return result

def collect(targets,root):
    rows=[];missing=[];identities=[]
    for i,t in enumerate(targets):
        p=root/f'{i:03d}'
        if not (p/'terminal.json').exists():missing.append(t['path']);continue
        assert load(p/'terminal.json')['status']=='complete_light_checked'
        identities.append(ev(p/'terminal.json'))
        for r in load(p/'panels.json'):
            rr=dict(r,family=t.get('family','CoAlign_complete_graph'),producer=t['path'])
            if rr['threshold']>1:rr['threshold']/=100
            rows.append(rr)
    return means(rows),missing,identities

def run():
    register('native_physical_tables',__file__,'Summarize completed A39 native producer identities and B native18+Where/ViT21, all sources/ROIs, without mixing quality populations or selecting favorable protocols. Accept B8400 only from independent_terminal. Partial native dose coverage is explicit; label-dose peaks are diagnostic, never input/method selection.',dict(GPU_forwards=0,CPU_threads=1,new_disk_MiB='<8 tables and summaries'))
    c=load(C00/'native_physical_dose_contract.json');a,amissing,aid=collect(c['SPD_registration_targets'],C00/'native_physical_dose_v1/SPD_native_registration');assert not amissing
    table(CYCLE/'SPD_native_registration_39_completed_source_means.csv',a)
    b,missing,bid=collect(load(C21/'native_dose_response_contract.json')['targets'],C21/'native_dose_response_v1/producer')
    bb,bmissing,bbid=collect(c['new_dose_targets'],C00/'native_physical_dose_v1/backbone_dose');assert not bmissing
    allb=b+bb;table(CYCLE/'SPD_native_all_backbone_dose_completed_means.csv',allb)
    peaks=[];group=defaultdict(dict)
    for r in allb:
        if r['protocol'] not in ('alpha000','alpha025','alpha050','alpha075','official'):continue
        key=tuple(r[k] for k in ('family','producer','roi','arm','source_group','threshold'));group[key][r['protocol']]=r
    protocols=['alpha000','alpha025','alpha050','alpha075','official']
    for key,values in group.items():
        if len(values)!=5:continue
        v=[values[p]['AP'] for p in protocols];maximum=max(v)
        peaks.append(dict(zip(('family','producer','roi','arm','source_group','threshold'),key),AP_alpha0=v[0],AP_alpha1=v[-1],alpha1_minus_alpha0_pp=v[-1]-v[0],peak_alphas=json.dumps([i*.25 for i,x in enumerate(v) if x==maximum]),peak_AP=maximum,curve_AP=json.dumps(v),all_four_waits=all(r['four_wait_complete'] for r in values.values()),scope='fixed_prediction_label_intervention_only_not_method_selection'))
    table(CYCLE/'SPD_native_dose_endpoint_peak_diagnostics.csv',peaks)
    # Compare arms only within the SAME producer/ROI/quality mask, never AP across backbones.
    contrasts=[];index=defaultdict(dict)
    for r in a:index[tuple(r[k] for k in ('family','producer','protocol','roi','source_group','threshold'))][r['arm']]=r
    for key,values in index.items():
        for x,y in [('ONLINE_K1','PUBLIC'),('K1_INPUT','NO_GRAPH'),('COALIGN_FULL','NO_GRAPH')]:
            if x in values and y in values:
                assert values[x]['waits']==values[y]['waits']
                contrasts.append(dict(zip(('family','producer','protocol','roi','source_group','threshold'),key),contrast=x+' minus '+y,delta_pp=values[x]['AP']-values[y]['AP'],n_waits=values[x]['n_waits'],scope='same_producer_quality_selected_population_not_full_main_AP'))
    table(CYCLE/'SPD_native_registration_within_producer_contrasts.csv',contrasts)
    bterm=C06/'SPD_dose_independent_v3/producer/independent_terminal.json';bt=load(bterm);assert bt['status']=='complete_independently_reviewed' and bt['panels']==8400 and all(bt['checks'].values())
    checks=load(C06/'SPD_dose_independent_v3/producer/independent_panel_checks.json');assert len(checks)==8400
    maxdelta=max(abs(r[k]) for r in checks for k in ('AP50_delta','AP70_delta'))
    atomic(CYCLE/'native_physical_synthesis.json',dict(created_at=now(),status='complete_light_checked',A_native_producers=len(aid),A_source_mean_rows=len(a),A_input_identities=aid,A_scope='Quality-conditioned official vs registration labels, only inf geometry changed, not full AP or physical truth; own full independent review not implied',B_native18_completed=len(bid),B_backbone_completed=len(bbid),B_native18_missing=missing,B_completed_mean_rows=len(allb),B_scope='All7 annotation-assisted label interventions, own native/common ROI and fixed prediction population; backbone/native complete only on individual terminals; no cross-ROI absolute ranking',B8400_independent=ev(bterm),B8400_max_AP_delta_pp=maxdelta,B8400_original_failed_versions_retained=True,B8400_scope='Independent labels/corners/IoU/ignore/VOC AP; existing saved predictions/layout shared, not detector/estimator retraining review',native_physical_contrasts=contrasts,release_closed=False,GPU_forwards=0))
    record('native_physical_tables',dict(status='complete_light_checked',terminal=ev(CYCLE/'native_physical_synthesis.json')))
    print(json.dumps(dict(A_producers=len(aid),B_native_completed=len(bid),B_backbone_completed=len(bbid),B_missing=missing,B8400_exact=maxdelta==0)))

if __name__=='__main__':run()
