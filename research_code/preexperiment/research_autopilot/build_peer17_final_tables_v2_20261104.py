"""Source-addressed final tables. Old controls are explicitly versioned.

Every numeric CSV cell has immutable source operands and review scope. Table
arithmetic is independently checked by the separate reviewer, not represented
as another independent experiment or statistical implementation.
"""
import argparse
import csv
import io
import os
import time
import numpy as np
from pathlib import Path
from peer17_v1_20261104 import ROOT,AUTO,AUDIT,BASE,atomic,ev,load,now,register,record
from peer16_v1_20261103 import BASE as P16
from peer15_v1_20261103 import BASE as P15
import run_peer17_final_recipe_v1_20261104 as recipe
OUT=ROOT/'docs/paper_draft_v0_20261103/final_tables_v2'
JOB=BASE/'final_tables_v2'

def pointer(x,keys):
    for k in keys:x=x[k]
    return x

def operand(path,keys=(),fmt='json'):
    return dict(source=ev(path),pointer=list(keys),format=fmt)

def value(o):
    path=ROOT/o['source']['path']
    x=load(path) if o['format']=='json' else list(csv.DictReader(path.open()))
    return float(pointer(x,o['pointer']))

def prepare():
    register('final_tables_v2',__file__,
        'Generate peer17 final tables from exact addressed sources after '
        'new final recipe scientific review. Retain negative controls, source '
        'units and old-recipe labels; no inferred missing cells. Independent '
        'numeric serialization/source/arithmetic review is separate from science.',
        dict(GPU_forwards=0,updates=0,CPU_threads=1,wall_h='finite26h dependency',disk_MiB='<20'),
        deadline='2026-11-05T12:00:00+08:00',output=str(OUT.relative_to(ROOT)),
        generator=ev(__file__),final_recipe=ev(recipe.OUT/'contract.json'),
        FFNet='Do not include in Table1: original3237 SPD check is not DAIR-C paper reproduction; exact paper-quantity selfcheck prerequisite not established. Retain full-common-GT adaptation in auxiliary table.')

def build():
    OUT.mkdir(exist_ok=True);cells=[];pending=[]
    def add(table,row,metric,ops,review,review_ref,scope,op='identity',factor=1.):
        vals=[value(o) for o in ops]
        result={'identity':lambda:vals[0],'mean':lambda:float(np.mean(vals)),
                'sample_SD':lambda:float(np.std(vals,ddof=1)),
                'max':lambda:max(vals),'difference':lambda:vals[0]-vals[1]}[op]()*factor
        cells.append(dict(table=table,row=row,metric=metric,value=result,operation=op,factor=factor,
            operands=ops,review_status=review,review_evidence=ev(review_ref) if review_ref and review_ref.exists() else None,
            scope=scope))
    def panels(table,path,review_ref,scope,fields=('ap50','ap70'),label=None,filter_row=None,status='independent_point_AP'):
        data=load(path)
        for i,r in enumerate(data):
            if filter_row and not filter_row(r):continue
            row=label(r,i) if label else '|'.join(str(r.get(k,'')) for k in ('role','domain','method','wait_ms','seed','source','threshold'))
            for f in fields:
                if f in r:add(table,row,f,[operand(path,(i,f))],status,review_ref,scope)
    p=P16/'E5_native_seed_AP50/panels.json';d=load(p)['per_seed']
    for w in (0,100,200,400):
        ids=[i for i,r in enumerate(d) if r['method']=='native' and r['wait_ms']==w]
        assert len(ids)==3
        for th in (50,70):
            ops=[operand(p,('per_seed',i,'AP'+str(th))) for i in ids]
            for op in ('mean','sample_SD'):
                add('table1','V2X-ViT_three_detector_seeds|'+str(w),'AP'+str(th)+'_'+op,ops,
                    'accepted_per_frame_ledgers;seed2023_raw_independent',P16/'E5_native_seed_AP50/terminal.json',
                    'uniform60 native strict/current commonGT; SD over detector seeds, not head seeds',op)
    for model in ('fcooper','traf'):
        p=P15/('external_raw_'+model)/'terminal.json';d=load(p)
        for w in (0,100,200,400):
            for th in (50,70):add('table1',model+'_native_author_weight|'+str(w),'AP'+str(th),
                [operand(p,('results',str(w),'metrics',str(th/100),'ap'))],
                'independent_raw_polygon_matching_AP',p,'native strict/current commonGT; single detector weight',factor=100.)
    # Exact released official late baselines, never compare a weaker column by accident.
    late=AUDIT/'cycle12_peer14_v1_20261103/paper_SPD_late_all_method_main_and_sources.csv'
    entries=list(csv.DictReader(late.open()))
    for w in (0,100,200,400):
        ids=[i for i,r in enumerate(entries) if r['role']=='val21' and r['protocol']=='official' and r['source_group']=='all' and r['method'] in ('official_late_no_comp','official_late_time_comp') and int(r['wait_ms'])==w]
        assert len(ids)==2,(w,ids)
        for th in (50,70):add('table1','stronger_late_per_column|'+str(w),'AP'+str(th),
            [operand(late,(i,'ap'+str(th)),'csv') for i in ids],
            'released_prior_independent_table',None,'columnwise max of the two official late baselines, no evaluation-tuned CoMET selection','max')
    for domain in recipe.DOMAINS:
        p=recipe.OUT/domain/'panels.json';rev=BASE/'final_recipe_independent'/domain/'terminal.json'
        if not p.exists() or not rev.exists() or load(rev)['status']!='complete_independently_reviewed':
            pending.append('new_final_recipe:'+domain);continue
        pp=load(p)
        panels('cross_detector_same_recipe',p,rev,'peer17 no-position + cal coarse tau; own heading/current-q inputs; head seeds not detector seeds')
        for role in ('val21','mechanism_holdout14'):
            for w in (0,100,200,400):
                ids=[i for i,r in enumerate(pp) if r['role']==role and r['wait_ms']==w];assert len(ids)==3
                for th in (50,70):
                    for op in ('mean','sample_SD'):
                        add('table1' if domain=='author' and role=='val21' else 'cross_detector_same_recipe',
                            'CoMET_'+domain+'|'+role+'|'+str(w),'AP'+str(th)+'_'+op,
                            [operand(p,(i,'ap'+str(th))) for i in ids],'independent_point_AP',rev,
                            'peer17 final recipe; SD over three fixed head seeds only',op)
        cp=recipe.OUT/domain/'calibration_panels.json'
        panels('readout_tau_cal_grid',cp,rev,'cal-only0.30:0.05:0.70; no val selection',label=lambda r,i:domain+'|'+str(r['tau'])+'|'+str(r['seed'])+'|'+str(r['wait_ms']))
        bp=BASE/'final_recipe_independent'/domain/'bootstrap.json'
        if bp.exists():panels('final_recipe_oldtau_CI',bp,rev,'2000 whole-sequence bootstrap; shared accepted statistic implementation, not independently reimplemented',
            fields=('delta_pp','CI95_low','CI95_high'),label=lambda r,i:domain+'|'+r['comparison']+'|'+str(r['threshold'])+'|'+str(r['wait_ms']),status='independent_point_AP;shared_bootstrap_kernel')
    for role in ('val21','mechanism_holdout14'):
        panels('temporal_aggregation',P16/'E3_temporal_baselines'/(role+'_panels.json'),P16/'E2_E3_independent_state_geometry_AP/terminal.json',
            'E3 frozen author full39 old-tau control, not relabeled peer17 final recipe',label=lambda r,i:role+'|'+r['method']+'|'+str(r['wait_ms']))
        panels('external_temporal_aggregation',P16/'E3_external_temporal_v3'/(role+'_panels.json'),P16/'E3_external_temporal_independent/terminal.json',
            'E3 original strict60 modules+shared original head; NOT full external systems or current author head',label=lambda r,i:role+'|'+r['method']+'|'+str(r['wait_ms']))
        panels('external_temporal_CI',P16/'E3_external_temporal_v3'/(role+'_bootstrap.json'),P16/'E3_external_temporal_independent/terminal.json',
            'paired2000 sequence bootstrap; original module recipe; shared statistic kernel',fields=('delta_pp','CI95_low','CI95_high'),
            label=lambda r,i:r['comparison']+'|'+str(r['threshold'])+'|'+str(r['wait_ms']),status='independent_point_AP;shared_bootstrap_kernel')
    bp=P16/'E2_E3_sequence_bootstrap/E3_author/panels.json'
    panels('temporal_CI',bp,P16/'E2_E3_independent_state_geometry_AP/terminal.json',
        'E3 frozen author old recipe; paired2000 sequence bootstrap, shared statistic implementation',
        fields=('delta_pp','CI95_low','CI95_high'),
        label=lambda r,i:r['comparison']+'|'+str(r['threshold'])+'|'+str(r['wait_ms']),status='independent_point_AP;shared_bootstrap_kernel')
    panels('readout_held_site',P16/'E7_location_disjoint/panels.json',P16/'E7_location_folds_independent/terminal.json',
        'six supplementary leave-one-site-out folds, no change to official split; old tau',label=lambda r,i:r['held_site']+'|'+r['kind']+'|'+r['method']+'|'+str(r['wait_ms']))
    panels('readout_held_site_CI',P16/'E7_location_folds_independent/bootstrap_panels.json',P16/'E7_location_folds_independent/terminal.json',
        'few-sequence intervals may degenerate; shared bootstrap kernel',fields=('delta_pp','CI95_low','CI95_high'),
        label=lambda r,i:str(i)+'|'+r['comparison']+'|'+str(r['wait_ms']),status='independent_point_AP;shared_bootstrap_kernel')
    p=recipe.OUT/'feature_decision.json'
    if p.exists():
        for i,r in enumerate(load(p)['domains']):add('readout_no_position',r['domain'],'no_position_minus_full39_AP70',
            [operand(p,('domains',i,'no_position_minus_full39_pp'))],'independent_point_AP;gate_arithmetic',P16/'E7_no_position_crossdomain_independent/terminal.json',
            'three-head-seed/four-wait paired oldtau result; global five-domain rule, not per-domain choice')
    panels('OPV_pose_time',P16/'E9_opv_pose_time/panels.json',P16/'E9_opv_pose_time_independent/terminal.json',
        'full2170 receiver/5985 sensor test, P1 controlled physical-time noise; not SPD final readout recipe',fields=('AP50','AP70'),
        label=lambda r,i:r['noise']+'|'+r['method']+'|'+str(r['wait_ms']))
    p=P16/'E1_physical_velocity_independent/panels.json'
    for i,r in enumerate(load(p)):
        for axis in ('x_speed','yaw_rate'):
            for f,ptr in [('r',(i,axis,'r')),('CI95_low',(i,axis,'CI95',0)),('CI95_high',(i,axis,'CI95',1))]:
                add('E1_speed_coupling',r['role']+'|'+r['quality']+'|'+axis,f,[operand(p,ptr)],
                    'independent_direct_sequence_resampling_and_pearson',P16/'E1_physical_velocity_independent/terminal.json',
                    'raw unlabelled ICP; not physical GT or unique constant-clock cause')
    panels('E8_source_contribution',P16/'E8_source_AP_units_v2/source_AP.json',P16/'E8_source_AP_units_v2/terminal.json',
        'corrected percent units; old recipe; other-source ignore AP is NOT additive decomposition',fields=('AP','GT','TP','FP'))
    panels('FFNet_adaptation_auxiliary',P16/'E11_full_common_GT/panels.json',P16/'E11_full_common_GT/terminal.json',
        'full31123GT, BEV, saved strict adaptation, not DAIR-C paper reproduction; excluded Table1',
        fields=('BEV_AP',),label=lambda r,i:str(i)+'|'+str(r.get('convention',''))+'|'+str(r.get('wait_ms',''))+'|'+str(r['threshold']))
    refs=[dict(name='CoAlign_unchanged',source=ev(AUDIT/'cycle12_peer14_v1_20261103/CoAlign_completed_composed_graph_all_wait_factors.csv'))] if (AUDIT/'cycle12_peer14_v1_20261103/CoAlign_completed_composed_graph_all_wait_factors.csv').exists() else []
    if not refs:pending.append('CoAlign unchanged table exact reference resolve')
    if pending:status='partial_explicit_pending'
    else:status='tables_generated_pending_independent_numeric_audit'
    for table in sorted(set(c['table'] for c in cells)):
        stream=io.StringIO();writer=csv.DictWriter(stream,fieldnames=['row','metric','value','review_status','scope','source_operands'])
        writer.writeheader()
        for c in cells:
            if c['table']==table:
                import json
                writer.writerow({k:c[k] for k in ('row','metric','value','review_status','scope')}|dict(source_operands=json.dumps(c['operands'],ensure_ascii=False)))
        target=OUT/(table+'.csv');temporary=target.with_name(target.name+'.tmp.'+str(os.getpid()))
        temporary.write_text(stream.getvalue());temporary.replace(target)
    atomic(OUT/'numeric_ledger.json',dict(created_at=now(),cells=cells))
    atomic(OUT/'manifest.json',dict(created_at=now(),status=status,pending=pending,
        generator=ev(__file__),numeric_ledger=ev(OUT/'numeric_ledger.json'),references=refs,
        no_automatic_main_tex_edits=True,FFNet_table1_included=False,
        remaining='Control E3/E8 tables keep original readout identities; no claim that every old diagnostic has been recomputed with peer17 final recipe. Independent statistics not globally complete.'))
    return dict(status=status,cells=len(cells),pending=pending)

def wait():
    start=time.monotonic();record('final_tables_v2',dict(status='waiting_final_recipe_scientific_review'))
    review=BASE/'final_recipe_independent/terminal.json'
    while not review.exists():
        assert time.monotonic()-start<26*3600
        time.sleep(30)
    result=build()
    import review_peer17_final_tables_v2_20261104 as reviewer
    audit=reviewer.run()
    status='complete_light_checked' if audit['status']=='passed' and not result['pending'] else 'partial_explicit_pending'
    atomic(JOB/'terminal.json',dict(created_at=now(),status=status,result=result,
        numeric_audit=ev(OUT/'independent_numeric_audit.json'),GPU_forwards=0,updates=0))
    record('final_tables_v2',dict(status=status,terminal=ev(JOB/'terminal.json'),owner=None))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','build','wait']);a=p.parse_args();print(globals()[a.mode]())
