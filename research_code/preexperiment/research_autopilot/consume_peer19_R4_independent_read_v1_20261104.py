"""Consume independent R4 and H1 AP; apply the frozen cal-only .3pp rule.

No new experiment, AP computation, fitting, selection, or cache duplication.
The production provisional read stays immutable. A separate accepted decision
and row-addressed CSV retain both roles, every arm and all head-level CIs.
"""
import argparse,csv,io,json,math,time,traceback
from peer19_v1_20261104 import *
NAME='R4_independent_read'
OUT=BASE/NAME
TABLE=ROOT/'docs/paper_draft_v0_20261103/final_tables_v3/R4_independent_read'

def prepare():
    register(NAME,__file__,__doc__,dict(GPU_forwards=0,updates=0,CPU_threads=1,
        wall_h='bounded36h dependencies; seconds after review',disk_MiB='<3 CSV and provenance only'),
        dependencies=['robust_external_independent','recipe_independent/author'],
        input_contracts={n:ev(BASE/n/'contract.json') for n in ('robust_external','robust_external_independent','recipe_independent_author')},
        rule='cal8 global robust arm only; val21 mean AP70 >= H1-.3pp weakens claim; no formal per-cell selection',
        deadline='2026-11-05T18:00:00+08:00',immutable_provisional_read=True)

def mean(values):
    assert values and all(math.isfinite(v) for v in values)
    return sum(values)/len(values)

def run():
    begin=time.monotonic();record(NAME,dict(status='waiting_R4_and_H1_independent_AP',resource_owner=False,owner=None))
    try:
        for root in ('robust_external_independent','recipe_independent/author'):
            wait(BASE/root/'terminal.json',36,accepted=('complete_independently_reviewed',))
        record(NAME,dict(status='running_preregistered_independent_consumption',resource_owner=True))
        cal=load(BASE/'robust_external/calibration.json');kind=cal['kind'];choices=cal['choices']
        assert cal['selection_roles']==['calibration8'] and cal['no_formal_selection']
        selected=max(choices,key=lambda k:choices[k]['AP70']);assert selected==cal['selected_global']
        registry=load(BASE/'recipe_v3_interface_recovery_v2/method_registry.json')
        assert kind==registry['selected_kind'] and not registry['N2_in_main_method']
        identities=load(BASE/'H1_paper_supplement/method_identity.json')['domains']['author']
        assert cal['q']==identities['q'] and cal['tau']==identities['tau']
        for seed,ref in cal['heads'].items():assert ref==identities['heads'][seed]['weights']==ev(ROOT/ref['path'])
        ext=load(BASE/'robust_external_independent/AP.json')
        h1=[r for r in load(BASE/'recipe_independent/author/AP.json') if r.get('phase')=='primary' and r.get('kind')==kind]
        expected={(role,module,filt,seed,w) for role in ('val21','mechanism_holdout14')
            for module in ('freealign','roco','cbm') for filt in ('instant','median','gated_KF')
            for seed in (2023,2024,2025) for w in (0,100,200,400)}
        assert len(ext)==len(expected) and {(r['role'],r['module'],r['filter'],r['seed'],r['wait_ms']) for r in ext}==expected
        assert len(h1)==24
        rows=[]
        for role in ('val21','mechanism_holdout14'):
            for threshold in (50,70):
                metric='AP'+str(threshold);hr=[r for r in h1 if r['role']==role];assert len(hr)==12
                hv=mean([r[metric] for r in hr])
                for arm in sorted({r['module']+'_'+r['filter'] for r in ext}):
                    er=[r for r in ext if r['role']==role and r['module']+'_'+r['filter']==arm];assert len(er)==12
                    value=mean([r[metric] for r in er])
                    rows.append(dict(role=role,threshold=threshold,arm=arm,cal_selected=arm==selected,
                        H1_AP=hv,external_AP=value,external_minus_H1=value-hv))
        main=next(r for r in rows if r['role']=='val21' and r['threshold']==70 and r['cal_selected'])
        provisional=load(BASE/'robust_external/preregistered_read.json')
        assert abs(main['external_minus_H1']-provisional['external_minus_H1'])<1e-8
        weaken=main['external_minus_H1']>=-.3
        assert weaken==provisional['weaken_claim']
        refs={n:ev(BASE/n) for n in ('robust_external/calibration.json','robust_external/preregistered_read.json',
            'robust_external/identity.json','robust_external_independent/AP.json','robust_external_independent/bootstrap.json',
            'robust_external_independent/terminal.json','recipe_independent/author/AP.json','recipe_independent/author/terminal.json',
            'H1_paper_supplement/method_identity.json')}
        TABLE.mkdir(parents=True,exist_ok=False)
        for filename in ('AP.json','bootstrap.json'):
            source=BASE/'robust_external_independent'/filename;values=load(source)
            stream=io.StringIO();writer=csv.DictWriter(stream,fieldnames=['source_row','values']);writer.writeheader()
            writer.writerows(dict(source_row=i,values=json.dumps(r,ensure_ascii=False)) for i,r in enumerate(values))
            target=TABLE/(filename[:-5]+'.csv');target.write_text(stream.getvalue())
            decoded=list(csv.DictReader(target.open()))
            assert len(decoded)==len(values) and all(int(r['source_row'])==i and json.loads(r['values'])==values[i] for i,r in enumerate(decoded))
        atomic(OUT/'all_arm_summary.json',rows)
        decision=dict(created_at=now(),status='accepted_preregistered_read',independent_pending=False,
            cal_selected_external=selected,selection_role='calibration8',formal_role='val21',
            H1_AP70=main['H1_AP'],external_AP70=main['external_AP'],external_minus_H1=main['external_minus_H1'],
            threshold_pp=-.3,weaken_claim=weaken,
            interpretation='Common-mode plus cross-message structure; observation source can change.' if weaken else
                'These existing registration modules plus the tested cal-selected robust temporal processing do not catch H1 in this setting; not a universal impossibility claim.',
            same_H1_weights_prefix=True,all_formal_arms_retained=True,no_formal_selection=True,
            independent_scope='Upstream independent polygon matching/AP and established independent sequence2000CI. Shared detections, scores and module fitting; no complete external system attribution.',
            sources=refs,root_table_ledger_unchanged=True)
        atomic(OUT/'accepted_decision.json',decision)
        atomic(TABLE/'manifest.json',dict(created_at=now(),decision=ev(OUT/'accepted_decision.json'),
            all_arms=ev(OUT/'all_arm_summary.json'),sources=refs,CSV_roundtrip_exact=True,no_new_AP_or_bootstrap=True))
        atomic(OUT/'terminal.json',dict(created_at=now(),status='complete_light_checked',decision=ev(OUT/'accepted_decision.json'),
            attachment=ev(TABLE/'manifest.json'),wall_s=time.monotonic()-begin))
        record(NAME,dict(status='complete_light_checked',terminal=ev(OUT/'terminal.json'),resource_owner=False,owner=None))
    except Exception as e:
        atomic(OUT/'terminal.json',dict(created_at=now(),status='failed_retained',error=repr(e),traceback=traceback.format_exc()))
        record(NAME,dict(status='failed_retained',terminal=ev(OUT/'terminal.json'),resource_owner=False,owner=None));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run']);globals()[p.parse_args().mode]()
