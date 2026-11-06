"""R1 input-coverage supplement: actual H1 strict60 calibration at all4 waits.

The frozen peer19 loader regressed to legacy cal0/400, although peer17 already
materialized/reviewed cal4. Reuse that exact cache; no new forward, learning,
association, or val/hold selection. Existing recipes remain immutable. Verify
q/tau/global head choice; a changed choice explicitly requires a new scientific
recipe and its downstream consumers, never a relabeling of current results.
"""
import argparse,time,traceback
import numpy as np
from peer19_v1_20261104 import *
import run_peer19_H1_assets_v1_20261104 as h1
import review_peer19_recipe_v3_20261104 as independent
from review_peer16_E4_E7_AP_v1_20261103 import check,geometry
NAME='strict_cal4_H1'
OUT=BASE/NAME
REVIEW=BASE/'strict_cal4_H1_independent'
INPUT=P17/'strict_cal4/calibration8_four_wait_actual_material.npz'

def prepare():
    for name in (NAME,'strict_cal4_H1_independent'):
        register(name,__file__,__doc__,dict(GPU_forwards=0,updates=0,CPU_threads=1,
            disk_MiB='<35 compact cal predictions/metrics; shared22MB actual cache referenced',wall_h='bounded8h'),
            cal4_input=ev(INPUT),cal4_review=ev(P17/'strict_cal4_independent/terminal.json'),
            current_recipe=ev(h1.REC/'strict60/calibration.json'),registry=ev(h1.REC/'method_registry.json'),
            expected_outcome='Missing100/200 may or may not change cal q/tau/global frozen-vs-refit choice. Never use formal roles to decide.',
            original_handoff_reread=True,deadline='2026-11-05T12:00:00+08:00')

def materials():
    assert load(P17/'strict_cal4_independent/terminal.json')['status']=='complete_independently_reviewed'
    assert ev(INPUT)==load(P17/'strict_cal4/input_identity.json')['cal4']
    with np.load(INPUT) as z:m={k:z[k] for k in z.files}
    assert set(map(int,m['wait_ms']))=={0,100,200,400}
    return m

def selection(rows,metric,field,grid):
    values={v:float(np.mean([r[metric] for r in rows if r[field]==v])) for v in grid}
    return max(grid,key=lambda v:(values[v],-v)),values

def global_choice(strict_means):
    others={k:[] for k in ('frozen','refit')}
    for d in h1.recipe.DOMAINS:
        if d=='strict60':continue
        c=load(h1.REC/d/'calibration.json')
        for k in others:others[k].append(c['choices'][k]['mean_AP70'])
    means={k:float(np.mean(others[k]+[strict_means[k]])) for k in others}
    return ('refit' if means['refit']>means['frozen'] else 'frozen'),means

def run():
    start=time.monotonic();record(NAME,dict(status='running_actual_H1_cal4_coverage',resource_owner=True))
    try:
        h1.context();m=materials();old=load(h1.REC/'strict60/calibration.json');oldtau=load(h1.recipe.prior.OUT/'strict60/calibration.json')['selected_tau'];rows=[]
        for q in h1.recipe.Q:
            d=h1.recipe.hard_state(m,q)['delta'];b,f=h1.recipe.geometry(m,d)
            for seed in h1.recipe.SEEDS:
                label='q'+str(q)+'_seed'+str(seed)
                pp,p=h1.recipe.evaluate(m,b,f,h1.recipe.head('strict60','frozen',seed),label,oldtau)
                rows.extend(dict(r,q=q,seed=seed,phase='q') for r in pp);np.savez_compressed(OUT/(label+'.npz'),**p)
            atomic(OUT/'q_panels.json',rows)
        q,grid=selection(rows,'ap70','q',h1.recipe.Q)
        atomic(OUT/'q_selection.json',dict(selected_q=q,grid=grid,old_q=old['q'],selection_roles=['calibration8']))
        taurows=[];chosen={};strict_means={}
        # Frozen/refit comparison is valid only at the already trained q.
        if q==old['q']:
            d=h1.recipe.hard_state(m,q)['delta'];np.savez_compressed(OUT/'selected_state.npz',delta=d)
            b,f=h1.recipe.geometry(m,d)
            for kind in ('frozen','refit'):
                for seed in h1.recipe.SEEDS:
                    preds={}
                    for tau in h1.recipe.prior.GRID:
                        label=kind+'_tau'+str(round(tau*100))+'_seed'+str(seed)
                        pp,p=h1.recipe.evaluate(m,b,f,h1.recipe.head('strict60',kind,seed),label,tau)
                        taurows.extend(dict(r,kind=kind,tau=tau,seed=seed,phase='tau') for r in pp);preds.update(p)
                    np.savez_compressed(OUT/(kind+'_seed'+str(seed)+'.npz'),**preds);atomic(OUT/'tau_panels.json',taurows)
                chosen[kind],g=selection([r for r in taurows if r['kind']==kind],'ap70','tau',h1.recipe.prior.GRID)
                strict_means[kind]=g[chosen[kind]]
            kind,means=global_choice(strict_means)
            same=all(chosen[k]==old['choices'][k]['tau'] for k in chosen) and kind==load(h1.REC/'method_registry.json')['selected_kind']
        else:kind=None;means={};same=False
        result=dict(created_at=now(),selected_q=q,old_q=old['q'],q_unchanged=q==old['q'],tau=chosen,
            global_kind=kind,global_cal8_means=means,strict_selected_mean_AP70=strict_means,
            recipe_identity_unchanged=bool(same),requires_new_recipe=not same,
            interpretation='Same actual q/tau/heads; formal predictions remain valid without rerunning.' if same else
                'Calibration-only choice changed or needs new geometry-refit heads. Current formal arrays remain old identity; register a corrected recipe and all affected consumers before final-table acceptance.',
            official_main_unchanged=True,formal_roles_read_for_selection=False,independent_pending=True)
        atomic(OUT/'decision.json',result)
        atomic(OUT/'terminal.json',dict(created_at=now(),status='complete_light_checked',decision=ev(OUT/'decision.json'),wall_s=time.monotonic()-start))
        record(NAME,dict(status='complete_light_checked',terminal=ev(OUT/'terminal.json'),resource_owner=False,owner=None))
    except Exception as e:fail(NAME,OUT,e);raise

def review():
    name='strict_cal4_H1_independent';record(name,dict(status='waiting_H1_actual_cal4',resource_owner=False,owner=None));start=time.monotonic()
    try:
        wait(OUT/'terminal.json',8);record(name,dict(status='running_independent_H1_cal4_AP',resource_owner=True));m=materials();rows=[]
        gold=load(OUT/'q_panels.json')
        for q in h1.recipe.Q:
            d,n,z=independent.primary_state(m,q);b=geometry(m,d)
            for seed in h1.recipe.SEEDS:
                label='q'+str(q)+'_seed'+str(seed)
                with np.load(OUT/(label+'.npz')) as p:rr=check(m,b,p,label,gold)
                assert all(abs(r['error'+str(t)])<1e-8 and r['TP_mismatch'+str(t)]==0 for r in rr for t in (50,70))
                rows.extend(dict(r,q=q,seed=seed,phase='q') for r in rr);atomic(REVIEW/'AP.json',rows)
        q,grid=selection(rows,'AP70','q',h1.recipe.Q);decision=load(OUT/'decision.json');assert q==decision['selected_q']
        chosen={};means={}
        if decision['q_unchanged']:
            d=independent.primary_state(m,q)[0];b=geometry(m,d);gold=load(OUT/'tau_panels.json')
            for kind in ('frozen','refit'):
                for seed in h1.recipe.SEEDS:
                    with np.load(OUT/(kind+'_seed'+str(seed)+'.npz')) as p:
                        for tau in h1.recipe.prior.GRID:
                            label=kind+'_tau'+str(round(tau*100))+'_seed'+str(seed);rr=check(m,b,p,label,gold)
                            assert all(abs(r['error'+str(t)])<1e-8 and r['TP_mismatch'+str(t)]==0 for r in rr for t in (50,70))
                            rows.extend(dict(r,kind=kind,tau=tau,seed=seed,phase='tau') for r in rr);atomic(REVIEW/'AP.json',rows)
                chosen[kind],g=selection([r for r in rows if r.get('kind')==kind],'AP70','tau',h1.recipe.prior.GRID);means[kind]=g[chosen[kind]]
            kind,globalmeans=global_choice(means)
            assert chosen==decision['tau'] and kind==decision['global_kind']
            assert all(abs(globalmeans[k]-decision['global_cal8_means'][k])<1e-8 for k in globalmeans)
        atomic(REVIEW/'decision.json',dict(decision,created_at=now(),independent_pending=False,AP=ev(REVIEW/'AP.json'),
            producer=ev(OUT/'decision.json'),scope='Independent all-ego KF and polygon matching/AP; same trained heads and candidate scores; full4wait cal ranking rechecked, no formal re-selection.'))
        atomic(REVIEW/'terminal.json',dict(created_at=now(),status='complete_independently_reviewed',decision=ev(REVIEW/'decision.json'),
            recipe_identity_unchanged=decision['recipe_identity_unchanged'],requires_new_recipe=decision['requires_new_recipe'],wall_s=time.monotonic()-start))
        record(name,dict(status='complete_independently_reviewed',terminal=ev(REVIEW/'terminal.json'),resource_owner=False,owner=None))
    except Exception as e:fail(name,REVIEW,e);raise

def fail(name,dest,e):
    atomic(dest/'terminal.json',dict(created_at=now(),status='failed_retained',error=repr(e),traceback=traceback.format_exc()))
    record(name,dict(status='failed_retained',terminal=ev(dest/'terminal.json'),resource_owner=False,owner=None))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run','review']);globals()[p.parse_args().mode]()
