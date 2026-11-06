"""Five-domain no-position gate; cal-only coarse protected-prefix threshold.

No detector forwards, head fitting, association or geometry changes. Store only
compact candidate IDs/scores/outcomes, referencing existing heads and inputs.
"""
import argparse
import time
import traceback
import numpy as np
from peer17_v1_20261104 import ROOT,BASE,atomic,ev,load,now,register,record
from peer16_v1_20261103 import BASE as P16
from run_peer16_cached_methods_v1_20261103 import materials,DOMAINS
import run_peer16_E7_no_position_crossdomain_v1_20261103 as old
from run_peer16_E7_readout_v1_20261103 import REMOVED
OUT=BASE/'final_recipe'
SEEDS=(2023,2024,2025)
GRID=tuple(round(x/100,2) for x in range(30,71,5))

def head(domain,kind,seed):
    root=P16/'E7_readout_v3' if domain=='author' else P16/'E7_no_position_crossdomain_v2'/domain
    return root/(kind+'_seed'+str(seed))

def decision():
    assert load(P16/'E4_E7_independent_AP/terminal.json')['status']=='complete_independently_reviewed'
    assert load(P16/'E7_no_position_crossdomain_independent/terminal.json')['status']=='complete_independently_reviewed'
    cross=load(P16/'E7_no_position_crossdomain_v2/panels.json');rows=[]
    for domain in DOMAINS:
        values={}
        for kind in ('full39','no_position'):
            a=[]
            for seed in SEEDS:
                pp=load(head(domain,kind,seed)/'panels.json') if domain=='author' else [x for x in cross if x['domain']==domain and x['kind']==kind and x['head_seed']==seed]
                a.append(float(np.mean([r['ap70'] for r in pp if r.get('role','val21')=='val21'])))
            values[kind]=a
        rows.append(dict(domain=domain,three_seed_mean_AP70=values,
            no_position_minus_full39_pp=float(np.mean(values['no_position'])-np.mean(values['full39']))))
    delta=float(np.mean([r['no_position_minus_full39_pp'] for r in rows]))
    return dict(created_at=now(),domains=rows,five_domain_mean_difference_pp=delta,
        preregistered_rule='peer17 five-domain mean loss <0.3pp',adopt_no_position=delta>-.3,
        old_full39_results_retained=True,not_per_cell_selection=True)

def prepare():
    d=decision()
    register('final_recipe',__file__,
        'Execute peer17 five-domain mean feature gate; reuse all existing paired3seed heads. '
        'Select each detector protected-prefix threshold using calibration8 four-wait '
        'AP70 averaged over fixed3headseeds, 0.30:0.05:0.70 grid from E7. '
        'Freeze before val21/hold14; no per-cell/seed/wait tuning. Effects may be negative.',
        dict(GPU_forwards=0,updates=0,CPU_threads=1,wall_h='finite16h',disk_MiB='<180 compact predictions; no GT/box/head copy'),
        feature_gate=d,domains=list(DOMAINS),seeds=list(SEEDS),tau_grid=list(GRID),
        tie_break='lower tau',selection_population='calibration8 only; average four waits and three fixed heads',
        semantics='tau_p changes protected readout prefix only, following E7 threshold contract; '
          'CV anchor membership, heading/current-q geometry and trained head targets/weights remain frozen. '
          'Do not silently change estimator anchors or retrain targets after threshold choice.',
        readout_inputs=[dict(domain=d,kind=k,seed=s,weights=ev(head(d,k,s)/'direct_u.pth'),
            source_map=ev(head(d,k,s)/'source_maps.json')) for d in DOMAINS for k in ('full39','no_position') for s in SEEDS],
        independent_predecessors=[ev(P16/x/'terminal.json') for x in ('E4_E7_independent_AP','E7_no_position_crossdomain_independent')],
        recipe='current heading, author q=.15 other q=.05, r=.6, CV anchors unchanged; selected no-position readout; cal coarse protected prefix',
        storage='Only compact candidate ID/score/TP; shared geometry reconstructed from immutable inputs; active final table/reviewer consumers.')
    for domain in DOMAINS:
        record('final_recipe_'+domain,dict(status='queued_dependency',resource_owner=False,owner=None,
            contract=ev(OUT/'contract.json'),scope='Finite serial domain child of final_recipe; not a second worker.'))

def protected(m,tau):
    return dict(m,protected=((m['source']==0)&(m['raw_scores']>=tau)).astype(np.int8))

def run_domain(domain,kind):
    dest=OUT/domain;dest.mkdir(exist_ok=False);start=time.monotonic()
    record('final_recipe_'+domain,dict(status='running_calibration_grid',resource_owner=True))
    try:
        data,_,q=materials(domain);m,_=data['calibration8'];boxes,features=old.geometry(m,q)
        if kind=='no_position':features[:,REMOVED]=0
        panels=[]
        for seed in SEEDS:
            score=old.author.refit.score_with_head(m,features,head(domain,kind,seed));pred={}
            for tau in GRID:
                name='tau'+str(round(tau*100))+'_seed'+str(seed)
                pp,p=old.author.refit.fixed.base.evaluate(protected(m,tau),{name:boxes},{name:score})
                pred.update({k:v for k,v in p.items() if '__' in k})
                panels.extend(dict(r,domain=domain,role='calibration8',tau=tau,seed=seed,kind=kind) for r in pp)
                atomic(dest/'calibration_panels.json',panels)
                atomic(dest/'status.json',dict(created_at=now(),phase='calibration',seed=seed,tau=tau,wall_s=time.monotonic()-start))
            np.savez_compressed(dest/('calibration_seed'+str(seed)+'_predictions.npz'),**pred)
        candidates=[dict(tau=t,mean_AP70=float(np.mean([r['ap70'] for r in panels if r['tau']==t]))) for t in GRID]
        tau=max(candidates,key=lambda r:(r['mean_AP70'],-r['tau']))['tau']
        atomic(dest/'calibration.json',dict(created_at=now(),selected_tau=tau,grid=candidates,
            kind=kind,q=q,roles_used=['calibration8'],seeds=list(SEEDS),selection_before_formal_evaluation=True))
        rows=[]
        for role in ('val21','mechanism_holdout14'):
            m,_=data[role];boxes,features=old.geometry(m,q)
            if kind=='no_position':features[:,REMOVED]=0
            for seed in SEEDS:
                name='CoMET_seed'+str(seed);score=old.author.refit.score_with_head(m,features,head(domain,kind,seed))
                pp,p=old.author.refit.fixed.base.evaluate(protected(m,tau),{name:boxes},{name:score})
                np.savez_compressed(dest/(role+'_seed'+str(seed)+'_predictions.npz'),**p)
                rows.extend(dict(r,domain=domain,role=role,seed=seed,kind=kind,tau=tau,q=q) for r in pp)
                atomic(dest/'panels.json',rows)
                atomic(dest/'status.json',dict(created_at=now(),phase='frozen_formal',role=role,seed=seed,wall_s=time.monotonic()-start))
        identity=dict(domain=domain,selected_kind=kind,selected_tau=tau,q=q,r=.6,
            own_input=True,current_heading=True,association_unchanged=True,GT_not_used_for_inference=True,
            heads=[dict(seed=s,weights=ev(head(domain,kind,s)/'direct_u.pth'),maps=ev(head(domain,kind,s)/'source_maps.json')) for s in SEEDS],
            shared_inputs='run_peer16_cached_methods_v1_20261103.materials(domain); exact domain, no substituted legacy branch')
        atomic(dest/'identity.json',identity)
        atomic(dest/'terminal.json',dict(created_at=now(),status='complete_light_checked',GPU_forwards=0,updates=0,
            wall_s=time.monotonic()-start,identity=ev(dest/'identity.json'),calibration=ev(dest/'calibration.json'),
            panels=ev(dest/'panels.json'),independent_pending=True,
            new_bytes=sum(p.stat().st_size for p in dest.iterdir() if p.is_file())))
        record('final_recipe_'+domain,dict(status='complete_light_checked',terminal=ev(dest/'terminal.json'),owner=None))
    except Exception as exc:
        atomic(dest/'terminal.json',dict(created_at=now(),status='failed_retained',error=repr(exc),traceback=traceback.format_exc()))
        record('final_recipe_'+domain,dict(status='failed_retained',terminal=ev(dest/'terminal.json'),owner=None));raise

def run():
    start=time.monotonic();record('final_recipe',dict(status='running_final_recipe',resource_owner=False,owner=None))
    d=decision();atomic(OUT/'feature_decision.json',d);kind='no_position' if d['adopt_no_position'] else 'full39'
    try:
        for domain in DOMAINS:
            assert time.monotonic()-start<16*3600
            run_domain(domain,kind)
        atomic(OUT/'terminal.json',dict(created_at=now(),status='complete_light_checked',GPU_forwards=0,updates=0,
            domains={d:ev(OUT/d/'terminal.json') for d in DOMAINS},wall_s=time.monotonic()-start,independent_pending=True))
        record('final_recipe',dict(status='complete_light_checked',terminal=ev(OUT/'terminal.json'),owner=None))
    except Exception as exc:
        atomic(OUT/'terminal.json',dict(created_at=now(),status='failed_retained',error=repr(exc),traceback=traceback.format_exc()))
        record('final_recipe',dict(status='failed_retained',terminal=ev(OUT/'terminal.json'),owner=None));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run']);globals()[p.parse_args().mode]()
