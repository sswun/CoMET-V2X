"""Independent five-domain R6e choice acceptance and conditional real assets.

Recompute choice from independent cal8 AP only. Original recipe and tables
retain identity. If clean wins, actually regenerate all core13 comparisons and
independent AP/CI at a separate root, not merely a renamed registry. Additional
module fairness/root-table integration remains explicit pending until executed.
"""
import argparse,inspect,traceback
import numpy as np
from peer20_population_core_v1_20261105 import *
import run_peer20_clean_head_v1_20261105 as clean
import run_peer20_population_v1_20261105 as core
RECOVERY=BASE/'R6e_namespace_recovery_v3'

def source(domain):return (RECOVERY if domain=='strict60' else clean.OUT)/domain
def reviewer(domain):return BASE/('R6e_namespace_v3_strict60_independent' if domain=='strict60' else 'R6e_'+domain+'_independent')

def prepare():
    register('R6e_choice_independent',__file__,__doc__,dict(GPU_forwards=0,updates=0,CPU_threads=1,disk_MiB='<2'),
        domains=list(h1.recipe.DOMAINS),choice='Same five-domain cal official AP70 mean; tie original; no population or formal-role selection')
    for domain in h1.recipe.DOMAINS:
        for r in (False,True):
            register('R6e_clean_assets_'+domain+('_independent' if r else ''),__file__,__doc__,dict(GPU_forwards=0,updates=0,CPU_threads=1,replicates=2000,disk_MiB='<220 only if clean selected; otherwise <1'),
                domain=domain,scientific_engine=ev(core.__file__),clean_head=str(source(domain)),no_new_fit=True,
                temporal_controls='Retain original cal-selected control parameters, do not introduce a second R6e selection round',
                remaining_if_clean='SourceAP/robust external fairness and current root-table integration require separate versioned consumers; never declare entire new recipe accepted here.')

def accept():
    name='R6e_choice_independent';dest=BASE/name
    try:
        wait(clean.OUT/'global_choice/terminal.json');means=[]
        for domain in h1.recipe.DOMAINS:
            wait(reviewer(domain)/'terminal.json');panels=load(reviewer(domain)/'AP.json')
            values={v:float(np.mean([r['AP70'] for r in panels if r['role']=='calibration8' and r['variant']==v])) for v in ('clean','original')}
            assert all(sum(r['role']=='calibration8' and r['variant']==v for r in panels)==12 for v in values)
            gold=load(source(domain)/'calibration_ready.json')['means'];assert all(abs(values[v]-gold[v])<1e-8 for v in values)
            means.append(dict(domain=domain,means=values,review=ev(reviewer(domain)/'terminal.json')))
        overall={v:float(np.mean([r['means'][v] for r in means])) for v in ('clean','original')};selected='clean' if overall['clean']>overall['original'] else 'original'
        provisional=load(clean.OUT/'global_choice/decision.json');assert selected==provisional['selected'] and all(abs(overall[v]-provisional['means'][v])<1e-8 for v in overall)
        atomic(dest/'decision.json',dict(created_at=now(),selected=selected,means=overall,domains=means,selection_roles=['calibration8'],formal_roles_not_used=True,
            old_registry_unchanged=True,old_main_tables_unchanged=True,
            heads={d:{str(s):ev(source(d)/f'seed{s}/direct_u.pth') for s in clean.SEEDS} for d in h1.recipe.DOMAINS},
            whole_recipe_acceptance=(selected=='original'),conditional_consumers='Clean core13 production and independent only; if clean selected require explicit further fairness/source/table consumers.'))
        atomic(dest/'terminal.json',dict(created_at=now(),status='complete_independently_reviewed',decision=ev(dest/'decision.json')))
        record(name,dict(status='complete_independently_reviewed',terminal=ev(dest/'terminal.json'),owner=None,resource_owner=False))
    except Exception as e:
        atomic(dest/'terminal.json',dict(created_at=now(),status='failed_retained',error=repr(e),traceback=traceback.format_exc()));record(name,dict(status='failed_retained',terminal=ev(dest/'terminal.json'),owner=None,resource_owner=False));raise

def assets(domain,review):
    name='R6e_clean_assets_'+domain+('_independent' if review else '');dest=BASE/name
    wait(BASE/'R6e_choice_independent/terminal.json')
    if load(BASE/'R6e_choice_independent/decision.json')['selected']=='original':
        atomic(dest/'terminal.json',dict(created_at=now(),status='complete_independently_reviewed',not_required=True,reason='Cal8 independent choice retains original H1 heads; no duplicate inference/scoring',choice=ev(BASE/'R6e_choice_independent/decision.json')))
        record(name,dict(status='complete_independently_reviewed',terminal=ev(dest/'terminal.json'),owner=None,resource_owner=False));return
    wait(BASE/('R6b_'+domain)/'terminal.json')
    prior=core.h1.recipe.head
    core.h1.recipe.head=lambda d,k,s:source(d)/f'seed{s}'
    # New worker processes only; do not alter running or frozen original jobs.
    code=inspect.getsource(core.run).replace("'R6b_'+domain", "'R6e_clean_assets_'+domain")
    code=code.replace("choices=temporal_choice(dataset['calibration8'][0],q,tau,heads,domain)","choices=load(BASE/('R6b_'+domain)/'calibration.json')['selected']")
    ns=dict(core.__dict__);exec(compile(code,__file__+'::clean_head_actual_assets','exec'),ns);ns['run'](domain,review)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','accept','assets','review'));p.add_argument('--domain',choices=h1.recipe.DOMAINS);a=p.parse_args()
    prepare() if a.mode=='prepare' else accept() if a.mode=='accept' else assets(a.domain,a.mode=='review')
