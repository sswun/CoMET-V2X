"""R6b/R6d missing native, stronger-late and external-module population panels.

No new detectors, solvers, fits or choices. Reuse actual immutable saved boxes.
Native systems are val21-only; matched modules/late also hold14. Each original
late column is reported; stronger-per-column rule is inherited from existing
official-all-frame AP, never optimized on a population or auxiliary label.
"""
import argparse,json,traceback
import numpy as np
from peer20_population_core_v1_20261105 import *
import peer20_saved_native_v1_20261105 as native
import run_peer20_auxiliary_GT_v1_20261105 as aux
import run_peer19_robust_external_v1_20261104 as robust
import run_peer17_module_recipe_v1_20261104 as modules
import bootstrap_peer17_main_contrasts_v1_20261104 as late
from run_peer16_E7_native_readout_v1_20261103 import native_root
from review_peer15_external_raw_AP_v1_20261103 import RUNS
from bootstrap_peer18_final_native_v1_20261104 import STRICT

FAMILIES=('modules','late','native_author','native_strict60','native_vit','native_vit2024','native_vit2025','native_fcooper','native_where')

def prepare():
    for family in FAMILIES:
        for review in (False,True):
            register('R6bd_external_'+family+('_independent' if review else ''),__file__,__doc__,
                dict(GPU_forwards=0,updates=0,CPU_threads=1,replicates=2000,disk_MiB='<250 compact scores/TP/draws; raw shared'),
                family=family,adapter=ev(native.__file__),tiers=['official','gap1s','gap3s','all_gaps'],
                populations=list(POPS),native_roles=['val21'],other_roles=['val21','mechanism_holdout14'],
                independence='Independent polygon/greedy matching and AP/bootstrap; raw detection, readout weights, label mapping and source adapters shared.',
                labels='Original official population fixed; inf-only exact-zero labels translated by noncausal filled reference only in auxiliary tiers; never inference.')

def native_input(family,review):
    method=family.removeprefix('native_');domain={'author':'author','vit':'v2xvit_seed2024','vit2024':'v2xvit_seed2024','vit2025':'v2xvit_seed2025','fcooper':'author','where':'where2comm'}.get(method,method)
    if method=='where':
        root=P18/'actual_native_inputs/where/public/evaluation'
        assert (root/'terminal.json').exists(),root
        cases=native.shard_cases(root,'PUBLIC');identity=[ev(root/'terminal.json'),ev(root/'records.json')]
    else:
        if method=='author':folders=[ROOT/RUNS['traf']/f'strict{w:03d}/run' for w in WAITS]
        elif method=='strict60':folders=[STRICT/f'strict{w:03d}/run' for w in WAITS]
        else:
            base=native_root(method)
            folders=[next(p for p in (base/f'delay_{w:03d}ms',base/f'strict{w:03d}',base/f'strict{w}') if (p/'frame_manifest.jsonl').exists()) for w in WAITS]
        identity=[dict(manifest=ev(p/'frame_manifest.jsonl'),status=ev(p/'status.json')) for p in folders]
        cases=native.manifest_cases(folders)
    label=family
    m,rows,p,b=native.assemble(domain,'val21',cases,label,review)
    yield 'val21',m,rows,p,label,b,identity

def inputs(family,review):
    if family.startswith('native_'):
        yield from native_input(family,review);return
    if family=='late':
        _,cs=late.boot.configs()
        for role in ('val21','mechanism_holdout14'):
            m,rows,p,boxes,_=late.boot.aux.official_late(role,cs['official_late'])
            for label,b in boxes.items():yield role,m,rows,p,label,b,dict(family_contract=cs['official_late'],stronger_rule='Frozen original all-frame per-column max, not new population selection; both columns retained')
        return
    wait(P19/'robust_external_independent/terminal.json')
    dataset,_,_=data('author');_,cs=robust.old.configs();spec=cs['modules']
    for role in ('val21','mechanism_holdout14'):
        for method in robust.METHODS:
            m,rows,b,f,valid,path=robust.material(role,method,dataset,spec)
            for seed in h1.recipe.SEEDS:
                label=method+'_U_seed'+str(seed);pred=modules.OUT/method/f'seed{seed}'/(role+'_predictions.npz')
                with np.load(pred) as z:p={k:z[k] for k in z.files}
                yield role,m,rows,p,label,b,dict(prediction=ev(pred),geometry=ev(path),module_not_system=True,readout='Original method-specific no-position matched head, not H1 weights')
            for filt in ('instant',*robust.GRIDS):
                tag=method+'_'+filt
                with np.load(robust.OUT/(role+'_'+tag+'_state.npz')) as z:
                    bb=robust.geometry(m,b,f,z['observation'],z['delta'])[0]
                for seed in h1.recipe.SEEDS:
                    label=tag+'_seed'+str(seed);pred=robust.OUT/(role+'_'+label+'_predictions.npz')
                    with np.load(pred) as z:p={k:z[k] for k in z.files}
                    yield role,m,rows,p,label,bb,dict(prediction=ev(pred),geometry=ev(path),module_not_system=True,readout='Exact H1 weights and tau',R4_gate='Original official -.3pp read unchanged; all population columns auxiliary')

def run(family,review=False):
    name='R6bd_external_'+family+('_independent' if review else '');dest=BASE/name;src=BASE/('R6bd_external_'+family);done=[]
    try:
        if review:wait(src/'terminal.json')
        record(name,dict(status='running_independent_external_population_and_aux' if review else 'running_external_population_and_aux',resource_owner=True))
        for role,m,rows,p,label,b,identity in inputs(family,review):
            tag=role+'_'+label;path=src/(tag+'_predictions.npz')
            if review:
                # Independently reconstruct matches, at frozen original ranks.
                check=aux.rematch(m,p,label,b,m['gt_boxes'],True)
                assert all(np.array_equal(v,p[k]) for k,v in check.items()),('official_TP',family,tag)
                with np.load(path) as z:assert all(np.array_equal(z[k],v) for k,v in p.items() if k.startswith(label+'__'))
            else:
                selected={k:v for k,v in p.items() if k.startswith(label+'__')}
                np.savez_compressed(path,**selected);atomic(src/(tag+'_identity.json'),identity)
            save_stats(dest,m,rows,p,label,tag,review,load(src/(tag+'_statistics.json')) if review else None)
            for tier,cap in aux.TIERS:
                gt,ident=aux.changed_truth(m,rows,role,cap);tp=aux.rematch(m,p,label,b,gt,review);key=tag+'_'+tier;tp_path=src/(key+'_TP.npz')
                if review:
                    with np.load(tp_path) as z:assert all(np.array_equal(z[k],v) for k,v in tp.items())
                else:
                    np.savez_compressed(tp_path,**tp);atomic(src/(key+'_identity.json'),ident)
                save_stats(dest,m,rows,dict(p,**tp),label,key,review,load(src/(key+'_statistics.json')) if review else None)
            done.append(dict(role=role,label=label,predictions=ev(path)))
            atomic(dest/'progress.json',dict(created_at=now(),completed=len(done),last=tag))
        atomic(dest/'identity.json',dict(completed=done,official_main_unchanged=True,native_val_only=True,all_population_columns_posthoc=True))
        status='complete_independently_reviewed' if review else 'complete_light_checked'
        atomic(dest/'terminal.json',dict(created_at=now(),status=status,completed=len(done),scope='External R6b plus all three R6d tiers; no main-table selection',identity=ev(dest/'identity.json')))
        record(name,dict(status=status,terminal=ev(dest/'terminal.json'),owner=None,resource_owner=False))
    except Exception as e:
        atomic(dest/'terminal.json',dict(created_at=now(),status='failed_retained',error=repr(e),traceback=traceback.format_exc()));record(name,dict(status='failed_retained',terminal=ev(dest/'terminal.json'),owner=None,resource_owner=False));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','run','review'));p.add_argument('--family',choices=FAMILIES);a=p.parse_args()
    prepare() if a.mode=='prepare' else run(a.family,a.mode=='review')
