"""Unify the external module readout recipe using cached module geometry."""
import argparse
import inspect
import time
import traceback
import numpy as np
from peer17_v1_20261104 import ROOT,BASE,atomic,ev,load,now,register,record
from peer16_v1_20261103 import BASE as P16
import run_peer16_E5_module_head_seeds_v1_20261103 as old
import run_peer17_final_recipe_v1_20261104 as final
from run_peer16_E7_readout_v1_20261103 import REMOVED
OUT=BASE/'module_recipe'
MODULES=('freealign','roco','cbm')

def prepare():
    assert final.decision()['adopt_no_position']
    register('module_recipe',__file__,
        'Peer17 adopts no-position. Reuse immutable official module fit/cal '
        'geometry and exact old balanced draws, fit3fixed headseeds with '
        'position inputs zeroed, evaluate current commonROI/val+hold with '
        'the same frozen author cal coarse protected threshold. Do not '
        'recompute modules, forwards or tune external solvers. Effects may decrease.',
        dict(GPU_forwards=0,module_calls=0,updates=18000,CPU_threads=1,wall_h='finite12h',disk_MiB='<60 head/score deltas'),
        original=ev(old.OUT/'terminal.json'),original_independent=ev(P16/'E5_module_head_seeds_independent/terminal.json'),
        final_recipe=ev(final.OUT/'contract.json'),removed_features=list(REMOVED),seeds=list(final.SEEDS),
        boundary='Official box module on old saved detector pool, not full external systems; record actual prefix/GT identity vs final author pool before comparisons.')

def run():
    start=time.monotonic();record('module_recipe',dict(status='running_cached_geometry_no_position_heads'))
    try:
        _,cs=old.configs();spec=cs['modules'];dataset=old.author.PackedDataset(ROOT/load(old.OLD)['datasets']['mechanism_holdout14'])
        bases={r:old.base_fit(dataset,r) for r in ('fit24','calibration8')};panels=[]
        for method in MODULES:
            dest=OUT/method;dest.mkdir(exist_ok=False);fitcal={}
            for role,m in bases.items():
                path=old.OUT/role/(role+('_freealign_materialized.npz' if method=='freealign' else '_module_material.npz'))
                with np.load(path) as z:b=z[method+'_boxes'];f=z[method+'_features'].copy()
                f[:,REMOVED]=0;fitcal[role]=old.matched.geometry_material(m,b,f)
            schedule=old.OUT/method/'schedule.npz'
            for seed in final.SEEDS:
                head=dest/('seed'+str(seed));code=inspect.getsource(old.matched.train_head)
                code=code.replace('torch.manual_seed(readout.MODEL_SEED)','torch.manual_seed(SEED)').replace('"seed": readout.MODEL_SEED','"seed": SEED')
                ns=dict(old.matched.__dict__,SEED=seed);exec(compile(code,__file__+'::same_budget_no_position','exec'),ns)
                ns['train_head'](fitcal['fit24'],schedule,head)
                old.matched.source_map(fitcal['calibration8'],old.matched.score(fitcal['calibration8'],head),head)
            dep=final.OUT/'author/calibration.json'
            while not dep.exists():
                assert time.monotonic()-start<12*3600;time.sleep(30)
            tau=load(dep)['selected_tau'];identities=[]
            for role in ('val21','mechanism_holdout14'):
                m,rows,_,boxes,_=old.modules(role,spec);b=boxes[method+'__raw']
                key=role+('_freealign' if method=='freealign' else '_module')
                with np.load(ROOT/spec['inputs'][key]['path']) as z:f=z[method+'_features'].copy()
                f[:,REMOVED]=0
                # Record actual common-prefix identity, without pretending that
                # a different module box population is the same candidate set.
                author_m,author_rows=final.old.materials('author')[0][role]
                checks={k:np.array_equal(m[k],author_m[k]) for k in ('source','raw_scores','condition_offsets','sequence','wait_ms','gt_offsets','gt_boxes')}
                identities.append(dict(role=role,common_arrays=checks,common_candidate_layout=all(checks.values()),module_geometry=spec['inputs'][key]))
                for seed in final.SEEDS:
                    name=method+'_U_seed'+str(seed);head=dest/('seed'+str(seed))
                    score=old.author.refit.score_with_head(m,f,head)
                    pp,p=old.author.refit.fixed.base.evaluate(final.protected(m,tau),{name:b},{name:score})
                    panels.extend(dict(r,role=role,module=method,head_seed=seed,tau=tau,kind='no_position') for r in pp)
                    np.savez_compressed(head/(role+'_predictions.npz'),**p)
                    atomic(OUT/'panels.json',panels);atomic(OUT/'status.json',dict(created_at=now(),module=method,role=role,seed=seed,wall_s=time.monotonic()-start))
            atomic(dest/'identity.json',dict(created_at=now(),roles=identities,old_schedule=ev(schedule),new_updates=6000,module_calls=0,GPU_forwards=0))
            atomic(dest/'terminal.json',dict(created_at=now(),status='complete_light_checked',identity=ev(dest/'identity.json'),updates=6000,independent_pending=True))
        atomic(OUT/'terminal.json',dict(created_at=now(),status='complete_light_checked',GPU_forwards=0,updates=18000,module_calls=0,wall_s=time.monotonic()-start,independent_pending=True))
        record('module_recipe',dict(status='complete_light_checked',terminal=ev(OUT/'terminal.json')))
    except Exception as exc:
        atomic(OUT/'terminal.json',dict(created_at=now(),status='failed_retained',error=repr(exc),traceback=traceback.format_exc()))
        record('module_recipe',dict(status='failed_retained',terminal=ev(OUT/'terminal.json')));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run']);globals()[p.parse_args().mode]()
