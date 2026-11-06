"""Supply actual missing fit/cal box-module inputs, then three matched U seeds.

These are existing official box modules on the old author pool, not complete
external systems. No detector inputs or val geometry are regenerated.
"""
import argparse
import inspect
import time
import traceback
import numpy as np
from peer16_v1_20261103 import ROOT,AUTO,BASE,atomic,ev,load,now,register,record
import run_peer15_author_recipe_v1_20261103 as author
import run_crosscheckpoint_temporal_matched39d_v1_20261029 as matched
from run_peer16_E7_shared_ego_prefix_v1_20261103 import balanced_native
from bootstrap_peer15_paper_v1_20261103 import configs
from run_peer10_aux_remaining_v2_20261101 import modules
OUT=BASE/'E5_module_head_seeds'
OLD=AUTO/'status_audit_20261029_spd_roco_cbm_same_pool/contract_recovery_v3.json'

def prepare():
    _,cs=configs();c=cs['modules']
    register('E5_module_head_seeds',__file__,
        'Existing val/hold module boxes do not supply fit/cal inputs for '
        'matched-head seed reliability. Actually run the unchanged official '
        'FreeAlign/RoCo/CBM box modules on original fit24/cal8 only, then '
        'three fixed39D/2000-step heads each; evaluate immutable val/hold '
        'geometry with current common ROI and all waits. No model choice, '
        'external tuning or new detector forwards. Retain failure fallbacks.',
        dict(GPU_forwards=0,updates=18000,CPU_threads=1,
             module_calls='3 x20036 fit/cal conditions',wall_h='finite28h',
             disk_MiB='<350 module geometry/features, heads/scores; shared GT referenced'),
        original_modules=ev(OLD),old_pool=c['inputs']['val21_common'],
        original_train_pool=ev(ROOT/'artifacts/research_autopilot/status_audit_20261021_1200/object_history_selector_v1/train_dataset/summary.json'),
        accepted_geometry=c['inputs'],seeds=[2023,2024,2025],
        readout='39x32x1 Adam.003,2000 steps; fit24 balanced sampling, cal8 source map only',
        boundary='Module-level old author detector population, not full systems, '
                 'not strict60 native head, no independent neural training claim.')

def base_fit(dataset,role):
    m=author.formal.materialize(dataset,role,np.zeros(2),np.zeros(2))
    gt=[dataset.pool(int(i))['gt_boxes'] for i in m['dataset_index']]
    m.update(gt_boxes=np.concatenate(gt),gt_offsets=np.r_[0,np.cumsum([len(x) for x in gt])])
    return m

def run():
    start=time.monotonic();record('E5_module_head_seeds',dict(status='running_missing_module_fitcal_inputs'))
    try:
        import run_spd_current_official_roco_cbm_same_pool_v1_20261029 as mod
        c=load(OLD);_,cs=configs();spec=cs['modules']
        roco=mod.roco.load_roco(ROOT/c['inputs']['roco_source']['path'])
        cbm=mod.cbm.load_cbm(ROOT/c['inputs']['cbm_source']['path'])
        root=ROOT/c['datasets']['mechanism_holdout14'];dataset=mod.freealign.PackedDataset(root)
        geometry={};refs={}
        for role in ('fit24','calibration8'):
            common=base_fit(dataset,role);dest=OUT/role;dest.mkdir(exist_ok=False)
            fa=mod.freealign.materialize_role(dataset,role,common,None,dest)
            rc=mod.materialize(dataset,role,common,roco,cbm,dest)
            assert fa['same_population'] and fa['legal_packet_times'] and rc['same_population'] and rc['legal_packet_times']
            refs[role]=dict(freealign=fa,module=rc)
            atomic(dest/'terminal.json',dict(created_at=now(),status='complete_light_checked',
                inputs=refs[role],GPU_forwards=0,GT_never_passed_to_box_modules=True))
            geometry[role]=common
        rows=[];done={}
        for method in ('freealign','roco','cbm'):
            dest=OUT/method;dest.mkdir(exist_ok=False);fitcal={}
            for role,common in geometry.items():
                path=OUT/role/(role+('_freealign_materialized.npz' if method=='freealign' else '_module_material.npz'))
                with np.load(path) as z:b=z[method+'_boxes'];f=z[method+'_features']
                fitcal[role]=matched.geometry_material(common,b,f)
            fit,cal=fitcal['fit24'],fitcal['calibration8']
            weights,batches,_,_,_=balanced_native(fit)
            schedule=dest/'schedule.npz';np.savez_compressed(schedule,weights=weights,batches=batches)
            for seed in (2023,2024,2025):
                head=dest/('seed'+str(seed));source=inspect.getsource(matched.train_head)
                source=source.replace('torch.manual_seed(readout.MODEL_SEED)','torch.manual_seed(SEED)').replace('"seed": readout.MODEL_SEED','"seed": SEED')
                ns=dict(matched.__dict__,SEED=seed);exec(compile(source,__file__+'::same_budget_module_U','exec'),ns)
                ns['train_head'](fit,schedule,head);matched.source_map(cal,matched.score(cal,head),head)
                for role in ('mechanism_holdout14','val21'):
                    m,_,_,boxes,_=modules(role,spec)
                    b=boxes[method+'__raw'];key=role+('_freealign' if method=='freealign' else '_module')
                    with np.load(ROOT/spec['inputs'][key]['path']) as z:f=z[method+'_features']
                    score=author.refit.score_with_head(m,f,head);name=method+'_U_seed'+str(seed)
                    pp,pred=author.refit.fixed.base.evaluate(m,{name:b},{name:score})
                    rows.extend(dict(r,role=role,module=method,head_seed=seed) for r in pp)
                    np.savez_compressed(head/(role+'_predictions.npz'),**pred)
                    atomic(OUT/'panels.json',rows)
                    atomic(OUT/'status.json',dict(created_at=now(),module=method,seed=seed,role=role,wall_s=time.monotonic()-start))
            atomic(dest/'terminal.json',dict(created_at=now(),status='complete_light_checked',
                heads=3,updates=6000,GPU_forwards=0,independent_pending=True))
            done[method]=ev(dest/'terminal.json')
        atomic(OUT/'terminal.json',dict(created_at=now(),status='complete_light_checked',
            completed=done,panels=ev(OUT/'panels.json'),GPU_forwards=0,updates=18000,
            wall_s=time.monotonic()-start,independent_pending=True))
        record('E5_module_head_seeds',dict(status='complete_light_checked',terminal=ev(OUT/'terminal.json')))
    except Exception as exc:
        atomic(OUT/'terminal.json',dict(created_at=now(),status='failed_retained',
            error=repr(exc),traceback=traceback.format_exc(),wall_s=time.monotonic()-start))
        record('E5_module_head_seeds',dict(status='failed_retained',terminal=ev(OUT/'terminal.json')))
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run']);globals()[p.parse_args().mode]()
