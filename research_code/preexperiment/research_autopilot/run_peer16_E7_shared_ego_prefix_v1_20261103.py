"""Unanswered exact shared-ego-prefix control, using existing cached ego boxes.

Native outputs are an additional virtual current-frame hypothesis; this is not
the same candidate population or a pure backbone comparison with late fusion.
"""
import argparse
import inspect
import time
import traceback
from collections import defaultdict
import numpy as np
from peer16_v1_20261103 import ROOT, BASE, atomic, ev, load, now, register, record
import run_peer16_E7_native_readout_v1_20261103 as old
from run_peer16_cached_methods_v1_20261103 import materials
from object_history_selector_core_v1_20261021 import common_mask,feature_matrix

OUT=BASE/'E7_shared_ego_prefix'
READY=BASE/'E7_native_readout_v2'
METHODS=old.METHODS
AUTHOR_DATA=None


def prepare():
    register('E7_shared_ego_prefix',__file__,
        'Native-prefix control alone cannot answer W5 original protected-prefix '
        'fairness. After its sole collector finishes each model, combine the '
        'same immutable TraF-author standalone ego candidates/protected prefix '
        'used by CoMET with saved native fused outputs. Fit3 fixed head seeds '
        'x2000 on fit24, map on cal8, evaluate all val waits without selection. '
        'Raw shared-prefix and frozen-author-U transfer controls included. '
        'No detector call, no held-role native evaluation or new micro-method.',
        dict(GPU_forwards=0,updates=30000,CPU_threads=2,
             wall_h='finite28h dependencies plus1-6h CPU',disk_MiB='<250'),
        ready=ev(READY/'contract.json'),
        native_inputs=ev(old.PROD/'contract.json'),
        author_prefix=ev(old.author.CONTRACT),
        features='Keep original ego features/flags; fused candidates use '
                 'identity pose/age0 and zero motion extras, source1 denotes '
                 'virtual fused hypothesis, NOT a physical roadside packet.',
        fair_scope='Exact same author standalone ego candidate/prefix geometry '
                   'as CoMET; fused extra candidate population necessarily '
                   'different, not a pure correction/module attribution.',
        statistics='Full independent geometry/AP and paired sequence CI successor.',
        storage='Shared raw/ego/GT referenced; only heads/schedules/scores saved.')


def material(method,role,tau):
    global AUTHOR_DATA
    if AUTHOR_DATA is None:AUTHOR_DATA=materials('author')[0]
    ego,rows=AUTHOR_DATA[role]
    native,refs=old.material(method,role,tau)
    lookup={(str(r['sequence']),int(r['wait_ms']),str(r['current_vehicle_frame'])):i
            for i,r in enumerate(rows)}
    assert len(lookup)==len(rows)
    arrays=defaultdict(list);co=[0];go=[0]
    for i,(seq,w,frame) in enumerate(zip(native['sequence'],native['wait_ms'],native['current_frame'])):
        j=lookup[str(seq),int(w),str(frame)]
        a,b=map(int,native['condition_offsets'][i:i+2]);ga,gb=map(int,native['gt_offsets'][i:i+2])
        ea,eb=map(int,ego['condition_offsets'][j:j+2]);g0,g1=map(int,ego['gt_offsets'][j:j+2])
        gt=native['gt_boxes'][ga:gb];eg=ego['gt_boxes'][g0:g1];eg=eg[common_mask(eg)]
        assert len(gt)==len(eg)
        assert np.array_equal(gt[np.lexsort(gt.T[::-1])],eg[np.lexsort(eg.T[::-1])]),('current_GT',method,role,i)
        ids=ea+np.flatnonzero(ego['source'][ea:eb]==0)
        bb=ego['cv_boxes'][ids];ss=ego['raw_scores'][ids]
        fixed=ego['protected'][ids].astype(np.int8)
        assert np.array_equal(fixed,ss>=tau),('original_prefix_threshold',role,i)
        nb=native['boxes'][a:b];ns=native['raw_scores'][a:b]
        nf=feature_matrix(nb,nb,ns,0.,np.eye(4),bb[fixed==1],ss[fixed==1])
        nf=np.column_stack((nf,np.zeros((len(nb),9))))
        arrays['boxes'].append(np.concatenate((bb,nb)).astype(np.float32))
        arrays['features'].append(np.concatenate((ego['cv_features'][ids],nf)).astype(np.float32))
        arrays['raw_scores'].append(np.r_[ss,ns])
        arrays['protected'].append(np.r_[fixed,np.zeros(len(nb),np.int8)])
        arrays['source'].append(np.r_[np.zeros(len(bb),np.int8),np.ones(len(nb),np.int8)])
        arrays['gt_boxes'].append(gt)
        co.append(co[-1]+len(bb)+len(nb));go.append(go[-1]+len(gt))
    result={k:np.concatenate(v) for k,v in arrays.items()}
    result.update({k:native[k].copy() for k in ('sequence','wait_ms','dataset_index',
        'global_index','receiver_timestamp_us','current_frame')})
    result.update(condition_offsets=np.asarray(co),gt_offsets=np.asarray(go),
                  anchor=np.zeros(co[-1],np.int8))
    return result,refs+[ev(old.author.CONTRACT)]


def balanced_native(fit):
    s=inspect.getsource(old.matched.readout.balanced_training_schedule)
    s=s.replace('def balanced_training_schedule(', 'def balanced_shared_prefix(')
    marker='        p = pool(materialized, condition)';assert s.count(marker)==1
    s=s.replace(marker,"        a,b=map(int,materialized['condition_offsets'][condition:condition+2])\n        p=dict(absolute=np.arange(a,b,dtype=np.int64),protected=materialized['protected'][a:b])")
    s=s.replace('materialized["global_index"]','materialized["dataset_index"]')
    ns=dict(old.matched.readout.__dict__)
    exec(compile(s,__file__+'::same_balanced_sampler','exec'),ns)
    return ns['balanced_shared_prefix'](fit)


def one(method):
    s=inspect.getsource(old.one)
    s=s.replace('matched.readout.balanced_training_schedule(fit)','balanced_native(fit)')
    s=s.replace('exact_ego_prefix_not_covered=True','exact_author_ego_prefix_shared=True')
    s=s.replace('Native-candidate prefix, same39D architecture/budget; not original standalone-ego prefix.',
        'Original author standalone ego prefix shared exactly; native fused extra hypotheses, same39D budget, not same candidate population.')
    ns=dict(old.__dict__,OUT=OUT,material=material,balanced_native=balanced_native)
    exec(compile(s,__file__+'::same_fixed_budget_head_training','exec'),ns)
    return ns['one'](method)


def run():
    start=time.monotonic();done={}
    record('E7_shared_ego_prefix',dict(status='finite_ready_first_shared_ego_prefix'))
    while len(done)<len(METHODS):
        assert time.monotonic()-start<28*3600,'finite28h shared-prefix chain'
        for method in METHODS:
            if method in done:continue
            t=READY/method/'terminal.json'
            if not t.exists():continue
            try:
                assert load(t)['status']=='complete_light_checked'
                done[method]=one(method)
            except Exception as exc:
                atomic(OUT/method/'terminal.json',dict(created_at=now(),status='failed_retained',
                    error=repr(exc),traceback=traceback.format_exc()))
                done[method]=ev(OUT/method/'terminal.json')
            atomic(OUT/'status.json',dict(created_at=now(),completed=done))
        if len(done)<len(METHODS):time.sleep(30)
    valid=all(load(ROOT/r['path'])['status']=='complete_light_checked' for r in done.values())
    status='complete_light_checked' if valid else 'failed_retained'
    atomic(OUT/'terminal.json',dict(created_at=now(),status=status,completed=done,
        GPU_forwards=0,wall_s=time.monotonic()-start,independent_pending=True))
    record('E7_shared_ego_prefix',dict(status=status,terminal=ev(OUT/'terminal.json')))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run'])
    globals()[p.parse_args().mode]()
