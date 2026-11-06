"""Native-candidate readout control, not a disguised CoMET geometry improvement.

This control protects confident NATIVE candidates, not separate ego boxes.
Thus the exact original ego-prefix variant remains separately unmeasured.
"""
import argparse
import inspect
import json
import subprocess
import time
import traceback
from collections import defaultdict
from pathlib import Path
import numpy as np
from peer16_v1_20261103 import ROOT,AUTO,BASE,REMOTE_ROOT,atomic,ev,load,now,register,record
import run_crosscheckpoint_temporal_matched39d_v1_20261029 as matched
import run_peer15_author_recipe_v1_20261103 as author
from object_history_selector_core_v1_20261021 import common_mask,feature_matrix
from review_peer15_external_raw_AP_v1_20261103 import RUNS,raw_path
import review_peer16_E6_pose_source_v1_20261103 as first
import review_peer16_E6_pose_source_v2_20261103 as second
import review_peer16_E6_pose_source_v3_20261103 as accepted
PROD=BASE/'E7_native_fitcal_cache';OUT=BASE/'E7_native_readout'
METHODS=('vit','fcooper','vit2024','vit2025','where')
def prepare():
    register('E7_native_readout',__file__,
        'Quantify how much a fixed39D/2000-update readout can help existing native fusion '
        'outputs, using only fit24 and cal8, three fixed head seeds. Also transfer the '
        'frozen CoMET author head without fitting. Retain all val outcomes, no val choice. '
        'This separately registered control uses native high-confidence prefix, NOT a '
        'separate ego prefix: original exact ego-prefix fused+U remains an explicit gap.',
        dict(GPU_forwards=0,CPU_threads=2,updates='5 models x3 head seeds x2000=30000',
             wall_h='finite24h dependencies +1-5h CPU',disk_MiB='<220 scores/candidate IDs/head weights, shared fused geometry/GT referenced'),
        producer=ev(PROD/'contract.json'),seeds=[2023,2024,2025],
        representation='Fused output in receiver frame, source==0/identity pose/zero age, hypothesis_is_ego1, no causal-motion features. Protected iff native score>=original author frozen tau. Geometry/labels immutable.',
        target='Native-candidate uncovered-current-GT utility atIoU70; same39x32x1/Adam.003/2000updates and sequence-frame-wait-candidate balanced sampling; cal8 only source logistic.',
        limitation='Not exact ego-prefix or same candidate population as late fusion; not an isolated CoMET module effect, not backbone retraining. Source/age flags for fused outputs are an explicit adapter convention.',
        successor='Independent native geometry/AP receipt per model after scientific terminal; no model wake.')
def native_root(method):
    if method.startswith('vit20'):
        return AUTO/'status_audit_20261025_autonomy'/('v2xvit_seed'+method[-4:]+'_terminal_synthesis_v1/inputs/strict')
    if method=='where':return ROOT/first.REF['where']
    return ROOT/RUNS[method]
def material(method,role,tau):
    arrays=defaultdict(list);co=[0];go=[0];sources=[]
    first.PROD=PROD;second.LABEL_CONTEXT.clear()
    gtfn=accepted.ns['current_gt']
    for wait in (0,100,200,400):
        if role=='val21':
            root=native_root(method)
            candidates=[root/('delay_'+str(wait).zfill(3)+'ms'),root/('strict'+str(wait).zfill(3)),root/('strict'+str(wait))]
            folder=next(p for p in candidates if (p/'frame_manifest.jsonl').exists())
            rows=[json.loads(l) for l in (folder/'frame_manifest.jsonl').read_text().splitlines() if l.strip()]
            assert len(rows)==3316;sources.append(ev(folder/'frame_manifest.jsonl'))
        else:
            folder=PROD/method;rows=[r for r in load(folder/f'wait{wait:03d}_records.json') if r['role']==role]
            assert len(rows)==(3762 if role=='fit24' else 1247)
            sources.append(ev(folder/f'wait{wait:03d}_records.json'))
        last=None;z=None
        for i,r in enumerate(rows):
            if role=='val21':
                p=raw_path(r,folder);assert ev(p)['sha256']==r['raw_sha256']
                with np.load(p) as raw:
                    b=raw['pred_boxes'];s=raw['pred_scores'];label=raw['pred_labels'];gt=raw['gt_boxes'];gl=raw['gt_labels']
                index=r['global_index'];seq=str(r['sequence'])
                frame=r.get('effective_vehicle_frame',r.get('effective_label_frame'))
            else:
                if r['shard']!=last:
                    if z is not None:z.close()
                    z=np.load(folder/r['shard']);last=r['shard']
                k=r['case'];b=z[f'c{k}_box3d_lidar'];s=z[f'c{k}_scores'];label=z[f'c{k}_label_preds']
                gt,gl=gtfn(method,dict(effective_label_frame=r['current_vehicle_frame']),r)
                index=r['dataset_index'];seq=str(r['sequence']);frame=r['current_vehicle_frame']
            keep=common_mask(b)&(label==0);b=b[keep].astype(np.float32);s=s[keep].astype(np.float64)
            gm=common_mask(gt)&(gl==0);gt=gt[gm].astype(np.float32)
            fixed=s>=tau;f=feature_matrix(b,b,s,0.,np.eye(4),b[fixed],s[fixed])
            extra=np.zeros((len(b),9));extra[:,0]=1.
            arrays['features'].append(np.column_stack((f,extra)).astype(np.float32))
            arrays['boxes'].append(b);arrays['raw_scores'].append(s);arrays['protected'].append(fixed.astype(np.int8))
            arrays['gt_boxes'].append(gt);co.append(co[-1]+len(b));go.append(go[-1]+len(gt))
            arrays['sequence'].append(seq);arrays['wait_ms'].append(wait);arrays['dataset_index'].append(index)
            arrays['global_index'].append(index);arrays['receiver_timestamp_us'].append(0)
            arrays['current_frame'].append(str(frame))
        if z is not None:z.close()
    result={k:np.concatenate(arrays[k]) for k in ('features','boxes','raw_scores','protected','gt_boxes')}
    result.update({k:np.asarray(arrays[k]) for k in ('sequence','wait_ms','dataset_index','global_index','receiver_timestamp_us','current_frame')})
    result.update(condition_offsets=np.asarray(co),gt_offsets=np.asarray(go),source=np.zeros(co[-1],np.int8),anchor=np.zeros(co[-1],np.int8))
    return result,sources
def one(method):
    dest=OUT/method;dest.mkdir(exist_ok=False);start=time.monotonic()
    c=load(author.CONTRACT);tau=float(load(ROOT/c['datasets']['mechanism_holdout14']/'summary.json')['protect_threshold'])
    fit,fi=material(method,'fit24',tau);cal,ci=material(method,'calibration8',tau)
    fit=matched.geometry_material(fit,fit['boxes'],fit['features']);cal=matched.geometry_material(cal,cal['boxes'],cal['features'])
    # Native candidates require a new balanced schedule, not reindexing late-fusion draws.
    weights,batches,mean,std,seq=matched.readout.balanced_training_schedule(fit)
    schedule=dest/'schedule.npz';np.savez_compressed(schedule,weights=weights,batches=batches)
    val,vi=material(method,'val21',tau);results=[]
    for seed in (2023,2024,2025):
        head=dest/('seed'+str(seed))
        s=inspect.getsource(matched.train_head).replace('torch.manual_seed(readout.MODEL_SEED)','torch.manual_seed(SEED)').replace('"seed": readout.MODEL_SEED','"seed": SEED')
        ns=dict(matched.__dict__,SEED=seed);exec(compile(s,__file__+'::fixed_native_readout_seed','exec'),ns)
        ns['train_head'](fit,schedule,head)
        matched.source_map(cal,matched.score(cal,head),head)
        score=author.refit.score_with_head(val,val['features'],head);name='native_U_seed'+str(seed)
        pp,pred=author.refit.fixed.base.evaluate(val,{name:val['boxes']},{name:score})
        results.extend(pp);np.savez_compressed(head/'val_predictions.npz',**pred)
        atomic(dest/'panels.json',results)
        atomic(dest/'status.json',dict(created_at=now(),stage='val_scoring',seed=seed,wall_s=time.monotonic()-start))
    authorhead=(ROOT/c['input_weight_identity']['geometry_heads']['kf']['weights']['path']).parent
    transfer=author.refit.score_with_head(val,val['features'],authorhead)
    pp,pred=author.refit.fixed.base.evaluate(val,{'native_prefix_raw':val['boxes'],'native_frozen_author_U':val['boxes']},
        {'native_prefix_raw':val['raw_scores'],'native_frozen_author_U':transfer})
    results.extend(pp);np.savez_compressed(dest/'val_control_predictions.npz',**pred)
    atomic(dest/'panels.json',results)
    atomic(dest/'input_identities.json',dict(fit=fi,cal=ci,val=vi,threshold=tau,
        frozen_author_head=ev(authorhead/'direct_u.pth'),schedule=ev(schedule),
        weight=load(PROD/'contract.json')['methods'][method]['weight_identity'],
        native_geometry_unchanged=True,exact_ego_prefix_not_covered=True))
    atomic(dest/'terminal.json',dict(created_at=now(),status='complete_light_checked',heads=3,updates=6000,
        GPU_forwards=0,wall_s=time.monotonic()-start,panels=ev(dest/'panels.json'),independent_pending=True,
        scope='Native-candidate prefix, same39D architecture/budget; not original standalone-ego prefix.'))
    return ev(dest/'terminal.json')
def run():
    start=time.monotonic();done={};record('E7_native_readout',dict(status='finite_ready_first_native_fitcal_readout'))
    while len(done)<len(METHODS) and time.monotonic()-start<24*3600:
        for method in METHODS:
            if method in done:continue
            t=PROD/method/'terminal.json'
            if method in ('vit2024','vit2025','where') and not t.exists():
                remote=REMOTE_ROOT/t.relative_to(ROOT)
                if subprocess.run(['ssh','remote-gpu','test','-f',str(remote)],capture_output=True).returncode==0:
                    t.parent.mkdir(parents=True,exist_ok=True)
                    subprocess.run(['rsync','-a','--ignore-existing','remote-gpu:'+str(remote.parent)+'/',str(t.parent)+'/'],check=True)
            if not t.exists():continue
            try:
                assert load(t)['status']=='predictions_complete_pending_independent_AP'
                done[method]=one(method)
            except Exception as exc:
                atomic(OUT/method/'terminal.json',dict(created_at=now(),status='failed_retained',error=repr(exc),traceback=traceback.format_exc()))
                done[method]=ev(OUT/method/'terminal.json')
            atomic(OUT/'status.json',dict(created_at=now(),completed=done))
        if len(done)<len(METHODS):time.sleep(30)
    status='complete_light_checked' if len(done)==len(METHODS) and all(load(ROOT/r['path'])['status']=='complete_light_checked' for r in done.values()) else 'failed_retained'
    atomic(OUT/'terminal.json',dict(created_at=now(),status=status,completed=done,GPU_forwards=0,wall_s=time.monotonic()-start,
        exact_standalone_ego_prefix_variant_still_open=True,independent_pending=True))
    record('E7_native_readout',dict(status=status,terminal=ev(OUT/'terminal.json')))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run']);globals()[p.parse_args().mode]()
