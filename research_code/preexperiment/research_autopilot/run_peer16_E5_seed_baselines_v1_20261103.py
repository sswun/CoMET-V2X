"""Additive AP50/AP70 seed accounting from accepted per-frame outcomes."""
import time
import numpy as np
from peer16_v1_20261103 import ROOT,AUTO,BASE,atomic,ev,load,now,register,record
from review_traf_v2xseq_adapted_fixed60_raw_20261006 import voc_ap
import json
OUT=BASE/'E5_native_seed_AP50'
def main():
    refs={}
    for seed in (2024,2025):
        p=AUTO/'status_audit_20261025_autonomy'/f'v2xvit_seed{seed}_terminal_synthesis_v1/analysis_v1'
        refs[str(seed)]={k:ev(p/k) for k in ['summary.json','independent_review.json','ego_frames.jsonl']+[f'strict_{w:03d}ms_frames.jsonl' for w in (0,100,200,400)]}
    register('E5_native_seed_AP50',__file__,'AP50 already exists as accepted per-frame matches; pool it without new model calls or changing pairing. Report3seed sampleSD and single-car AP, retaining protocol/seed identities.',dict(GPU_forwards=0,CPU_threads=1,updates=0,disk_MiB='<20 compact score/TP ledgers',wall_h='<.1'),inputs=refs,seed2023=ev(ROOT/'artifacts/research_autopilot/status_audit_20261103_peer15/external_raw_vit/terminal.json'),scope='Seed accounting and existing accepted ledger AP, not a new raw matching audit. Other-method ego/head-seed/subtraction/CI components of E5 remain open.')
    start=time.monotonic();record('E5_native_seed_AP50',dict(status='running_accepted_outcome_pooling'));panels=[]
    for seed in (2024,2025):
        ss=load(ROOT/refs[str(seed)]['summary.json']['path']);assert load(ROOT/refs[str(seed)]['independent_review.json']['path'])['status']=='passed'
        for w in (None,0,100,200,400):
            key='ego_frames.jsonl' if w is None else f'strict_{w:03d}ms_frames.jsonl'
            p=ROOT/refs[str(seed)][key]['path'];assert ev(p)==refs[str(seed)][key]
            rows=[json.loads(x) for x in p.read_text().splitlines()];assert len(rows)==3316 and sum(r['gt'] for r in rows)==31123
            arrays=dict(sequence=np.asarray([r['sequence'] for r in rows]),global_index=np.asarray([r['global_index'] for r in rows]),GT_count=np.asarray([r['gt'] for r in rows]));pp=dict(seed=seed,wait_ms=w,method='ego' if w is None else 'native',GT=31123,checkpoint_sha256=ss['checkpoint_sha256'],accepted_ledger=ev(p))
            for th in (50,70):
                scores=np.concatenate([r['scores'+str(th)] for r in rows]);tp=np.concatenate([r['tp'+str(th)] for r in rows]);fp=np.concatenate([r['fp'+str(th)] for r in rows]);value=voc_ap(tp.tolist(),fp.tolist(),scores.tolist(),31123,True)
                pp['AP'+str(th)]=100*value['ap'];arrays.update({f'iou{th}_scores':scores,f'iou{th}_tp':tp,f'iou{th}_offset':np.r_[0,np.cumsum([len(r['scores'+str(th)]) for r in rows])],f'iou{th}_frame_gt':arrays['GT_count']})
            np.savez_compressed(OUT/f'seed{seed}_{"ego" if w is None else "wait"+str(w)}_matches.npz',**arrays);panels.append(pp)
    first=load(ROOT/'artifacts/research_autopilot/status_audit_20261103_peer15/external_raw_vit/results.json')
    for w,r in first.items():panels.append(dict(seed=2023,wait_ms=int(w),method='native',AP50=100*r['metrics']['0.5']['ap'],AP70=100*r['metrics']['0.7']['ap'],source='peer15 independently recomputed raw predictions'))
    pooled=[]
    for w in (0,100,200,400):
        p=dict(wait_ms=w,seeds=[2023,2024,2025],paradigm='uniform60 native strict input, val21')
        for th in (50,70):
            v=np.asarray([next(r for r in panels if r['seed']==s and r['wait_ms']==w)['AP'+str(th)] for s in p['seeds']]);p['AP'+str(th)]=dict(mean=float(v.mean()),sample_SD=float(v.std(ddof=1)),per_seed=v.tolist())
        pooled.append(p)
    atomic(OUT/'panels.json',dict(per_seed=panels,three_seed=pooled));atomic(OUT/'terminal.json',dict(created_at=now(),status='complete_light_checked',panels=ev(OUT/'panels.json'),GPU_forwards=0,wall_s=time.monotonic()-start,independent_raw_rematching=False,accepted_per_frame_matching_reused=True));record('E5_native_seed_AP50',dict(status='complete_light_checked',terminal=ev(OUT/'terminal.json')))
if __name__=='__main__':main()
