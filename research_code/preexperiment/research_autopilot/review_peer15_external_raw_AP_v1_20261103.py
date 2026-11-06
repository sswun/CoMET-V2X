"""Independent raw geometry/matching/VOC AP50+70; no detector forwards."""
import argparse, hashlib, json, time, traceback
from pathlib import Path
import numpy as np
import review_traf_v2xseq_adapted_fixed60_raw_20261006 as independent
from audit_v2xseq_strict_common_roi_20261010 import mask,ROI
from peer15_v1_20261103 import ROOT,BASE,AUTO,REMOTE_ROOT,atomic,ev,load,now,register,record

RUNS=dict(vit='artifacts/runs/v2xseq_v2xvit_uniform_sensor_fixed60_eval_v2_recovery_20261014',fcooper='artifacts/runs/v2xseq_fcooper_uniform_sensor_fixed60_eval_v1_local_20261012',traf='artifacts/research_autopilot/status_audit_20261010_2100/traf_native4_full_successor_v2')
SOURCES=dict(vit='status_audit_20261014_1500/vit_eval_recovery_v3/terminal_problem_analysis/summary.json',fcooper='status_audit_20261013_1500/external_recovery_v1/fcooper_chain_v2/terminal_problem_analysis/summary.json',traf='status_audit_20261019_1200/strict_source_rank_effect_v4/summary.json')

def prepare():
    for method in RUNS:
        register('external_raw_'+method,Path(__file__).resolve(),'Saved raw predictions must independently reproduce both accepted AP50 and AP70 in the same common ROI/current-GT population; all discrepancies retained, no new forward or tuning.',dict(GPU_forwards=0,CPU_threads=1,frames=13264,new_disk_MiB=12),method=method,accepted_source=ev(AUTO/SOURCES[method]),manifest_pattern=RUNS[method],ROI=ROI,reviewer=ev(independent.__file__),bootstrap_input='compact score/tp/gt arrays per frame and sequence, references raw manifests and hashes')

def expected(method,wait):
    s=load(AUTO/SOURCES[method])
    if method=='traf':
        m=s['existing_native4_full_reference']['strict'+str(wait).zfill(3)]['metrics']
        return {str(t):m[str(t)]['global_score_sort'] for t in (.5,.7)}
    m=s['results']['delay_'+str(wait).zfill(3)+'ms']
    return {str(t):m['common_AP'+str(int(t*100))] for t in (.5,.7)}

def raw_path(row,arm):
    p=Path(row['raw_path'])
    if p.exists():return p
    for root in (str(ROOT)+'/',str(REMOTE_ROOT)+'/'):
        if str(p).startswith(root):
            candidate=ROOT/str(p)[len(root):]
            if candidate.exists():return candidate
    return arm/'raw'/p.name

def run(method):
    out=BASE/('external_raw_'+method);c=load(out/'contract.json');assert c['script']==ev(__file__)
    start=time.monotonic();cpu=time.process_time();results={};selfcheck=independent.self_test();assert selfcheck['status']=='passed'
    try:
        for wait in (0,100,200,400):
            arm=ROOT/RUNS[method]/(('strict'+str(wait).zfill(3)+'/run') if method=='traf' else 'delay_'+str(wait).zfill(3)+'ms')
            manifest=arm/'frame_manifest.jsonl';rows=[json.loads(l) for l in manifest.read_text().splitlines() if l.strip()];assert len(rows)==3316
            assert [r['global_index'] for r in rows]==list(range(3316));assert len({r['sequence'] for r in rows})==21
            ledger={str(t):dict(scores=[],tp=[],fp=[],gt=0,offset=[0],frame_gt=[]) for t in (.5,.7)};digest=hashlib.sha256()
            for i,row in enumerate(rows):
                p=raw_path(row,arm);assert independent.sha256(p)==row['raw_sha256'],p
                with np.load(p,allow_pickle=False) as z:arr={k:z[k] for k in ('pred_boxes','pred_scores','pred_labels','gt_boxes','gt_labels')}
                assert all(np.isfinite(v).all() for v in arr.values())
                pm,gm=mask(arr['pred_boxes']),mask(arr['gt_boxes']);gt=arr['gt_boxes'][gm];gl=arr['gt_labels'][gm]
                digest.update(gt.tobytes());digest.update(gl.tobytes())
                for t in (.5,.7):
                    scores,tp,fp,n=independent.match_frame(arr['pred_boxes'][pm],arr['pred_scores'][pm],arr['pred_labels'][pm],gt,gl,t)
                    d=ledger[str(t)];d['scores'].extend(scores);d['tp'].extend(tp);d['fp'].extend(fp);d['gt']+=n;d['offset'].append(len(d['scores']));d['frame_gt'].append(n)
                if i%128==0:atomic(out/'status.json',dict(created_at=now(),status='independent_raw_geometry_matching',method=method,wait_ms=wait,frames=i+1,wall_s=time.monotonic()-start))
            metrics={t:independent.voc_ap(d['tp'],d['fp'],d['scores'],d['gt'],True) for t,d in ledger.items()}
            accepted=expected(method,wait);comparison={t:dict(recomputed=metrics[t]['ap'],accepted=accepted[t]['ap'],delta=metrics[t]['ap']-accepted[t]['ap'],within_1e_10=abs(metrics[t]['ap']-accepted[t]['ap'])<=1e-10) for t in metrics}
            arrays=dict(sequence=np.asarray([r['sequence'] for r in rows]),global_index=np.asarray([r['global_index'] for r in rows]))
            for t,d in ledger.items():
                key='iou'+str(int(float(t)*100))
                for k in ('scores','tp','offset','frame_gt'):arrays[key+'_'+k]=np.asarray(d[k],dtype=np.float64 if k=='scores' else np.int64)
            shard=out/('wait'+str(wait).zfill(3)+'_matches.npz');np.savez_compressed(shard,**arrays)
            results[str(wait)]=dict(metrics=metrics,comparison=comparison,manifest=ev(manifest),compact_matches=ev(shard),current_GT_digest=digest.hexdigest(),frames=len(rows),sequences=21)
            atomic(out/'results.json',results)
        passed=all(v['within_1e_10'] for r in results.values() for v in r['comparison'].values())
        terminal=dict(created_at=now(),status='complete_independently_recomputed' if passed else 'completed_discrepancy_retained',method=method,results=results,reviewer_selftest=selfcheck,all_raw_SHA_checked=True,new_GPU_forwards=0,independent_geometry_matching_VOC=True,wall_s=time.monotonic()-start,CPU_s=time.process_time()-cpu,no_model_wakeup=True)
        atomic(out/'terminal.json',terminal);record('external_raw_'+method,dict(status=terminal['status'],terminal=ev(out/'terminal.json')))
    except Exception as e:
        atomic(out/'terminal.json',dict(created_at=now(),status='failed_retained',error=repr(e),traceback=traceback.format_exc(),results=results,wall_s=time.monotonic()-start));record('external_raw_'+method,dict(status='failed_retained',terminal=ev(out/'terminal.json')));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','vit','fcooper','traf']);a=p.parse_args();prepare() if a.mode=='prepare' else run(a.mode)
