"""Sole collector + independent raw nominal-pair AP, matching current GT by hash."""
import argparse,hashlib,json,shlex,subprocess,time,traceback
from pathlib import Path
import numpy as np
from peer16_v1_20261103 import ROOT,BASE,REMOTE_ROOT,atomic,ev,load,now,register,record
import review_traf_v2xseq_adapted_fixed60_raw_20261006 as ind
from audit_v2xseq_strict_common_roi_20261010 import mask
from review_peer15_external_raw_AP_v1_20261103 import raw_path,RUNS
OUT=BASE/'E6_independent';PROD=BASE/'E6_pose_source_v2'
REF=dict(RUNS,where='artifacts/runs/v2xseq_where2comm_sensor_arrival_domain_fixed60_eval_v1_20261010')
def prepare():
    register('E6_independent',__file__,'Independently reconstruct corners/IoU/matching/VOC50/70 for all E6 models. Verify current GT content exactly against accepted main strict population, not nominal source/offset semantics conflation.',dict(GPU_forwards=0,CPU_threads=1,wall_h='finite20h ready-first all3 producers',disk_MiB='<30 compact cluster outcomes; one remote output collector'),producer=ev(PROD/'contract.json'),references=REF)
def review(method):
    src=PROD/method;out=OUT/method;out.mkdir(exist_ok=False);result={}
    ref=ROOT/REF[method]/'delay_000ms';manifest=ref/'frame_manifest.jsonl'
    if not manifest.exists():
        # Accepted Where sensor/strict0 directory uses the named arm.
        candidates=[ROOT/REF[method]/'strict000/frame_manifest.jsonl',ROOT/REF[method]/'strict0/frame_manifest.jsonl']
        manifest=next(p for p in candidates if p.exists());ref=manifest.parent
    rr=[json.loads(l) for l in manifest.read_text().splitlines() if l.strip()];assert len(rr)==3316
    for w in (0,100,200,400):
        rows=load(src/f'wait{w:03d}_records.json');assert len(rows)==3316
        scores=[];tps={50:[],70:[]};off=[0];gtcounts=[];last=None;z=None
        for i,(r,base) in enumerate(zip(rows,rr)):
            assert r['global_index']==base['global_index']==i and str(r['sequence'])==str(base['sequence'])
            raw=raw_path(base,ref)
            assert ind.sha256(raw)==base['raw_sha256']
            with np.load(raw) as a:gt=a['gt_boxes'];gl=a['gt_labels']
            whole=np.column_stack((gt,gl))
            # Accepted old extract_frame_gt can store concatenated labels in
            # float64 or float32. Explicit content equality is checked by both hashes.
            hashes={hashlib.sha256(whole.astype(t).tobytes()).hexdigest() for t in (np.float32,np.float64)}
            assert r['current_GT_shape']==list(whole.shape) and r['current_GT_sha256'] in hashes,('current_GT',method,w,i)
            if last!=r['shard']:
                if z is not None:z.close()
                z=np.load(src/r['shard']);last=r['shard']
            k=r['case'];b=z[f'c{k}_box3d_lidar'];s=z[f'c{k}_scores'];l=z[f'c{k}_label_preds'];pm,gm=mask(b),mask(gt)
            for th in (50,70):
                ss,tp,fp,ng=ind.match_frame(b[pm],s[pm],l[pm],gt[gm],gl[gm],th/100)
                tps[th].extend(tp)
            scores.extend(ss);off.append(len(scores));gtcounts.append(ng)
            if i%128==0:atomic(out/'status.json',dict(created_at=now(),status='independent_geometry_GT_matching',wait_ms=w,frames=i+1))
        if z is not None:z.close()
        metrics={str(th):ind.voc_ap(tps[th],(1-np.asarray(tps[th])).tolist(),scores,sum(gtcounts),True) for th in (50,70)}
        np.savez_compressed(out/f'wait{w:03d}_matches.npz',sequence=np.asarray([r['sequence'] for r in rows]),score=np.asarray(scores),tp50=np.asarray(tps[50]),tp70=np.asarray(tps[70]),offset=np.asarray(off),frame_GT=np.asarray(gtcounts))
        result[str(w)]=dict(metrics=metrics,frames=3316,GT=sum(gtcounts),current_GT_exact=True,manifest=ev(manifest))
        atomic(out/'panels.json',result)
    atomic(out/'terminal.json',dict(created_at=now(),status='complete_independently_recomputed',results=result,annotation_assisted=True,not_deployment=True,GPU_forwards=0));return ev(out/'terminal.json')
def run():
    start=time.monotonic();done={};record('E6_independent',dict(status='waiting_ready_first_3_GPU_producers'))
    while len(done)<3 and time.monotonic()-start<20*3600:
        remote=PROD/'fcooper';rel=str(remote.relative_to(ROOT))
        if not (remote/'terminal.json').exists():
            r=subprocess.run(['ssh','remote-gpu','test -f '+shlex.quote(str(REMOTE_ROOT/rel/'terminal.json'))],capture_output=True)
            if r.returncode==0:
                remote.mkdir(parents=True,exist_ok=True)
                subprocess.run(['rsync','-a','--ignore-existing','remote-gpu:'+str(REMOTE_ROOT/rel)+'/',str(remote)+'/'],check=True)
        for method in ('vit','where','fcooper'):
            p=PROD/method/'terminal.json'
            if method in done or not p.exists():continue
            try:
                assert load(p)['status']=='predictions_complete_pending_independent_AP',load(p)
                done[method]=dict(status='complete_independently_recomputed',terminal=review(method))
            except Exception as e:
                done[method]=dict(status='failed_retained',error=repr(e),traceback=traceback.format_exc());atomic(OUT/method/'terminal.json',dict(created_at=now(),**done[method]))
            atomic(OUT/'status.json',dict(created_at=now(),status='finite_ready_first_review',completed=done))
        if len(done)<3:time.sleep(30)
    status='complete_independently_recomputed' if len(done)==3 and all(d['status']=='complete_independently_recomputed' for d in done.values()) else 'failed_retained'
    atomic(OUT/'terminal.json',dict(created_at=now(),status=status,results=done,wall_s=time.monotonic()-start));record('E6_independent',dict(status=status,terminal=ev(OUT/'terminal.json')))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run']);globals()[p.parse_args().mode]()
