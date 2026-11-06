"""Zero-forward native SPD alpha chain; actual own ROI and shared GT identities."""
import argparse,os,time,traceback
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
from collections import defaultdict
from contextlib import ExitStack
import numpy as np
from peer14_cycle21_v1_20261102 import ROOT,AUTO,AUDIT,CYCLE,C15,C18,atomic,ev,load,now,register,record
BASE=CYCLE/'native_dose_response_v1';NAME='native_dose_response'
PROTOCOLS=('alpha000','alpha025','alpha050','alpha075','official','official_seqmean','official_MA5')

def prepare():
    BASE.mkdir(exist_ok=False)
    initial=['cycle15_peer11_v1/e60_val','k4_eval_public','k4_eval_online_k1','seed2_public/evaluation','seed2_online_k1/evaluation','cycle15_peer11_v1/seed3_public/evaluation','cycle15_peer11_v1/seed3_online/evaluation']
    targets=[dict(path=n,family='TraF_native',bounds=[0,-38.4,-3,96,38.4,1]) for n in initial]
    for n in ('adapted_evaluation','online_adapted_evaluation','author_evaluation'):targets.append(dict(path=str((C15/'coalign_full_v1'/n).relative_to(AUDIT)),family='CoAlign_shared_TraF_GT',bounds=[-100.8,-40,-3.5,100.8,40,1.5]))
    for s in (20261103,20261104):
        for train in ('public','online'):
            for kind in ('graph','reference'):targets.append(dict(path=str((C18/'coalign_confirmation_v1'/('seed'+str(s))/(train+'_'+kind)/'evaluation').relative_to(AUDIT)),family='CoAlign_shared_TraF_GT',bounds=[-100.8,-40,-3.5,100.8,40,1.5]))
    register(NAME,__file__,'Same fixed-prediction alpha interventions for saved native TraF and complete CoAlign arms. Official AP must replay before curve acceptance. No cross-ROI/backbone absolute rankings; negative/no-flip outcomes retained, no inference/training/model choice.',dict(GPU_forwards=0,CPU_threads=1,wall_h='finite36h known producer wait and sequential scoring',new_disk_MiB='<80 panels only; all GT/predictions referenced'))
    c=load(CYCLE/(NAME+'_contract.json'));c.update(targets=targets,protocols=PROTOCOLS,source_kernel=ev(ROOT/'preexperiment/research_autopilot/run_peer07_native_cvh_source_split_ap_v1_20261031.py'),TraF_source_contract=ev(AUDIT/'cycle03_v1/contract.json'),CoAlign_population_contract=ev(C18/'coalign_sources_ROI_v2/contract.json'),scope='First frozen batch: all3seed TraF PUBLIC/ONLINE plus e60, CoAlign first author/adapted/online and two confirmation seeds all actual arms. Where/ViT/native upper/other saved historical native remain explicit next-batch gap, not complete B.',labels='Inf-exclusive only XY +=(alpha-1)*official offset. seqmean/MA5 offline annotation-only references. Original GT masks/classes/z/HWL/yaw/order and own prediction ROI unchanged. Both original-native and common ROI panels; CoAlign native name means shared TraF GT, NOT author complete native GT.',storage='No saved duplicate GT/predictions/outcomes; per-producer metrics/identities only.')
    # Use the actual original range; fail closed if its contract shape changes.
    bounds=load(AUDIT/'cycle03_v1/contract.json')['native_source_aux']['native_eval_range']
    for t in targets:
        if t['family']=='TraF_native':t['bounds']=bounds
    atomic(CYCLE/(NAME+'_contract.json'),c)

def compute(t,offsets,slow,out):
    import run_peer07_native_cvh_source_split_ap_v1_20261031 as base
    from run_peer10_native_source_aux_v1_20261101 import outcome
    p=AUDIT/t['path'];term=load(p/'terminal.json');assert term['status']=='native_AP_complete_light_checked';records=load(p/'records.json');arms=term['arms'];reference={int(r['wait_ms']):r for r in load(p/'panels.json')};panels=[];labelcache={};identities=[];bounds=t['bounds']
    waits=sorted({int(r['wait_ms']) for r in records});labelroot=ROOT/'artifacts/data/v2x_seq_spd_main_domain_shared_v1/V2X-Seq-SPD/cooperative/label'
    for wait in waits:
        selected=[r for r in records if int(r['wait_ms'])==wait];stats=defaultdict(lambda:dict(scores=[],tp=[],fp=[],gt=0));shardpath=p/f'wait{wait:03d}.npz';identities.append(ev(shardpath))
        with ExitStack() as stack:
            z=stack.enter_context(np.load(shardpath));g=None
            if 'GT_shard' in selected[0]:
                gp=ROOT/selected[0]['GT_shard'];assert all(r['GT_shard']==selected[0]['GT_shard'] for r in selected);g=stack.enter_context(np.load(gp));identities.append(ev(gp))
            for i,r in enumerate(selected):
                n=r['shard_case'];gt=(g[f"c{r['GT_case']}_gt"] if g is not None else z[f'c{n}_gt']);frame=str(r['current_vehicle_frame'])
                if frame not in labelcache:labelcache[frame]=load(labelroot/(frame+'.json'))
                codes,error=base.source_codes(gt,labelcache[frame]);assert error<.001
                by={}
                for protocol,alpha in zip(PROTOCOLS[:5],[0,.25,.5,.75,1]):a=gt[:,:7].copy();a[codes==2,:2]+=(alpha-1)*offsets[frame];by[protocol]=a
                for protocol in PROTOCOLS[5:]:a=gt[:,:7].copy();a[codes==2,:2]+=slow[protocol][frame]-offsets[frame];by[protocol]=a
                assert np.array_equal(by['official'],gt[:,:7])
                for a in by.values():assert np.array_equal(a[codes!=2],gt[codes!=2,:7]) and np.array_equal(a[:,2:],gt[:,2:7])
                for roi in ('native','common'):
                    mask=gt[:,7]==0
                    if roi=='common':mask &=base.common_mask(gt[:,:7])
                    polygons={protocol:base.polygons(a[mask]) for protocol,a in by.items()}
                    for arm in arms:
                        bb=z[f'c{n}_{arm}_box3d_lidar'];ss=z[f'c{n}_{arm}_scores'];cl=z[f'c{n}_{arm}_label_preds'];keep=(cl==0)&(bb[:,0]>=bounds[0])&(bb[:,0]<=bounds[3])&(bb[:,1]>=bounds[1])&(bb[:,1]<=bounds[4])
                        if roi=='common':keep &=base.common_mask(bb)
                        bb=bb[keep];ss=ss[keep];order=np.argsort(-ss);ss=ss[order].astype(np.float64);predpolys=base.polygons(bb[order])
                        for protocol in PROTOCOLS:
                            overlap=base.matrix(predpolys,polygons[protocol])
                            for group in ('all','veh','coop','inf','veh_coop'):
                                for th in (50,70):
                                    tp,fp,ignore,count=outcome(overlap,codes[mask],group,th/100);valid=ignore==0;v=stats[protocol,roi,arm,group,th];v['gt']+=count;v['scores'].append(ss[valid]);v['tp'].append(tp[valid]);v['fp'].append(fp[valid])
                if i%128==0:atomic(out/'status.json',dict(status='native_alpha',producer=t['path'],wait_ms=wait,conditions=i+1,total=len(selected),created_at=now()))
        for (protocol,roi,arm,group,th),v in stats.items():
            m=base.metric(np.concatenate(v['tp']),np.concatenate(v['fp']),np.concatenate(v['scores']),v['gt']);row=dict(protocol=protocol,roi=roi,arm=arm,source_group=group,threshold=th,wait_ms=wait,**m)
            if protocol=='official' and roi=='native' and group=='all':assert abs(m['ap']-100*reference[wait]['metrics'][arm]['global_score_sort']['ap'+str(th)])<1e-8,('official native replay',t['path'],row)
            panels.append(row)
        atomic(out/'panels.json',panels)
    atomic(out/'input_identity.json',dict(terminal=ev(p/'terminal.json'),records=ev(p/'records.json'),panels=ev(p/'panels.json'),shards=identities,family=t['family'],bounds=bounds,actual_arms=arms,GT_never_duplicated=True))
    atomic(out/'terminal.json',dict(status='complete_light_checked',created_at=now(),panels=ev(out/'panels.json'),input=ev(out/'input_identity.json'),official_AP_replay_exact=True,GPU_forwards=0,independent_review_pending=True))

def run():
    c=load(CYCLE/(NAME+'_contract.json'));assert c['script']==ev(__file__);out=BASE/'producer';out.mkdir(exist_ok=False);start=time.monotonic();results=[];record(NAME,dict(status='running_native_alpha_chain'))
    meta=load(ROOT/'artifacts/data/v2x_seq_spd_main_domain_shared_v1/V2X-Seq-SPD/cooperative/data_info.json');groups=defaultdict(list);offsets={}
    for r in meta:frame=str(r['vehicle_frame']);offsets[frame]=np.array([r['system_error_offset'][k] for k in ('delta_x','delta_y')]);groups[str(r['vehicle_sequence'])].append(frame)
    slow={p:{} for p in PROTOCOLS[5:]}
    for seq,frames in groups.items():
        frames.sort();a=np.array([offsets[f] for f in frames])
        for i,f in enumerate(frames):slow['official_seqmean'][f]=a.mean(0);slow['official_MA5'][f]=a[max(0,i-2):i+3].mean(0)
    for i,t in enumerate(c['targets']):
        sub=out/str(i).zfill(3);sub.mkdir()
        try:
            p=AUDIT/t['path']
            while not (p/'terminal.json').exists():
                if time.monotonic()-start>36*3600:raise TimeoutError('finite36h producer wait')
                atomic(out/'status.json',dict(status='waiting_known_native_producer',producer=t['path'],completed=len(results),created_at=now()));time.sleep(60)
            compute(t,offsets,slow,sub);results.append(dict(producer=t['path'],terminal=ev(sub/'terminal.json')))
        except Exception as e:
            atomic(sub/'terminal.json',dict(status='failed_retained',created_at=now(),error=repr(e),traceback=traceback.format_exc()));results.append(dict(producer=t['path'],failed_terminal=ev(sub/'terminal.json')))
        atomic(out/'results.json',results)
    status='complete_light_checked' if all('terminal' in r for r in results) else 'failed_retained'
    atomic(out/'terminal.json',dict(status=status,created_at=now(),results=ev(out/'results.json'),scope=c['scope'],wall_s=time.monotonic()-start,GPU_forwards=0,no_model_wakeup=True));record(NAME,dict(status=status,terminal=ev(out/'terminal.json')))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run']);a=p.parse_args();prepare() if a.mode=='prepare' else run()
