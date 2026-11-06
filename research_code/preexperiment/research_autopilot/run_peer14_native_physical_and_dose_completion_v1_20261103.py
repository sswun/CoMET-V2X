"""Saved-native registration subsets and unmeasured backbone dose curves."""
import argparse,os,time,traceback
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
from collections import defaultdict
from contextlib import ExitStack
import numpy as np
from peer14_cycle00_v1_20261103 import ROOT,AUDIT,CYCLE,C21,C15,atomic,ev,load,now,register,record
import run_peer14_native_dose_response_v1_20261102 as dose
import run_peer14_registration_read_v1_20261102 as registration
BASE=CYCLE/'native_physical_dose_v1'

def prepare():
    BASE.mkdir(exist_ok=False);targets=[];bounds=load(AUDIT/'cycle03_v1/contract.json')['native_source_aux']['native_eval_range']
    where=[AUDIT/'cycle06_peer12_v1_20261102'/f'where_{t}_v6/val21/evaluation' for t in ('public','online_k1')]+[AUDIT/'cycle06_peer12_v1_20261102/where_e60_val21_v6/native/evaluation']
    where+=list((AUDIT/'cycle09_peer12_v1_20261102').glob('where_confirmation_gpu*/**/val21/evaluation'))
    where+=list((AUDIT/'cycle09_peer12_v1_20261102').glob('where_confirmation_gpu*/official*val21/evaluation'))
    vit=[AUDIT/'cycle09_peer12_v1_20261102/vit_native_v3'/h/'val21/evaluation' for h in ('gpu0','gpu1','remote')]
    vit+=list((C15/'vit_confirmation_v2').glob('gpu*/*/val21/evaluation'));vb=load(C15/'vit_source_identity_v4/contract.json')['native_eval_range']
    for family,paths,bb in [('Where_native',where,bounds),('ViT_native',vit,vb)]:
        for p in sorted(set(paths)):
            if not (p/'terminal.json').exists():continue
            assert load(p/'terminal.json')['status']=='native_AP_complete_light_checked';targets.append(dict(path=str(p.relative_to(AUDIT)),family=family,bounds=bb))
    assert targets
    oldtargets=load(C21/'native_dose_response_contract.json')['targets']
    register('native_physical_dose',__file__,'Fill real native A/B gaps with saved predictions only. V2 and SPD registration label AP is QUALITY-CONDITIONED and paired with official on identical population, never full main AP or physical truth. Added Where/ViT alpha curves retain each actual ROI/class population and original official endpoint replay, no label/seed/grid selection.',dict(GPU_forwards=0,updates=0,CPU_threads_per_lane=1,wall_h='finite36h known registration/native producer waits and sequential CPU scoring',new_disk_MiB='<100 metrics only; GT/pred/raw referenced'),dose_source=ev(dose.__file__),registration_source=ev(registration.__file__),new_dose_targets=targets,SPD_registration_targets=oldtargets+targets,V2_native_targets=['cycle15_peer11_v1/v2_public','cycle15_peer11_v1/v2_online'],scope='Existing native val21 backbone outputs only; first18 retained according to original contracts, no new native hold inference. No ROI recrop after label movement, no official geometry fill for rejected frames. ICP coverage<30% forbids labels.')
    for n in ('backbone_dose','SPD_native_registration','V2_native_registration'):record(n,dict(status='registered_ready',contract=ev(CYCLE/'native_physical_dose_contract.json')))
    record('native_physical_dose',dict(status='complete_light_checked',administrative_contract_only=True,scientific_children_open=True))

def offsets_and_slow():
    meta=load(ROOT/'artifacts/data/v2x_seq_spd_main_domain_shared_v1/V2X-Seq-SPD/cooperative/data_info.json');groups=defaultdict(list);offsets={}
    for r in meta:f=str(r['vehicle_frame']);offsets[f]=np.array([r['system_error_offset'][k] for k in ('delta_x','delta_y')]);groups[str(r['vehicle_sequence'])].append(f)
    slow={p:{} for p in dose.PROTOCOLS[5:]}
    for frames in groups.values():
        frames.sort();a=np.array([offsets[f] for f in frames])
        for i,f in enumerate(frames):slow['official_seqmean'][f]=a.mean(0);slow['official_MA5'][f]=a[max(0,i-2):i+3].mean(0)
    return offsets,slow

def wait(path,start):
    while not path.exists():
        if time.monotonic()-start>36*3600:raise TimeoutError('finite36h known input')
        time.sleep(30)

def SPD_one(t,accepted,offsets,out):
    import run_peer07_native_cvh_source_split_ap_v1_20261031 as kernel
    from run_peer10_native_source_aux_v1_20261101 import outcome
    p=AUDIT/t['path'];assert load(p/'terminal.json')['status']=='native_AP_complete_light_checked';records=load(p/'records.json');arms=load(p/'terminal.json')['arms'];panels=[];pop=[];cache={};bounds=t['bounds']
    for w in sorted({r['wait_ms'] for r in records}):
        allrows=[r for r in records if r['wait_ms']==w];rows=[r for r in allrows if (str(r['sequence']),str(r['current_vehicle_frame'])) in accepted];pop.append(dict(wait_ms=w,accepted_conditions=len(rows),original_conditions=len(allrows)))
        if not rows:continue
        stats=defaultdict(lambda:dict(scores=[],tp=[],fp=[],gt=0))
        with ExitStack() as stack:
            z=stack.enter_context(np.load(p/f'wait{w:03d}.npz'));g=stack.enter_context(np.load(ROOT/rows[0]['GT_shard'])) if 'GT_shard' in rows[0] else None
            for j,r in enumerate(rows):
                n=r['shard_case'];gt=g[f"c{r['GT_case']}_gt"] if g is not None else z[f'c{n}_gt'];frame=str(r['current_vehicle_frame']);ref=accepted[str(r['sequence']),frame]
                if frame not in cache:cache[frame]=load(ROOT/'artifacts/data/v2x_seq_spd_main_domain_shared_v1/V2X-Seq-SPD/cooperative/label'/(frame+'.json'))
                codes,error=kernel.source_codes(gt,cache[frame]);assert error<.001;remote=codes==2;anchored=gt[:,:7].copy();theta=ref['yaw'];R=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]]);anchored[remote,:2]=(gt[remote,:2].astype(float)-offsets[frame])@R.T+ref['delta_xy'];anchored[remote,2]+=ref['delta_z'];anchored[remote,6]+=theta
                assert np.array_equal(anchored[~remote],gt[~remote,:7]) and np.array_equal(anchored[:,3:6],gt[:,3:6])
                for roi in ('native','common'):
                    mask=gt[:,7]==0
                    if roi=='common':mask &=kernel.common_mask(gt[:,:7])
                    pp={k:kernel.polygons(v[mask]) for k,v in [('official_quality_subset',gt[:,:7]),('registration_SE2_z_quality_subset',anchored)]}
                    for arm in arms:
                        bb=z[f'c{n}_{arm}_box3d_lidar'];ss=z[f'c{n}_{arm}_scores'];cl=z[f'c{n}_{arm}_label_preds'];keep=(cl==0)&(bb[:,0]>=bounds[0])&(bb[:,0]<=bounds[3])&(bb[:,1]>=bounds[1])&(bb[:,1]<=bounds[4])
                        if roi=='common':keep &=kernel.common_mask(bb)
                        bb=bb[keep];ss=ss[keep];order=np.argsort(-ss);ss=ss[order].astype(float);pol=kernel.polygons(bb[order])
                        for protocol,gg in pp.items():
                            ov=kernel.matrix(pol,gg)
                            for group in ('all','veh','coop','inf','veh_coop'):
                                for th in (.5,.7):
                                    tp,fp,ig,count=outcome(ov,codes[mask],group,th);valid=ig==0;v=stats[protocol,roi,arm,group,th];v['scores'].extend(ss[valid]);v['tp'].extend(tp[valid]);v['fp'].extend(fp[valid]);v['gt']+=count
                if j%128==0:atomic(out/'status.json',dict(created_at=now(),wait_ms=w,conditions=j+1,total=len(rows),status='native_registration_subset'))
        for (protocol,roi,arm,group,th),v in stats.items():panels.append(dict(protocol=protocol,roi=roi,arm=arm,source_group=group,threshold=th,wait_ms=w,**kernel.metric(v['tp'],v['fp'],v['scores'],v['gt'])))
        atomic(out/'panels.json',panels)
    atomic(out/'population.json',pop);atomic(out/'terminal.json',dict(status='complete_light_checked',created_at=now(),producer=ev(p/'terminal.json'),panels=ev(out/'panels.json') if panels else None,population=ev(out/'population.json'),GPU_forwards=0,quality_subset_only=True,not_full_main_AP=True,independent_pending=True))

def V2(out,start):
    import run_peer14_v2_dose_response_v1_20261102 as v2
    old=v2.old;term,accepted=registration.load_registration('V2');c=load(CYCLE/'native_physical_dose_contract.json');frames=old.frame_map(ROOT/load(old.BASE/'contract.json')['inputs']['lineage']['path']);trs={(str(r['split']),str(r['sequence']),str(r['timestamp'])):r for r in load(old.LABELS/'replacement_transforms.json')};classmap=load(old.BASE/'native_population_v6_contract.json')['class_map'];panels=[];pop=[]
    for name in c['V2_native_targets']:
        p=AUDIT/name;records=load(p/'records.json');train=name.rsplit('_',1)[1]
        for role in ('val21','mechanism_holdout14'):
            coverage=term['coverage']['V2__'+role]
            if not coverage['reference_labels_allowed']:pop.append(dict(role=role,training=train,status='coverage_below30_no_labels'));continue
            split='val' if role=='val21' else 'train'
            for w in (0,400):
                full=[r for r in records if r['role']==role and r['wait_ms']==w];rows=[r for r in full if (str(r['sequence']),str(r['timestamp'])) in accepted];pop.append(dict(role=role,training=train,wait_ms=w,accepted_conditions=len(rows),original_conditions=len(full)));stats={a:defaultdict(lambda:dict(scores=[],flags=[],gt=0)) for a in ('PUBLIC','ONLINE_K1')}
                with np.load(p/f'{role}_wait{w:03d}.npz') as z:
                    for r in rows:
                        n=r['shard_case'];g=z[f'c{n}_gt'];gt=g[g[:,7]==0,:7];key=(split,str(r['sequence']),str(r['timestamp']));remote,_=v2.labels(frames[key],trs[key],False,gt,classmap);ref=accepted[key[1:]];theta=ref['yaw'];R=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]]);a=gt.copy();a[remote,:2]=gt[remote,:2]@R.T+ref['delta_xy'];a[remote,2]+=ref['delta_z'];a[remote,6]+=theta;assert np.array_equal(a[~remote],gt[~remote]);labels=dict(official_quality_subset=gt,registration_SE2_z_quality_subset=a)
                        for arm in stats:
                            bb=z[f'c{n}_{arm}_box3d_lidar'];ss=z[f'c{n}_{arm}_scores'];cl=z[f'c{n}_{arm}_label_preds'];keep=(cl==0)&(bb[:,0]>=-100)&(bb[:,0]<=100)&(bb[:,1]>=-40)&(bb[:,1]<=40);bb=bb[keep];ss=ss[keep];order=np.argsort(-ss);old.add(stats[arm],bb[order],ss[order],labels,remote,w,r['sequence'])
                for arm,st in stats.items():panels+=old.finish(st,dict(family='V2_native_TraF',role=role,method=train+'_trained__'+arm,training=train,input=arm))
                atomic(out/'panels.json',panels)
    atomic(out/'population.json',pop);return panels

def run(mode):
    c=load(CYCLE/'native_physical_dose_contract.json');assert c['script']==ev(__file__) and c['dose_source']==ev(dose.__file__);out=BASE/mode;out.mkdir();start=time.monotonic();results=[];record(mode,dict(status='running_native_physical_or_dose_chain'))
    try:
        offsets,slow=offsets_and_slow()
        if mode=='V2_native_registration':V2(out,start)
        else:
            if mode=='SPD_native_registration':
                wait(registration.INPUTS['SPD']/'terminal.json',start);term,accepted=registration.load_registration('SPD')
                # Every SPD native target is val21. Do not create labels below gate.
                if not term['coverage']['SPD__val21']['reference_labels_allowed']:
                    atomic(out/'terminal.json',dict(status='complete_light_checked',created_at=now(),scope='coverage_below30_no_reference_labels',registration=ev(registration.INPUTS['SPD']/'terminal.json')));record(mode,dict(status='complete_light_checked',terminal=ev(out/'terminal.json')));return
            targets=c['new_dose_targets'] if mode=='backbone_dose' else c['SPD_registration_targets']
            for i,t in enumerate(targets):
                sub=out/f'{i:03d}';sub.mkdir()
                try:
                    wait(AUDIT/t['path']/'terminal.json',start)
                    dose.compute(t,offsets,slow,sub) if mode=='backbone_dose' else SPD_one(t,accepted,offsets,sub)
                    results.append(dict(producer=t['path'],terminal=ev(sub/'terminal.json')))
                except Exception as e:atomic(sub/'terminal.json',dict(status='failed_retained',created_at=now(),error=repr(e),traceback=traceback.format_exc()));results.append(dict(producer=t['path'],failure=ev(sub/'terminal.json')))
                atomic(out/'results.json',results)
        status='complete_light_checked' if all('failure' not in r for r in results) else 'failed_retained';atomic(out/'terminal.json',dict(status=status,created_at=now(),results=results,wall_s=time.monotonic()-start,GPU_forwards=0,independent_review_pending=True,no_model_wakeup=True));record(mode,dict(status=status,terminal=ev(out/'terminal.json')))
    except Exception as e:atomic(out/'terminal.json',dict(status='failed_retained',created_at=now(),error=repr(e),traceback=traceback.format_exc()));record(mode,dict(status='failed_retained',terminal=ev(out/'terminal.json')));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','backbone_dose','SPD_native_registration','V2_native_registration']);a=p.parse_args();prepare() if a.mode=='prepare' else run(a.mode)
