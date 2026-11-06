"""Frozen raw-ICP input reference in TraF; not a deployable estimator.

Cross-perception reference to the already measured CoAlign ICP arms. No K2
variant or fitting, no new native hold14. Original GT is referenced, not copied.
"""
import argparse,sys,time,traceback,subprocess,json
from pathlib import Path
from peer14_cycle12_v1_20261103 import ROOT,AUDIT,CYCLE,C09,C21,REMOTE_ROOT,atomic,ev,load,now,register,record
import run_peer12_native_profile_v1_20261101 as profile
BASE=CYCLE/'traf_physical_reference_v1'
ARM='RAW_ICP_REFERENCE'
GTROOT=AUDIT/'cycle15_peer11_v1/e60_val'

def plans():
    return dict(gpu0=[('seed1_public',AUDIT/'k4_public/final.pth'),('seed2_public',AUDIT/'seed2_public/training/final.pth')],
        gpu1=[('seed1_online',AUDIT/'k4_online/final.pth'),('seed2_online',AUDIT/'seed2_online_k1/training/final.pth')],
        remote=[('e60',ROOT/'artifacts/checkpoints/traf_align_v2xseq_0f6f98de/net_epoch60.pth'),('seed3_public',AUDIT/'cycle15_peer11_v1/seed3_public/training/final.pth'),('seed3_online',AUDIT/'cycle15_peer11_v1/seed3_online/training/final.pth')])

def source(profiling):
    s=profile.adapted_source()
    if not profiling:
        old="selected_all=[r for r in cases if int(r['wait_ms'])==wait];selected=[selected_all[int(j)] for j in np.linspace(0,len(selected_all)-1,16,dtype=int)]"
        assert s.count(old)==1;s=s.replace(old,"selected=[r for r in cases if int(r['wait_ms'])==wait]")
    changes=[("if arm=='K2_SE2':","if arm=='RAW_ICP_REFERENCE':"),
        ('torch.set_num_threads(1)','torch.set_num_threads(2)'),
        ("model.eval();panels=[];records=[]","model.eval();panels=[];records=[];gtindex={(int(r['wait_ms']),int(r['global_index'])):r['shard_case'] for r in load(GTROOT/'records.json')}"),
        ("for wait in (0,100,200,400):","for wait in (0,100,200,400):\n            gtref=np.load(GTROOT/f'wait{wait:03d}.npz')"),
        ("if current_gt is None:current_gt=gt;shard[f'c{ordinal}_gt']=gt","if current_gt is None:\n                        current_gt=gt;assert np.array_equal(gt,gtref[f'c{gtindex[wait,index]}_gt']),('shared GT array mismatch',wait,index)"),
        ("'shard_case':ordinal,'forward_ms':forward_ms","'shard_case':ordinal,'GT_shard':str((GTROOT/f'wait{wait:03d}.npz').relative_to(ROOT)),'GT_case':gtindex[wait,index],'forward_ms':forward_ms"),
        ("del datasets,shard","gtref.close();del datasets,shard"),
        ("'official_reference_annotation_assisted':True","'official_reference_annotation_assisted':False,'raw_ICP_synchronized_nondeployable_reference':True,'scientific_AP':not profiling")]
    for a,b in changes:assert s.count(a)==1,(a,s.count(a));s=s.replace(a,b)
    compile(s,__file__+'::frozen_native_reference','exec');return s

def prepare():
    import numpy as np
    BASE.mkdir(exist_ok=False);source(True);source(False)
    c=load(AUDIT/'contract_v1.json');rows=[json.loads(s) for s in (ROOT/c['inputs']['val_conditions']['path']).read_text().splitlines()]
    with np.load(ROOT/c['inputs']['val21_material']['path']) as z:cases=[rows[int(i)] for i in z['dataset_index']]
    raw=C21/'raw_registration_v1/producer';accepted=set(load(raw/'accepted_indices.json'));field={}
    for r in map(json.loads,(raw/'registrations.jsonl').open()):
        if r['dataset']!='SPD' or r['index'] not in accepted:continue
        D=np.eye(4);a=r['yaw'];D[:2,:2]=[[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]];D[:2,3]=r['delta_xy'];D[2,3]=r['delta_z'];field[str(r['sequence']),str(r['frame'])]=D
    matrices=np.stack([field.get((str(r['sequence']),str(r['current_vehicle_frame'])),np.eye(4)) for r in cases]);assert len(matrices)==13264
    np.savez_compressed(BASE/'raw_ICP_val21_manifest.npz',sequence=np.asarray([str(r['sequence']) for r in cases]),decision_us=np.asarray([r['receiver_timestamp_us'] for r in cases],np.int64),wait_ms=np.asarray([r['wait_ms'] for r in cases],np.int64),global_index=np.asarray([r['global_index'] for r in cases],np.int64),matrix=matrices)
    register('traf_physical_reference',__file__,
        'Does label-independent synchronized ICP improve fixed native perception under official, auxiliary and registration-anchored labels consistently across TraF and CoAlign? CoAlign ICP already loses vsK1 by >0.25pp, TraF is unmeasured. Apply the exact same accepted SE2+z field with PUBLIC fallback to e60 and all three existing PUBLIC/ONLINE paired weights; retain all signs. No new estimator, no fit/cal selection or training. Synchronous current ICP may violate deadline and is not truth or deployable.',
        dict(scientific_forwards=dict(gpu0=26528,gpu1=26528,remote=39792),profile_conditions_per_model=64,profile_and_full_replays_per_model=8,training_updates=0,CPU_threads=2,wall_h='local2-4,remote2-3 after prior unique GPU owner; remote expected beyond15h',new_disk_GiB='<2 unique compressed predictions across7 models; all original GT referenced, no pointcloud/detector cache copies'),
        evaluator=ev(ROOT/'preexperiment/research_autopilot/run_peer10_native_evaluate_v1_20261101.py'),profile_adapter=ev(profile.__file__),native_contract=ev(AUDIT/'contract_v1.json'),manifest=ev(BASE/'raw_ICP_val21_manifest.npz'),registration_terminal=ev(raw/'terminal.json'),registration_rows=ev(raw/'registrations.jsonl'),accepted=ev(raw/'accepted_indices.json'),GT_records=ev(GTROOT/'records.json'),GT_shards={str(w):ev(GTROOT/f'wait{w:03d}.npz') for w in (0,100,200,400)},
        plans={lane:[dict(name=n,weight=ev(w)) for n,w in pp] for lane,pp in plans().items()},ICP_fallback='Rejected/unavailable -> identity residual applied to public source-to-current-ego. Accepted synchronized current-cloud SE2+z applied to each available remote packet. Same field as CoAlign physical/composed reference; no source-time truth claim.',
        native_bounds=load(AUDIT/'cycle03_v1/contract.json')['native_source_aux']['native_eval_range'],existing_reader_profile='Prior native read_parse7-9%<30%, no LRU; new64 profile retained before full',raw_pointcloud_labels_never_used=True,not_new_K2_variant=True)
    atomic(BASE/'build_checks.json',dict(created_at=now(),manifest_conditions=13264,accepted_frame_keys=len(field),accepted_conditions=sum((str(r['sequence']),str(r['current_vehicle_frame'])) in field for r in cases),matrix_bottom_rows_valid=bool(np.all(matrices[:,3,:]==[0,0,0,1])),profile_full_compile=True,GPU_forwards=0))
    for lane in plans():record('traf_physical_'+lane,dict(status='registered_finite_successor',contract=ev(CYCLE/'traf_physical_reference_contract.json')))
    record('traf_physical_reference',dict(status='complete_light_checked',administrative_only=True,scientific_children_open=True))

def evaluate(weight,out,gpu,profiling):
    sys.argv=[__file__,'--workspace',str(ROOT),'--contract',str(AUDIT/'contract_v1.json'),'--output',str(out),'--weight',str(weight),'--gpu',str(gpu),'--arms',ARM,'--se2-manifest',str(BASE/'raw_ICP_val21_manifest.npz')]
    ns=dict(__name__='frozen_traf_ICP_reference',__file__=__file__,install_reader=profile.install_reader,attach_projection_timer=profile.attach_projection_timer,cache_enabled=False,GTROOT=GTROOT,load=load,profiling=profiling)
    exec(compile(source(profiling),__file__+'::reference','exec'),ns);ns['main']()

def run(lane,gpu):
    c=load(CYCLE/'traf_physical_reference_contract.json')
    for k in ('script','evaluator','profile_adapter','manifest','native_contract','GT_records'):assert ev(ROOT/c[k]['path'])==c[k],k
    assert all(ev(ROOT[v['path']])==v for v in c['GT_shards'].values())
    out=BASE/lane;out.mkdir(exist_ok=False);start=time.monotonic();results=[]
    prior='embodied-peer14-c09-graph-ceilings-'+lane+'-v1-20261103.service';record('traf_physical_'+lane,dict(status='waiting_unique_prior_GPU_owner',prior_unit=prior))
    try:
        while subprocess.check_output(['systemctl','--user','show',prior,'-p','ActiveState'],text=True).strip() in ('ActiveState=active','ActiveState=activating','ActiveState=deactivating'):
            if time.monotonic()-start>18*3600:raise TimeoutError('finite18h priorGPU owner wait')
            atomic(out/'status.json',dict(created_at=now(),status='waiting_unique_prior_GPU_owner',prior=prior));time.sleep(30)
        assert load(C09/'coalign_graph_ceilings_v1'/lane/'terminal.json')['status']=='complete_light_checked'
        record('traf_physical_'+lane,dict(status='running_fixed_ICP_nondeployable_reference'))
        for plan in c['plans'][lane]:
            w=ROOT/plan['weight']['path'];assert ev(w)==plan['weight'];dest=out/plan['name'];dest.mkdir()
            for profiling in (True,False):
                target=dest/('evaluation_profile' if profiling else 'evaluation');evaluate(w,target,gpu,profiling)
                assert load(target/'terminal.json')['cases']==(64 if profiling else 13264)
                replays=[load(p) for p in target.glob('replay_*.json')]
                assert len(replays)==4 and all(p['shape_equal'] and all(v==0 for v in p['max_abs_error'].values()) for p in replays)
            atomic(dest/'evaluation/reference_identity.json',dict(created_at=now(),weight=plan['weight'],contract=ev(CYCLE/'traf_physical_reference_contract.json'),noncausal_synchronized_cloud_reference=True,no_GT_in_ICP=True,GT_referenced_only=True,not_deployable=True))
            results.append(ev(dest/'evaluation/terminal.json'));atomic(out/'results.json',results)
        atomic(out/'terminal.json',dict(status='complete_light_checked',created_at=now(),results=results,scientific_forwards=13264*len(results),wall_s=time.monotonic()-start,no_model_wakeup=True));record('traf_physical_'+lane,dict(status='complete_light_checked',terminal=ev(out/'terminal.json')))
    except Exception as e:
        atomic(out/'terminal.json',dict(status='failed_retained',created_at=now(),error=repr(e),traceback=traceback.format_exc(),results=results,wall_s=time.monotonic()-start));record('traf_physical_'+lane,dict(status='failed_retained',terminal=ev(out/'terminal.json')));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','gpu0','gpu1','remote']);p.add_argument('--gpu',type=int,default=0);a=p.parse_args();prepare() if a.mode=='prepare' else run(a.mode,a.gpu)
