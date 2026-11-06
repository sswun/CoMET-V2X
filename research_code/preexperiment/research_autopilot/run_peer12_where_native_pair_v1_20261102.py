"""Where native1 same-budget K4: own q005 state, native fusion/head/readout.

CPU loader qualification precedes finite two-GPU training/evaluation lanes.
No changes to original scripts, source weights, labels, split or running lanes.
"""
import argparse, copy, inspect, os, subprocess, sys, time, traceback
from pathlib import Path
from peer12_cycle06_v1_20261102 import ROOT,AUTO,AUDIT,CYCLE,C03,atomic,ev,load,now,task,update

BASE=ROOT/'preexperiment/research_baselines'
SOURCE=ROOT/'third_party/third_party/related_works/TraF-Align'
WEIGHT=ROOT/'artifacts/research_autopilot/innovation_validation_20261015/frozen_inputs/where_uniform60.pth'
FRAME=ROOT/'artifacts/research_autopilot/status_audit_20261028_1910/where_task_geometry_fresh39d_v1/producer_v1'
HISTORY=ROOT/'artifacts/research_autopilot/status_audit_20261026_where_current_geometry_execution_v1/history_adapter_v4'

def build(root):
    sys.path[:0]=[str(BASE),str(SOURCE)]
    from datasets import build_dataset
    from hypes_yaml import yaml_utils
    config=SOURCE/'hypes_yaml/dair_v2x_seq/Benchmarks/dair_v2x_seq_where2comm.yaml'
    split=SOURCE/'datasets/Basedataset/V2XSeq_dataset_split_official.yaml'
    author=ROOT/'artifacts/data/v2x_seq_spd_traf_align_author21_v1/V2X-Seq-SPD'
    shared=ROOT/'artifacts/data/v2x_seq_spd_main_domain_shared_v1/V2X-Seq-SPD'
    h=yaml_utils.load_yaml(str(config));h.update(root_dir=str(author)+os.sep,split_dir=str(split),delay_ego=0)
    h['train_params'].update(train_batch_size=1,val_batch_size=1)
    h['wild_setting']['agent_i_delay']=0;h['dataset']['augment']=False
    h=yaml_utils.check_pillar_params(h)
    assert h['dataset']['frame_his']==h['dataset']['cav_frame_his']==1
    assert h['model']['core_method']=='where2comm' and h['fusion']['args']['proj_first']
    return h,build_dataset,[author,shared],config,split

def install(ds,np,roots,wait,mode='strict'):
    assert mode=='strict'
    sys.path.insert(0,str(BASE))
    import v2xseq_recorded_arrival_20261009 as old
    from traf_native4_recorded_arrival_20261010 import Catalogue as FullCatalogue
    from evaluate_traf_v2xseq_adapted_fixed60_raw_20261006 import install_corrected_index
    full=FullCatalogue(roots)
    ds.native4_catalogue=full
    class CompleteCatalogue:
        def __init__(self,root):self.root=Path(root);self.info=full.info
        def select(self,sequence,frame,delay):
            decision=int(self.info['vehicle-side'][frame]['pointcloud_timestamp'])
            history=full.history('infrastructure-side',sequence,decision,delay)
            remote=history[0] if history else None
            return dict(protocol_id=old.PROTOCOL_ID,sequence=sequence,effective_label_frame=frame,effective_vehicle_frame=frame,
                effective_infrastructure_frame=remote['frame_id'] if remote else None,vehicle_pointcloud_timestamp_us=decision,
                infrastructure_pointcloud_timestamp_us=remote['sensor_time_us'] if remote else None,configured_wait_ms=delay,
                actual_age_ms=remote['age_ms'] if remote else None,no_packet_fallback=remote is None,
                calibration_policy='published_sensor_extrinsics_only_no_pair_offset')
    s=inspect.getsource(old.install)
    changes=[('Catalogue(root)','CompleteCatalogue(root)'),
        ('catalogue.root / "infrastructure-side" / info["pointcloud_path"]','full.locate("infrastructure-side",info,"pointcloud_path")'),
        ('catalogue.root / "infrastructure-side" / info["calib_virtuallidar_to_world_path"]','full.locate("infrastructure-side",info,"calib_virtuallidar_to_world_path")'),
        ('catalogue.root / "vehicle-side" / ego_info[k]','full.locate("vehicle-side",ego_info,k)'),
        ('dataset.arrival_input_audit = audit','dataset.arrival_input_audit = audit\n        dataset.native4_audit = dict(audit,decision_us=audit["vehicle_pointcloud_timestamp_us"],wait_ms=delay_ms,current_vehicle_frame=audit["effective_vehicle_frame"])')]
    for a,b in changes:assert s.count(a)==1,(a,s.count(a));s=s.replace(a,b)
    ns=dict(old.__dict__,CompleteCatalogue=CompleteCatalogue,full=full)
    exec(compile(s,__file__+'::metadata_complete_strict1','exec'),ns)
    ns['install'](ds,np,install_corrected_index,wait,roots[0])

def batch_audit(ds,batch,np):
    a=ds.native4_audit;paths=[str(p) for p in batch['lidar_path']]
    expected=[a['current_vehicle_frame'],a['effective_infrastructure_frame'] or '']
    assert [Path(p).stem if p else '' for p in paths]==expected
    assert a['effective_label_frame']==a['current_vehicle_frame']
    assert a['no_packet_fallback'] or a['actual_age_ms']>=a['wait_ms']
    assert np.allclose(batch['time_delays'][0].numpy(),a['actual_time_delay_frames'],rtol=0,atol=2e-5)
    assert not a['remote_labels_loaded'] and not a['system_error_offset_used']
    assert len(paths)==2 and batch['record_len'].tolist() in ([1],[2])
    return dict(a,model_input_paths=paths)

def inference_input(batch):
    return {k:batch[k] for k in ('processed_lidar','record_len','pairwise_t_matrix','time_delays')}

def prepare():
    import numpy as np
    for n in ('current_handoff.md','main_chat_directive.md'):(AUTO/n).read_text()
    out=CYCLE/'where_inputs';out.mkdir(exist_ok=False)
    parent=load(AUDIT/'contract_v1.json');refs={};identities=[]
    # Own Where frame-state observations, not TraF cache/state or old method weights.
    s=(ROOT/'preexperiment/research_autopilot/run_crosscheckpoint_current_temporal_fixed_readout_controls_v1_20261029.py').read_text()
    ns={'__name__':'where_q005_materializer','__file__':__file__}
    exec(compile(s,__file__+'::frozen_where_q005','exec'),ns)
    for role in ('fit24','mechanism_holdout14','val21'):
        p=FRAME/(role+'_frame_inputs.npz');identities.append(ev(p))
        with np.load(p) as z:f={k:z[k].copy() for k in z.files}
        delta,_=ns['run_state'](f,f['sequence'],f['wait_ms'],'q005',{})
        split='val' if role=='val21' else 'train'
        ledger=HISTORY/(split+'_dataset/conditions.jsonl');identities.append(ev(ledger))
        rows=[__import__('json').loads(x) for x in ledger.read_text().splitlines()]
        selected=[i for i,w in enumerate(f['wait_ms']) if role!='fit24' or int(w) in (0,400)]
        cases=[]
        for i in selected:
            r=dict(rows[int(f['dataset_index'][i])]);r['receiver_timestamp_us']=int(f['receiver_timestamp_us'][i])
            assert (r['role'],str(r['sequence']),int(r['wait_ms']))==(role,str(f['sequence'][i]),int(f['wait_ms'][i]))
            cases.append(r)
        # jsonl is an immutable, small normalized identity ledger, no boxes/GT/features.
        ledger_out=out/(role+'_conditions.jsonl')
        with ledger_out.open('x') as stream:
            for r in cases:stream.write(__import__('json').dumps(r,separators=(',',':'))+'\n')
        material=out/(role+'_identity_delta.npz')
        np.savez_compressed(material,**{k:f[k][selected] for k in ('sequence','receiver_timestamp_us','wait_ms')},dataset_index=np.arange(len(selected)),q005_delta=delta[selected],q005_base_delta=delta[selected])
        for key in ({'fit24':'fit24_material','val21':'val21_material','mechanism_holdout14':'mechanism_holdout14_material'}[role],):refs[key]=ev(material)
        refs[{'fit24':'fit_q005_frame','val21':'val21_gate_material','mechanism_holdout14':'mechanism_holdout14_gate_material'}[role]]=ev(material)
        refs[{'fit24':'fit_conditions','val21':'val_conditions','mechanism_holdout14':'train_conditions'}[role]]=ev(ledger_out)
    contract=dict(parent,inputs=refs,K4=dict(parent['K4'],arms=['PUBLIC','ONLINE_K1'],initial_weight=ev(WEIGHT)))
    contract.update(created_at=now(),script=ev(__file__),instructions={n:ev(AUTO/n) for n in ('current_handoff.md','main_chat_directive.md')},own_where_observations=identities,
        prediction_before_result='Test whether own-input q005 correction-aware fusion/head adaptation transfers to native Where1 under paired PUBLIC training control. Preserve all signs; no TraF correction/head substitution; no evaluation-based recipe selection.',
        estimated_cost={'per_arm_updates':2000,'per_arm_microbatches':8000,'per_arm_formal_forwards':69048,'profile_forwards':408,'GPU_wall_h_per_arm':'3-7 under thermal quota','new_disk_GiB':'<1.2 for paired final+rolling weights and sole compressed native predictions'},
        source_scripts={n:ev(ROOT/'preexperiment/research_autopilot'/n) for n in ('run_peer10_k4_native_finetune_v1_20261101.py','run_peer10_native_evaluate_v1_20261101.py','run_peer12_native_profile_v1_20261101.py')},
        deployment='strict latest legal single packet, public sensor extrinsics + own causal q005 translation; original Where native scoring/NMS; no GT/future inference',storage='reference own Where observations; compact identity/delta only; retain final/formal predictions and rolling state while active',no_model_wakeup=True)
    atomic(CYCLE/'where_contract.json',contract)
    for arm in ('PUBLIC','ONLINE_K1'):
        update('peer12_where_native_'+arm.lower()+'_v1_20261102',task('K4'),dict(status='registered_after_CPU_qualification',contract=ev(CYCLE/'where_contract.json'),prediction_before_result=contract['prediction_before_result'],estimated_cost=contract['estimated_cost']))
    update('peer12_where_native_CPU_qualification_v1_20261102',task('K4'),dict(status='registered_ready',contract=ev(CYCLE/'where_contract.json'),estimated_cost={'GPU_forwards':0,'CPU_threads':1,'CPU_wall_min':'2-10','new_disk_MiB':10}))

def qualify():
    import numpy as np,torch
    out=CYCLE/'where_qualification';out.mkdir(exist_ok=False);start=time.monotonic();checks=[]
    c=load(CYCLE/'where_contract.json')
    h,builder,roots,_,_=build(ROOT)
    from utils import train_utils
    model=train_utils.create_model(h).float();model.load_state_dict(torch.load(WEIGHT,map_location='cpu'),strict=True);del model
    for role in ('fit24','mechanism_holdout14','val21'):
        cases=[__import__('json').loads(x) for x in (CYCLE/'where_inputs'/(role+'_conditions.jsonl')).read_text().splitlines()]
        for wait in sorted({int(r['wait_ms']) for r in cases}):
            ds=builder(copy.deepcopy(h),set='val' if role=='val21' else 'train');ds.train=False;install(ds,np,roots,wait)
            original=ds.retrieve_base_data
            selected=[r for r in cases if int(r['wait_ms'])==wait];boundaries=[]
            for seq in sorted({r['sequence'] for r in selected}):
                rs=[r for r in selected if r['sequence']==seq];boundaries.extend((rs[0],rs[-1]))
            for r in boundaries:
                np.random.seed(2023+int(r['global_index']));batch=ds.collate_batch([ds[int(r['global_index'])]]);a=batch_audit(ds,batch,np)
                assert (a['sequence'],a['decision_us'])==(str(r['sequence']),int(r['receiver_timestamp_us']))
                # Same current GT under a nonzero left translation, no label exposure to inference.
                before=batch['label_dict'].clone()
                def corrected(index,cur_ego_pose_flag=False):
                    data=original(index,cur_ego_pose_flag=cur_ego_pose_flag)
                    for k,v in data.items():
                        if isinstance(k,int) and not v['ego']:
                            m=np.eye(4);m[:2,3]=[.25,-.125];v['params']['transformation_matrix']=m@np.asarray(v['params']['transformation_matrix'])
                    return data
                ds.retrieve_base_data=corrected;np.random.seed(2023+int(r['global_index']));after=ds.collate_batch([ds[int(r['global_index'])]]);ds.retrieve_base_data=original
                assert torch.equal(before,after['label_dict'])
                checks.append(dict(role=role,sequence=r['sequence'],wait_ms=wait,index=r['global_index'],current_GT_equal=True,legal_packet_checked=True))
            atomic(out/'status.json',dict(status='actual_native_loader_batches',completed_batches=len(checks)*2,role=role,wait_ms=wait,wall_s=time.monotonic()-start))
    assert not torch.cuda.is_initialized()
    atomic(out/'terminal.json',dict(status='complete_light_checked',created_at=now(),checks=checks,strict_weight_load=True,actual_batches=len(checks)*2,GPU_forwards=0,updates=0,wall_s=time.monotonic()-start,contract=ev(CYCLE/'where_contract.json'),not_training_or_AP=True))
    update('peer12_where_native_CPU_qualification_v1_20261102',task('K4'),dict(status='complete_light_checked',terminal=ev(out/'terminal.json')))

def training_source():
    s=(ROOT/'preexperiment/research_autopilot/run_peer10_k4_native_finetune_v1_20261101.py').read_text()
    for a,b in [('from qualify_traf_native4_cpu_v2_20261010 import build','from run_peer12_where_native_pair_v1_20261102 import build'),('import traf_native4_recorded_arrival_20261010 as adapter','import run_peer12_where_native_pair_v1_20261102 as adapter'),('from loss.dtf_point_pillars_loss import DTFPointPillarsLoss','from loss.point_pillars_loss import PointPillarsLoss as DTFPointPillarsLoss'),('torch.set_num_threads(1)','torch.set_num_threads(2)')]:
        assert s.count(a)==1;s=s.replace(a,b)
    return s

def source_for(kind,hold,profile):
    from run_peer12_native_profile_v1_20261101 import adapted_source
    from run_peer10_k4_hold14_native_v1_20261101 import CHANGES
    s=adapted_source() if profile else (ROOT/'preexperiment/research_autopilot/run_peer10_native_evaluate_v1_20261101.py').read_text()
    changes=[('from qualify_traf_native4_cpu_v2_20261010 import build','from run_peer12_where_native_pair_v1_20261102 import build'),('import traf_native4_recorded_arrival_20261010 as adapter','import run_peer12_where_native_pair_v1_20261102 as adapter'),('torch.set_num_threads(1)','torch.set_num_threads(2)')]
    if hold:changes+=CHANGES
    if not profile:changes += [('h,builder,roots,_,_=build(root);official=','h,builder,roots,_,_=build(root);reader=install_reader(cache_enabled);official='),('if refresh.exists():subprocess.run','if False and refresh.exists():subprocess.run')]
    for a,b in changes:assert s.count(a)==1,(a,s.count(a));s=s.replace(a,b)
    return s

def lane(arm,gpu):
    unit='embodied-peer12-factor-gap-gpu0-v1-20261102.service' if gpu==0 else 'embodied-peer12-mixed-factor-gpu1-v1-20261102.service'
    out=CYCLE/('where_'+arm.lower());out.mkdir(exist_ok=False);tid='peer12_where_native_'+arm.lower()+'_v1_20261102'
    while subprocess.check_output(['systemctl','--user','show',unit,'-p','ActiveState'],text=True).strip() in ('ActiveState=active','ActiveState=activating'):
        atomic(out/'status.json',dict(status='waiting_known_GPU_predecessor',unit=unit,created_at=now()));time.sleep(30)
    q=CYCLE/'where_qualification/terminal.json'
    while not q.exists():time.sleep(15)
    assert load(q)['status']=='complete_light_checked'
    update(tid,task('K4'),dict(status='running_native_training_then_profiled_full_evaluation',unit='embodied-peer12-where-'+str(gpu)+'-v1-20261102.service'))
    start=time.monotonic()
    try:
        c=load(CYCLE/'where_contract.json');assert ev(__file__)==c['script']
        sys.argv=[__file__,'--workspace',str(ROOT),'--contract',str(CYCLE/'where_contract.json'),'--output',str(out/'training'),'--arm',arm,'--gpu',str(gpu)]
        ns={'__name__':'where_native_paired_train','__file__':__file__};s=training_source();atomic(out/'training_source_identity.json',{'adapted_from':c['source_scripts']['run_peer10_k4_native_finetune_v1_20261101.py'],'adapter':ev(__file__)})
        exec(compile(s,__file__+'::native1_train','exec'),ns);ns['main']()
        from run_peer12_native_additions_v1_20261102 import run_cell
        from run_peer12_native_profile_v1_20261101 import install_reader,attach_projection_timer
        # Retain the existing 64-condition profile/LRU bitwise gate unchanged.
        s=inspect.getsource(run_cell).replace("str(AUDIT/'contract_v1.json')","str(CYCLE/'where_contract.json')")
        ns=dict(ROOT=ROOT,AUDIT=AUDIT,CYCLE=CYCLE,load=load,atomic=atomic,ev=ev,now=now,time=time,sys=sys,__file__=__file__,source_for=source_for)
        # run_cell only reads c for provenance; it does not use the original lag identities.
        s=s.replace("c=load(CYCLE/'contract.json')","c=load(CYCLE/'where_contract.json')")
        exec(compile(s,__file__+'::native1_profiled_cell','exec'),ns)
        for role in ('val21','hold14'):ns['run_cell']('where',out/'training/final.pth',gpu,out/role,role=='hold14')
        atomic(out/'terminal.json',dict(status='complete_light_checked',created_at=now(),wall_s=time.monotonic()-start,contract=ev(CYCLE/'where_contract.json'),actual_weight=ev(out/'training/final.pth'),no_model_wakeup=True))
        update(tid,task('K4'),dict(status='complete_light_checked',terminal=ev(out/'terminal.json')))
    except Exception as e:
        atomic(out/'terminal.json',dict(status='failed_retained',created_at=now(),error=repr(e),traceback=traceback.format_exc(),wall_s=time.monotonic()-start));update(tid,task('K4'),dict(status='failed_retained',terminal=ev(out/'terminal.json')));raise

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','qualify','public','online']);a=p.parse_args()
    if a.mode=='prepare':return prepare()
    if a.mode=='qualify':
        try:return qualify()
        except Exception as e:
            atomic(CYCLE/'where_qualification/terminal.json',dict(status='failed_retained',created_at=now(),error=repr(e),traceback=traceback.format_exc(),GPU_forwards=0,updates=0))
            update('peer12_where_native_CPU_qualification_v1_20261102',task('K4'),dict(status='failed_retained',terminal=ev(CYCLE/'where_qualification/terminal.json')))
            raise
    lane('PUBLIC' if a.mode=='public' else 'ONLINE_K1',0 if a.mode=='public' else 1)

if __name__=='__main__':main()
