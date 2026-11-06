"""Third native backbone, own observations; finite successor, no new KF variant."""
import argparse
import inspect
import json
import os
import sys
import time
import traceback
from peer12_cycle06_v1_20261102 import ROOT, AUTO, AUDIT, CYCLE as C06, atomic, ev, load, now, task, update
from run_peer12_where_confirmation_v1_20261102 import CYCLE as C09, attempt, record
import run_peer12_where_native_pair_v1_20261102 as adapter
import run_peer12_where_native_execution_v6_20261102 as execution

CYCLE = C09/'vit_native'
FRAME = AUTO/'status_audit_20261028_1820/v2xvit_seed2024_task_geometry_fresh39d_v1/producer_v1'
HISTORY = AUTO/'status_audit_20261025_autonomy/current_geometry_detector_seed_bridge_v1/seed2024/history_v1'
WEIGHT = AUTO/'status_audit_20261025_autonomy/v2xvit_seed2024_matched_common_method_v1/inputs/seed2024_net_epoch60.pth'
TID = 'peer12_vit_native_{}_v1_20261102'


def build(root):
    namespace = dict(adapter.__dict__)
    source = inspect.getsource(BASE_BUILD)
    assert source.count('dair_v2x_seq_where2comm.yaml')==1
    source = source.replace('dair_v2x_seq_where2comm.yaml','dair_v2x_seq_v2xvit.yaml')
    source = source.replace("=='where2comm'", "=='v2xvit'")
    exec(compile(source,__file__+'::author_vit_config','exec'),namespace)
    h,builder,roots,config,split = namespace['build'](root)
    h['root_dir'] = str(roots[1])+os.sep
    return h,builder,roots,config,split


BASE_BUILD = adapter.build


def inference_input(batch):
    # Both extra tensors are native ViT metadata, not annotation inputs.
    # Strict loader has identity spatial correction and actual legal ages.
    return {k:batch[k] for k in ('processed_lidar','record_len','pairwise_t_matrix','time_delays','prior_encoding','spatial_correction_matrix')}


def setup():
    execution.setup()
    adapter.build = build
    adapter.inference_input = inference_input
    base_audit = adapter.batch_audit
    def vit_audit(dataset, batch, np):
        result = base_audit(dataset,batch,np)
        prior = batch['prior_encoding'].detach().cpu().numpy()
        spatial = batch['spatial_correction_matrix'].detach().cpu().numpy()
        assert np.isfinite(prior).all() and np.isfinite(spatial).all()
        assert np.allclose(spatial,np.eye(4),rtol=0,atol=1e-7)
        assert np.allclose(prior[0,:len(result['actual_time_delay_frames']),1],result['actual_time_delay_frames'],rtol=0,atol=2e-5)
        return result
    adapter.batch_audit = vit_audit
    c = load(CYCLE/'contract.json')
    assert ev(__file__)==c['script'] and ev(WEIGHT)==c['K4']['initial_weight']
    return c


def prepare():
    import numpy as np
    from run_crosscheckpoint_current_temporal_fixed_readout_controls_v1_20261029 import run_state
    refs = {}
    for name in ('current_handoff.md','main_chat_directive.md'):
        (AUTO/name).read_text()
        refs[name]=ev(AUTO/name)
    CYCLE.mkdir(exist_ok=False)
    out=CYCLE/'inputs'
    out.mkdir()
    inputs={}
    observations=[]
    for role in ('fit24','mechanism_holdout14','val21'):
        framepath=FRAME/(role+'_frame_inputs.npz')
        with np.load(framepath) as archive:
            frame={k:archive[k].copy() for k in archive.files}
        delta,_=run_state(frame,frame['sequence'],frame['wait_ms'],'q005',{})
        ledger=HISTORY/('val_dataset' if role=='val21' else 'train_dataset')/'conditions.jsonl'
        rows=[json.loads(line) for line in ledger.read_text().splitlines()]
        selected=[i for i,w in enumerate(frame['wait_ms']) if role!='fit24' or int(w) in (0,400)]
        cases=[]
        for i in selected:
            row=dict(rows[int(frame['dataset_index'][i])])
            row['receiver_timestamp_us']=int(frame['receiver_timestamp_us'][i])
            assert (row['role'],str(row['sequence']),int(row['wait_ms']))==(role,str(frame['sequence'][i]),int(frame['wait_ms'][i]))
            cases.append(row)
        ledger_out=out/(role+'_conditions.jsonl')
        with ledger_out.open('x') as stream:
            for row in cases: stream.write(json.dumps(row,separators=(',',':'))+'\n')
        material=out/(role+'_identity_delta.npz')
        np.savez_compressed(material,**{k:frame[k][selected] for k in ('sequence','wait_ms','receiver_timestamp_us')},
                            dataset_index=np.arange(len(selected)),q005_delta=delta[selected],q005_base_delta=delta[selected])
        material_names={'fit24':('fit24_material','fit_q005_frame'),'val21':('val21_material','val21_gate_material'),
                        'mechanism_holdout14':('mechanism_holdout14_material','mechanism_holdout14_gate_material')}
        for name in material_names[role]: inputs[name]=ev(material)
        inputs[{'fit24':'fit_conditions','val21':'val_conditions','mechanism_holdout14':'train_conditions'}[role]]=ev(ledger_out)
        observations.extend([ev(framepath),ev(ledger)])
    c=load(C06/'where_execution_contract_v6.json')
    c.update(created_at=now(),script=ev(__file__),inputs=inputs,instructions=refs,own_vit_observations=observations,
             prediction_before_result='TraF positive, Where seed1 hold negative and val null, V2 negative. Test the unchanged q005 correction-aware native training hypothesis on a third actual backbone against its own PUBLIC same-budget fine-tuning. Not a new estimator or a small-effect mechanism revision; no full heading/U claim.',
             input_metadata='Native ViT prior_encoding from strict actual time delays/velocity/type, identity spatial correction; no GT/future in model input.',
             initial_checkpoint_origin='Our completed sensor/uniform full60 ViT seed2024, SHA7f899b8d; not an author public-weight reproduction.',
             estimated_cost=dict(per_arm_updates=2000,per_arm_microbatches=8000,per_arm_scientific_forwards=69024,
                                 per_arm_profile_forwards=408,per_arm_replay_forwards=24,CPU_loader_batches=752,
                                 GPU_wall_h='2-5 per lane after Where chains; actual profile controls pacing',new_disk_GiB='<2 sole predictions/final/active resumes'),
             storage='Own existing observations referenced, only small identity/delta and one native producer per identity. No copied old boxes/GT/head.',no_model_wakeup=True)
    c['K4']['initial_weight']=ev(WEIGHT)
    c['K4']['seed']=20261101
    atomic(CYCLE/'contract.json',c)
    for label in ('qualification','gpu0','gpu1','remote'):
        record(TID.format(label),dict(status='registered_finite_successor',contract=ev(CYCLE/'contract.json'),
               prediction_before_result=c['prediction_before_result'],estimated_cost=c['estimated_cost']))
    record(TID.format('sources'),dict(status='registered_finite_successor',contract=ev(CYCLE/'contract.json'),
        estimated_cost=dict(GPU_forwards=0,CPU_threads=1,CPU_wall_h='4-12 after paired producer',new_disk_GiB='<0.5 flags/metrics')),'K5')


def qualify():
    setup()
    source=inspect.getsource(adapter.qualify)
    changes=[("CYCLE/'where_contract.json'","CYCLE/'contract.json'"),
             ("CYCLE/'where_qualification'","CYCLE/'qualification'"),
             ("CYCLE/'where_inputs'","CYCLE/'inputs'"),
             ("model=train_utils.create_model(h).float();model.load_state_dict(torch.load(WEIGHT,map_location='cpu'),strict=True);del model",'torch.set_num_threads(1)')]
    for before,after in changes:
        assert source.count(before)==1,(before,source.count(before))
        source=source.replace(before,after)
    namespace=dict(adapter.__dict__,CYCLE=CYCLE,WEIGHT=WEIGHT)
    # Qualification's batch_audit must use v6's actual age-list adaptation.
    namespace['update']=lambda *args: None
    exec(compile(source,__file__+'::CPU_loader_only','exec'),namespace)
    try:
        namespace['qualify']()
        proof=load(CYCLE/'qualification/terminal.json')
        proof.update(strict_weight_load=False,strict_weight_load_note='GPU first-step strict model load is pending; CPU qualification only executes loaders/current-GT semantics.',actual_GPU_qualification_pending=True)
        atomic(CYCLE/'qualification/terminal.json',proof)
        record(TID.format('qualification'),dict(status='complete_light_checked',terminal=ev(CYCLE/'qualification/terminal.json')))
    except Exception as error:
        atomic(CYCLE/'qualification/terminal.json',dict(status='failed_retained',error=repr(error),traceback=traceback.format_exc(),GPU_forwards=0,updates=0))
        record(TID.format('qualification'),dict(status='failed_retained',terminal=ev(CYCLE/'qualification/terminal.json')))
        raise


def cell(weight,gpu,out,hold):
    source=inspect.getsource(execution.cell)
    assert source.count("CYCLE/'where_execution_contract_v6.json'")==1
    source=source.replace("CYCLE/'where_execution_contract_v6.json'","CYCLE/'contract.json'")
    namespace=dict(execution.__dict__,CYCLE=CYCLE)
    exec(compile(source,__file__+'::native_vit_evaluator','exec'),namespace)
    namespace['cell'](weight,gpu,out,hold,'PUBLIC,ONLINE_K1,OFFICIAL_REFERENCE')


def lane(label):
    gpu=0 if label in ('gpu0','remote') else 1
    arm={'gpu0':'PUBLIC','gpu1':'ONLINE_K1','remote':'OFFICIAL_REFERENCE'}[label]
    out=CYCLE/label
    out.mkdir(exist_ok=False)
    predecessor='embodied-peer12-where-confirm-'+label+'-v1-20261102.service'
    execution.wait_unit(predecessor,out)
    proof=CYCLE/'qualification/terminal.json'
    while not proof.exists():
        atomic(out/'status.json',dict(status='waiting_actual_CPU_qualification',created_at=now()))
        time.sleep(30)
    try:
        assert load(proof)['status']=='complete_light_checked'
        setup()
        record(TID.format(label),dict(status='running_actual_native_training_then_full_AP',unit='embodied-peer12-vit-native-'+label+'-v1-20261102.service'))
        sys.argv=[__file__,'--workspace',str(ROOT),'--contract',str(CYCLE/'contract.json'),'--output',str(out/'training'),'--arm',arm,'--gpu',str(gpu)]
        namespace=dict(__name__='native_ViT_fixed_pair_train',__file__=__file__)
        exec(compile(adapter.training_source(),__file__+'::native_train','exec'),namespace)
        namespace['main']()
        results=[]
        for role in ('val21','hold14'):
            dest=out/role
            results.append(attempt(dest,cell,out/'training/final.pth',gpu,dest,role=='hold14'))
        status='complete_light_checked' if all(r['status']=='complete_light_checked' for r in results) else 'failed_retained'
        atomic(out/'terminal.json',dict(status=status,created_at=now(),arm=arm,results=results,no_model_wakeup=True))
        record(TID.format(label),dict(status=status,terminal=ev(out/'terminal.json')))
    except Exception as error:
        atomic(out/'terminal.json',dict(status='failed_retained',created_at=now(),error=repr(error),traceback=traceback.format_exc()))
        record(TID.format(label),dict(status='failed_retained',terminal=ev(out/'terminal.json')))
        raise


def sources():
    from run_peer12_native_source_additions_v1_20261101 import compute_namespace
    namespace,contract=compute_namespace()
    namespace['CYCLE']=CYCLE
    out=CYCLE/'sources'
    out.mkdir(exist_ok=False)
    results=[]
    # Local pair; remote reference has its own separate CPU consumer/receipt.
    for label in ('gpu0','gpu1'):
        for role in ('val21','hold14'):
            parent=CYCLE/label
            producer=parent/role/'evaluation'
            while not (producer/'terminal.json').exists() and not (parent/'terminal.json').exists():
                atomic(out/'status.json',dict(status='waiting_finite_native_ViT',producer=str(producer),created_at=now()))
                time.sleep(30)
            try:
                assert load(producer/'terminal.json')['status']=='native_AP_complete_light_checked'
                name=str(producer.relative_to(AUDIT))
                namespace['compute'](name,contract)
                results.append(dict(status='complete_light_checked',terminal=ev(CYCLE/'native_source_aux'/name/'terminal.json')))
            except Exception as error:
                results.append(dict(status='failed_retained',error=repr(error),traceback=traceback.format_exc()))
    status='complete_light_checked' if all(r['status']=='complete_light_checked' for r in results) else 'failed_retained'
    atomic(out/'terminal.json',dict(status=status,created_at=now(),results=results,no_model_wakeup=True))
    record(TID.format('sources'),dict(status=status,terminal=ev(out/'terminal.json')),'K5')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['prepare','qualify','gpu0','gpu1','remote','sources'])
    mode=parser.parse_args().mode
    if mode=='prepare':prepare()
    elif mode=='qualify':qualify()
    elif mode=='sources':sources()
    else:lane(mode)
