"""Actual author native4 forward with exact SE2 or translation input plugins."""
from __future__ import annotations
import argparse,copy,hashlib,json,subprocess,sys,time,traceback
from pathlib import Path
from peer10_tournament_v1_20261101 import atomic,sha,now

def install(ds,np,wait,corrections,label):
    original=ds.retrieve_base_data
    def retrieve(index,cur_ego_pose_flag=False):
        data=original(index,cur_ego_pose_flag=cur_ego_pose_flag);a=ds.native4_audit;key=(str(a['sequence']),int(a['decision_us']),int(wait))
        if key not in corrections:raise RuntimeError(('correction join miss',key))
        v=corrections[key];assert int(v['index'])==int(a['global_index'])
        if not a['no_packet_fallback']:
            remote=next(k for k,v in data.items() if isinstance(k,int) and not v['ego']);data[remote]['params']['transformation_matrix']=v['matrix']@np.asarray(data[remote]['params']['transformation_matrix'])
        a['peer10_correction']={'label':label,'matrix':v['matrix'].tolist(),'no_packet_fallback':a['no_packet_fallback'],'composition':'exact_SE2_current_ego @ public_source_to_current_ego'}
        return data
    ds.retrieve_base_data=retrieve

def main():
    p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);p.add_argument('--contract',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--weight',type=Path,required=True);p.add_argument('--gpu',type=int,default=0);p.add_argument('--se2-manifest',type=Path);p.add_argument('--arms',default='PUBLIC,ONLINE_K1,OFFICIAL_REFERENCE');a=p.parse_args();root=a.workspace;out=a.output;out.mkdir(exist_ok=False);start=time.monotonic();state={'status':'initializing','forwards':0,'cases':0,'weight_sha256':sha(a.weight),'contract_sha256':sha(a.contract)};atomic(out/'status.json',state)
    try:
        import numpy as np
        import torch
        sys.path[:0]=[str(root/'preexperiment/research_baselines'),str(root/'preexperiment/research_autopilot')]
        from qualify_traf_native4_cpu_v2_20261010 import build
        import traf_native4_recorded_arrival_20261010 as adapter
        import traf_align_v2v4real_pilot as common
        if not getattr(torch,'_embodied_thermal_pacing',False):raise RuntimeError('thermal missing')
        c=json.loads(a.contract.read_text());refs=c['inputs'];arms=a.arms.split(',')
        for name in ('val21_material','val21_gate_material','val_conditions'):
            if sha(root/refs[name]['path'])!=refs[name]['sha256']:raise RuntimeError(('SHA drift',name))
        with np.load(root/refs['val21_material']['path']) as z:identity={k:z[k].copy() for k in ('sequence','receiver_timestamp_us','wait_ms','dataset_index')}
        with np.load(root/refs['val21_gate_material']['path']) as z:delta=z['q005_base_delta'].copy()
        rows=[json.loads(s) for s in (root/refs['val_conditions']['path']).read_text().splitlines()];cases=[]
        for i,idx in enumerate(identity['dataset_index']):
            r=rows[int(idx)];assert (str(r['sequence']),int(r['receiver_timestamp_us']),int(r['wait_ms']))==(str(identity['sequence'][i]),int(identity['receiver_timestamp_us'][i]),int(identity['wait_ms'][i]));cases.append(r)
        assert len(cases)==13264
        h,builder,roots,_,_=build(root);official={str(r['vehicle_frame']):r for r in json.loads((roots[0]/'cooperative/data_info.json').read_text())}
        maps={arm:{} for arm in arms}
        se2={}
        if a.se2_manifest:
            with np.load(a.se2_manifest) as z:
                for seq,stamp,wait,index,matrix in zip(z['sequence'],z['decision_us'],z['wait_ms'],z['global_index'],z['matrix']):se2[(str(seq),int(stamp),int(wait))]={'index':int(index),'matrix':matrix}
            state['se2_manifest_sha256']=sha(a.se2_manifest);assert len(se2)==13264
        for i,r in enumerate(cases):
            key=(str(r['sequence']),int(r['receiver_timestamp_us']),int(r['wait_ms']))
            for arm in arms:
                if arm=='K2_SE2':maps[arm][key]=se2[key];continue
                matrix=np.eye(4)
                if arm=='ONLINE_K1':matrix[:2,3]=delta[i]
                elif arm=='OFFICIAL_REFERENCE':
                    d=official[str(r['current_vehicle_frame'])]['system_error_offset'];matrix[:2,3]=[d['delta_x'],d['delta_y']]
                elif arm!='PUBLIC':raise ValueError(arm)
                maps[arm][key]={'index':int(r['global_index']),'matrix':matrix}
        from utils import train_utils,box_utils,eval_utils
        torch.set_num_threads(1);torch.cuda.set_device(a.gpu);device=torch.device('cuda',a.gpu);torch.manual_seed(2023);np.random.seed(2023);torch.cuda.manual_seed_all(2023);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
        model=train_utils.create_model(h).float().to(device);model.load_state_dict(torch.load(a.weight,map_location='cpu'),strict=True);model.eval();panels=[];records=[]
        weights_before={k:hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest() for k,v in model.state_dict().items()}
        for wait in (0,100,200,400):
            datasets={};stats={};shard={};selected=[r for r in cases if int(r['wait_ms'])==wait]
            for arm in arms:
                ds=builder(copy.deepcopy(h),set='val');adapter.install(ds,np,roots,wait,'strict');install(ds,np,wait,maps[arm],arm);datasets[arm]=ds;stats[arm]=common.blank_stat('vehicle',[.5,.7])
            for ordinal,r in enumerate(selected):
                index=int(r['global_index']);current_gt=None;forward_ms={}
                for arm in arms:
                    ds=datasets[arm];np.random.seed(2023+index);batch=ds.collate_batch([ds[index]]);audit=ds.native4_audit
                    assert (audit['sequence'],int(audit['decision_us']))==(str(r['sequence']),int(r['receiver_timestamp_us']))
                    if ordinal in (0,len(selected)-1):adapter.batch_audit(ds,batch,np)
                    gt=common.extract_frame_gt(torch,batch['label_dict'],h).numpy()
                    if current_gt is None:current_gt=gt;shard[f'c{ordinal}_gt']=gt
                    else:assert np.array_equal(current_gt,gt)
                    def forward():
                        inputs=copy.deepcopy(adapter.inference_input(batch));inputs['label_dict']=torch.empty((0,13),dtype=batch['label_dict'].dtype)
                        with torch.no_grad():_,d=model(train_utils.to_device(inputs,device))
                        return {k:d[0][k].detach().cpu().clone() for k in ('box3d_lidar','scores','label_preds')}
                    torch.cuda.synchronize();tick=time.monotonic();pred=forward();torch.cuda.synchronize();forward_ms[arm]=(time.monotonic()-tick)*1000;state['forwards']+=1
                    if ordinal==0:
                        repeat=forward();state['forwards']+=1;diff={k:float(torch.max(torch.abs(pred[k]-repeat[k]))) if pred[k].numel() and pred[k].shape==repeat[k].shape else 0. if pred[k].shape==repeat[k].shape else None for k in pred}
                        # Numerical differences are reported, not rejected by a bit-exact effect gate.
                        atomic(out/f'replay_wait{wait}_{arm}.json',{'shape_equal':all(pred[k].shape==repeat[k].shape for k in pred),'max_abs_error':diff,'same_environment':True});del repeat
                    assert all(torch.isfinite(v).all() for v in pred.values())
                    stat,_=common.evaluate_frame(torch,box_utils,eval_utils,pred,batch['label_dict'],h,'vehicle',[.5,.7]);common.merge_stat(stats[arm],stat)
                    for k,v in pred.items():shard[f'c{ordinal}_{arm}_{k}']=v.numpy()
                    del batch,pred;torch.cuda.empty_cache()
                records.append({'sequence':r['sequence'],'decision_us':int(r['receiver_timestamp_us']),'current_vehicle_frame':r['current_vehicle_frame'],'wait_ms':wait,'global_index':index,'shard':f'wait{wait:03d}.npz','shard_case':ordinal,'forward_ms':forward_ms})
                state.update(status='evaluating',cases=len(records),wait_ms=wait,wall_s=time.monotonic()-start);atomic(out/'status.json',state)
            np.savez_compressed(out/f'wait{wait:03d}.npz',**shard);panels.append({'wait_ms':wait,'cases':len(selected),'metrics':{arm:common.stat_metrics(eval_utils,stat,'vehicle',[.5,.7]) for arm,stat in stats.items()}});atomic(out/'panels.json',panels);atomic(out/'records.json',records);del datasets,shard
        weights_after={k:hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest() for k,v in model.state_dict().items()};assert weights_before==weights_after
        atomic(out/'terminal.json',{**state,'status':'native_AP_complete_light_checked','created_at':now(),'weight_unchanged':True,'arms':arms,'panel_file_sha256':sha(out/'panels.json'),'forward_GT_input':False,'official_reference_annotation_assisted':True,'source_AP_pending':True})
        # Reporting only: never dispatch an event or language-model wakeup.
        refresh=root/'preexperiment/research_autopilot/refresh_peer10_scoreboard_v1_20261101.py'
        if refresh.exists():subprocess.run([sys.executable,str(refresh)],cwd=root,check=False)
    except Exception as exc:atomic(out/'terminal.json',{**state,'status':'failed_retained','error':repr(exc),'traceback':traceback.format_exc(),'wall_s':time.monotonic()-start});raise
if __name__=='__main__':main()
