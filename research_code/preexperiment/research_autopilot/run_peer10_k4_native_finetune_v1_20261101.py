"""Strict native4 current-GT fusion/head adaptation, fixed budget and final AP."""
from __future__ import annotations
import argparse, copy, hashlib, json, os, random, sys, time, traceback
from pathlib import Path
from peer10_tournament_v1_20261101 import atomic, sha, now

def install_correction(ds,np,arm,online,official):
    original=ds.retrieve_base_data
    def retrieve(index,cur_ego_pose_flag=False):
        data=original(index,cur_ego_pose_flag=cur_ego_pose_flag);a=ds.native4_audit
        key=(str(a['sequence']),int(a['decision_us']),int(a['wait_ms']))
        if arm=='ONLINE_K1':
            if key not in online:raise RuntimeError(('online condition missing',key))
            d=online[key]
        elif arm=='OFFICIAL_REFERENCE':
            raw=official[a['current_vehicle_frame']]['system_error_offset'];d=np.asarray([raw['delta_x'],raw['delta_y']],np.float64)
        else:d=np.zeros(2)
        if not a['no_packet_fallback']:
            remote=next(k for k,v in data.items() if isinstance(k,int) and not v['ego'])
            m=np.eye(4);m[:2,3]=d
            data[remote]['params']['transformation_matrix']=m@np.asarray(data[remote]['params']['transformation_matrix'])
        a['k4_correction']={'arm':arm,'delta_xy':d.tolist(),'annotation_assisted':arm=='OFFICIAL_REFERENCE'}
        return data
    ds.retrieve_base_data=retrieve

def main():
    p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);p.add_argument('--contract',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--arm',choices=['PUBLIC','ONLINE_K1','OFFICIAL_REFERENCE'],required=True);p.add_argument('--gpu',type=int,default=0);a=p.parse_args()
    root=a.workspace.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    state={'status':'initializing','arm':a.arm,'created_at':now(),'updates':0,'microbatches':0,'contract_sha256':sha(a.contract)};atomic(out/'status.json',state)
    try:
        import numpy as np
        import torch
        sys.path[:0]=[str(root/'preexperiment/research_baselines'),str(root/'preexperiment/research_autopilot')]
        from qualify_traf_native4_cpu_v2_20261010 import build
        import traf_native4_recorded_arrival_20261010 as adapter
        import traf_align_v2v4real_pilot as common
        if not getattr(torch,'_embodied_thermal_pacing',False):raise RuntimeError('thermal pacing missing')
        contract=json.loads(a.contract.read_text());c=contract['K4'];refs=contract['inputs']
        for name in ('fit24_material','fit_q005_frame','fit_conditions'):
            if sha(root/refs[name]['path'])!=refs[name]['sha256']:raise RuntimeError(('input drift',name))
        with np.load(root/refs['fit24_material']['path']) as z:material={k:z[k].copy() for k in ('sequence','wait_ms','receiver_timestamp_us','dataset_index')}
        with np.load(root/refs['fit_q005_frame']['path']) as z:delta=z['q005_delta'].copy()
        rows=[json.loads(s) for s in (root/refs['fit_conditions']['path']).read_text().splitlines()]
        online={};train_ids=[]
        for i,idx in enumerate(material['dataset_index']):
            r=rows[int(idx)];key=(str(material['sequence'][i]),int(material['receiver_timestamp_us'][i]),int(material['wait_ms'][i]))
            assert key==(str(r['sequence']),int(r['receiver_timestamp_us']),int(r['wait_ms']))
            assert key not in online;online[key]=delta[i].astype(np.float64)
        h,builder,roots,_,_=build(root)
        from utils import train_utils,box_utils,eval_utils
        from loss.dtf_point_pillars_loss import DTFPointPillarsLoss
        torch.set_num_threads(1);torch.cuda.set_device(a.gpu);device=torch.device('cuda',a.gpu)
        random.seed(c['seed']);np.random.seed(c['seed']);torch.manual_seed(c['seed']);torch.cuda.manual_seed_all(c['seed']);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
        official_rows=json.loads((roots[0]/'cooperative/data_info.json').read_text());official={str(r['vehicle_frame']):r for r in official_rows}
        datasets={}
        for wait in c['training_waits_ms']:
            ds=builder(copy.deepcopy(h),set='train');ds.train=False
            # Split remains train46. Only input loading mode is strict/eval, not model mode.
            adapter.install(ds,np,roots,wait,'strict');install_correction(ds,np,a.arm,online,official);datasets[wait]=ds
        ds=datasets[0]
        previous=0
        for scene_id,cumulative in enumerate(ds.len_record):
            scene=ds.scenario_database[scene_id];ego=next(k for k,v in scene.items() if isinstance(k,int) and v['ego']);keys=[k for k in scene[ego] if k not in ('ego','agent')]
            for local,index in enumerate(range(previous,int(cumulative))):
                r=scene[ego][keys[local]];frame=str(r['timestamp']);seq=str(r['sequence']);decision=int(ds.native4_catalogue.info['vehicle-side'][frame]['pointcloud_timestamp'])
                for wait in c['training_waits_ms']:
                    if (seq,decision,wait) in online:train_ids.append((index,wait,seq,decision))
            previous=int(cumulative)
        assert len(train_ids)==len(online),(len(train_ids),len(online))
        train_ids.sort(key=lambda r:(r[2],r[3],r[1]));order=np.random.RandomState(c['seed']).permutation(len(train_ids));schedule=[train_ids[i] for i in order]
        atomic(out/'sampling_identity.json',{'conditions':len(schedule),'sha256':hashlib.sha256(json.dumps(schedule,separators=(',',':')).encode()).hexdigest(),'split':'fit24 within official train46','waits':c['training_waits_ms'],'historical_GT':False,'source_GT':False})
        if sha(root/c['initial_weight']['path'])!=c['initial_weight']['sha256']:raise RuntimeError('initial weight drift')
        model=train_utils.create_model(h).float().to(device);model.load_state_dict(torch.load(root/c['initial_weight']['path'],map_location='cpu'),strict=True)
        for name,param in model.named_parameters():param.requires_grad_(not name.startswith(('reader.','backbone.')))
        model.train();model.reader.eval();model.backbone.eval()
        criterion=DTFPointPillarsLoss(h).to(device)
        optim=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=c['lr']);optim.zero_grad(set_to_none=True)
        fixed_before={n:hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest() for n,v in model.state_dict().items() if n.startswith(('reader.','backbone.'))}
        ledger=(out/'training.jsonl').open('a',buffering=1)
        for micro in range(c['updates']*c['gradient_accumulation']):
            index,wait,seq,decision=schedule[micro%len(schedule)];ds=datasets[wait];np.random.seed(c['seed']+micro)
            batch=ds.collate_batch([ds[index]]);audit=ds.native4_audit
            assert (audit['sequence'],int(audit['decision_us']))==(seq,decision)
            if micro<8:adapter.batch_audit(ds,batch,np)
            # Whitelist only strict model inputs and current GT target; no historical supervision.
            inputs=adapter.inference_input(batch);inputs['label_dict']=batch['label_dict']
            inputs=train_utils.to_device(inputs,device);pred=model(inputs)
            loss=criterion.get_cls_layer_loss(pred)+criterion.get_box_reg_layer_loss(pred)
            if not torch.isfinite(loss):raise RuntimeError(('nonfinite loss',micro))
            value=float(loss.detach());(loss/c['gradient_accumulation']).backward()
            if (micro+1)%c['gradient_accumulation']==0:
                norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],10.,error_if_nonfinite=True);optim.step();optim.zero_grad(set_to_none=True);state['updates']+=1
                ledger.write(json.dumps({'update':state['updates'],'microbatch':micro+1,'loss':value,'gradient_norm':float(norm),'sequence':seq,'decision_us':decision,'wait_ms':wait,'wall_s':time.monotonic()-start})+'\n')
                if state['updates']==4:atomic(out/'qualification.json',{'status':'passed_actual_backward','updates':4,'loss':value,'gradient_norm':float(norm),'not_formal_AP':True,'continuing_fixed_budget':True})
                if state['updates']%100==0:
                    tmp=out/'resume.tmp.pth';torch.save({'model':model.state_dict(),'optimizer':optim.state_dict(),'updates':state['updates'],'torch_rng':torch.get_rng_state(),'numpy_rng':np.random.get_state()},tmp);tmp.replace(out/'resume.pth')
            state.update(status='training',microbatches=micro+1,last_loss=value,wall_s=time.monotonic()-start,cuda_peak_bytes=int(torch.cuda.max_memory_allocated(device)));atomic(out/'status.json',state)
            del inputs,pred,loss,batch;torch.cuda.empty_cache()
        ledger.close()
        fixed_after={n:hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest() for n,v in model.state_dict().items() if n.startswith(('reader.','backbone.'))}
        assert fixed_before==fixed_after,'frozen backbone/reader changed'
        torch.save(model.state_dict(),out/'final.pth');state.update(status='trained_final_fixed_budget',final_sha256=sha(out/'final.pth'));atomic(out/'status.json',state)
        # Evaluation is delegated to the same strict native runner below in the chain.
        atomic(out/'terminal.json',{**state,'status':'training_complete_light_checked','training_wall_s':time.monotonic()-start,'parameter_identity':{'frozen_reader_backbone_exact':True},'evaluation_pending':True})
    except Exception as exc:
        state.update(status='failed_retained',error=repr(exc),traceback=traceback.format_exc(),wall_s=time.monotonic()-start);atomic(out/'status.json',state);atomic(out/'terminal.json',state);raise
if __name__=='__main__':main()
