"""Actual SPD stage1 -> original CoAlign graph -> author multiscale fusion.

Adapted SPD training is explicitly not original DAIR training reproduction.
Only fit sensor labels train stage1; inference has no annotation inputs.
"""
import argparse,copy,json,os,random,subprocess,sys,time,traceback
from pathlib import Path
from peer13_cycle15_v1_20261102 import ROOT,AUTO,AUDIT,CYCLE,atomic,ev,load,now,update,task

SOURCE=ROOT/'third_party/external/CoAlign_64d1987affe4fb881c439cc498c30ab01bec67e8'
ASSETS=AUDIT/'cycle06_peer12_v1_20261102/coalign_author_acquisition'
BASE=CYCLE/'coalign_full_v1'
REMOTE=str(ROOT).startswith('/home/user/')

def record(label,values):
    tid='peer13_coalign_full_'+label+'_v1_20261102'
    if REMOTE:
        p=BASE/'remote_queue_delta.json';rows=load(p) if p.exists() else {}
        rows[tid]=dict(id=tid,parent_task_id=task('K5'),**values);atomic(p,rows)
    else:update(tid,task('K5'),values)

def runtime():
    import numpy as np,torch
    np.float=float;np.int=int;np.bool=bool
    torch.set_num_threads(2)
    sys.path[:0]=[str(SOURCE),str(ROOT/'preexperiment/research_baselines'),str(ROOT/'.conda/coalign-g2o-python-20261010')]
    sys.path.extend([str(ROOT/'.conda/cost-mmcv-20261002'),str(ROOT/'.conda/codyntrust-deps-20261004')])
    from codyntrust_bounded_qualification_20261004 import prepare_runtime
    deps=prepare_runtime(build=False)
    from codyntrust_d3d_math_adapter_20261004 import install_legacy_mathh
    install_legacy_mathh(ROOT/'.conda/codyntrust-native-20261004/d3d_math_v030/codyntrust_d3d_math_v030.so',ROOT/'artifacts/research_autopilot/status_audit_20261004_1200/codyntrust_d3d_v030_offline_review.json')
    from opencood.hypes_yaml.yaml_utils import load_yaml
    cfg=load_yaml(str(SOURCE/'opencood/hypes_yaml/dairv2x/lidar_only_with_noise/coalign/pointpillar_uncertainty.yaml'))
    return np,torch,cfg,deps

def catalogue():
    from traf_native4_recorded_arrival_20261010 import Catalogue
    return Catalogue([ROOT/'artifacts/data/v2x_seq_spd_traf_align_author21_v1/V2X-Seq-SPD',ROOT/'artifacts/data/v2x_seq_spd_main_domain_shared_v1/V2X-Seq-SPD'])

def prepare():
    refs={}
    for name in ('current_handoff.md','main_chat_directive.md'):
        (AUTO/name).read_text();refs[name]=ev(AUTO/name)
    BASE.mkdir(exist_ok=False)
    c=load(AUDIT/'contract_v1.json');fit=ROOT/c['inputs']['fit_conditions']['path']
    rows=[json.loads(s) for s in fit.read_text().splitlines()]
    with __import__('numpy').load(ROOT/c['inputs']['fit24_material']['path']) as z:seqs=set(map(str,z['sequence']))
    sys.path.insert(0,str(ROOT/'preexperiment/research_baselines'));cat=catalogue();sensors=[]
    for side,entries in cat.info.items():
        for frame,info in sorted(entries.items()):
            if str(info['sequence_id']) not in seqs:continue
            p=cat.locate(side,info,'pointcloud_path');label=cat.locate(side,info,'label_lidar_std_path')
            sensors.append(dict(side=side,frame=frame,sequence=str(info['sequence_id']),pointcloud=ev(p),label=ev(label)))
    assert sensors and len(seqs)==24
    atomic(BASE/'fit_sensor_manifest.json',dict(sequences=sorted(seqs),sensors=sensors))
    cfg=SOURCE/'opencood/hypes_yaml/dairv2x/lidar_only_with_noise/coalign/pointpillar_uncertainty.yaml'
    contract=dict(created_at=now(),script=ev(__file__),instructions=refs,fit_manifest=ev(BASE/'fit_sensor_manifest.json'),
        shared_inputs=c['inputs'],stage1_author_config=ev(cfg),fusion_config=ev(ASSETS/'fusion_DAIR/config.yaml'),fusion_weight=ev(ASSETS/'fusion_DAIR/net_epoch_bestval_at13.pth'),
        prediction_before_result='Full uncertainty-stage1+author graph and multiscale fusion may recover PUBLIC pose error or fail under strict asynchronous moving targets. Retain full/no-graph/K1-input arms on identical val21, both label protocols; never substitute cached DAIR index boxes.',
        stage1=dict(seed=20261102,epochs=12,batch=4,optimizer='author Adam lr .001 eps1e-10 wd1e-4',scheduler='adapted multistep epochs8,11 gamma .1',initialization='fresh original PointPillarUncertainty all parameters; no author stage1 checkpoint exists',selection='fixed epoch12, no val/cal/hold selection',augmentation='original DAIR single-agent flip/rotation/scaling',labels='native per-sensor LiDAR vehicle classes, fit24 sequences only',budget_note='Explicit adapted SPD fit budget12 epochs, not author DAIR50epoch reproduction'),
        evaluation=dict(role='val21 only',wait_ms=[0,100,200,400],arms=['COALIGN_FULL','NO_GRAPH','K1_INPUT'],geometry='strict latest legal vehicle/current and infrastructure/arrival packet, public sensor extrinsics only; graph residual SE2 in current receiver frame',K1_identity='existing exact TraF q005 native input, no heading/U substitution',labels=['official','interpolated','no_interpolation'],inference_GT=False,deadline='2026-11-04T12:00:00+08:00; incomplete -> module only'),
        estimated_cost=dict(stage1_sensor_count=len(sensors),stage1_forward_examples=12*(len(sensors)//4)*4,stage1_updates=12*(len(sensors)//4),fusion_scientific_forwards=39792,stage1_eval_forwards='2 per legal condition, repeated packets may cache identity-matched boxes',GPU_wall_h='stage1 8-24 estimated; full eval 2-6, actual status supersedes',new_disk_GiB='<5 final+rolling weights and sole compact predictions; no pointcloud copy'),
        storage='Shared fit manifests/data, val identity ledger and author assets referenced. Preserve epoch12 and active rolling; older epoch model not retained. Official predictions one copy; source/aux consumers zero-forward.',no_model_wakeup=True)
    atomic(BASE/'contract.json',contract)
    for label in ('stage1','evaluation','sources'):
        record(label,dict(status='registered_finite_successor',contract=ev(BASE/'contract.json'),prediction_before_result=contract['prediction_before_result'],estimated_cost=contract['estimated_cost']))

def processor(cfg,train):
    from opencood.data_utils.pre_processor import build_preprocessor
    from opencood.data_utils.post_processor import build_postprocessor
    return build_preprocessor(cfg['preprocess'],train),build_postprocessor(cfg['postprocess'],train)

def sensor_sample(row,cfg,pre,post,anchors,np,augment=None):
    from opencood.utils.pcd_utils import pcd_to_np,shuffle_points,mask_points_by_range,mask_ego_points
    pcd=mask_ego_points(mask_points_by_range(shuffle_points(pcd_to_np(str(ROOT/row['pointcloud']['path']))),cfg['preprocess']['cav_lidar_range']))
    values=[]
    for obj in load(ROOT/row['label']['path']):
        if obj['type'].lower() not in ('car','van','truck','bus'):continue
        x=obj['3d_location'];d=obj['3d_dimensions']
        values.append([x['x'],x['y'],x['z'],d['h'],d['w'],d['l'],obj['rotation']])
    b=np.asarray(values,np.float32).reshape(-1,7);bounds=cfg['postprocess']['gt_range']
    b=b[(b[:,0]>=bounds[0])&(b[:,0]<=bounds[3])&(b[:,1]>=bounds[1])&(b[:,1]<=bounds[4])]
    centres=np.zeros((cfg['postprocess']['max_num'],7),np.float32);mask=np.zeros(len(centres),np.float32)
    assert len(b)<=len(centres);centres[:len(b)]=b;mask[:len(b)]=1
    if augment:
        d=augment.forward(dict(lidar_np=pcd,object_bbx_center=centres,object_bbx_mask=mask))
        pcd,centres,mask=d['lidar_np'],d['object_bbx_center'],d['object_bbx_mask']
    return pre.preprocess(pcd),post.generate_label(gt_box_center=centres,anchors=anchors,mask=mask)

def save_model(path,value,torch):
    tmp=path.with_suffix('.tmp');torch.save(value,tmp);tmp.replace(path)

def stage1(gpu):
    out=BASE/'stage1';out.mkdir(exist_ok=False);start=time.monotonic();steps=0;examples=0
    record('stage1',dict(status='running_actual_fit_uncertainty_training'))
    try:
        np,torch,cfg,deps=runtime();c=load(BASE/'contract.json');assert ev(__file__)==c['script']
        from opencood.tools import train_utils
        from opencood.data_utils.augmentor.data_augmentor import DataAugmentor
        torch.cuda.set_device(gpu);device=torch.device('cuda',gpu)
        seed=c['stage1']['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
        pre,post=processor(cfg,True);anchors=post.generate_anchor_box();augment=DataAugmentor(cfg['data_augment'],True)
        model=train_utils.create_model(cfg).float().to(device);criterion=train_utils.create_loss(cfg).to(device)
        opt=train_utils.setup_optimizer(cfg,model);scheduler=torch.optim.lr_scheduler.MultiStepLR(opt,[8,11],gamma=.1)
        rows=load(BASE/'fit_sensor_manifest.json')['sensors'];rng=np.random.default_rng(seed)
        atomic(out/'identity.json',dict(contract=ev(BASE/'contract.json'),manifest=ev(BASE/'fit_sensor_manifest.json'),dependencies=deps,train_sensor_examples=len(rows),source_file_SHA_verified=all(ev(ROOT/r[k]['path'])==r[k] for r in rows for k in ('pointcloud','label')),no_nonfit_labels=True))
        for epoch in range(12):
            model.train();order=rng.permutation(len(rows));losses=[]
            for k in range(0,len(rows)-3,4):
                selected=[rows[int(i)] for i in order[k:k+4]]
                vals=[sensor_sample(r,cfg,pre,post,anchors,np,augment) for r in selected]
                batch=dict(processed_lidar=pre.collate_batch([v[0] for v in vals]));labels=post.collate_batch([v[1] for v in vals])
                batch=train_utils.to_device(batch,device);labels=train_utils.to_device(labels,device)
                opt.zero_grad(set_to_none=True);prediction=model(batch);loss=criterion(prediction,labels)
                assert torch.isfinite(loss);loss.backward()
                grads=[p.grad for p in model.parameters() if p.grad is not None];assert grads and all(torch.isfinite(x).all() for x in grads)
                assert model.unc_head.weight.grad is not None and torch.isfinite(model.unc_head.weight.grad).all()
                opt.step();steps+=1;examples+=4;losses.append(float(loss.detach()))
                with (out/'train_ledger.jsonl').open('a') as f:f.write(json.dumps(dict(epoch=epoch+1,step=steps,examples=examples,sensors=[dict(side=r['side'],frame=r['frame'],sequence=r['sequence']) for r in selected],loss=losses[-1]))+'\n')
                if steps%16==0 or steps<=4:
                    atomic(out/'status.json',dict(status='training',created_at=now(),epoch=epoch+1,steps=steps,examples=examples,loss=losses[-1],wall_s=time.monotonic()-start))
                if steps%128==0:
                    save_model(out/'rolling.pt',dict(model=model.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),epoch=epoch,order=order,next_k=k+4,steps=steps,examples=examples,python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),sampler_rng=rng.bit_generator.state),torch)
            scheduler.step();save_model(out/'latest_epoch.pth',model.state_dict(),torch)
            atomic(out/'epoch_status.json',dict(completed_epoch=epoch+1,steps=steps,examples=examples,mean_loss=float(np.mean(losses)),weight=ev(out/'latest_epoch.pth'),wall_s=time.monotonic()-start))
        save_model(out/'final.pth',model.state_dict(),torch)
        atomic(out/'terminal.json',dict(status='training_complete_light_checked',created_at=now(),epochs=12,steps=steps,examples=examples,weight=ev(out/'final.pth'),wall_s=time.monotonic()-start,complete_CoAlign=False,not_original_DAIR_reproduction=True,no_model_wakeup=True))
        record('stage1',dict(status='complete_light_checked',terminal=ev(out/'terminal.json')))
    except Exception as error:
        atomic(out/'terminal.json',dict(status='failed_retained',created_at=now(),error=repr(error),traceback=traceback.format_exc(),steps=steps,examples=examples,wall_s=time.monotonic()-start,complete_CoAlign=False));record('stage1',dict(status='failed_retained',terminal=ev(out/'terminal.json')));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','stage1']);p.add_argument('--gpu',type=int,default=0);a=p.parse_args()
    if a.mode=='prepare':prepare()
    else:stage1(a.gpu)
