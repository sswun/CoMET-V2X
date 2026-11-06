"""Full strict CoAlign input/fusion path and explicitly separate fit adaptation."""
import argparse,copy,json,sys,time,traceback,subprocess
from collections import OrderedDict,defaultdict
from peer13_cycle15_v1_20261102 import ROOT,AUTO,AUDIT,CYCLE,C09,atomic,ev,load,now,update,task
import run_peer13_coalign_full_v1_20261102 as stage

BASE=stage.BASE
REMOTE=stage.REMOTE

def record(label,values):stage.record(label,values)

def prepare():
    for n in ('current_handoff.md','main_chat_directive.md'):(AUTO/n).read_text()
    c=load(BASE/'contract.json')
    spec=dict(created_at=now(),script=ev(__file__),stage1_contract=ev(BASE/'contract.json'),
        stage1_recovery=ev(BASE/'stage1_recovery_v2_contract.json'),
        inference='Actual author DAIR intensity reader /256, per-agent CPU voxelization (proj_first=False), native uncertainty detector, author 1.5m/1000iteration SE2 graph and native multiscale attention/head/NMS. Ego identity fixed; original roll/pitch/z retained, no GT/official offsets in inference.',
        main_identity='Author DAIR fusion checkpoint plus own SPD fit stage1; full strict-SPD adaptation, not original published-DAIR replication.',
        extra_fusion_identity='Separate same-budget PUBLIC fusion adaptation: author DAIR checkpoint, fit24 2000updates x4, AdamW lr1e-4, original classification/regression/direction loss, feature extractor eval/frozen, fusion/shrink/head trainable. Not author training replication, no comparison to unadapted TraF as same budget.',
        paths=dict(local_GT_reference=str((C09/'vit_native_v3/gpu0/val21/evaluation').relative_to(ROOT)),remote_GT_reference=str((C09/'vit_native_v3/remote/val21/evaluation').relative_to(ROOT))),
        estimated_cost=dict(author_fusion_forwards=39792,adapted_fusion_forwards=39792,stage1_eval_forwards='2 per legal condition before cache reuse',adapted_updates=2000,adapted_microbatches=8000,profile_conditions=64,wall_h='1-4 evaluation each lane + .5-2 fit adaptation; stage1 finite dependency',new_disk_GiB='<4 sole predictions/no copied GT; final and active resume'),
        storage='GT references sole existing seed1 ViT native current-frame GT; labels are never a model input. Raw lidar not cached. Identical stage1 packet predictions cached in memory by side/frame with max256 entries; no new pointcloud/feature duplication.',
        prediction_before_result=c['prediction_before_result'],no_model_wakeup=True)
    atomic(BASE/'pipeline_contract.json',spec)
    for label in ('author_evaluation','fusion_fit_and_evaluation','author_sources','adapted_sources'):
        record(label,dict(status='registered_finite_successor',contract=ev(BASE/'pipeline_contract.json'),prediction_before_result=spec['prediction_before_result'],estimated_cost=spec['estimated_cost']))

def config(np,torch):
    from opencood.hypes_yaml.yaml_utils import load_yaml
    cfg=load_yaml(str(stage.ASSETS/'fusion_DAIR/config.yaml'))
    return cfg

def conditions(c,role,np):
    key=role+'_material';m=stage.ROOT/c['shared_inputs'][key]['path']
    with np.load(m) as z:ids={k:z[k].copy() for k in ('sequence','wait_ms','receiver_timestamp_us','dataset_index')}
    ledger=ROOT/c['shared_inputs']['fit_conditions' if role=='fit24' else 'val_conditions']['path']
    rows=[json.loads(s) for s in ledger.read_text().splitlines()];result=[]
    for seq,w,t,idx in zip(ids['sequence'],ids['wait_ms'],ids['receiver_timestamp_us'],ids['dataset_index']):
        r=rows[int(idx)];assert (str(r['sequence']),int(r['wait_ms']),int(r['receiver_timestamp_us']))==(str(seq),int(w),int(t));result.append(r)
    return result

def pair(row,cat,cfg,pre,np):
    from opencood.utils.pcd_utils import read_pcd,shuffle_points,mask_points_by_range,mask_ego_points
    histories={side:cat.history(side,str(row['sequence']),int(row['receiver_timestamp_us']),int(row['wait_ms']) if side=='infrastructure-side' else 0)[:1] for side in ('vehicle-side','infrastructure-side')}
    assert histories['vehicle-side'][0]['frame_id']==str(row['current_vehicle_frame'])
    pc=[];trans=[];ident=[];ego_pose=np.asarray(row['current_ego_pose_lidar_to_world'],float)
    for side,h in histories.items():
        if not h:continue
        loaded=cat.load(side,h,np)[0];pose=np.asarray(loaded['source_pose']);T=np.linalg.solve(ego_pose,pose)
        if side=='vehicle-side':assert np.allclose(T,np.eye(4),atol=1e-8)
        else:
            assert loaded['sensor_time_us']<=int(row['receiver_timestamp_us'])-int(row['wait_ms'])*1000
            s=row.get('source');assert s is not None and str(s['source_frame'])==loaded['frame_id']
        raw=read_pcd(loaded['path'])[0];raw=mask_ego_points(mask_points_by_range(shuffle_points(raw),cfg['preprocess']['cav_lidar_range']))
        pc.append(pre.preprocess(raw));trans.append(T);ident.append(dict(side=side,frame=loaded['frame_id'],timestamp_us=loaded['sensor_time_us'],pointcloud_SHA256=loaded['pointcloud_sha256']))
    assert len(pc)==(2 if row.get('source') else 1)
    return pc,trans,ident

def model_input(pc,trans,pre,np,torch,device):
    from opencood.tools import train_utils
    matrix=np.tile(np.eye(4),(1,2,2,1,1))
    for i in range(len(trans)):
        for j in range(len(trans)):matrix[0,i,j]=np.linalg.solve(trans[j],trans[i])
    data=dict(processed_lidar=pre.collate_batch(pc),record_len=torch.tensor([len(pc)],dtype=torch.long),pairwise_t_matrix=torch.from_numpy(matrix).float())
    return train_utils.to_device(data,device)

def native_labels(m,index,cfg,post,anchors,np):
    a,b=map(int,m['gt_offsets'][index:index+2]);boxes=m['gt_boxes'][a:b].copy();bounds=cfg['postprocess']['gt_range']
    boxes=boxes[(boxes[:,0]>=bounds[0])&(boxes[:,0]<=bounds[3])&(boxes[:,1]>=bounds[1])&(boxes[:,1]<=bounds[4])]
    padded=np.zeros((max(len(boxes),cfg['postprocess']['max_num']),7),np.float32);mask=np.zeros(len(padded),np.float32);padded[:len(boxes)]=boxes;mask[:len(boxes)]=1
    return post.collate_batch([post.generate_label(gt_box_center=padded,anchors=anchors,mask=mask)])

def fit(gpu):
    out=BASE/'fusion_fit';out.mkdir(exist_ok=False);start=time.monotonic();updates=0;micro=0
    try:
        np,torch,_,deps=stage.runtime();cfg=config(np,torch);c=load(BASE/'contract.json')
        from opencood.tools import train_utils
        torch.cuda.set_device(gpu);device=torch.device('cuda',gpu);seed=20261102
        torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);np.random.seed(seed)
        pre,post=stage.processor(cfg,False);anchors=post.generate_anchor_box();cat=stage.catalogue();rows=conditions(c,'fit24',np)
        with np.load(ROOT/c['shared_inputs']['fit24_material']['path']) as z:m={k:z[k].copy() for k in ('gt_boxes','gt_offsets')}
        model=train_utils.create_model(cfg).float().to(device);model.load_state_dict(torch.load(stage.ASSETS/'fusion_DAIR/net_epoch_bestval_at13.pth',map_location='cpu'),strict=True)
        for name,p in model.named_parameters():p.requires_grad_(name.startswith(('fusion_net.','shrink_conv.','cls_head.','reg_head.','dir_head.')))
        criterion=train_utils.create_loss(cfg).to(device);optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4)
        order=np.random.default_rng(seed).permutation(len(rows));record('fusion_fit_and_evaluation',dict(status='running_PUBLIC_same_budget_fusion_fit'))
        for step in range(2000):
            model.eval()
            for name in ('fusion_net','shrink_conv','cls_head','reg_head','dir_head'):
                if hasattr(model,name):getattr(model,name).train()
            optimizer.zero_grad(set_to_none=True);losses=[]
            for sub in range(4):
                i=int(order[micro%len(order)]);np.random.seed(2023+int(rows[i]['global_index']))
                pc,trans,ident=pair(rows[i],cat,cfg,pre,np);batch=model_input(pc,trans,pre,np,torch,device)
                labels=train_utils.to_device(native_labels(m,i,cfg,post,anchors,np),device)
                output=model(batch);loss=criterion(output,labels);assert torch.isfinite(loss);(loss/4).backward();losses.append(float(loss.detach()));micro+=1
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None);optimizer.step();updates+=1
            with (out/'train_ledger.jsonl').open('a') as f:f.write(json.dumps(dict(updates=updates,microbatches=micro,loss=losses))+'\n')
            if updates%16==0 or updates<=4:atomic(out/'status.json',dict(status='same_budget_PUBLIC_training',created_at=now(),updates=updates,microbatches=micro,wall_s=time.monotonic()-start))
            if updates%128==0:stage.save_model(out/'rolling.pt',dict(model=model.state_dict(),optimizer=optimizer.state_dict(),updates=updates,microbatches=micro),torch)
        stage.save_model(out/'final.pth',model.state_dict(),torch)
        atomic(out/'terminal.json',dict(status='training_complete_light_checked',created_at=now(),updates=updates,microbatches=micro,weight=ev(out/'final.pth'),wall_s=time.monotonic()-start,no_model_wakeup=True))
    except Exception as error:
        atomic(out/'terminal.json',dict(status='failed_retained',created_at=now(),error=repr(error),traceback=traceback.format_exc(),updates=updates,microbatches=micro,wall_s=time.monotonic()-start));raise

def evaluate(gpu,adapted=False,profile=False):
    name='adapted_evaluation' if adapted else 'author_evaluation';out=BASE/(name+('_profile' if profile else ''));out.mkdir(exist_ok=False);start=time.monotonic();forwards=0
    try:
        np,torch,stagecfg,deps=stage.runtime();cfg=config(np,torch);c=load(BASE/'contract.json');contract=load(BASE/'pipeline_contract.json')
        from opencood.tools import train_utils
        from opencood.utils import box_utils
        from opencood.utils.transformation_utils import tfm_to_pose,pose_to_tfm
        from opencood.models.sub_modules.box_align_v2 import box_alignment_relative_sample_np
        from run_peer07_native_cvh_source_split_ap_v1_20261031 import polygons,matrix,metric
        from run_peer10_native_source_aux_v1_20261101 import outcome
        torch.cuda.set_device(gpu);device=torch.device('cuda',gpu);torch.manual_seed(2023);np.random.seed(2023);torch.cuda.manual_seed_all(2023)
        torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
        pre,post=stage.processor(cfg,False);_,stagepost=stage.processor(stagecfg,False)
        anchors=torch.from_numpy(post.generate_anchor_box()).to(device);stageanchors=torch.from_numpy(stagepost.generate_anchor_box()).to(device)
        fusion=train_utils.create_model(cfg).float().to(device);weight=BASE/'fusion_fit/final.pth' if adapted else stage.ASSETS/'fusion_DAIR/net_epoch_bestval_at13.pth'
        fusion.load_state_dict(torch.load(weight,map_location='cpu'),strict=True);fusion.eval()
        detector=train_utils.create_model(stagecfg).float().to(device);stageweight=BASE/'stage1_v2/final.pth';detector.load_state_dict(torch.load(stageweight,map_location='cpu'),strict=True);detector.eval()
        cat=stage.catalogue();rows=conditions(c,'val21',np);assert len(rows)==13264
        gtroot=ROOT/contract['paths']['remote_GT_reference' if REMOTE else 'local_GT_reference'];gtrecords=load(gtroot/'records.json');gtindex={(int(r['wait_ms']),int(r['global_index'])):int(r['shard_case']) for r in gtrecords};assert len(gtindex)==len(rows)
        with np.load(ROOT/c['shared_inputs']['val21_gate_material']['path']) as z:delta=z['q005_base_delta'].copy()
        cache=OrderedDict();arms=['COALIGN_FULL','NO_GRAPH','K1_INPUT'];records=[];panels=[];profile_rows=[];graph_calls=0
        atomic(out/'input_identity.json',dict(contract=ev(BASE/'pipeline_contract.json'),fusion_weight=ev(weight),stage1_weight=ev(stageweight),GT_records=ev(gtroot/'records.json'),GT_shards={str(w):ev(gtroot/f'wait{w:03d}.npz') for w in (0,100,200,400)},GT_not_in_model_input=True,raw_reader='author DAIR read_pcd intensity/256',proj_first=False))
        for wait in (0,100,200,400):
            ids=[i for i,r in enumerate(rows) if int(r['wait_ms'])==wait]
            if profile:ids=[ids[int(k)] for k in np.linspace(0,len(ids)-1,16,dtype=int)]
            shard={};stats=defaultdict(lambda:dict(tp=[],fp=[],scores=[],gt=0));gtfile=gtroot/f'wait{wait:03d}.npz'
            with np.load(gtfile) as ground:
                for ordinal,i in enumerate(ids):
                    row=rows[i];np.random.seed(2023+int(row['global_index']));tick=time.monotonic();pc,trans,ident=pair(row,cat,cfg,pre,np);data_s=time.monotonic()-tick
                    public=[x.copy() for x in trans];graph=[x.copy() for x in trans];tick=time.monotonic();stagecorners=[];unc=[]
                    for j,source in enumerate(ident):
                        key=(source['side'],source['frame'])
                        if key in cache:value=cache.pop(key);cache[key]=value
                        else:
                            single=model_input([pc[j]],[np.eye(4)],pre,np,torch,device)
                            with torch.no_grad():out1=detector(single);corners,boxes,uncertainty=stagepost.post_process_stage1(out1,stageanchors)
                            value=(np.empty((0,8,3)),np.empty((0,3))) if corners is None else (corners[0].detach().cpu().numpy(),uncertainty[0].detach().cpu().numpy())
                            cache[key]=value
                            while len(cache)>256:cache.popitem(last=False)
                        stagecorners.append(value[0]);unc.append(value[1])
                    stage_s=time.monotonic()-tick;tick=time.monotonic()
                    if len(public)>1 and all(len(x) for x in stagecorners):
                        poses=np.asarray([tfm_to_pose(x) for x in public]);refined=box_alignment_relative_sample_np(stagecorners,poses.copy(),uncertainty_list=unc,**stagecfg['box_align']['args']);graph_calls+=1
                        poses[:,[0,1,4]]=refined;converted=pose_to_tfm(poses)
                        graph=[np.linalg.solve(converted[0],x) for x in converted]
                    graph_s=time.monotonic()-tick;k1=[x.copy() for x in public]
                    if len(k1)>1:k1[1][:2,3]+=delta[i]
                    variants=dict(COALIGN_FULL=graph,NO_GRAPH=public,K1_INPUT=k1);gi=gtindex[wait,int(row['global_index'])];gt=ground[f'c{gi}_gt']
                    # GT is a sole shared reference, not stored in new shards.
                    predtimes={}
                    for arm,transforms in variants.items():
                        inputs=model_input(pc,transforms,pre,np,torch,device)
                        def forward():
                            with torch.no_grad():output=fusion(inputs);corners,scores=post.post_process({'ego':dict(transformation_matrix=torch.eye(4,device=device),anchor_box=anchors)},{'ego':output})
                            if corners is None:return np.empty((0,7),np.float32),np.empty(0,np.float32)
                            return box_utils.corner_to_center(corners.detach().cpu().numpy(),'lwh').astype(np.float32),scores.detach().cpu().numpy()
                        tick=time.monotonic();boxes,scores=forward();torch.cuda.synchronize();predtimes[arm]=time.monotonic()-tick;forwards+=1
                        if ordinal==0:
                            bb,ss=forward();forwards+=1
                            atomic(out/f'replay_wait{wait}_{arm}.json',dict(shape_equal=bb.shape==boxes.shape and ss.shape==scores.shape,max_box_error=float(np.max(np.abs(bb-boxes),initial=0)) if bb.shape==boxes.shape else None,max_score_error=float(np.max(np.abs(ss-scores),initial=0)) if ss.shape==scores.shape else None,same_environment=True))
                        assert np.isfinite(boxes).all() and np.isfinite(scores).all()
                        for key,value in [('box3d_lidar',boxes),('scores',scores),('label_preds',np.zeros(len(boxes),np.int64))]:shard[f'c{ordinal}_{arm}_{key}']=value
                        bounds=[-100.8,-40,-3.5,100.8,40,1.5];keep=(boxes[:,0]>=bounds[0])&(boxes[:,0]<=bounds[3])&(boxes[:,1]>=bounds[1])&(boxes[:,1]<=bounds[4]);order=np.argsort(-scores[keep]);pred=polygons(boxes[keep][order]);s=scores[keep][order];ov=matrix(pred,polygons(gt[gt[:,7]==0,:7]))
                        for threshold in (.5,.7):
                            tp,fp,ig,n=outcome(ov,np.zeros(ov.shape[1],np.int8),'all',threshold);v=stats[arm,threshold];v['tp'].extend(tp);v['fp'].extend(fp);v['scores'].extend(s);v['gt']+=n
                    records.append(dict(sequence=row['sequence'],current_vehicle_frame=row['current_vehicle_frame'],decision_us=int(row['receiver_timestamp_us']),wait_ms=wait,global_index=int(row['global_index']),shard_case=ordinal,GT_shard=str(gtfile.relative_to(ROOT)),GT_case=gi,input_packets=ident,public_T=[x.tolist() for x in public],graph_T=[x.tolist() for x in graph]))
                    profile_rows.append(dict(wait_ms=wait,index=i,data_total_s=data_s,stage1_s=stage_s,graph_s=graph_s,forward_s=predtimes))
                    if ordinal%32==0:atomic(out/'status.json',dict(status='full_system_evaluating',created_at=now(),cases=len(records),forwards=forwards,wait_ms=wait,graph_calls=graph_calls,wall_s=time.monotonic()-start))
            np.savez_compressed(out/f'wait{wait:03d}.npz',**shard)
            metrics={arm:dict(global_score_sort={f'ap{int(t*100)}':metric(stats[arm,t]['tp'],stats[arm,t]['fp'],stats[arm,t]['scores'],stats[arm,t]['gt'])['ap']/100 for t in (.5,.7)}) for arm in arms}
            panels.append(dict(wait_ms=wait,cases=len(ids),metrics=metrics));atomic(out/'panels.json',panels);atomic(out/'records.json',records)
        atomic(out/'profile_rows.json',profile_rows)
        atomic(out/'terminal.json',dict(status='native_AP_complete_light_checked',created_at=now(),cases=len(records),forwards=forwards,graph_calls=graph_calls,arms=arms,complete_CoAlign=not profile,scientific_AP=not profile,GT_shared_reference=True,wall_s=time.monotonic()-start,no_model_wakeup=True))
        if not profile:record('adapted_evaluation' if adapted else 'author_evaluation',dict(status='complete_light_checked',terminal=ev(out/'terminal.json')))
    except Exception as error:
        atomic(out/'terminal.json',dict(status='failed_retained',created_at=now(),error=repr(error),traceback=traceback.format_exc(),forwards=forwards,wall_s=time.monotonic()-start,complete_CoAlign=False));raise

def wait_stage1(out):
    while not (BASE/'stage1_v2/terminal.json').exists():
        if not REMOTE:
            subprocess.run(['rsync','-a','--include=terminal.json','--include=final.pth','--exclude=*','remote-gpu:/path/to/remote_project/'+str((BASE/'stage1_v2').relative_to(ROOT))+'/',str(BASE/'stage1_v2')+'/'],check=True)
        atomic(out/'status.json',dict(status='waiting_finite_stage1_final',created_at=now()));time.sleep(30)
    assert load(BASE/'stage1_v2/terminal.json')['status']=='training_complete_light_checked'

def lane(mode,gpu):
    out=BASE/(mode+'_lane');out.mkdir(exist_ok=False)
    try:
        if mode=='adapted':
            from run_peer12_vit_confirmation_v1_20261102 import wait_unit
            wait_unit('embodied-peer13-vit-gpu1-v2-20261102.service',out);fit(gpu)
        wait_stage1(out);evaluate(gpu,mode=='adapted',True);evaluate(gpu,mode=='adapted',False)
        atomic(out/'terminal.json',dict(status='complete_light_checked',created_at=now(),no_model_wakeup=True))
    except Exception as error:
        atomic(out/'terminal.json',dict(status='failed_retained',created_at=now(),error=repr(error),traceback=traceback.format_exc()));record('fusion_fit_and_evaluation' if mode=='adapted' else 'author_evaluation',dict(status='failed_retained',terminal=ev(out/'terminal.json')));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','author','adapted']);p.add_argument('--gpu',type=int,default=0);a=p.parse_args()
    prepare() if a.mode=='prepare' else lane(a.mode,a.gpu)
