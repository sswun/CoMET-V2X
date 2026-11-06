"""Unmeasured input-estimator x complete CoAlign composition, fixed weights.

No new estimator or training. K1 exact existing q005 input, K2 existing causal
SE2; raw ICP is synchronized/nondeployable. All feed unchanged author graph.
"""
import argparse,inspect,time,traceback,subprocess
from peer14_cycle06_v1_20261103 import ROOT,AUDIT,CYCLE,C03,C15,C18,atomic,ev,load,now,register,record
import run_peer13_coalign_seed_confirmation_v1_20261102 as confirm
import run_peer14_coalign_physical_inputs_v2_20261103 as physical
BASE=CYCLE/'coalign_composed_graph_v1'
KINDS=('K1_GRAPH','K2_GRAPH','ICP_GRAPH_REFERENCE')

def replace(s,a,b):
    assert s.count(a)==1,(a,s.count(a));return s.replace(a,b)

def plans(lane):
    seed={'gpu0':20261102,'gpu1':20261104,'remote':20261103}[lane]
    if seed==20261102:
        stage=C15/'coalign_full_v1/stage1_v2/final.pth'
        weights={t:C15/'coalign_full_v1'/f for t,f in [('public','fusion_fit/final.pth'),('online','online_fusion_fit/final.pth')]}
    else:
        stage=C18/'coalign_confirmation_v1'/('seed'+str(seed))/'stage1/final.pth'
        weights={t:C18/'coalign_confirmation_v1'/('seed'+str(seed))/(t+'_fit/final.pth') for t in ('public','online')}
    return seed,stage,weights

def namespace(lane,training,kind):
    import numpy as np
    seed,sw,weights=plans(lane);rows,k2,icp=physical.ns['fields']()
    def corrected(trans,row,delta):
        result=[x.copy() for x in trans]
        if len(result)<2:return result
        if kind=='K1_GRAPH':result[1][:2,3]+=delta
        elif kind=='K2_GRAPH':result[1]=k2[str(row['current_vehicle_frame']),int(row['wait_ms'])]@result[1]
        else:result[1]=icp.get((str(row['sequence']),str(row['current_vehicle_frame'])),np.eye(4))@result[1]
        return result
    old=confirm.old;dest=BASE/lane/training/kind/'evaluation';dest.parent.mkdir(parents=True,exist_ok=True)
    ns=dict(old.__dict__,OUTPUT=dest,FUSION_WEIGHT=weights[training],STAGE_WEIGHT=sw,scorers=confirm.fixed.scorers,corrected=corrected,ARM=kind,NEW_CONTRACT=CYCLE/'coalign_composed_graph_contract.json',record=lambda *a:None)
    s=inspect.getsource(old.evaluate)
    changes=[("name='adapted_evaluation' if adapted else 'author_evaluation';out=BASE/(name+('_profile' if profile else ''))","name='composed_graph';out=OUTPUT.with_name(OUTPUT.name+'_profile') if profile else OUTPUT"),
        ("from run_peer07_native_cvh_source_split_ap_v1_20261031 import polygons,matrix,metric\n        from run_peer10_native_source_aux_v1_20261101 import outcome","polygons,matrix,metric,outcome=scorers()"),
        ("load(BASE/'pipeline_contract.json')","load(BASE/'pipeline_contract_v3.json')"),("ev(BASE/'pipeline_contract.json')","ev(NEW_CONTRACT)"),
        ("weight=BASE/'fusion_fit/final.pth' if adapted else stage.ASSETS/'fusion_DAIR/net_epoch_bestval_at13.pth'","weight=FUSION_WEIGHT"),
        ("stageweight=BASE/'stage1_v2/final.pth'","stageweight=STAGE_WEIGHT"),
        ("arms=['COALIGN_FULL','NO_GRAPH','K1_INPUT']","arms=[ARM]"),
        ("public=[x.copy() for x in trans];graph=[x.copy() for x in trans]","original_public=[x.copy() for x in trans];trans=corrected(trans,row,delta[i]);public=[x.copy() for x in trans];graph=[x.copy() for x in trans]"),
        ("**stagecfg['box_align']['args']","**cfg['box_align']['args']"),
        ("corner_to_center(corners.detach().cpu().numpy(),'lwh')","corner_to_center(corners.detach().cpu().numpy(),'hwl')"),
        ("variants=dict(COALIGN_FULL=graph,NO_GRAPH=public,K1_INPUT=k1);gi=","variants={ARM:graph};gi="),
        ("public_T=[x.tolist() for x in public]","public_T=[x.tolist() for x in original_public],estimator_input_T=[x.tolist() for x in public]"),
        ("complete_CoAlign=not profile","complete_CoAlign=not profile,composed_initialization=ARM,ICP_nondeployable=(ARM=='ICP_GRAPH_REFERENCE')")]
    for a,b in changes:s=replace(s,a,b)
    exec(compile(s,__file__+'::fixed_graph_composition','exec'),ns)
    return ns

def prepare():
    BASE.mkdir(exist_ok=False)
    register('coalign_composed_graph',__file__,
        'The prior K2/ICP runs omit stage1 and graph. Test the unanswered composition: exact frozen K1 or K2 initialization then unchanged full CoAlign uncertainty/graph/fusion, plus nondeployable raw-ICP initialization. Compare each with reused same-seed PUBLIC graph and no-graph estimator rows. Three fixed seeds, PUBLIC/ONLINE fair weights, all four val waits. No new K2 variant, no per-cell selection, no new hold inference. Graph can erase, retain or harm estimator benefit.',
        dict(scientific_fusion_forwards_per_lane=79584,training_updates=0,CPU_threads=2,profile_conditions_per_evaluation=64,replay_forwards_per_evaluation=8,estimated_wall_h_per_lane='4-10 including stage1/author graph; six finite evaluations each beyond09h',new_disk_GiB='<3 per lane unique compressed predictions and records; no copied GT/data'),
        evaluator=ev(confirm.old.__file__),scoring=ev(confirm.fixed.__file__),K2_manifest=ev(AUDIT/'k2_k3_formal_resume_v3/K2_val21_native_manifest.npz'),physical_contract=ev(C03/'coalign_physical_inputs_recovery_v2_contract.json'),
        ICP_scope='synchronized current raw cloud reference, potentially after deadline, not deployable and not truth; rejected -> identity residual',
        weights={lane:dict(seed=plans(lane)[0],stage=ev(plans(lane)[1]),fusion={t:ev(w) for t,w in plans(lane)[2].items()}) for lane in ('gpu0','gpu1','remote')})
    for lane in ('gpu0','gpu1','remote'):
        for t in ('public','online'):
            for k in KINDS:namespace(lane,t,k)
        record('composed_'+lane,dict(status='registered_finite_successor',contract=ev(CYCLE/'coalign_composed_graph_contract.json')))
    record('composed_sources',dict(status='registered_finite_successor'))
    record('coalign_composed_graph',dict(status='complete_light_checked',administrative_only=True,scientific_children_open=True))

def run(lane,gpu):
    c=load(CYCLE/'coalign_composed_graph_contract.json');assert c['script']==ev(__file__)
    seed,sw,weights=plans(lane)
    assert c['weights'][lane]['stage']==ev(sw)
    assert all(c['weights'][lane]['fusion'][t]==ev(w) for t,w in weights.items())
    out=BASE/lane;out.mkdir(exist_ok=True);start=time.monotonic();record('composed_'+lane,dict(status='waiting_prior_unique_GPU_owner'))
    try:
        prior='embodied-peer14-c03-physical-inputs-'+('public' if lane=='gpu0' else 'online')+'-recovery-v2-v1-20261103.service'
        if lane!='remote':
            while 'ActiveState=active' in subprocess.check_output(['systemctl','--user','show',prior,'-p','ActiveState'],text=True):
                if time.monotonic()-start>8*3600:raise TimeoutError('finite prior GPU wait')
                time.sleep(20)
        record('composed_'+lane,dict(status='running_fixed_estimator_complete_graph_composition'))
        for training in ('public','online'):
            for kind in KINDS:
                ns=namespace(lane,training,kind);ns['evaluate'](gpu,True,True)
                profile=ns['OUTPUT'].with_name('evaluation_profile')
                assert load(profile/'terminal.json')['cases']==64
                assert all(load(p)['max_box_error']==load(p)['max_score_error']==0 and load(p)['shape_equal'] for p in profile.glob('replay_*.json'))
                ns['evaluate'](gpu,True,False)
        atomic(out/'terminal.json',dict(status='complete_light_checked',created_at=now(),wall_s=time.monotonic()-start,scientific_fusion_forwards=79584,no_model_wakeup=True))
        record('composed_'+lane,dict(status='complete_light_checked',terminal=ev(out/'terminal.json')))
    except Exception as e:
        atomic(out/'terminal.json',dict(status='failed_retained',created_at=now(),error=repr(e),traceback=traceback.format_exc(),wall_s=time.monotonic()-start))
        record('composed_'+lane,dict(status='failed_retained',terminal=ev(out/'terminal.json')));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','gpu0','gpu1','remote']);p.add_argument('--gpu',type=int,default=0);a=p.parse_args()
    prepare() if a.mode=='prepare' else run(a.mode,a.gpu)
