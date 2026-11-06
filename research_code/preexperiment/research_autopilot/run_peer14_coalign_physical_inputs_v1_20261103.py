"""Existing K2 causal SE2 and raw-ICP reference on fixed CoAlign weights.

K2 is unchanged; ICP uses synchronized raw pairs and is NONDEPLOYABLE reference.
No uncertainty or graph calls; do not present reference arms as complete systems.
"""
import argparse,json,time,traceback
from peer14_cycle03_v1_20261103 import ROOT,AUDIT,CYCLE,C00,C21,C18,C15,REMOTE_ROOT,atomic,ev,load,now,register,record
import run_peer14_coalign_reference_completion_v1_20261102 as ref
BASE=CYCLE/'coalign_physical_inputs_v1'

def fields():
    import numpy as np
    rows=ref.confirm.old.conditions(load(ref.confirm.old.BASE/'contract.json'),'val21',np)
    path=AUDIT/'k2_k3_formal_resume_v3/K2_val21_native_manifest.npz'
    result={}
    with np.load(path) as z:
        assert len(rows)==len(z['matrix'])==13264
        for i,r in enumerate(rows):
            assert (str(r['sequence']),int(r['receiver_timestamp_us']),int(r['wait_ms']),int(r['global_index']))==(str(z['sequence'][i]),int(z['decision_us'][i]),int(z['wait_ms'][i]),int(z['global_index'][i]))
            result[str(r['current_vehicle_frame']),int(r['wait_ms'])]=z['matrix'][i].copy()
    raw=C21/'raw_registration_v1/producer';accepted=set(load(raw/'accepted_indices.json'));icp={}
    for r in map(json.loads,(raw/'registrations.jsonl').open()):
        if r['dataset']!='SPD' or r['index'] not in accepted:continue
        theta=r['yaw'];D=np.eye(4);D[:2,:2]=[[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]];D[:2,3]=r['delta_xy'];D[2,3]=r['delta_z'];icp[str(r['sequence']),str(r['frame'])]=D
    return rows,result,icp

def build(weight,out):
    import numpy as np
    rows,k2,icp=fields();ns=ref.build('public','slow');out.parent.mkdir(parents=True,exist_ok=True)
    def variants(public,row):
        frame=str(row['current_vehicle_frame']);key=(str(row['sequence']),frame)
        matrices={'K2_SE2_INPUT':k2[frame,int(row['wait_ms'])],'RAW_ICP_REFERENCE':icp.get(key,np.eye(4))}
        result={}
        for arm,D in matrices.items():
            v=[x.copy() for x in public]
            if len(v)>1:v[1]=D@v[1]
            result[arm]=v
        return result
    ns.update(OUTPUT=out,FUSION_WEIGHT=weight,ARMS=['K2_SE2_INPUT','RAW_ICP_REFERENCE'],make_variants=variants,REFERENCE_CONTRACT=CYCLE/'coalign_physical_inputs_contract.json')
    return ns

def prepare():
    BASE.mkdir(exist_ok=False)
    rows,k2,icp=fields()
    register('coalign_physical_inputs',__file__,
        'Peer14 A branch now passes in both quality-selected val populations: unchanged K2 is reconsidered only as physical estimator. On fixed CoAlign author/three fair PUBLIC/ONLINE weights, measure exact K2 input and label-independent raw-ICP reference; compare reused K1/NO_GRAPH/full graph and official/aux/registration labels. ICP is noncausal synchronized-pair proxy, not deployment. No per-cell/seed selection, no new estimator variant.',
        dict(scientific_forwards_per_local_lane=106112,training_updates=0,CPU_threads=2,profile='64 conditions and same-environment replay each weight',estimated_h='about3 per local lane from actual preceding 99480 in2.65h, plus GPU1 official-weight wait',new_disk_GiB='<3 compressed unique predictions, shared GT only'),
        original_reference=ev(ref.__file__),K2_manifest=ev(AUDIT/'k2_k3_formal_resume_v3/K2_val21_native_manifest.npz'),
        K2_fit_freeze=ev(AUDIT/'k2_k3_formal_resume_v3/fit_freeze.json'),registration=ev(C21/'raw_registration_v1/producer/terminal.json'),
        cached_A_SPD=ev(C21/'registration_read_v1/SPD/terminal.json'),cached_A_V2=ev(C21/'registration_read_v1/V2/terminal.json'),
        ICP_rejected_policy='PUBLIC identity residual on rejected/unavailable synchronized current frame. Full original val population predictions; paired quality-only registration-label AP separately, never official fill on rejected labels.',
        scope='strict SPD val21 only; CoAlign fusion/native readout with TraF-backed existing legal K2 estimation; no independent-CoAlign-observation claim. ICP left upright SE2+z matches A label definition; no roll/pitch residual.',
        no_GT_in_K2=True,ICP_reference_only=True)
    plans={}
    for t in ('public','online'):
        paths=[C15/'coalign_full_v1'/('fusion_fit/final.pth' if t=='public' else 'online_fusion_fit/final.pth')]
        paths += [C18/'coalign_confirmation_v1'/('seed'+str(s))/(t+'_fit/final.pth') for s in (20261103,20261104)]
        if t=='public':paths.insert(0,ref.confirm.stage.ASSETS/'fusion_DAIR/net_epoch_bestval_at13.pth')
        else:paths.append(C00/'coalign_upper_parent_recovery_v2/official_fit/final.pth')
        plans[t]=[dict(name='weight'+str(i),weight=str(p.relative_to(ROOT))) for i,p in enumerate(paths)]
        record('physical_inputs_'+t,dict(status='registered_ready',contract=ev(CYCLE/'coalign_physical_inputs_contract.json')))
    c=load(CYCLE/'coalign_physical_inputs_contract.json');c['plans']=plans;atomic(CYCLE/'coalign_physical_inputs_contract.json',c)
    testrow=next(r for r in rows if r.get('source') is not None)
    import numpy as np
    v=build(ROOT/plans['public'][0]['weight'],BASE/'build_only/evaluation')['make_variants']([np.eye(4),np.eye(4)],testrow)
    assert np.array_equal(v['K2_SE2_INPUT'][1],k2[str(testrow['current_vehicle_frame']),int(testrow['wait_ms'])])
    assert all(np.allclose(a[1][3],[0,0,0,1]) for a in v.items())
    atomic(BASE/'build_checks.json',dict(created_at=now(),manifest_rows_exact=13264,exact_K2_matrix_join=True,homogeneous_composition=True,scientific_GPU_forwards=0,not_CUDA_qualification=True))
    record('coalign_physical_inputs',dict(status='complete_light_checked',administrative_contract_only=True,scientific_children_open=True))

def run(training,gpu):
    c=load(CYCLE/'coalign_physical_inputs_contract.json');assert c['script']==ev(__file__) and c['original_reference']==ev(ref.__file__) and c['K2_manifest']==ev(ROOT/c['K2_manifest']['path'])
    out=BASE/training;out.mkdir(exist_ok=False);start=time.monotonic();record('physical_inputs_'+training,dict(status='running_K2_and_nondeployable_ICP_inputs'))
    try:
        for plan in c['plans'][training]:
            weight=ROOT/plan['weight'];tick=time.monotonic()
            while not weight.exists():
                rel=weight.relative_to(ROOT);dest=weight.parent;dest.mkdir(parents=True,exist_ok=True)
                import subprocess
                if subprocess.run(['ssh','remote-gpu','test','-f',str(REMOTE_ROOT/rel)],capture_output=True).returncode==0:subprocess.run(['rsync','-a','--ignore-existing','remote-gpu:'+str(REMOTE_ROOT/rel),str(dest)+'/'],check=True)
                if time.monotonic()-tick>18*3600:raise TimeoutError('finite18h known OFFICIAL-fit reference dependency')
                if not weight.exists():time.sleep(30)
            p=out/plan['name']/'evaluation';ns=build(weight,p)
            for profile in (True,False):ns['evaluate'](gpu,True,profile)
            atomic(p/'physical_input_identity.json',dict(weight=ev(weight),contract=ev(CYCLE/'coalign_physical_inputs_contract.json'),K2_manifest=c['K2_manifest'],K2_legal_causal=True,ICP_noncausal_synchronized_pointcloud_reference=True,ICP_no_GT=True,not_complete_system=True,stage1_calls=0,graph_calls=0))
        atomic(out/'terminal.json',dict(status='complete_light_checked',created_at=now(),wall_s=time.monotonic()-start,no_model_wakeup=True));record('physical_inputs_'+training,dict(status='complete_light_checked',terminal=ev(out/'terminal.json')))
    except Exception as e:
        atomic(out/'terminal.json',dict(status='failed_retained',created_at=now(),error=repr(e),traceback=traceback.format_exc()));record('physical_inputs_'+training,dict(status='failed_retained',terminal=ev(out/'terminal.json')));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','public','online']);p.add_argument('--gpu',type=int,default=0);a=p.parse_args();prepare() if a.mode=='prepare' else run(a.mode,a.gpu)
