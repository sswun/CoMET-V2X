"""Unanswered annotation ceiling decomposition WITH full author graph.

Fixed three seed PUBLIC/ONLINE weights. No new estimator/training, no hold14.
The earlier slow/full ceilings had zero stage1/graph calls and cannot answer it.
"""
import argparse,time,traceback,subprocess
from peer14_cycle09_v1_20261103 import ROOT,AUDIT,CYCLE,C06,C21,atomic,ev,load,now,register,record
import run_peer14_coalign_composed_graph_v1_20261103 as composed
import run_peer14_coalign_upper_completion_v1_20261103 as upper
BASE=CYCLE/'coalign_graph_ceilings_v1'
KINDS=('OFFICIAL_SEQMEAN_GRAPH_REFERENCE','OFFICIAL_MA5_GRAPH_REFERENCE','OFFICIAL_GRAPH_REFERENCE')

def namespace(lane,training,kind):
    official,_=upper.fields();slow=upper.ref.SLOW
    ns=composed.namespace(lane,training,kind)
    def corrected(trans,row,delta):
        result=[x.copy() for x in trans];frame=str(row['current_vehicle_frame'])
        field=official if kind==KINDS[2] else slow['official_seqmean' if kind==KINDS[0] else 'official_MA5']
        if len(result)>1:result[1][:2,3]+=field[frame]
        return result
    out=BASE/lane/training/kind/'evaluation';out.parent.mkdir(parents=True,exist_ok=True)
    ns.update(OUTPUT=out,corrected=corrected,NEW_CONTRACT=CYCLE/'coalign_graph_ceilings_contract.json')
    return ns

def prepare():
    BASE.mkdir(exist_ok=False)
    register('coalign_graph_ceilings',__file__,
        'Does unchanged complete stage1/graph/fusion recover or discard the slow/full annotation offset ceiling? Compare paired existing PUBLIC graph and K1_GRAPH per seed/weight, plus reused no-graph ceilings. Three globally fixed nondeployable reference inputs, no per-cell selection. Both signs retained; this is K5 full-system ceiling analysis, NOT a new K2/K1 mechanism or deployable method.',
        dict(scientific_fusion_forwards_per_lane=79584,training_updates=0,CPU_threads=2,profile_conditions_per_evaluation=64,replay_fusion_forwards_per_evaluation=8,estimated_wall_h_per_lane='3.5-7 after unique current owner; covers12h and beyond',new_disk_GiB='<3 unique compressed prediction/pose/profile per lane; no raw/GT/cache copies'),
        composed_script=ev(composed.__file__),upper_script=ev(upper.__file__),slow=ev(upper.ref.BASE/'slow_offsets_frozen.json'),metadata=ev(ROOT/'artifacts/data/v2x_seq_spd_main_domain_shared_v1/V2X-Seq-SPD/cooperative/data_info.json'),
        weights={l:dict(seed=composed.plans(l)[0],stage=ev(composed.plans(l)[1]),fusion={t:ev(w) for t,w in composed.plans(l)[2].items()}) for l in ('gpu0','gpu1','remote')},
        arms=KINDS,annotation_reference_only=True,stage1_graph_fusion=True,reference_input='annotation-assisted full/sequence-mean/centered MA5 in vehicle lidar XY; future labels used ONLY in named nondeployable reference arms',GT_population='original val21 exact same GT/ROI, official labels unchanged')
    for lane in ('gpu0','gpu1','remote'):record('graph_ceilings_'+lane,dict(status='registered_finite_successor',contract=ev(CYCLE/'coalign_graph_ceilings_contract.json')))
    record('coalign_graph_ceilings',dict(status='complete_light_checked',administrative_only=True,scientific_children_open=True))

def run(lane,gpu):
    c=load(CYCLE/'coalign_graph_ceilings_contract.json')
    for k in ('script','composed_script','upper_script','slow','metadata'):assert ev(ROOT/c[k]['path'])==c[k],k
    _,sw,weights=composed.plans(lane)
    assert ev(sw)==c['weights'][lane]['stage'] and all(ev(w)==c['weights'][lane]['fusion'][t] for t,w in weights.items())
    out=BASE/lane;out.mkdir(exist_ok=False);start=time.monotonic()
    prior='embodied-peer14-c06-composed-'+(lane if lane!='remote' else 'remote-recovery-v2')+'-v1-20261103.service'
    record('graph_ceilings_'+lane,dict(status='waiting_unique_prior_GPU_owner',prior_unit=prior))
    try:
        while True:
            state=subprocess.check_output(['systemctl','--user','show',prior,'-p','ActiveState'],text=True).strip()
            if state not in ('ActiveState=active','ActiveState=activating','ActiveState=deactivating'):break
            if time.monotonic()-start>16*3600:raise TimeoutError('finite16h GPU owner wait')
            atomic(out/'status.json',dict(created_at=now(),status='waiting_unique_prior_GPU_owner',prior=prior));time.sleep(20)
        assert load(composed.BASE/lane/'terminal.json')['status']=='complete_light_checked','prior science must succeed before next GPU owner'
        record('graph_ceilings_'+lane,dict(status='running_annotation_reference_complete_graph'))
        for training in ('public','online'):
            for kind in KINDS:
                ns=namespace(lane,training,kind);ns['evaluate'](gpu,True,True)
                pp=ns['OUTPUT'].with_name('evaluation_profile');v=load(pp/'terminal.json')
                assert v['cases']==64 and v['graph_calls']>0
                replays=[load(p) for p in pp.glob('replay_*.json')]
                assert replays and all(r['shape_equal'] and r['max_box_error']==r['max_score_error']==0 for r in replays)
                ns['evaluate'](gpu,True,False)
                atomic(ns['OUTPUT']/'reference_identity.json',dict(created_at=now(),kind=kind,annotation_reference_only=True,not_deployable=True,stage1_graph_fusion=True,contract=ev(CYCLE/'coalign_graph_ceilings_contract.json'),stage=ev(sw),fusion=ev(weights[training])))
        atomic(out/'terminal.json',dict(status='complete_light_checked',created_at=now(),scientific_fusion_forwards=79584,wall_s=time.monotonic()-start,no_model_wakeup=True))
        record('graph_ceilings_'+lane,dict(status='complete_light_checked',terminal=ev(out/'terminal.json')))
    except Exception as e:
        atomic(out/'terminal.json',dict(status='failed_retained',created_at=now(),error=repr(e),traceback=traceback.format_exc(),wall_s=time.monotonic()-start));record('graph_ceilings_'+lane,dict(status='failed_retained',terminal=ev(out/'terminal.json')));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','gpu0','gpu1','remote']);p.add_argument('--gpu',type=int,default=0);a=p.parse_args();prepare() if a.mode=='prepare' else run(a.mode,a.gpu)
