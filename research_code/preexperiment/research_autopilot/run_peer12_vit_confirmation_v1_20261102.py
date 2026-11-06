"""Fixed paired ViT seeds and exact no-training native baseline, finite lanes."""
import argparse
import copy
import inspect
import subprocess
import sys
import time
import traceback
from peer12_cycle12_v1_20261102 import ROOT,AUTO,AUDIT,C09,CYCLE,atomic,ev,load,now,update,task
import run_peer12_vit_native_execution_v3_20261102 as current
import run_peer12_where_native_execution_v6_20261102 as evaluator
import run_peer12_where_native_pair_v1_20261102 as adapter

REMOTE=str(ROOT).startswith('/home/user/')
BASE=CYCLE/'vit_confirmation'
TID='peer12_vit_confirmation_{}_v1_20261102'


def record(label,values,candidate='K4'):
    tid=TID.format(label)
    if REMOTE:
        path=CYCLE/'vit_confirmation_remote_queue_delta.json'
        rows=load(path) if path.exists() else {}
        rows[tid]=dict(id=tid,parent_task_id=task(candidate),**values)
        atomic(path,rows)
    else:
        update(tid,task(candidate),values)


def prepare():
    instructions={}
    for name in ('current_handoff.md','main_chat_directive.md'):
        (AUTO/name).read_text();instructions[name]=ev(AUTO/name)
    BASE.mkdir(exist_ok=False)
    c=copy.deepcopy(load(current.CYCLE/'contract.json'))
    dependencies=[ev(ROOT/'preexperiment/research_autopilot'/name) for name in (
        'run_peer12_vit_confirmation_v1_20261102.py','run_peer12_vit_native_execution_v3_20261102.py',
        'run_peer12_vit_native_pair_v1_20261102.py','run_peer12_where_native_execution_v6_20261102.py',
        'run_peer12_where_native_pair_v1_20261102.py','run_peer10_k4_native_finetune_v1_20261101.py',
        'run_peer10_native_evaluate_v1_20261101.py','run_peer12_native_profile_v1_20261101.py')]
    c.update(created_at=now(),script=ev(__file__),instructions=instructions,dependencies=dependencies,
        predecessor=ev(current.CYCLE/'contract.json'),
        prediction_before_result='TraF three-seed fair gains, Where val null/hold negative across all three paired seeds, V2 negative. First ViT fair pair is still unmeasured. Pre-register fixed seeds2/3 to quantify native training variability without selecting by AP/sign. Exact own-full60 no-training baseline separates plugin from fine-tuning effects.',
        seed_selection='20261102 local PUBLIC/ONLINE pair;20261103 remote PUBLIC then ONLINE pair; final checkpoint, all signs retained; no extra OFFICIAL seeds.',
        identity='Same own ViT observations/seed2024 full60 weight, single-frame1+1, q005 translation, native head/NMS and prior/spatial metadata. Not full late-fusion heading/U and not author public-weight reproduction.',
        reuse_audit='Existing ViT1+4 matched/cache results have different native1+1 readout. Current C09 ViT v3 has only three trained arms, no exact initial-weight native1+1 three-input all-role baseline. Do not rerun old matched/cache predictions.',
        estimated_cost=dict(per_trained_arm_updates=2000,per_arm_microbatches=8000,
            per_arm_scientific_forwards=69024,per_arm_profile_forwards=408,per_arm_replays=24,
            remote_no_training_baseline_scientific_forwards=69024,
            local_lane_wall_h='2-4 after current seed1',remote_lane_wall_h='5-9 baseline plus paired seed3 after current OFFICIAL reference',
            new_disk_GiB='<5 sole predictions/final+active recovery; consumers bound'),
        storage='Reference unique C09 ViT input ledgers/observations/initial weight. No copied GT/features; preserve final weights/formal predictions and active resume only. Source consumers reference sole native producers.',no_model_wakeup=True)
    atomic(BASE/'contract.json',c)
    for seed in (20261101,20261102,20261103):
        seeded=copy.deepcopy(c);seeded['K4']['seed']=seed
        atomic(BASE/('seed'+str(seed)+'_contract.json'),seeded)
    for label in ('gpu0','gpu1','remote'):
        record(label,dict(status='registered_finite_successor',contract=ev(BASE/'contract.json'),
            prediction_before_result=c['prediction_before_result'],estimated_cost=c['estimated_cost']))
    record('initial_weight_plugin',dict(status='registered_finite_successor',contract=ev(BASE/'contract.json'),
        estimated_cost=dict(scientific_forwards=69024,updates=0),prediction_before_result='No-training own checkpoint baseline, not K4 same-budget gain.'),'K1')
    for label in ('sources','remote_sources'):
        record(label,dict(status='registered_finite_successor',contract=ev(BASE/'contract.json'),
            estimated_cost=dict(GPU_forwards=0,CPU_threads=1,wall_h='6-20 finite consumers under shared quota',new_disk_GiB='<0.8 outcomes/metrics only')),'K5')
    # Compile the adapter namespace without CPU/GPU data execution.
    current.namespace()


def setup():
    c=load(BASE/'contract.json');assert c['script']==ev(__file__)
    for reference in c['dependencies']:
        assert reference==ev(ROOT/reference['path'])
    namespace=current.namespace()
    namespace['setup']()
    return c


def wait_unit(unit,out):
    while subprocess.check_output(['systemctl','--user','show',unit,'-p','ActiveState'],text=True).strip() in ('ActiveState=active','ActiveState=activating'):
        atomic(out/'status.json',dict(status='waiting_known_finite_predecessor',unit=unit,created_at=now()))
        time.sleep(30)


def cell(weight,gpu,out,hold,seed):
    source=inspect.getsource(evaluator.cell)
    before="CYCLE/'where_execution_contract_v6.json'"
    assert source.count(before)==1
    source=source.replace(before,'evaluation_contract')
    namespace=dict(evaluator.__dict__,CYCLE=BASE,evaluation_contract=BASE/('seed'+str(seed)+'_contract.json'))
    exec(compile(source,__file__+'::unchanged_native_ViT_profiled_cell','exec'),namespace)
    namespace['cell'](weight,gpu,out,hold,'PUBLIC,ONLINE_K1,OFFICIAL_REFERENCE')


def train(seed,arm,gpu,out):
    sys.argv=[__file__,'--workspace',str(ROOT),'--contract',str(BASE/('seed'+str(seed)+'_contract.json')),
        '--output',str(out),'--arm',arm,'--gpu',str(gpu)]
    namespace=dict(__name__='native_ViT_fixed_seed_confirmation',__file__=__file__)
    exec(compile(adapter.training_source(),__file__+'::same_native_training','exec'),namespace)
    namespace['main']()


def pair(seed,arm,gpu,out):
    out.mkdir(exist_ok=False);train(seed,arm,gpu,out/'training');results=[]
    for role in ('val21','hold14'):
        results.append(attempt(out/role,cell,out/'training/final.pth',gpu,out/role,role=='hold14',seed))
    status='complete_light_checked' if all(r['status']=='complete_light_checked' for r in results) else 'failed_retained'
    atomic(out/'terminal.json',dict(status=status,created_at=now(),seed=seed,arm=arm,results=results,no_model_wakeup=True))


def attempt(out,function,*args):
    try:
        function(*args)
        status=load(out/'terminal.json')['status']
        return dict(status=status,path=str(out.relative_to(ROOT)),terminal=ev(out/'terminal.json'))
    except Exception as error:
        out.mkdir(parents=True,exist_ok=True)
        if not (out/'terminal.json').exists():
            atomic(out/'terminal.json',dict(status='failed_retained',created_at=now(),error=repr(error),traceback=traceback.format_exc()))
        return dict(status='failed_retained',path=str(out.relative_to(ROOT)),terminal=ev(out/'terminal.json'))


def lane(label):
    out=BASE/label;out.mkdir(exist_ok=False)
    wait_unit('embodied-peer12-vit-native-'+label+'-v3-20261102.service',out)
    gpu=1 if label=='gpu1' else 0;setup();results=[]
    record(label,dict(status='running_fixed_seed_confirmations',unit='embodied-peer12-vit-confirm-'+label+'-v1-20261102.service'))
    if label=='remote':
        record('initial_weight_plugin',dict(status='running_exact_native_initial_checkpoint_baseline'),'K1')
        plugin=[]
        for role in ('val21','hold14'):
            dest=out/'initial_weight_plugin'/role
            plugin.append(attempt(dest,cell,ROOT/load(BASE/'contract.json')['K4']['initial_weight']['path'],gpu,dest,role=='hold14',20261101))
        status='complete_light_checked' if all(r['status']=='complete_light_checked' for r in plugin) else 'failed_retained'
        atomic(out/'initial_weight_plugin/terminal.json',dict(status=status,created_at=now(),results=plugin,updates=0,no_model_wakeup=True))
        record('initial_weight_plugin',dict(status=status,terminal=ev(out/'initial_weight_plugin/terminal.json')),'K1')
        results.extend(plugin)
        pairs=[(20261103,'PUBLIC'),(20261103,'ONLINE_K1')]
    else:
        pairs=[(20261102,'PUBLIC' if label=='gpu0' else 'ONLINE_K1')]
    for seed,arm in pairs:
        dest=out/('seed'+str(seed)+'_'+arm.lower())
        results.append(attempt(dest,pair,seed,arm,gpu,dest))
        atomic(out/'status.json',dict(status='finite_confirmation_lane',created_at=now(),results=results))
    status='complete_light_checked' if all(r['status']=='complete_light_checked' for r in results) else 'failed_retained'
    atomic(out/'terminal.json',dict(status=status,created_at=now(),results=results,no_model_wakeup=True))
    record(label,dict(status=status,terminal=ev(out/'terminal.json')))


def sources(remote=False):
    from run_peer12_native_source_additions_v1_20261101 import compute_namespace
    namespace,contract=compute_namespace();namespace['CYCLE']=BASE
    label='remote_sources' if remote else 'sources';out=BASE/label;out.mkdir(exist_ok=False)
    targets=[]
    if remote:
        for model in ('initial_weight_plugin','seed20261103_public','seed20261103_online_k1'):
            for role in ('val21','hold14'):targets.append((BASE/'remote'/model/role/'evaluation',BASE/'remote'))
    else:
        for lane_name,arm in (('gpu0','public'),('gpu1','online_k1')):
            for role in ('val21','hold14'):targets.append((BASE/lane_name/('seed20261102_'+arm)/role/'evaluation',BASE/lane_name))
    record(label,dict(status='running_finite_native_source_consumers'),'K5');results=[]
    for producer,parent in targets:
        while not (producer/'terminal.json').exists() and not (parent/'terminal.json').exists():
            atomic(out/'status.json',dict(status='waiting_registered_prediction',producer=str(producer.relative_to(ROOT)),created_at=now()))
            time.sleep(30)
        try:
            assert load(producer/'terminal.json')['status']=='native_AP_complete_light_checked'
            name=str(producer.relative_to(AUDIT));namespace['compute'](name,contract)
            results.append(dict(status='complete_light_checked',terminal=ev(BASE/'native_source_aux'/name/'terminal.json')))
        except Exception as error:
            results.append(dict(status='failed_retained',error=repr(error),traceback=traceback.format_exc()))
    status='complete_light_checked' if all(r['status']=='complete_light_checked' for r in results) else 'failed_retained'
    atomic(out/'terminal.json',dict(status=status,created_at=now(),results=results,no_model_wakeup=True))
    record(label,dict(status=status,terminal=ev(out/'terminal.json')),'K5')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','gpu0','gpu1','remote','sources','remote-sources'])
    mode=parser.parse_args().mode
    if mode=='prepare':prepare()
    elif mode in ('sources','remote-sources'):sources(mode=='remote-sources')
    else:lane(mode)
