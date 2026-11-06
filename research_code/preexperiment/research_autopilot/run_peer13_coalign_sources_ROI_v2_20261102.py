"""Exact CoAlign prediction-ROI replay, preserving its shared target population."""
import argparse,copy,os,time,traceback
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
from peer13_cycle18_v1_20261102 import ROOT,AUTO,AUDIT,CYCLE,C15,atomic,ev,load,now,update,task
import run_peer13_coalign_confirmation_sources_v1_20261102 as old
import run_peer13_coalign_seed_confirmation_v1_20261102 as producer
BASE=CYCLE/'coalign_sources_ROI_v2'
TID='peer13_coalign_sources_ROI_{}_v2_20261102'
RANGE=[-100.8,-40,-3.5,100.8,40,1.5]

def record(host,v):
    tid=TID.format(host)
    if str(ROOT).startswith('/home/user/'):
        p=BASE/'remote_queue_delta.json';r=load(p) if p.exists() else {};r[tid]=dict(id=tid,parent_task_id=task('K5'),**v);atomic(p,r)
    else:update(tid,task('K5'),v)

def prepare():
    refs={}
    for n in ('current_handoff.md','main_chat_directive.md'):(AUTO/n).read_text();refs[n]=ev(AUTO/n)
    BASE.mkdir(exist_ok=False)
    atomic(BASE/'contract.json',dict(created_at=now(),script=ev(__file__),adapter=ev(old.__file__),producer_contract=ev(producer.BASE/'contract.json'),instructions=refs,
        correction='C15 generic source spec used TraF prediction XY[0,96], while actual frozen CoAlign evaluator kept XY[-100.8,100.8]. Official replay therefore failed. Saved models/predictions/GT/official AP unchanged. Correct CPU consumer only; preserve old failures and costs.',
        native_eval_range=RANGE,population='CoAlign predictions use author prediction ROI but frozen GT is referenced from TraF-native loader. The native-named legacy panels are within-method shared-TraF-GT/CoAlign-prediction-ROI controls, NOT author-native complete-GT AP or cross-backbone rankings. Common ROI[0,-38.4,96,38.4] masks both predictions and targets and is the formal common-population comparison.',
        prediction_before_result='Correct prediction filtering should exactly replay old official AP before auxiliary/source acceptance. Signed graph/no-graph/K1 effects under common/source/auxiliary scopes remain unselected.',
        cost=dict(GPU_forwards=0,updates=0,CPU_threads=1,new_disk_GiB='<2 compressed outcomes; no GT/prediction copies'),timeout_h=32,no_model_wakeup=True))
    for host in ('local','remote'):record(host,dict(status='registered_finite_successor',contract=ev(BASE/'contract.json')))

def run(host):
    c=load(BASE/'contract.json');assert c['script']==ev(__file__) and c['adapter']==ev(old.__file__)
    out=BASE/host;out.mkdir(exist_ok=False);start=time.monotonic();ns,spec=old.namespace();spec=copy.deepcopy(spec);spec['native_source_aux']['native_eval_range']=c['native_eval_range'];ns['CYCLE']=BASE
    first=['author_evaluation'] if host=='remote' else ['adapted_evaluation','online_adapted_evaluation']
    targets=[C15/'coalign_full_v1'/n for n in first]
    kind='reference' if host=='remote' else 'graph'
    targets += [producer.BASE/('seed'+str(s))/(t+'_'+kind)/'evaluation' for s in (20261103,20261104) for t in ('public','online')]
    results=[]
    try:
        record(host,dict(status='running_finite_correct_ROI_source_chain',contract=ev(BASE/'contract.json')))
        for p in targets:
            while not (p/'terminal.json').exists():
                if time.monotonic()-start>c['timeout_h']*3600:raise TimeoutError('finite32h source wait')
                atomic(out/'status.json',dict(status='waiting_exact_CoAlign_prediction',producer=str(p.relative_to(ROOT)),created_at=now(),results=results));time.sleep(30)
            assert load(p/'terminal.json')['status']=='native_AP_complete_light_checked'
            name=str(p.relative_to(AUDIT));ns['compute'](name,spec)
            result=BASE/'native_source_aux'/name
            atomic(result/'ROI_population_identity.json',dict(contract=ev(BASE/'contract.json'),native_prediction_range=RANGE,native_GT='sole referenced TraF-native loader population; not complete CoAlign author-native GT',common_comparison='same prediction+GT mask [0,-38.4,96,38.4]',official_AP_replay_required=True,kind=kind if p in targets[len(first):] else 'all_three_arms'))
            results.append(dict(producer=name,terminal=ev(result/'terminal.json')))
            atomic(out/'status.json',dict(status='completed_producer_source_scope',created_at=now(),results=results))
        atomic(out/'terminal.json',dict(status='complete_light_checked',created_at=now(),results=results,no_model_wakeup=True));record(host,dict(status='complete_light_checked',terminal=ev(out/'terminal.json')))
    except Exception as e:
        atomic(out/'terminal.json',dict(status='failed_retained',created_at=now(),error=repr(e),traceback=traceback.format_exc(),results=results));record(host,dict(status='failed_retained',terminal=ev(out/'terminal.json')));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','local','remote']);a=p.parse_args();prepare() if a.mode=='prepare' else run(a.mode)
