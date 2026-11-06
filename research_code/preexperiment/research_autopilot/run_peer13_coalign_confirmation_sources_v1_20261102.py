"""Finite zero-forward consumers for the fixed CoAlign seed tournament."""
import argparse,inspect,os,time,traceback
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
from peer13_cycle18_v1_20261102 import ROOT,AUTO,AUDIT,CYCLE,atomic,ev,load,now
import run_peer13_coalign_seed_confirmation_v1_20261102 as producer
from run_peer12_native_source_additions_v1_20261101 import compute_namespace
import run_peer10_native_source_aux_v1_20261101 as scorer
BASE=CYCLE/'coalign_confirmation_sources_v1'

def namespace():
    ns,c=compute_namespace();ns['CYCLE']=BASE
    # The original fixed scorer owns the source/auxiliary population contract.
    source=inspect.getsource(scorer.compute)
    changes=[("gt=shard[f'c{idx}_gt']","gt=gtshard[f\"c{r['GT_case']}_gt\"]"),
        ("with np.load(producer/f'wait{wait:03d}.npz',allow_pickle=False) as shard:","with np.load(producer/f'wait{wait:03d}.npz',allow_pickle=False) as shard, np.load(ROOT/next(r['GT_shard'] for r in records if r['wait_ms']==wait),allow_pickle=False) as gtshard:")]
    for a,b in changes:assert source.count(a)==1,a;source=source.replace(a,b)
    exec(compile(source,__file__+'::immutable_shared_GT_source_consumer','exec'),ns)
    return ns,c

def prepare():
    refs={}
    for n in ('current_handoff.md','main_chat_directive.md'):(AUTO/n).read_text();refs[n]=ev(AUTO/n)
    BASE.mkdir(exist_ok=False);namespace()
    atomic(BASE/'contract.json',dict(created_at=now(),script=ev(__file__),producer_contract=ev(producer.BASE/'contract.json'),scorer=ev(scorer.__file__),namespace_builder=ev(ROOT/'preexperiment/research_autopilot/run_peer12_native_source_additions_v1_20261101.py'),instructions=refs,
        prediction_before_result='Fixed three-seed full/no-graph/K1 rankings may depend on the source or official/auxiliary label protocol. No AP-based label, seed or estimator selection.',
        population='Exact TraF-native sole GT references per condition; author CoAlign HWL/own NMS. Same frozen SPD label-only replacements; official replay and source<1mm required.',
        cost=dict(GPU_forwards=0,CPU_threads=1,new_disk_GiB='<1 compact outcomes; no copied GT or predictions'),timeout_h=30,no_model_wakeup=True))
    for host in ('local','remote'):producer.record('sources_'+host,dict(status='registered_finite_successor',contract=ev(BASE/'contract.json')))

def run(host):
    c=load(BASE/'contract.json');assert c['script']==ev(__file__) and c['producer_contract']==ev(producer.BASE/'contract.json') and c['scorer']==ev(scorer.__file__)
    out=BASE/host;out.mkdir(exist_ok=False);start=time.monotonic();ns,spec=namespace();results=[]
    kind='reference' if host=='remote' else 'graph'
    try:
        producer.record('sources_'+host,dict(status='running_finite_source_chain',contract=ev(BASE/'contract.json')))
        for seed in (20261103,20261104):
            for training in ('public','online'):
                p=producer.BASE/('seed'+str(seed))/(training+'_'+kind)/'evaluation';term=p/'terminal.json'
                while not term.exists():
                    lane='remote' if host=='remote' else ('gpu0' if seed==20261103 else 'gpu1')
                    failed=producer.BASE/lane/'terminal.json'
                    if failed.exists() and load(failed)['status']=='failed_retained':raise RuntimeError(('failed producer',str(failed)))
                    if time.monotonic()-start>c['timeout_h']*3600:raise TimeoutError('finite30h source wait')
                    atomic(out/'status.json',dict(status='waiting_known_prediction',producer=str(p.relative_to(ROOT)),created_at=now(),results=results));time.sleep(30)
                assert load(term)['status']=='native_AP_complete_light_checked'
                ns['compute'](str(p.relative_to(AUDIT)),spec)
                result=BASE/'native_source_aux'/p.relative_to(AUDIT)/'terminal.json';assert load(result)['status']=='complete_light_checked'
                results.append(dict(seed=seed,training=training,kind=kind,terminal=ev(result)))
                atomic(out/'status.json',dict(status='scoring_finite_seed_sources',created_at=now(),results=results))
        atomic(out/'terminal.json',dict(status='complete_light_checked',created_at=now(),results=results,no_model_wakeup=True));producer.record('sources_'+host,dict(status='complete_light_checked',terminal=ev(out/'terminal.json')))
    except Exception as e:
        atomic(out/'terminal.json',dict(status='failed_retained',created_at=now(),error=repr(e),traceback=traceback.format_exc(),results=results));producer.record('sources_'+host,dict(status='failed_retained',terminal=ev(out/'terminal.json')));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','local','remote']);a=p.parse_args();prepare() if a.mode=='prepare' else run(a.mode)
