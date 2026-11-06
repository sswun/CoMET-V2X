"""Seven unique ICP prediction producers, three ready-first CPU consumers."""
import argparse,os,time,traceback,subprocess
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
from peer14_cycle12_v1_20261103 import ROOT,AUDIT,CYCLE,REMOTE_ROOT,atomic,ev,load,now,register,record
import run_peer14_traf_physical_reference_v1_20261103 as producer
import run_peer14_coalign_variable_reference_sources_v2_20261103 as source
import run_peer14_native_physical_and_dose_completion_v1_20261103 as physical
BASE=CYCLE/'traf_reference_consumers_v1'

def targets():
    bounds=load(CYCLE/'traf_physical_reference_contract.json')['native_bounds']
    return [dict(path=str((producer.BASE/lane/n/'evaluation').relative_to(AUDIT)),lane=lane,name=n,family='TraF_native_synchronized_ICP_reference',bounds=bounds) for lane,pp in producer.plans().items() for n,w in pp]

def prepare():
    BASE.mkdir(exist_ok=False)
    register('traf_reference_consumers',__file__,'All seven frozen ICP native inputs: original official native AP replay, fixed official/auxiliary five source masks, paired quality-selected registration geometry, and seven-point label-dose response. These are input-reference rows, not a new deployable method. One remote prediction collector only, GT/predictions shared.',dict(GPU_forwards=0,training_updates=0,CPU_threads=1,timeout_h=36,new_disk_GiB='<1 metrics/outcomes plus unique remote prediction collection'),targets=targets(),producer_contract=ev(CYCLE/'traf_physical_reference_contract.json'))
    for mode in ('sources','registration','dose'):record('traf_reference_'+mode,dict(status='registered_finite_successor',contract=ev(CYCLE/'traf_reference_consumers_contract.json')))
    record('traf_reference_consumers',dict(status='complete_light_checked',administrative_only=True,scientific_children_open=True))

def collect(t):
    p=AUDIT/t['path']
    if (p/'terminal.json').exists():return True
    if t['lane']!='remote':return False
    remote=REMOTE_ROOT/p.relative_to(ROOT)
    if subprocess.run(['ssh','remote-gpu','test','-f',str(remote/'terminal.json')],capture_output=True).returncode:return False
    p.mkdir(parents=True,exist_ok=True);subprocess.run(['rsync','-a','--ignore-existing','remote-gpu:'+str(remote)+'/',str(p)+'/'],check=True)
    return (p/'terminal.json').exists()

def run(mode):
    c=load(CYCLE/'traf_reference_consumers_contract.json');assert ev(__file__)==c['script'];out=BASE/mode;out.mkdir(exist_ok=False);start=time.monotonic();results={};pending=list(enumerate(c['targets']))
    if mode=='sources':
        ns,spec=source.namespace();ns['CYCLE']=out;spec['native_source_aux']['native_eval_range']=load(CYCLE/'traf_physical_reference_contract.json')['native_bounds']
    else:
        offsets,slow=physical.offsets_and_slow()
        if mode=='registration':_,accepted=physical.registration.load_registration('SPD')
    record('traf_reference_'+mode,dict(status='running_finite_ready_first_reference_consumer'))
    try:
        while pending:
            progressed=False
            for i,t in list(pending):
                p=AUDIT/t['path'];ready=collect(t) if mode=='sources' else (p/'terminal.json').exists()
                if not ready:continue
                progressed=True;sub=out/f'{i:03d}';sub.mkdir()
                try:
                    assert load(p/'terminal.json')['status']=='native_AP_complete_light_checked'
                    if mode=='sources':ns['compute'](t['path'],spec);term=out/'native_source_aux'/t['path']/'terminal.json'
                    elif mode=='registration':physical.SPD_one(t,accepted,offsets,sub);term=sub/'terminal.json'
                    else:physical.dose.compute(t,offsets,slow,sub);term=sub/'terminal.json'
                    assert load(term)['status']=='complete_light_checked';results[str(i)]=dict(producer=t['path'],terminal=ev(term))
                except Exception as e:
                    atomic(sub/'failure.json',dict(status='failed_retained',created_at=now(),error=repr(e),traceback=traceback.format_exc()));results[str(i)]=dict(producer=t['path'],failure=ev(sub/'failure.json'))
                pending.remove((i,t));atomic(out/'results.json',results)
            atomic(out/'status.json',dict(created_at=now(),status='finite_ready_first_reference_consumer',completed=len(results),pending=len(pending)))
            if pending:
                if time.monotonic()-start>36*3600:raise TimeoutError('finite36h reference consumer')
                if not progressed:time.sleep(60)
        status='complete_light_checked' if all('terminal' in r for r in results.values()) else 'failed_retained'
        atomic(out/'terminal.json',dict(status=status,created_at=now(),results=ev(out/'results.json'),wall_s=time.monotonic()-start,GPU_forwards=0,no_model_wakeup=True));record('traf_reference_'+mode,dict(status=status,terminal=ev(out/'terminal.json')))
    except Exception as e:
        atomic(out/'terminal.json',dict(status='failed_retained',created_at=now(),error=repr(e),traceback=traceback.format_exc()));record('traf_reference_'+mode,dict(status='failed_retained',terminal=ev(out/'terminal.json')));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','sources','registration','dose']);a=p.parse_args();prepare() if a.mode=='prepare' else run(a.mode)
