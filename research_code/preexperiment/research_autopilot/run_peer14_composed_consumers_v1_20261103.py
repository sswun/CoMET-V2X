"""Finite full-composition source/registration/dose and remote collection chain."""
import argparse,os,time,traceback,subprocess
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
from peer14_cycle06_v1_20261103 import ROOT,AUDIT,CYCLE,REMOTE_ROOT,atomic,ev,load,now,register,record
import run_peer14_coalign_composed_graph_v1_20261103 as producer
import run_peer14_coalign_variable_reference_sources_v2_20261103 as source
import run_peer14_native_physical_and_dose_completion_v1_20261103 as physical
BASE=CYCLE/'composed_consumers_v1'

def targets():
    return [dict(path=str((producer.BASE/lane/t/k/'evaluation').relative_to(AUDIT)),lane=lane,training=t,kind=k,family='CoAlign_complete_graph_composition',bounds=[-100.8,-40,-3.5,100.8,40,1.5]) for t in ('public','online') for k in producer.KINDS for lane in ('gpu0','gpu1','remote')]

def prepare():
    BASE.mkdir(exist_ok=False)
    register('composed_consumers',__file__,
        'Score every completed full graph composition with original official replay, five source masks, both auxiliary protocols, paired accepted registration population and fixed seven-point annotation intervention. Collect completed remote products to same host-relative paths without overwriting existing content; no model wake. Different seed/model identities never merged as same cache.',
        dict(GPU_forwards=0,updates=0,CPU_threads=1,timeout_h=36,new_disk_GiB='<3 compressed source outcomes/metrics; unique remote predictions collected once; no copied GT'),targets=targets(),producer_contract=ev(CYCLE/'coalign_composed_graph_contract.json'))
    for mode in ('sources','registration','dose'):record('composed_'+mode,dict(status='registered_finite_successor',contract=ev(CYCLE/'composed_consumers_contract.json')))
    record('composed_consumers',dict(status='complete_light_checked',administrative_only=True,scientific_children_open=True))

def collect(t):
    p=AUDIT/t['path']
    if (p/'terminal.json').exists():return True
    if t['lane']!='remote':return False
    remote=REMOTE_ROOT/p.relative_to(ROOT)
    if subprocess.run(['ssh','remote-gpu','test','-f',str(remote/'terminal.json')],capture_output=True).returncode:return False
    p.mkdir(parents=True,exist_ok=True)
    subprocess.run(['rsync','-a','--ignore-existing','remote-gpu:'+str(remote)+'/',str(p)+'/'],check=True)
    return (p/'terminal.json').exists()

def run(mode):
    c=load(CYCLE/'composed_consumers_contract.json');assert c['script']==ev(__file__);out=BASE/mode;out.mkdir();start=time.monotonic();results={};record('composed_'+mode,dict(status='running_finite_ready_first_CPU_chain'))
    if mode=='sources':ns,spec=source.namespace();ns['CYCLE']=out
    else:
        offsets,slow=physical.offsets_and_slow()
        if mode=='registration':_,accepted=physical.registration.load_registration('SPD')
    try:
        pending=list(enumerate(c['targets']))
        while pending:
            if time.monotonic()-start>36*3600:raise TimeoutError('finite36h composed consumer')
            progressed=False
            for i,t in list(pending):
                # Only source lane writes remote predictions; other lanes use that sole copy.
                p=AUDIT/t['path'];ready=collect(t) if mode=='sources' else (p/'terminal.json').exists()
                if not ready:continue
                progressed=True;sub=out/f'{i:03d}';sub.mkdir()
                try:
                    assert load(p/'terminal.json')['status']=='native_AP_complete_light_checked'
                    if mode=='sources':ns['compute'](t['path'],spec);term=out/'native_source_aux'/t['path']/'terminal.json'
                    elif mode=='registration':physical.SPD_one(t,accepted,offsets,sub);term=sub/'terminal.json'
                    else:physical.dose.compute(t,offsets,slow,sub);term=sub/'terminal.json'
                    assert load(term)['status']=='complete_light_checked'
                    results[str(i)]=dict(producer=t['path'],terminal=ev(term))
                except Exception as e:
                    atomic(sub/'failure.json',dict(status='failed_retained',created_at=now(),error=repr(e),traceback=traceback.format_exc()));results[str(i)]=dict(producer=t['path'],failure=ev(sub/'failure.json'))
                pending.remove((i,t));atomic(out/'results.json',results)
            atomic(out/'status.json',dict(created_at=now(),status='finite_ready_first_consumer',completed=len(results),pending=len(pending)))
            if not progressed:time.sleep(30)
        status='complete_light_checked' if all('terminal' in r for r in results.values()) else 'failed_retained'
        atomic(out/'terminal.json',dict(status=status,created_at=now(),results=ev(out/'results.json'),wall_s=time.monotonic()-start,GPU_forwards=0,no_model_wakeup=True));record('composed_'+mode,dict(status=status,terminal=ev(out/'terminal.json')))
    except Exception as e:
        atomic(out/'terminal.json',dict(status='failed_retained',created_at=now(),error=repr(e),traceback=traceback.format_exc()));record('composed_'+mode,dict(status='failed_retained',terminal=ev(out/'terminal.json')));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','sources','registration','dose']);a=p.parse_args();prepare() if a.mode=='prepare' else run(a.mode)
