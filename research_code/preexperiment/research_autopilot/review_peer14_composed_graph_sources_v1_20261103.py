"""Finite ready-first full independent review of eighteen actual graph outputs."""
import argparse,os,time,traceback
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
from peer14_cycle09_v1_20261103 import ROOT,AUDIT,CYCLE,C06,atomic,ev,load,now,register,record
import review_peer13_coalign_full_sources_v1_20261102 as old
import run_peer14_composed_consumers_v1_20261103 as consumers
BASE=CYCLE/'composed_full_independent_v1'
GOLD=C06/'composed_consumers_v1/sources/native_source_aux'

def prepare():
    BASE.mkdir(exist_ok=False)
    register('composed_full_independent',__file__,'Full independent float64 corners/IoU/primary sources/ignore/ranking/VOC for all18 composed graphs as their actual source outcomes complete, canonical threshold mismatches retained. Distinct from old no-graph/reference reviews and not duplicate GPU replay. No estimator changes.',dict(GPU_forwards=0,CPU_threads=1,metric_panels=4320,timeout_h=36,new_disk_MiB='<35 review checks/identities, predictions/GT/outcomes zero-copy references'),targets=consumers.targets(),reviewer=ev(old.__file__),independent_kernel=ev(old.original.__file__),source_contract=ev(C06/'composed_consumers_contract.json'))

def run():
    c=load(CYCLE/'composed_full_independent_contract.json');assert ev(__file__)==c['script'] and ev(old.__file__)==c['reviewer'] and ev(old.original.__file__)==c['independent_kernel']
    start=time.monotonic();pending=list(enumerate(c['targets']));results={};record('composed_full_independent',dict(status='running_finite_ready_first_independent4320'))
    try:
        while pending:
            progressed=False
            for i,t in list(pending):
                p=AUDIT/t['path'];gold=GOLD/t['path']
                if not (p/'terminal.json').exists() or not (gold/'terminal.json').exists():continue
                progressed=True;out=BASE/f'{i:03d}'
                try:
                    assert load(p/'terminal.json')['status']=='native_AP_complete_light_checked' and load(gold/'terminal.json')['status']=='complete_light_checked'
                    ns=old.namespace('local');ns.update(REVIEW_OUT=out,GOLDROOT=GOLD,REVIEW_CONTRACT=CYCLE/'composed_full_independent_contract.json',EXPECTED_PANELS=240,wait_ready=lambda p:None,update=lambda *a:None)
                    ns['REVIEW_SPEC']=dict(ns['REVIEW_SPEC'],producers=[t['path']]);ns['main']();v=load(out/'terminal.json');results[str(i)]=dict(producer=t['path'],status=v['status'],terminal=ev(out/'terminal.json'))
                except Exception as e:
                    out.mkdir(parents=True,exist_ok=True);atomic(out/'wrapper_failure.json',dict(status='failed_retained',created_at=now(),error=repr(e),traceback=traceback.format_exc()));results[str(i)]=dict(producer=t['path'],status='failed_retained',failure=ev(out/'wrapper_failure.json'))
                pending.remove((i,t));atomic(BASE/'results.json',results)
            atomic(BASE/'status.json',dict(created_at=now(),status='ready_first_independent4320',completed=len(results),pending=len(pending)))
            if pending:
                if time.monotonic()-start>36*3600:raise TimeoutError('finite36h composed independent')
                if not progressed:time.sleep(60)
        status='complete_independently_reviewed' if all(r['status']=='complete_independently_reviewed' for r in results.values()) else 'failed_retained'
        atomic(BASE/'terminal.json',dict(created_at=now(),status=status,results=ev(BASE/'results.json'),whole_release_closed=False,no_model_wakeup=True));record('composed_full_independent',dict(status=status,terminal=ev(BASE/'terminal.json')))
    except Exception as e:
        atomic(BASE/'terminal.json',dict(status='failed_retained',created_at=now(),error=repr(e),traceback=traceback.format_exc(),results=results));record('composed_full_independent',dict(status='failed_retained',terminal=ev(BASE/'terminal.json')));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run']);globals()[p.parse_args().mode]()
