"""Full independent ceiling review after current composed review; no wakeups."""
import argparse,inspect,time
from peer14_cycle12_v1_20261103 import ROOT,CYCLE,C09,atomic,ev,load,now,register,record
import review_peer14_composed_graph_sources_v1_20261103 as previous
import run_peer14_graph_ceiling_consumers_v1_20261103 as consumers
BASE=CYCLE/'graph_ceiling_full_independent_v1'
GOLD=C09/'graph_ceiling_consumers_v1/sources/native_source_aux'

def prepare():
    BASE.mkdir(exist_ok=False)
    register('graph_ceiling_full_independent',__file__,'All18 unchanged complete graph annotation-ceiling predictions/source outcomes independently reconstructed from primary GT provenance and float64 geometry/VOC. Queue after actual current composed4320 review to protect main CPU work. Threshold-sensitive failures retained; no canonical overwrite or extra GPU replay.',dict(GPU_forwards=0,CPU_threads=1,metric_panels=4320,timeout_h=48,new_disk_MiB='<35 review only, GT/pred/outcome references'),targets=consumers.targets(),reviewer=ev(previous.old.__file__),independent_kernel=ev(previous.old.original.__file__),source_contract=ev(C09/'graph_ceiling_consumers_contract.json'),predecessor=ev(C09/'composed_full_independent_contract.json'))

def run():
    start=time.monotonic();record('graph_ceiling_full_independent',dict(status='waiting_current_composed_full_independent_terminal'))
    dep=C09/'composed_full_independent_v1/terminal.json'
    while not dep.exists():
        if time.monotonic()-start>36*3600:raise TimeoutError('finite36h predecessor review')
        atomic(BASE/'status.json',dict(created_at=now(),status='waiting_current_composed_full_independent_terminal',predecessor=str(dep.relative_to(ROOT))));time.sleep(60)
    # Previous numerical failure does not invalidate independent reconstruction of new arms.
    ns=dict(previous.__dict__,CYCLE=CYCLE,BASE=BASE,GOLD=GOLD)
    s=inspect.getsource(previous.run).replace('composed_full_independent_contract.json','graph_ceiling_full_independent_contract.json').replace("'composed_full_independent'","'graph_ceiling_full_independent'").replace('36*3600','48*3600').replace('finite36h composed independent','finite48h ceiling independent')
    exec(compile(s,__file__+'::full_ceiling_reconstruction','exec'),ns);ns['__file__']=__file__;ns['run']()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run']);globals()[p.parse_args().mode]()
