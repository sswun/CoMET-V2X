"""Reuse frozen full-graph scoring with distinct finite target identities."""
import argparse,inspect
from peer14_cycle09_v1_20261103 import ROOT,AUDIT,CYCLE,REMOTE_ROOT,atomic,ev,load,now,register,record
import run_peer14_composed_consumers_v1_20261103 as prior
import run_peer14_coalign_graph_ceilings_v1_20261103 as producer
BASE=CYCLE/'graph_ceiling_consumers_v1'

def targets():
    return [dict(path=str((producer.BASE/l/t/k/'evaluation').relative_to(AUDIT)),lane=l,training=t,kind=k,family='CoAlign_complete_graph_annotation_ceiling',bounds=[-100.8,-40,-3.5,100.8,40,1.5]) for t in ('public','online') for k in producer.KINDS for l in ('gpu0','gpu1','remote')]

def prepare():
    BASE.mkdir(exist_ok=False)
    register('graph_ceiling_consumers',__file__,'Full-system nondeployable slow/full input references need actual five-source official/auxiliary, registration-quality and fixed-dose scoring. Reuse exact existing scorer, preserve negative results and sole same-relative-path remote collection, no cached predictions relabelled as graph without calls.',dict(GPU_forwards=0,CPU_threads=1,timeout_h=36,new_disk_GiB='<3 compressed metrics/outcomes; prediction collection once, no copied raw/GT'),targets=targets(),producer=ev(CYCLE/'coalign_graph_ceilings_contract.json'),scorer=ev(prior.__file__))
    for m in ('sources','registration','dose'):record('graph_ceiling_'+m,dict(status='registered_finite_successor'))
    record('graph_ceiling_consumers',dict(status='complete_light_checked',administrative_only=True,scientific_children_open=True))

def run(mode):
    ns=dict(prior.__dict__,CYCLE=CYCLE,BASE=BASE,record=lambda name,v:record(name.replace('composed_','graph_ceiling_'),v))
    src=inspect.getsource(prior.run).replace("composed_consumers_contract.json","graph_ceiling_consumers_contract.json")
    ns['__file__']=__file__
    exec(compile(src,__file__+'::unchanged_scoring_new_targets','exec'),ns);ns['run'](mode)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','sources','registration','dose']);a=p.parse_args();prepare() if a.mode=='prepare' else run(a.mode)
