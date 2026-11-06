"""Explicit complete qualification namespace; reuse inputs, retire zero-batch v2."""
import argparse
from peer12_cycle06_v1_20261102 import ROOT,AUTO,atomic,ev,load,now
from run_peer12_where_confirmation_v1_20261102 import CYCLE as C09,record

ORIGINAL=ROOT/'preexperiment/research_autopilot/run_peer12_vit_native_pair_v1_20261102.py'
CYCLE=C09/'vit_native_v3'
TID='peer12_vit_native_{}_v3_20261102'


def namespace():
    source=ORIGINAL.read_text()
    replacements=[("CYCLE = C09/'vit_native'","CYCLE = C09/'vit_native_v3'"),
        ("TID = 'peer12_vit_native_{}_v1_20261102'","TID = 'peer12_vit_native_{}_v3_20261102'"),
        ("assert source.count(before)==1,(before,source.count(before))",
         "assert source.count(before)==(2 if before==\"CYCLE/'where_contract.json'\" else 1),(before,source.count(before))"),
        ("namespace=dict(adapter.__dict__,CYCLE=CYCLE,WEIGHT=WEIGHT)",
         "namespace=dict(adapter.__dict__,CYCLE=CYCLE,WEIGHT=WEIGHT,C09=C09)"),
        ("'embodied-peer12-vit-native-'+label+'-v1-20261102.service'", "'embodied-peer12-vit-native-'+label+'-v3-20261102.service'")]
    for before,after in replacements:
        assert source.count(before)==1,(before,source.count(before))
        source=source.replace(before,after)
    source=source.replace("CYCLE/'inputs'", "C09/'vit_native/inputs'")
    ns=dict(__name__='vit_native_explicit_v3',__file__=__file__)
    exec(compile(source,__file__+'::complete_CPU_namespace','exec'),ns)
    return ns


def prepare():
    refs={}
    for name in ('current_handoff.md','main_chat_directive.md'):
        (AUTO/name).read_text()
        refs[name]=ev(AUTO/name)
    CYCLE.mkdir(exist_ok=False)
    c=load(C09/'vit_native_v2/contract.json')
    c.update(created_at=now(),script=ev(__file__),instructions=refs,recovered_source=ev(ORIGINAL),
             predecessor=ev(C09/'vit_native_v2/contract.json'),
             recovery='v2 CPU qualifier shallow namespace lacked C09 for sole-input reference. Both v1/v2 failed before dataset batches, zero GPU/training. v3 provides explicit namespace and retains unique original inputs.')
    atomic(CYCLE/'contract.json',c)
    namespace()
    for label in ('qualification','qualification_remote','gpu0','gpu1','remote','sources','remote_source'):
        record(TID.format(label),dict(status='registered_finite_successor',contract=ev(CYCLE/'contract.json'),
               prediction_before_result=c['prediction_before_result'],estimated_cost=c['estimated_cost']),
               'K5' if 'source' in label else 'K4')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['prepare','qualify','qualify-remote','gpu0','gpu1','remote','sources'])
    mode=parser.parse_args().mode
    if mode=='prepare':prepare()
    else:
        ns=namespace()
        if mode=='qualify-remote':
            ns['record']=lambda tid,values,*rest:record(tid.replace('_qualification_','_qualification_remote_'),values,*rest)
            ns['qualify']()
        elif mode=='qualify':ns['qualify']()
        elif mode=='sources':ns['sources']()
        else:ns['lane'](mode)
