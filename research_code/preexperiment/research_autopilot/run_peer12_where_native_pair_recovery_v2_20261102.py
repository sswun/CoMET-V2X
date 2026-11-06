"""Separate recovery: CPU loader only; native constructor/strict load on GPU."""
import argparse, inspect, sys, traceback
import run_peer12_where_native_pair_v1_20261102 as old
from peer12_cycle06_v1_20261102 import ROOT,AUTO,AUDIT,CYCLE,C03,atomic,ev,load,now,task,update

def prepare():
    for n in ('current_handoff.md','main_chat_directive.md'):(AUTO/n).read_text()
    c=dict(load(CYCLE/'where_contract.json'),script=ev(__file__),created_at=now(),original_contract=ev(CYCLE/'where_contract.json'),
        recovery='CPU qualification does not instantiate author model because constructor creates CUDA anchors; inspect finite checkpoint tensors on CPU, exact strict model load plus first four finite backward updates remain in each GPU lane. No model/algorithm patch.',
        original_failure=ev(CYCLE/'where_qualification/terminal.json'))
    atomic(CYCLE/'where_contract_recovery_v2.json',c)
    for arm in ('public','online_k1'):
        update('peer12_where_native_'+arm+'_v1_20261102',task('K4'),dict(status='failed_retained',GPU_forwards=0,updates=0,reason='Unstarted lane superseded before execution after CPU-only constructor qualification failed',superseded_by='peer12_where_native_'+arm+'_v2_20261102'))
        update('peer12_where_native_'+arm+'_v2_20261102',task('K4'),dict(status='registered_after_loader_qualification',contract=ev(CYCLE/'where_contract_recovery_v2.json'),prediction_before_result=c['prediction_before_result'],estimated_cost=c['estimated_cost']))
    update('peer12_where_native_CPU_qualification_v2_20261102',task('K4'),dict(status='registered_ready',contract=ev(CYCLE/'where_contract_recovery_v2.json'),GPU_forwards=0))

def namespace():
    ns=dict(old.__dict__);ns['__file__']=__file__
    s=inspect.getsource(old.qualify)
    changes=[("out=CYCLE/'where_qualification'","out=CYCLE/'where_qualification_v2'"),
        ("c=load(CYCLE/'where_contract.json')","c=load(CYCLE/'where_contract_recovery_v2.json')"),
        ("model=train_utils.create_model(h).float();model.load_state_dict(torch.load(WEIGHT,map_location='cpu'),strict=True);del model","weights=torch.load(WEIGHT,map_location='cpu');assert all(torch.isfinite(v).all() for v in weights.values());del weights"),
        ('strict_weight_load=True','strict_weight_load=False,CPU_checkpoint_finite=True,actual_model_strict_load_pending_GPU=True'),
        ("peer12_where_native_CPU_qualification_v1_20261102","peer12_where_native_CPU_qualification_v2_20261102")]
    for a,b in changes:assert s.count(a)==1,(a,s.count(a));s=s.replace(a,b)
    exec(compile(s,__file__+'::loader_only','exec'),ns)
    s=inspect.getsource(old.lane)
    for a,b in [("'where_'+arm.lower()","'where_'+arm.lower()+'_v2'"),("'_v1_20261102'","'_v2_20261102'"),("where_qualification/terminal.json","where_qualification_v2/terminal.json"),("c=load(CYCLE/'where_contract.json')","c=load(CYCLE/'where_contract_recovery_v2.json')")]:
        assert s.count(a)==1,(a,s.count(a));s=s.replace(a,b)
    exec(compile(s,__file__+'::paired_native_GPU','exec'),ns)
    return ns

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','qualify','public','online']);a=p.parse_args()
    if a.mode=='prepare':return prepare()
    ns=namespace()
    if a.mode=='qualify':
        try:return ns['qualify']()
        except Exception as e:
            out=CYCLE/'where_qualification_v2/terminal.json';atomic(out,dict(status='failed_retained',created_at=now(),error=repr(e),traceback=traceback.format_exc(),GPU_forwards=0,updates=0))
            update('peer12_where_native_CPU_qualification_v2_20261102',task('K4'),dict(status='failed_retained',terminal=ev(out)));raise
    ns['lane']('PUBLIC' if a.mode=='public' else 'ONLINE_K1',0 if a.mode=='public' else 1)

if __name__=='__main__':main()
