"""Restore exact previously approved TraF legacy-NumPy/thermal environment.

v2 waiters were retired before any model forward after real CPU preflight found
missing NumPy aliases. v1/v2, waiting receipts and all science scripts retained.
"""
import argparse,inspect,runpy
from peer14_cycle12_v1_20261103 import ROOT,CYCLE,atomic,ev,load,now,register,record
import run_peer14_traf_physical_reference_v2_20261103 as previous
SHIM=ROOT/'preexperiment/research_autopilot/compat_traf_numpy124_thermal_v4/sitecustomize.py'
ns=dict(previous.ns)
s=inspect.getsource(previous.old.run).replace("ROOT[v['path']]","ROOT/v['path']").replace('traf_physical_reference_contract.json','traf_physical_reference_recovery_v3_contract.json').replace("'traf_physical_'+lane","'traf_physical_recovery_v3_'+lane")
ns['__file__']=__file__
exec(compile(s,__file__+'::same_science_approved_environment','exec'),ns)
# evaluate() and generated source also carry the actual v3 execution identity.
es=inspect.getsource(previous.old.evaluate)
exec(compile(es,__file__+'::same_evaluator','exec'),ns)

def prepare():
    c=load(CYCLE/'traf_physical_reference_recovery_v2_contract.json');assert c['script']==ev(previous.__file__)
    extras={k:v for k,v in c.items() if k not in ('created_at','script','instructions','prediction_before_result','cost','official_main_unchanged','no_model_wakeup')}
    extras.update(approved_legacy_numpy_thermal_shim=ev(SHIM),environment_repair='Restore original np.float=float and np.int=int legacy compatibility, approved v4 wrapper calls existing thermal hook. No algorithm, data, precision, weight or GPU scheduling change.',retired_waiters='v2 retired explicitly while all three status=waiting_unique_prior_GPU_owner, before model initialization/forward; prior C09 graph units untouched')
    register('traf_physical_reference_recovery_v3',__file__,c['prediction_before_result'],c['cost'],**extras)
    for lane in ('gpu0','gpu1','remote'):
        p=CYCLE/('traf_physical_waiter_v2_retired_'+lane+'.json');atomic(p,dict(status='failed_retained',created_at=now(),scientific_GPU_forwards=0,model_initialized=False,reason='Qualified CPU batch exposed missing original legacy-NumPy shim; all three owned waiter units stopped before scientific start, directories retained',successor='peer14_cycle12_traf_physical_recovery_v3_'+lane+'_v1_20261103'))
        record('traf_physical_recovery_v2_'+lane,dict(status='failed_retained',terminal=ev(p)))
        record('traf_physical_recovery_v3_'+lane,dict(status='registered_finite_successor',contract=ev(CYCLE/'traf_physical_reference_recovery_v3_contract.json')))
    record('traf_physical_reference_recovery_v3',dict(status='complete_light_checked',administrative_only=True,scientific_children_open=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','gpu0','gpu1','remote']);p.add_argument('--gpu',type=int,default=0);a=p.parse_args()
    if a.mode=='prepare':prepare()
    else:
        c=load(CYCLE/'traf_physical_reference_recovery_v3_contract.json');assert ev(SHIM)==c['approved_legacy_numpy_thermal_shim'];runpy.run_path(str(SHIM),run_name='approved_native_compatibility')
        import numpy as np
        assert np.float is float and np.int is int
        ns['run'](a.mode,a.gpu)
