"""Exact checkpoint-layout repair; follows full OPV test, no scientific change."""
import argparse, inspect, runpy, time, subprocess, traceback
from pathlib import Path
import run_peer14_traf_physical_reference_v3_20261103 as v3
from peer15_v1_20261103 import ROOT, BASE, C12, C09, atomic, ev, load, now, register, record

OUT=BASE/'traf_physical_layout_v4'

def qualified_load(model, weight, output):
    import torch
    state=torch.load(weight,map_location='cpu'); expected=model.state_dict(); changed=[]
    for key,value in list(state.items()):
        if key not in expected or tuple(value.shape)==tuple(expected[key].shape):continue
        assert key.startswith('backbone.blocks.') and key.endswith('.conv2.weight'),key
        assert value.ndim==4 and tuple(value.permute(3,0,1,2).shape)==tuple(expected[key].shape),(key,value.shape,expected[key].shape)
        converted=value.permute(3,0,1,2).contiguous()
        assert torch.equal(converted.permute(1,2,3,0).contiguous(),value),key
        changed.append(dict(key=key,source_shape=list(value.shape),runtime_shape=list(converted.shape),inverse_exact=True))
        state[key]=converted
    model.load_state_dict(state,strict=True)
    atomic(Path(output)/'checkpoint_layout_receipt.json',dict(created_at=now(),weight=ev(weight),changes=changed,strict_load=True,original_checkpoint_unchanged=True,transform='only shape-mismatched backbone.blocks.*.conv2.weight: permute(3,0,1,2), exact inverse verified'))

def source(profiling):
    s=v3.previous.old.source(profiling)
    old="model.load_state_dict(torch.load(a.weight,map_location='cpu'),strict=True)"
    assert s.count(old)==1
    return s.replace(old,"__import__('run_peer15_traf_physical_layout_v4_20261103').qualified_load(model,a.weight,out)")

ns=dict(v3.ns,source=source,__file__=__file__)
s=inspect.getsource(v3.previous.old.run).replace("ROOT[v['path']]","ROOT/v['path']").replace('traf_physical_reference_contract.json','traf_physical_reference_recovery_v4_contract.json').replace("'traf_physical_'+lane","'traf_physical_recovery_v4_'+lane")
exec(compile(s,__file__+'::same_science','exec'),ns)
exec(compile(inspect.getsource(v3.previous.old.evaluate),__file__+'::same_evaluate','exec'),ns)

def prepare():
    prior=load(C12/'traf_physical_reference_recovery_v3_contract.json')
    extras={k:v for k,v in prior.items() if k not in ('created_at','script','instructions','prediction_before_result','cost','official_main_unchanged','no_model_wakeup')}
    extras.update(prior_contract=ev(C12/'traf_physical_reference_recovery_v3_contract.json'),new_GPU_order='Wait for peer15 OPV full-test v2 terminal, then unchanged e60/public3/online3 chain',repair='Exact inverse-checked spconv channel-axis permutation, not precision/weight-value/algorithm changes; existing runtime-layout weights unchanged')
    register('traf_physical_layout_v4',Path(__file__).resolve(),prior['prediction_before_result'],prior['cost'],**extras)
    # Existing source consumers expect the original output root. New recovery has
    # its own contract and retains the failed zero-forward output before reuse.
    atomic(C12/'traf_physical_reference_recovery_v4_contract.json',dict(prior,created_at=now(),script=ev(__file__),layout_repair_contract=ev(OUT/'contract.json')))
    source(True);source(False)

def remote():
    start=time.monotonic();prior='embodied-peer15-opv-full-test-v2-v1-20261103.service'
    record('traf_physical_layout_v4',dict(status='waiting_OPV_full_test_v2_owner',prior_unit=prior))
    while subprocess.check_output(['systemctl','--user','show',prior,'-p','ActiveState'],text=True).strip() in ('ActiveState=active','ActiveState=activating','ActiveState=deactivating'):
        assert time.monotonic()-start<14*3600,'finite14h OPV wait'
        atomic(OUT/'waiting.json',dict(created_at=now(),status='waiting_unique_OPV_owner',prior_unit=prior));time.sleep(30)
    assert (BASE/'opv_full_test_v2/producer/terminal.json').exists(),'OPV terminal required; do not overlap'
    lane=C12/'traf_physical_reference_v1/remote';retained=lane.with_name('remote_failed_checkpoint_layout_v3_retained')
    assert load(lane/'terminal.json')['status']=='failed_retained'
    assert not retained.exists();lane.rename(retained)
    atomic(OUT/'failure_adjudication.json',dict(created_at=now(),retained_terminal=ev(retained/'terminal.json'),scientific_forwards=0,reason='e60 conv2 checkpoint channel layout differs from installed spconv runtime; strict load failed before any forward',original_terminal_unchanged=True))
    runpy.run_path(str(v3.SHIM),run_name='approved_native_compatibility')
    record('traf_physical_layout_v4',dict(status='running_layout_qualified_same_ICP_science'))
    ns['run']('remote',0)
    record('traf_physical_layout_v4',dict(status='complete_light_checked',terminal=ev(lane/'terminal.json')))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','remote']);a=p.parse_args();globals()[a.mode]()
