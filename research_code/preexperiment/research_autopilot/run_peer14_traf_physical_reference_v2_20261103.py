"""Bootstrap-only path fix; frozen v1 science unchanged, failed units retained."""
import argparse
from pathlib import Path
from peer14_cycle12_v1_20261103 import ROOT,CYCLE,atomic,ev,load,now,record,register
import run_peer14_traf_physical_reference_v1_20261103 as old
text=Path(old.__file__).read_text()
assert text.count("ROOT[v['path']]")==1
text=text.replace("ROOT[v['path']]","ROOT/v['path']")
text=text.replace('traf_physical_reference_contract.json','traf_physical_reference_recovery_v2_contract.json').replace("'traf_physical_'+lane","'traf_physical_recovery_v2_'+lane")
ns=dict(__name__='bootstrap_repaired_traf_reference',__file__=__file__)
exec(compile(text,__file__+'::frozen_v1','exec'),ns)

def prepare():
    oldc=load(CYCLE/'traf_physical_reference_contract.json');assert oldc['script']==ev(old.__file__)
    extras={k:v for k,v in oldc.items() if k not in ('created_at','script','instructions','prediction_before_result','cost','official_main_unchanged','no_model_wakeup')}
    extras['bootstrap_repair']=dict(original_script=ev(old.__file__),original_contract=ev(CYCLE/'traf_physical_reference_contract.json'),scope='Path subscription typo to join in pre-execution GT SHA check only; zero GPU/model/data changes',failed_scientific_forwards=0)
    register('traf_physical_reference_recovery_v2',__file__,oldc['prediction_before_result'],oldc['cost'],**extras)
    for lane in old.plans():
        log=(CYCLE/('traf-physical-'+lane+'_resource/command.log'))
        if lane=='remote':
            import subprocess
            log.parent.mkdir(exist_ok=True)
            subprocess.run(['rsync','-a','--ignore-existing','remote-gpu:'+str(old.REMOTE_ROOT/log.relative_to(ROOT)),str(log)],check=True)
        assert 'PosixPath' in log.read_text()
        failure=CYCLE/('traf_physical_bootstrap_failure_'+lane+'.json');atomic(failure,dict(created_at=now(),status='failed_retained',scientific_GPU_forwards=0,model_loaded=False,reason='GT Path joined by subscription before creating science directory',command_log=ev(log)))
        record('traf_physical_'+lane,dict(status='failed_retained',terminal=ev(failure)))
        record('traf_physical_recovery_v2_'+lane,dict(status='registered_finite_successor',contract=ev(CYCLE/'traf_physical_reference_recovery_v2_contract.json')))
    record('traf_physical_reference_recovery_v2',dict(status='complete_light_checked',administrative_only=True,scientific_children_open=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','gpu0','gpu1','remote']);p.add_argument('--gpu',type=int,default=0);a=p.parse_args();prepare() if a.mode=='prepare' else ns['run'](a.mode,a.gpu)
