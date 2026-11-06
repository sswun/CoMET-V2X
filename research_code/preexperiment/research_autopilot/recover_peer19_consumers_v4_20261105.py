"""Rebind interrupted v3 consumers to strict resource-recovery reviewer.

Only evidence paths change; scientific producer and failed review remain intact.
Dependencies are ready before loading; peer20 removed intermediate deadlines.
"""
import argparse
from pathlib import Path
from peer20_v1_20261105 import *

def run(module,mode):
    path=ROOT/'preexperiment/research_autopilot'/(module+'.py')
    source=path.read_text().replace('recipe_independent_interface_recovery_v2','recipe_independent_resource_recovery_v5')
    ns=dict(__file__=str(path),__name__='peer19_versioned_resource_consumer')
    exec(compile(source,str(path)+'::resource_review_binding_v4','exec'),ns)
    name='resource_recovery_'+module+'_'+mode
    ns['wait']=lambda path,*a,**kw:wait(path,accepted=kw.get('accepted',('complete_light_checked','complete_independently_reviewed')))
    ns['record']=lambda unused,values:record(name,values)
    ns[mode]()
    atomic(BASE/name/'terminal.json',dict(created_at=now(),status='complete_light_checked',
        original_script=ev(path),strict_reviewer=ev(P19/'recipe_independent_resource_recovery_v5/strict60/terminal.json'),
        scientific_acceptance='See canonical consumer output; execution receipt is not whole-paper acceptance'))
    record(name,dict(status='complete_light_checked',terminal=ev(BASE/name/'terminal.json'),owner=None,resource_owner=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--module',required=True);p.add_argument('--mode',default='run')
    run(**vars(p.parse_args()))
