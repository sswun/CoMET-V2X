"""Engineering-only gate JSON recovery and finite downstream reconnection.

The independent v1 gate finished all numerical/boolean comparisons, then JSON
rejected numpy.bool_. Preserve its failed terminal and frozen script. Normalize
NumPy scalar types only at the output boundary; all arithmetic/gates unchanged.
Run frozen direct-contrast/table consumers in new task roots with new dependency
paths. No fitting, forwards, additional selection or scientific modifications.
"""
import argparse
import inspect
import numpy as np
from peer18_v1_20261104 import *
import review_peer18_adoption_independent_v1_20261104 as gate
import compare_peer18_accepted_component_v1_20261104 as direct
import extend_peer18_bounded_paper_tables_v1_20261104 as tables

JOBS={
    'gate':('adoption_independent_serialization_v2',gate),
    'direct':('accepted_component_direct_contrasts_recovery_v2',direct),
    'tables':('bounded_paper_tables_recovery_v2',tables),
}
REMAP={
    'adoption_independent_v1':'adoption_independent_serialization_v2',
    'accepted_component_direct_contrasts_v1':'accepted_component_direct_contrasts_recovery_v2',
    'bounded_paper_tables_v1':'bounded_paper_tables_recovery_v2',
}

def plain(v):
    if isinstance(v,np.generic):return v.item()
    if isinstance(v,dict):return {k:plain(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [plain(x) for x in v]
    return v

def prepare(job):
    name,module=JOBS[job]
    dependencies=(['candidate_statistics_independent_launch_v2_'+str(s) for s in (0,2023,2024,2025)] if job=='gate' else
                  ['adoption_independent_serialization_v2'] if job=='direct' else
                  ['adoption_independent_serialization_v2','accepted_component_direct_contrasts_recovery_v2','A6_rigid_sensitivity_v1'])
    register(name,__file__,__doc__,dict(GPU_forwards=0,updates=0,CPU_threads=1,
        panels=2400 if job=='gate' else 390 if job=='direct' else 'accepted evidence only',
        disk_MiB='<40 shared source references; no cache duplication',wall_h='bounded27h dependencies'),
        frozen_engine=ev(module.__file__),
        predecessor=ev(BASE/'adoption_independent_v1/terminal.json'),
        dependencies={n:ev(BASE/n/'contract.json') for n in dependencies},
        invariant='Only JSON scalar normalization and task/dependency paths change. Original gate comparisons and source arithmetic unchanged.')

def run(job):
    name,module=JOBS[job]
    contract=load(BASE/name/'contract.json')
    assert contract['frozen_engine']==ev(module.__file__)
    source=inspect.getsource(module)
    for before,after in REMAP.items():source=source.replace(before,after)
    ns=dict(__name__='peer18_serialization_recovery_'+job,__file__=__file__)
    exec(compile(source,__file__+'::frozen_'+job,'exec'),ns)
    ns['atomic']=lambda path,value:atomic(path,plain(value))
    ns['run']()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run']);p.add_argument('--job',choices=JOBS,required=True)
    a=p.parse_args();globals()[a.mode](a.job)
