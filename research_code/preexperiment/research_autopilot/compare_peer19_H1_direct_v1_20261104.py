"""R1/R2 actual H1 vs exact accepted native/stronger-late/module references.

390 independently integrated paired sequence2000CI panels; no added old AP
deltas, no N2/N0 relabeling. Wait actual H1 independent saved-state AP review.
"""
import argparse,inspect
from peer19_v1_20261104 import *
import compare_peer18_accepted_component_v1_20261104 as old
OUT=BASE/'H1_direct_contrasts'
REC=BASE/'recipe_v3_interface_recovery_v2'

def current(domain,role,seed,variant):
    review=BASE/('recipe_independent_interface_recovery_v2' if domain=='strict60' else 'recipe_independent')/domain
    wait(review/'terminal.json',34,accepted=('complete_independently_reviewed',))
    return REC/domain/(role+'_'+variant.removeprefix('H1_')+'_seed'+str(seed)+'_predictions.npz')

def prepare():
    register('H1_direct_contrasts',__file__,__doc__,dict(GPU_forwards=0,updates=0,CPU_threads=1,panels=390,replicates=2000,wall_h='bounded36h',disk_MiB='<30 compact CI'),
        kernel=ev(old.ind.__file__),frozen_contrast_engine=ev(old.__file__),
        main_recipe=ev(BASE/'recipe_global_choice_interface_recovery_v2/contract.json'),
        module_scope='Earlier method-specific geometry-matched no-position readout3heads, equal opportunity but not identical weights. Separate R4 identical H1 weights is primary fairness control.')

def run():
    source=inspect.getsource(old.run).replace("'accepted_component_direct_contrasts_v1'","'H1_direct_contrasts'")
    source=source.replace("BASE/'adoption_independent_v1/terminal.json'","BASE/'recipe_global_choice_interface_recovery_v2/terminal.json'")
    source=source.replace("assert load(dep)['status']=='complete_independently_reviewed'","assert load(dep)['status']=='complete_light_checked'")
    source=source.replace("decision=load(dep)['recommendation']","decision='CoMET_all_ego_H1'")
    source=source.replace("v='N4' if decision=='CoMET_Net_N4' else load(BASE/'adoption_independent_v1/method_registry.json')['component']","v='H1_'+load(REC/'method_registry.json')['selected_kind']")
    source=source.replace("assert v in ('N2','N3','N4'),('No prior learned formal predictions for this component',v)","assert v in ('H1_frozen','H1_refit')")
    # Baseline sources explicitly stay peer18; only current H1 and outputs change.
    ns=dict(old.__dict__,BASE=BASE,OUT=OUT,REC=REC,current=current,record=record)
    import inspect as ins
    exec(compile(ins.getsource(old.contrast),__file__+'::independent_current_H1_CI','exec'),ns)
    exec(compile(source,__file__+'::actual_H1_direct_comparisons','exec'),ns);ns['run']()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run']);globals()[p.parse_args().mode]()
