"""Engineering-only recovery: omit unused optional anchor metadata; reuse q cal.

No fabricated anchor array. geometry_material/train_head/source_map never read
this metadata; anchor observations remain reconstructed from source==0 in H1.
All numerical geometry, labels, fitting, selection and formal scoring unchanged.
Original strict60 failure and coordinator/reviewer failures are retained.
"""
import argparse,inspect,os
from peer19_v1_20261104 import *
import run_peer19_recipe_v3_20261104 as prod
import review_peer19_recipe_v3_20261104 as review
OLD=prod.OUT
REC=BASE/'recipe_v3_interface_recovery_v2'
NAMES=dict(recipe_strict60='recipe_strict60_interface_recovery_v2',
    recipe_global_choice='recipe_global_choice_interface_recovery_v2',
    recipe_independent_strict60='recipe_independent_strict60_interface_recovery_v2')

def prepare():
    REC.mkdir(exist_ok=False)
    for d in prod.DOMAINS:
        if d!='strict60':(REC/d).symlink_to(OLD/d,target_is_directory=True)
    for key,name in NAMES.items():
        register(name,__file__,__doc__,dict(GPU_forwards=0,updates=6000 if key=='recipe_strict60' else 0,
            CPU_threads=1,wall_h='bounded30h',disk_MiB='<650 compact outputs/heads; old qcal shared by symlink'),
            frozen_producer=ev(prod.__file__),frozen_reviewer=ev(review.__file__),
            frozen_geometry=ev(prod.matched.__file__),old_failure=ev((BASE/'recipe_global_choice' if key=='recipe_global_choice' else (BASE/'recipe_independent/strict60' if key.startswith('recipe_independent') else OLD/'strict60'))/'terminal.json'),
            q_calibration=ev(OLD/'strict60/q_calibration.json'),no_scientific_change=True)

def setup():
    prod.OUT=REC
    original=record
    def mapped(name,values):original(NAMES.get(name,name),values)
    prod.record=mapped;review.record=mapped
    source=inspect.getsource(prod.matched.geometry_material)
    assert source.count('"protected", "anchor",')==1
    source=source.replace('"protected", "anchor",','"protected",')
    ns=dict(prod.matched.__dict__);exec(compile(source,__file__+'::omit_unused_anchor_metadata','exec'),ns)
    prod.matched.geometry_material=ns['geometry_material']
    review.OUT=BASE/'recipe_independent_interface_recovery_v2'

def run():
    setup();dest=REC/'strict60'
    source=inspect.getsource(prod.run)
    source=source.replace('calq=[]',"calq=load(OLD/'strict60/q_calibration_panels.json')")
    assert source.count('for q in Q:')==1
    source=source.replace('for q in Q:','for q in ():')
    marker="record(name,dict(status='running_H1_fitcal_recipe',resource_owner=True))"
    source=source.replace(marker,marker+"\n    for p in (OLD/'strict60').glob('q*_cal_predictions.npz'):(dest/p.name).symlink_to(p)\n    atomic(dest/'q_calibration_panels.json',load(OLD/'strict60/q_calibration_panels.json'))\n    atomic(dest/'engineering_recovery.json',dict(original_failure=ev(OLD/'strict60/terminal.json'),shared_qcal=ev(OLD/'strict60/q_calibration.json'),unused_anchor_metadata_omitted=True))")
    ns=dict(prod.__dict__,OLD=OLD);exec(compile(source,__file__+'::resume_completed_qcal','exec'),ns);ns['run']('strict60')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run','choose','review']);a=p.parse_args()
    if a.mode=='prepare':prepare()
    elif a.mode=='run':run()
    else:
        setup()
        if a.mode=='choose':prod.choose()
        else:review.run('strict60')
