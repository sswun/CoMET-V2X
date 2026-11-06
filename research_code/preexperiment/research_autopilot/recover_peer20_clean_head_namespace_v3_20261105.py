"""R6e zero-update namespace correction; preserve recipe REC separately.

v2 accidentally shadowed canonical recipe root before training. No updates or
forward results from that attempt. Exactly the original decontamination design;
only strict60's unused anchor metadata and versioned output path differ.
"""
import argparse,inspect
from peer20_v1_20261105 import *
import run_peer20_clean_head_v1_20261105 as old
import recover_peer20_clean_head_interface_v2_20261105 as prior
RECOVERY=BASE/'R6e_namespace_recovery_v3'

def prepare():
    for suffix in ('strict60','strict60_independent','global_choice'):
        register('R6e_namespace_v3_'+suffix,__file__,__doc__,dict(GPU_forwards=0,updates=6000 if suffix=='strict60' else 0,CPU_threads=1,disk_MiB='<100'),
            original=ev(old.__file__),failed_v2=ev(BASE/'R6e_interface_v2_strict60/terminal.json'),
            canonical_recipe=str(old.REC),recovered_output=str(RECOVERY/'strict60'),science_unchanged=True)

def setup():
    # Reuse only the verified omission of unused metadata, not v2's root map.
    source=inspect.getsource(old.h1.recipe.matched.geometry_material)
    assert source.count('"protected", "anchor",')==1
    source=source.replace('"protected", "anchor",','"protected",')
    ns=dict(old.h1.recipe.matched.__dict__);exec(compile(source,__file__+'::unused_metadata','exec'),ns)
    old.h1.recipe.matched.geometry_material=ns['geometry_material']
    source=inspect.getsource(old.execute).replace("name='R6e_'+domain+('_independent' if review else '')", "name='R6e_namespace_v3_'+domain+('_independent' if review else '')")
    source=source.replace('src=OUT/domain', 'src=RECOVERY/domain')
    ns=dict(old.__dict__,RECOVERY=RECOVERY);assert ns['REC']==old.REC and ns['REC']!=RECOVERY
    exec(compile(source,__file__+'::strict_namespace','exec'),ns);return ns['execute']

def choose():
    source=inspect.getsource(old.choose).replace("name='R6e_global_choice'", "name='R6e_namespace_v3_global_choice'")
    source=source.replace("path=OUT/d/'calibration_ready.json'", "path=(RECOVERY if d=='strict60' else OUT)/d/'calibration_ready.json'")
    ns=dict(old.__dict__,RECOVERY=RECOVERY);exec(compile(source,__file__+'::unchanged_cal_rule','exec'),ns);ns['choose']()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','run','review','choose'));a=p.parse_args()
    prepare() if a.mode=='prepare' else choose() if a.mode=='choose' else setup()('strict60',a.mode=='review')
