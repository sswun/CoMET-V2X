"""R6e strict60 interface-only recovery; omit unused optional anchor metadata.

Original failure/output retained. No numerical, sampling, training or selection
change. Other four domains continue unmodified. Shared global choice is written
by this sole replacement coordinator, with explicit mixed-version source paths.
"""
import argparse,inspect,time
from peer20_v1_20261105 import *
import run_peer20_clean_head_v1_20261105 as old
REC=BASE/'R6e_interface_recovery_v2'

def prepare():
    for suffix in ('strict60','strict60_independent','global_choice'):
        register('R6e_interface_v2_'+suffix,__file__,__doc__,dict(GPU_forwards=0,updates=6000 if suffix=='strict60' else 0,
            CPU_threads=1,disk_MiB='<100 shared schedules/heads and scores'),original=ev(old.__file__),
            failure=ev(BASE/'R6e_strict60/terminal.json'),science_unchanged=True,
            output=str(REC/'strict60'),global_choice=str(old.OUT/'global_choice'))

def setup():
    source=inspect.getsource(old.h1.recipe.matched.geometry_material)
    assert source.count('"protected", "anchor",')==1
    source=source.replace('"protected", "anchor",','"protected",')
    ns=dict(old.h1.recipe.matched.__dict__);exec(compile(source,__file__+'::unused_metadata','exec'),ns)
    old.h1.recipe.matched.geometry_material=ns['geometry_material']
    source=inspect.getsource(old.execute).replace("name='R6e_'+domain+('_independent' if review else '')", "name='R6e_interface_v2_'+domain+('_independent' if review else '')")
    source=source.replace('src=OUT/domain', 'src=REC/domain')
    ns=dict(old.__dict__,REC=REC);exec(compile(source,__file__+'::strict_interface','exec'),ns)
    return ns['execute']

def choose():
    # Original coordinator is stopped and recorded superseded before launch.
    source=inspect.getsource(old.choose).replace("name='R6e_global_choice'", "name='R6e_interface_v2_global_choice'")
    source=source.replace("path=OUT/d/'calibration_ready.json'", "path=(REC if d=='strict60' else OUT)/d/'calibration_ready.json'")
    ns=dict(old.__dict__,REC=REC);exec(compile(source,__file__+'::same_cal_rule','exec'),ns);ns['choose']()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','run','review','choose'));a=p.parse_args()
    prepare() if a.mode=='prepare' else choose() if a.mode=='choose' else setup()('strict60',a.mode=='review')
