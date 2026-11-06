"""New root-resolved native1 recovery; no source/label/algorithm edits.

The native4 metadata view has no vehicle-side/velodyne aliases. Where's
released ego retrieval must use the already staged main-domain data view.
"""
import os
import run_peer12_where_native_pair_v1_20261102 as old
from peer12_cycle06_v1_20261102 import ROOT,CYCLE,ev,update,task
BASE_BUILD=old.build

def build(root):
    h,builder,roots,config,split=BASE_BUILD(root)
    h['root_dir']=str(roots[1])+os.sep
    return h,builder,roots,config,split

def main():
    old.build=build
    original=ROOT/'preexperiment/research_autopilot/run_peer12_where_native_pair_recovery_v2_20261102.py'
    s=original.read_text().replace('_v2','_v4')
    marker='assert s.count(a)==1,(a,s.count(a));s=s.replace(a,b)'
    assert s.count(marker)==2
    s=s.replace(marker,'assert s.count(a)==(2 if a=="c=load(CYCLE/\'where_contract.json\')" and "def lane" in s else 1),(a,s.count(a));s=s.replace(a,b)')
    ns={'__name__':'where_recovery_v4','__file__':__file__}
    exec(compile(s,__file__+'::resolved_native1_ego_PCD_root','exec'),ns)
    if 'prepare' in __import__('sys').argv:
        for arm in ('public','online_k1'):
            update('peer12_where_native_'+arm+'_v3_20261102',task('K4'),dict(status='failed_retained',GPU_forwards=0,updates=0,reason='Unstarted lanes superseded after actual CPU loader revealed absent author-view velodyne alias; shared-view PCD already exists',failure=ev(CYCLE/'where_qualification_v3/terminal.json'),superseded_by='peer12_where_native_'+arm+'_v4_20261102'))
    ns['main']()

if __name__=='__main__':main()
