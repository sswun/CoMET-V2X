"""R6a independent recovery: raw frame track IDs are not unique.

Original failed review retained. Independently scan all token-identified raw
objects for the joint track+token key rather than a lossy track-only dict.
Production source tokens, geometry, zero definition and all results unchanged.
"""
import argparse,inspect
import audit_peer20_raw_labels_v1_20261105 as old
from peer20_v1_20261105 import *
OUT=BASE/'R6a_raw_labels_independent_v2'

def prepare():
    register('R6a_raw_labels_independent_v2',__file__,__doc__,dict(GPU_forwards=0,CPU_threads=1,disk_MiB='<1 checks only'),
        predecessor=ev(old.REVIEW/'terminal.json'),producer=ev(old.OUT/'terminal.json'),
        correction='Track-only dictionary overwrote multiple raw objects with same track_id. Joint track+token scan is lossless; no geometry association, no threshold change.')

def run():
    ns=dict(old.__dict__,REVIEW=OUT)
    code=inspect.getsource(old.calculate).replace('bytrack.get(track) if independent',"next((o for o in bytoken.values() if str(o['track_id'])==track and str(o['token'])==token),None) if independent")
    exec(compile(code,__file__+'::independent_joint_key_scan','exec'),ns)
    code=inspect.getsource(old.execute).replace("name='R6a_raw_labels_independent' if review", "name='R6a_raw_labels_independent_v2' if review")
    exec(compile(code,__file__+'::independent_execute','exec'),ns)
    ns['execute'](True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','run'));globals()[p.parse_args().mode]()
