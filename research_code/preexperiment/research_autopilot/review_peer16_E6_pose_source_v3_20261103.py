"""Preserve the producer's explicitly eight-column empty-label schema."""
import argparse, inspect
from peer16_v1_20261103 import BASE,ev,register
import review_peer16_E6_pose_source_v2_20261103 as old
OUT=BASE/'E6_independent_v3'
ns=dict(old.__dict__,**old.ns)
ns.update(OUT=OUT,__file__=__file__,old=old.old)
s=inspect.getsource(old.current_gt)
marker="    assert list(full.shape)==record['current_GT_shape']"
assert s.count(marker)==1
s=s.replace(marker,"    if not len(full):full=np.empty((0,8),np.float32)\n"+marker)
exec(compile(s,__file__+'::empty_label_schema','exec'),ns)
import review_peer16_E6_pose_source_v1_20261103 as v1
s=inspect.getsource(v1.review)
a=s.index('            whole=np.column_stack((gt,gl))');b=s.index("            if last!=r['shard']:",a)
s=s[:a]+"            rebuilt,labels=current_gt(method,base,r)\n            assert np.array_equal(rebuilt[mask(rebuilt)],gt[mask(gt)]) and np.array_equal(labels[mask(rebuilt)],gl[mask(gt)]),('common_current_GT',method,w,i)\n"+s[b:]
ns.update(LABEL_CONTEXT=old.LABEL_CONTEXT,runpy=old.runpy,SimpleNamespace=old.SimpleNamespace)
exec(compile(s,__file__+'::GT_schema_review','exec'),ns)
s=inspect.getsource(v1.run).replace("'E6_independent'","'E6_independent_v3'")
exec(compile(s,__file__+'::finite_review','exec'),ns)
def prepare():
    register('E6_independent_v3',__file__,'Same full current-GT digest audit as v2, including extract_frame_gt explicit(0,8) empty tensor. Both prior failures retained; no detector replay.',dict(GPU_forwards=0,CPU_threads=1,wait_h=20,disk_MiB='<30'),producer=ev(v1.PROD/'contract.json'),predecessor=ev(old.OUT/'where/terminal.json'))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run']);a=p.parse_args();prepare() if a.mode=='prepare' else ns['run']()
