"""Full-test receiver pose clock perturbations; zero new detector forwards."""
import argparse, inspect
from collections import defaultdict
import numpy as np
from scipy.spatial.transform import Rotation
from peer16_v1_20261103 import BASE,atomic,ev,load,register
import run_peer15_opv_full_read_v1_20261103 as old
OUT=BASE/'E9_opv_pose_time'
FAMILIES=('ego_pose_time_050ms','ego_pose_time_100ms','ego_pose_time_150ms')
def pose_time_arrays(rows,arrays,delta):
    """Time-skew only receiver localization; collaborator world poses stay clean."""
    last={};result=[];audit=[]
    for i,(r,a) in enumerate(zip(rows,arrays)):
        seq=r['sequence'];T=a['ego_pose_world'];S=T.copy();known=seq in last
        if known:
            prev=last[seq];dt=.1
            S[:3,3]=T[:3,3]-delta*(T[:3,3]-prev[:3,3])/dt
            omega=Rotation.from_matrix(prev[:3,:3].T@T[:3,:3]).as_rotvec()/dt
            S[:3,:3]=T[:3,:3]@Rotation.from_rotvec(-delta*omega).as_matrix()
        last[seq]=T
        remote=np.stack([np.linalg.inv(S)@T@R for R in a['remote_to_ego']])
        aa=dict(a,ego_pose_world=S,remote_to_ego=remote);result.append(aa)
        assert np.allclose(S@remote,T@a['remote_to_ego'],atol=1e-10,rtol=0)
        audit.append(dict(receiver_index=i,history_available=known,delta_ms=1000*delta,pose=S.tolist()))
    return result,audit
def delta_geometry(m,meta,noisy_arrays):
    from run_peer16_cached_methods_v1_20261103 import observation,joint_geometry
    rows=[dict(sequence=x['sequence'],wait_ms=x['wait_ms'],receiver_timestamp_us=int(m['receiver_timestamp_us'][i]),current_ego_pose_lidar_to_world=noisy_arrays[x['receiver_index']]['ego_pose_world'].tolist()) for i,x in enumerate(meta)]
    return joint_geometry(m,rows,observation(m),'V2_joint',.05,0.,{})
ns=dict(old.__dict__,OUT=OUT,__file__=__file__,FAMILIES=FAMILIES,pose_time_arrays=pose_time_arrays,delta_geometry=delta_geometry)
ns['METHODS']=old.METHODS+('CoMET_delta_heading_U',)
s=inspect.getsource(old.run)
s=s.replace("'opv_full_read'","'E9_opv_pose_time'")
s=s.replace('        rows=collect();assert len(rows)==2170',"        rows=load(INPUT/'records.json');assert len(rows)==2170\n        assert load(INPUT/'terminal.json')['full_standard_test']")
s=s.replace("        for family in ('SPD_structured','iid_02m_02deg'):\n            noise=noises(rows,arrays,family)","        for family in FAMILIES:\n            delay=int(family[-5:-2])/1000\n            noisy_arrays,audit=pose_time_arrays(rows,arrays,delay)\n            atomic(OUT/(family+'_pose_audit.json'),audit)\n            noise={(i,sender):np.eye(4) for i,r in enumerate(rows) for sender in range(1,len(r['sensors']))}")
s=s.replace('m,b,f,d,meta,n=material(rows,arrays,detections,noise,wait)','m,b,f,d,meta,n=material(rows,noisy_arrays,detections,noise,wait)')
s=s.replace('                scores={k:refit.score_with_head',"                db,df,clock=delta_geometry(m,meta,noisy_arrays)\n                methods[METHODS[3]]=(m,db,df)\n                scores={k:refit.score_with_head")
s=s.replace("delta=d,anchor_count=n)","delta=d,anchor_count=n,joint_clock_state=clock)")
s=s.replace("assert c['script']==ev(__file__)","assert c['script']==ev(__file__)")
exec(compile(s,__file__+'::full_test_cached_pose_time','exec'),ns)
def prepare():
    register('E9_opv_pose_time',__file__,
        'Receiver localization time skew creates a common displacement/rotation field. CoMET and joint delta may recover it or fail; preserve all3 fixed clock doses at0/400ms, unchanged full2170 test and all CAVs. No test tuning.',
        dict(GPU_forwards=0,CPU_threads=1,updates=0,wall_h='1-6',disk_MiB='<180 compact predictions and pose audits; full detector shards referenced'),
        full_detector=ev(old.BASE/'opv_full_test_v2/producer/terminal.json'),baseline=ev(old.OUT/'terminal.json'),baseline_independent=ev(old.BASE/'opv_full_independent/terminal.json'),
        noise='Receiver-only public pose backward constant-linear/angular-velocity extrapolation50/100/150ms; first frame identity without history. Exact clean sender world poses preserved. 10Hz simulator clock, not measured SPD clock.',
        joint='Same frozen E2 V2 [.1s prior,.005s Q,.05m XY Q,.6m R], matched614b head transfer; no eval-GT clock fit',
        waits=[0,400],reuse='SPD_structured/iid panels remain in peer15 full read; not repeated',selection='None on test; all doses retained; official SPD protocol unchanged')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run']);a=p.parse_args();prepare() if a.mode=='prepare' else ns['run']()
