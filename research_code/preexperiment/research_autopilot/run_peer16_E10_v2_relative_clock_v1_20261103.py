"""V2V4Real current-cache joint clock transfer with two moving public poses."""
import argparse
import inspect
import time
import traceback
from collections import defaultdict
import numpy as np
from peer16_v1_20261103 import ROOT, AUTO, BASE, atomic, ev, load, now, register, record
import run_peer10_v2_transfer_v1_20261101 as transfer
import run_peer16_cached_methods_v1_20261103 as clock
from run_peer16_E1_timing_audit_v1_20261103 import corr
fac=transfer.fac
INPUT=AUTO/'status_audit_20261030_v2v4real_current_q005_heading_anchor_gate_factorial_v1'
OUT=BASE/'E10_v2v4real_relative_clock'

def prepare():
    register('E10_v2v4real_relative_clock',__file__,
             'Joint clock structure may transfer or fail on moving-peer data. Evaluate every registered arm, original V2 population and own matched head; never swap in SPD geometry/head or choose evaluation cells.',
             dict(GPU_forwards=0,CPU_threads=1,updates=0,wall_h='1-4',disk_MiB='<80 compact predictions/clock states'),
             input_contract=ev(INPUT/'contract_recovery_v6.json'),
             head=ev(INPUT/'producer_v1/q005_cv_matched39d_head/direct_u.pth'),
             arms=['stale_same_U','CoMET_current','delta_ego_only','delta_relative_motion'],
             relative_model='A common pose-clock displacement at both agents: m=Re.T(ve-vs)+ws*J*pivot+(we-ws)*J*r. Backward differences from previously available public sender and receiver poses per sequence/wait. This common-clock assumption is a transferred hypothesis, not established physical truth.',
             timing='Existing V2 10Hz frame proxy, not independently measured network/pose timestamps. Missing/large-gap history yields zero motion; cold-start cases retained.',
             parameters='Exact E2 prior0.1s,Qclock0.005s,Qxy0.05m,R0.6m; no V2 tuning or GT clock regression',
             decision='val mean AP70>=stale-.5pp and hold>stale for second-real-domain claim; otherwise retain limitation. Independent review pending.')

def relative_dynamics(rows):
    previous={};out={}
    for r in rows:
        seq=str(r['sequence']);wait=int(r['wait_ms']);t=int(r['receiver_timestamp_us'])
        E=np.asarray(r['current_ego_pose_lidar_to_world'],float);s=r.get('source')
        key=(seq,wait);old=previous.get(key);v=np.zeros(2);omega=0.;ok=False
        if s:
            S=np.asarray(s['source_pose_lidar_to_world'],float);st=int(s['source_timestamp_us'])
            assert st<=t
            if old:
                pt,pE,pst,pS,vs,ws=old;dt=(t-pt)/1e6;ds=(st-pst)/1e6
                if 0<ds<=.35:
                    vs=(S[:2,3]-pS[:2,3])/ds
                    a=np.arctan2(S[1,0],S[0,0])-np.arctan2(pS[1,0],pS[0,0]);ws=np.arctan2(np.sin(a),np.cos(a))/ds
                if 0<dt<=.35 and 0<=ds<=.35:
                    ve=(E[:2,3]-pE[:2,3])/dt
                    a=np.arctan2(E[1,0],E[0,0])-np.arctan2(pE[1,0],pE[0,0]);we=np.arctan2(np.sin(a),np.cos(a))/dt
                    pivot=np.linalg.solve(E,S)[:2,3]
                    v=E[:2,:2].T@(ve-vs)+ws*np.array([-pivot[1],pivot[0]])
                    omega=float(we-ws);ok=True
            else:vs=np.zeros(2);ws=0.
            previous[key]=(t,E,st,S,vs,ws)
        # Distinguish waits: E2's original (seq,t) dynamics key otherwise collides.
        out[seq+'::wait'+str(wait),t]=(v,omega,ok)
    return out

def joint(m,rows,relative):
    rr=[dict(r,sequence=str(r['sequence'])+'::wait'+str(r['wait_ms'])) for r in rows]
    dyn=relative_dynamics(rows) if relative else clock.dynamics(rr)
    source=inspect.getsource(clock.joint_geometry)
    source=source.replace('dyn=dynamics(rows)','dyn=PRECOMPUTED')
    source=source.replace('author.heading.build_heading(m,np.zeros((len(rows),2)))','fac.heading_geometry(m,np.zeros((len(rows),2)))')
    ns=dict(clock.__dict__,PRECOMPUTED=dyn,fac=fac)
    exec(compile(source,__file__+'::joint_state','exec'),ns)
    return ns['joint_geometry'](m,rr,clock.observation(m),'V2_joint',.05,0.,{}),dyn

def run():
    start=time.monotonic();cpu=time.process_time();panels=[];correlations=[]
    record('E10_v2v4real_relative_clock',dict(status='running_actual_V2_public_pose_relative_clock'))
    try:
        c=load(INPUT/'contract_recovery_v6.json');head=INPUT/'producer_v1/q005_cv_matched39d_head'
        for role in ('mechanism_holdout14','val21'):
            ds=fac.PackedDataset(ROOT/c['datasets']['val' if role=='val21' else 'train'])
            m,rows=transfer.material(c,role,ds)
            with np.load(INPUT/'producer_v1'/(role+'_q005_state.npz')) as z:
                assert np.array_equal(z['state_key_sequence'],m['sequence']);d=z['q005_delta'].copy()
            b,f=fac.heading_geometry(m,d)
            boxes=dict(stale_same_U=m['stale_boxes'],CoMET_current=b)
            scores=dict(stale_same_U=clock.author.refit.score_with_head(m,m['stale_features'],head),
                        CoMET_current=clock.author.refit.score_with_head(m,f,head))
            states={}
            for name,relative in [('delta_ego_only',False),('delta_relative_motion',True)]:
                (b,f,x),dyn=joint(m,rows,relative);boxes[name]=b;scores[name]=clock.author.refit.score_with_head(m,f,head);states[name]=x
                if relative:
                    grouped=defaultdict(list)
                    for r,(ids,z) in zip(rows,clock.observation(m)):
                        v,w,ok=dyn[str(r['sequence'])+'::wait'+str(r['wait_ms']),int(r['receiver_timestamp_us'])]
                        if ok:
                            for residual in z:grouped[int(r['wait_ms'])].append((str(r['sequence']),v[0],residual[0]))
                    for wait,values in grouped.items():
                        seq=np.asarray([x[0] for x in values]);a=np.asarray([x[1:] for x in values]);centered=a.copy()
                        for s in set(seq):centered[seq==s]-=a[seq==s].mean(0)
                        correlations.append(dict(role=role,wait_ms=wait,anchors=len(values),label_free_relative_velocity_residual_r=corr(a[:,0],a[:,1]),within_sequence_r=corr(centered[:,0],centered[:,1])))
            fac.METHODS=tuple(boxes);pp,p=fac.evaluate_v2(ds,m,boxes,scores)
            previous=clock.author.formal.load_npz(INPUT/'producer_v1'/(role+'_factorial_predictions.npz'))
            checks={k:np.array_equal(p['CoMET_current__'+k],previous['heading_q005_base__'+k]) for k in ('candidate','score','tp50','tp70','offset')}
            assert all(checks.values()),checks
            atomic(OUT/(role+'_baseline_replay.json'),checks)
            panels.extend(dict(x,role=role) for x in pp);atomic(OUT/'panels.json',panels)
            np.savez_compressed(OUT/(role+'_predictions.npz'),**p)
            np.savez_compressed(OUT/(role+'_clock_states.npz'),**states)
            atomic(OUT/'correlations.json',correlations)
        atomic(OUT/'terminal.json',dict(created_at=now(),status='complete_light_checked',GPU_forwards=0,
               wall_s=time.monotonic()-start,CPU_s=time.process_time()-cpu,panels=ev(OUT/'panels.json'),
               original_K1_replay_exact=True,independent_pending=True,no_eval_selection=True))
        record('E10_v2v4real_relative_clock',dict(status='complete_light_checked',terminal=ev(OUT/'terminal.json')))
    except Exception as e:
        atomic(OUT/'terminal.json',dict(created_at=now(),status='failed_retained',error=repr(e),traceback=traceback.format_exc()))
        record('E10_v2v4real_relative_clock',dict(status='failed_retained',terminal=ev(OUT/'terminal.json')));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run']);globals()[p.parse_args().mode]()
