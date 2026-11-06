"""Causal, label-free R8b/R8d estimators; no evaluation or perturbation inputs.

Review paths reconstruct motion, matching and KF arithmetic separately. The
common upstream boxes/online poses and fixed hyperparameters remain shared.
"""
import ast,hashlib
from collections import defaultdict,deque
import numpy as np
from scipy.optimize import linear_sum_assignment
BOX_KEYS=('stale_boxes','cv_boxes','huber_boxes','supervised_constant_boxes','kf_boxes')

def matrix(value):return np.asarray(ast.literal_eval(value) if isinstance(value,str) else value,float)

def motion(rows,review=False):
    frames={}
    for r in rows:
        key=str(r['sequence']),int(r['receiver_timestamp_us']);T=matrix(r['current_ego_pose_lidar_to_world'])
        if key in frames:assert np.array_equal(frames[key],T)
        frames[key]=T
    result={};previous={}
    for (s,t),B in sorted(frames.items()):
        u=np.zeros(2)
        if s in previous:
            pt,A=previous[s];dt=(t-pt)/1e6;assert dt>0
            if review:
                displacement=B[:3,3]-A[:3,3];forward=sum(B[j,0]*displacement[j] for j in range(3))
                angle=float(np.arctan2(B[1,0],B[0,0])-np.arctan2(A[1,0],A[0,0]))
                angle=(angle+np.pi)%(2*np.pi)-np.pi;u[:]=forward/dt,angle/dt
            else:
                d=B[:3,:3].T@(B[:3,3]-A[:3,3])
                yaw=np.arctan2(B[1,0],B[0,0])-np.arctan2(A[1,0],A[0,0]);yaw=(yaw+np.pi)%(2*np.pi)-np.pi
                u[:]=d[0]/dt,yaw/dt
        result[s,t]=u;previous[s]=(t,B)
    return np.asarray([result[str(r['sequence']),int(r['receiver_timestamp_us'])] for r in rows])

def match(remote,ego,shift=None,review=False):
    if not len(remote) or not len(ego):return np.empty((0,2)),0
    # Keep float32 matching, then float64 residual, exactly as selected H1.
    moved=remote.copy()
    if shift is not None:moved[:,:2]+=np.asarray(shift,dtype=moved.dtype)
    if review:
        distance=np.sqrt(np.sum((moved[:,None,:2]-ego[None,:,:2])**2,axis=2))
        ri,ei=linear_sum_assignment(distance);use=[]
        for a,b in zip(ri,ei):
            if distance[a,b]<=2.5 and int(np.argmin(distance[a]))==b and int(np.argmin(distance[:,b]))==a:use.append((a,b))
        ri=np.asarray([p[0] for p in use],int);ei=np.asarray([p[1] for p in use],int)
    else:
        from run_peer16_E1_timing_audit_v1_20261103 import match as original
        ri,ei=original(moved,ego,2.5)
    residual=ego[ei,:2].astype(float)-moved[ri,:2].astype(float)
    return residual,len(ri)

def sets(m):
    out=[]
    for a,b in zip(m['condition_offsets'][:-1],m['condition_offsets'][1:]):
        a,b=int(a),int(b);src=m['source'][a:b];out.append((m['cv_boxes'][a:b][src==1],m['cv_boxes'][a:b][src==0]))
    return out

def observations(m,review=False):
    residual=[match(r,e,review=review)[0] for r,e in sets(m)]
    return np.asarray([len(x) for x in residual]),np.asarray([x.mean(0) if len(x) else np.zeros(2) for x in residual])

def filter_state(m,n,z,q,mean=None,p0=1.,review=False):
    states={};out=[];steps=[];seen=[]
    for i,(s,w,count,measure) in enumerate(zip(m['sequence'],m['wait_ms'],n,z)):
        key=str(s),int(w);x,p,k,ever=states.get(key,(np.zeros(2),p0,0,False));x=x.copy();pp=p+q*q
        mu=np.zeros(2) if mean is None else mean[i]
        if count:
            observation=measure-mu;gain=pp/(pp+.6*.6/count)
            if review:x=np.asarray([x[j]+gain*(observation[j]-x[j]) for j in range(2)])
            else:x+=gain*(observation-x)
            p=(1-gain)*pp;ever=True
        else:p=pp
        states[key]=(x,p,k+1,ever);out.append(x+mu);steps.append(k);seen.append(ever)
    return np.asarray(out),np.asarray(steps),np.asarray(seen)

def fit_motion(m,rows,q,speed=False,review=False):
    u=motion(rows,review);n,z=observations(m,review);d,k,seen=filter_state(m,n,z,q,review=review)
    use=(k>50)&seen;assert use.sum()>10
    X=np.column_stack((np.ones(use.sum()),u[use,:1] if speed else u[use]));Y=d[use]
    coef=np.linalg.lstsq(X,Y,rcond=None)[0]
    # The speed ablation has y intercept only, rather than a fitted y speed.
    if speed:
        coef[1,1]=0.;coef[0,1]=Y[:,1].mean()
    residual=Y-X@coef;p0=float(residual.var(axis=0).mean());assert p0>0
    return dict(coef=coef.tolist(),p0=p0,fit_messages=int(use.sum()),speed=speed,
        fitted_on='fit24 converged label-free H1 states; zero-based step>50 and an anchor already seen',no_labels=True)

def motion_state(m,rows,q,fit,review=False):
    u=motion(rows,review);X=np.column_stack((np.ones(len(rows)),u[:,:1] if fit['speed'] else u))
    mu=X@np.asarray(fit['coef']);n,z=observations(m,review)
    return filter_state(m,n,z,q,mu,fit['p0'],review)[0]

def acquire(window,review=False):
    values=[];messages=[]
    for k,(r,e) in enumerate(window):
        d=(e[None,:,:2].astype(float)-r[:,None,:2].astype(float)).reshape(-1,2)
        use=np.linalg.norm(d,axis=1)<=10.;values.extend(d[use]);messages.extend([k]*int(use.sum()))
    if len(values)<6:return None
    d=np.asarray(values);messages=np.asarray(messages);edges=np.arange(-10,10.5,.5)
    if review:
        H=np.zeros((40,40),float)
        for x,y in d:
            a=min(39,int(np.searchsorted(edges,x,side='right')-1));b=min(39,int(np.searchsorted(edges,y,side='right')-1));H[a,b]+=1
    else:H=np.histogram2d(d[:,0],d[:,1],bins=(edges,edges))[0]
    order=np.argsort(H.ravel())[::-1];a,b=np.unravel_index(order[0],H.shape);center=np.asarray((-9.75+.5*a,-9.75+.5*b))
    mask=np.linalg.norm(d-center,axis=1)<=1.
    if mask.sum()<6 or len(set(messages[mask]))<3:return None
    second=0.
    for f in order[1:]:
        i,j=np.unravel_index(f,H.shape)
        if max(abs(i-a),abs(j-b))>1:second=H[i,j];break
    value=d[mask].mean(0)
    if np.linalg.norm(value)<2. or (second>0 and H[a,b]/second<1.5):return None
    return value

def acquisition_state(m,q,enabled=True,review=False):
    result=[];events=[];states={};counts=[];everlock=[];stream_triggers=defaultdict(int)
    for i,((r,e),s,w) in enumerate(zip(sets(m),m['sequence'],m['wait_ms'])):
        key=str(s),int(w);x,p,xacq,lost,hist,ever=states.get(key,(np.zeros(2),1.,np.zeros(2),0,deque(maxlen=5),False))
        hist.append((r,e));res,n=match(r,e,None if not np.any(xacq) else xacq,review)
        pp=p+q*q;x=x.copy()
        if n:
            z=xacq+res.mean(0);gain=pp/(pp+.6*.6/n);x+=gain*(z-x);p=(1-gain)*pp;lost=0;ever=True
        else:
            p=pp;lost=lost+1 if min(len(r),len(e))>=3 else 0
            if enabled and lost>=5:
                value=acquire(list(hist),review)
                if value is not None:
                    xacq=value;x=value.copy();p=.25;lost=0;stream_triggers[key]+=1
                    events.append(dict(index=i,sequence=key[0],wait_ms=key[1],coarse_offset=value.tolist()))
        states[key]=(x,p,xacq,lost,hist,ever);result.append(x.copy());counts.append(n);everlock.append(ever)
    return np.asarray(result),dict(events=events,triggers=len(events),streams=len(states),
        trigger_by_stream=[dict(sequence=s,wait_ms=w,count=stream_triggers[s,w]) for s,w in states],
        never_locked_streams=sum(not v[-1] for v in states.values()),anchors=counts,ever_locked=everlock)

def inject(m,kind,amplitude=0.,seed=2026111401):
    """Perturbation harness only. Estimators get boxes, never injected delta."""
    seqs=sorted(set(map(str,m['sequence'])));directions={}
    # Sequence-keyed PCG64 makes the same sequence's direction invariant to
    # detector/split coverage and processing/shard order, as well as wait.
    for s in seqs:
        words=np.frombuffer(hashlib.sha256(s.encode()).digest()[:16],dtype='<u4')
        rng=np.random.default_rng(np.random.SeedSequence([seed,*map(int,words)]));t=rng.uniform(0,2*np.pi)
        directions[s]=np.array([np.cos(t),np.sin(t)])
    times={s:np.asarray(m['receiver_timestamp_us'])[m['sequence'].astype(str)==s] for s in seqs}
    shift=[]
    for s,t in zip(m['sequence'],m['receiver_timestamp_us']):
        s=str(s);lo,hi=float(times[s].min()),float(times[s].max());fraction=(float(t)-lo)/(hi-lo) if hi>lo else 0.
        a=amplitude if kind=='T1' else 2*fraction if kind=='T2' else 2.*(float(t)>=(lo+hi)/2) if kind=='T3' else 0.
        shift.append(directions[s]*a)
    shift=np.asarray(shift);out=dict(m);remote=m['source']==1
    repeated=np.repeat(shift,np.diff(m['condition_offsets']).astype(int),axis=0)
    for key in BOX_KEYS:
        assert key in m,key
        out[key]=m[key].copy();out[key][remote,:2]+=repeated[remote].astype(out[key].dtype)
        assert np.array_equal(out[key][~remote],m[key][~remote])
    assert out['gt_boxes'] is m['gt_boxes']
    return out,shift
