"""R6a: direct raw annotation lineage, not detector matching or physical truth.

Token/track joins precede geometry; raw roadside locations are transformed by
public calibrations. No nearest-neighbour acceptance or offset sign fitting.
The independent branch uses track+token crosscheck and a separately assembled
inverse rotation/translation chain, not the production homogeneous solve.
"""
import argparse,hashlib,time,traceback
from collections import Counter,defaultdict
import numpy as np
from peer20_v1_20261105 import *

NAME='R6a_raw_labels';OUT=BASE/NAME;REVIEW=BASE/'R6a_raw_labels_independent'
DATA=ROOT/'artifacts/data/v2x_seq_spd_main_domain_shared_v1/V2X-Seq-SPD'
ROLES=('fit24','calibration8','mechanism_holdout14','val21')

def roles():
    c=load(AUDIT/'contract_v1.json');result={}
    for role in ROLES:
        with np.load(ROOT/c['inputs'][role+'_material']['path']) as z:
            for seq in set(map(str,z['sequence'])):
                assert seq not in result
                result[seq]=role
    return result

def prepare():
    for name in (NAME,'R6a_raw_labels_independent'):
        register(name,__file__,__doc__,dict(GPU_forwards=0,updates=0,CPU_threads=1,
            disk_MiB='<25 compact lineage residuals; raw labels/calibrations referenced',wall_h='finite all4 roles'),
            metadata=ev(DATA/'cooperative/data_info.json'),role_source=ev(AUDIT/'contract_v1.json'),
            acceptance='Exact-zero raw-inf-to-coop-inf public projection residual median <.05m, with join coverage/misses explicit. No physical-truth claim.',
            zero_definition='delta_x==0 and delta_y==0 exactly; co-labelled=(coop object count>0), distinct masks even when discordant',
            independent_scope='Separate calibration algebra and track/token joins; all residuals/counts independently recomputed')

def xyz(o):return np.asarray([o['3d_location'][k] for k in ('x','y','z')],float)
def trans(j,nested=False):
    if nested:j=j['transform']
    return np.asarray(j['rotation'],float),np.asarray(j['translation'],float).reshape(3)

def calculate(independent=False):
    rolemap=roles();info=load(DATA/'cooperative/data_info.json')
    meta={side:{str(r['frame_id']):r for r in load(DATA/side/'data_info.json')} for side in ('vehicle-side','infrastructure-side')}
    opened={};labels={};records=[];frames=[];special=[];miss=Counter()
    def read(p):
        key=str(p.relative_to(ROOT))
        if key not in opened:opened[key]=ev(p)
        return load(p)
    def raw(side,frame):
        key=side,frame
        if key not in labels:
            item=meta[side][frame];objs=read(DATA/side/item['label_lidar_std_path'])
            labels[key]=({str(o['token']):o for o in objs},{str(o['track_id']):o for o in objs})
        return labels[key]
    for row in info:
        seq=str(row['vehicle_sequence'])
        if seq not in rolemap:continue
        role=rolemap[seq];vf=str(row['vehicle_frame']);vm=meta['vehicle-side'][vf]
        off=np.array([float(row['system_error_offset']['delta_'+k]) for k in ('x','y')]);zero=bool(np.all(off==0))
        objs=read(DATA/'cooperative/label'/(vf+'.json'));counts=Counter(o['from_side'] for o in objs)
        frames.append(dict(role=role,sequence=seq,frame=vf,timestamp_us=int(vm['pointcloud_timestamp']),
            zero=zero,coop=counts['coop'],inf=counts['inf'],veh=counts['veh'],offset=off.tolist()))
        A=read(DATA/'vehicle-side'/vm['calib_lidar_to_novatel_path']);B=read(DATA/'vehicle-side'/vm['calib_novatel_to_world_path'])
        r1,t1=trans(A,True);r2,t2=trans(B)
        for obj in objs:
            side=str(obj['from_side'])
            if side not in ('inf','coop'):continue
            inf=str(obj.get('inf_frame_id',row['infrastructure_frame']));im=meta['infrastructure-side'][inf]
            bytoken,bytrack=raw('infrastructure-side',inf);token=str(obj['inf_token']);track=str(obj['inf_track_id'])
            source=bytrack.get(track) if independent else bytoken.get(token)
            if source is None:miss[role+'_'+side+'_missing_lineage']+=1;continue
            assert str(source['token'])==token and str(source['track_id'])==track,(vf,inf,token,track)
            C=read(DATA/'infrastructure-side'/im['calib_virtuallidar_to_world_path']);r3,t3=trans(C)
            point=xyz(source)
            if independent:
                world=r3.dot(point)+t3
                nov=np.linalg.inv(r2).dot(world-t2)
                projected=np.linalg.inv(r1).dot(nov-t1)
            else:
                V=np.eye(4);V[:3,:3]=r2@r1;V[:3,3]=r2@t1+t2
                I=np.eye(4);I[:3,:3]=r3;I[:3,3]=t3
                projected=(np.linalg.solve(V,I)@np.r_[point,1])[:3]
            residual=xyz(obj)-projected
            records.append((role,seq,vf,side,token,zero,*residual,*off))
            if role=='calibration8' and zero and side=='coop':
                vtok,vtrack=raw('vehicle-side',str(obj.get('veh_frame_id',vf)))
                vehicle=vtok.get(str(obj['veh_token']))
                special.append(dict(frame=vf,sequence=seq,inf_frame=inf,inf_token=token,veh_token=str(obj['veh_token']),
                    raw_road_to_coop_residual=residual.tolist(),raw_veh_to_coop_residual=(xyz(obj)-xyz(vehicle)).tolist() if vehicle else None,
                    raw_inf_to_vehicle=(xyz(vehicle)-projected).tolist() if vehicle else None,
                    cooperative_location=xyz(obj).tolist(),raw_inf_location=point.tolist(),
                    inf_timestamp_us=int(im['pointcloud_timestamp']),veh_timestamp_us=int(vm['pointcloud_timestamp'])))
    summary={};groups=defaultdict(list)
    for r in records:groups[r[0],r[3],bool(r[5])].append(r)
    for (role,side,zero),rows in sorted(groups.items()):
        a=np.array([r[6:] for r in rows],float);res=a[:,:3];offset=a[:,3:];norm=np.linalg.norm(res[:,:2],axis=1);after=np.linalg.norm(res[:,:2]-offset,axis=1)
        summary[role+'_'+side+('_zero' if zero else '_nonzero')]=dict(n=len(rows),
            public_xy_median_m=float(np.median(norm)),public_xy_max_m=float(norm.max()),
            offset_corrected_xy_median_m=float(np.median(after)),offset_corrected_xy_max_m=float(after.max()),
            corrected_fraction_lt_005=float(np.mean(after<.05)),public_fraction_lt_005=float(np.mean(norm<.05)))
    census={};runs=[]
    for role in ROLES:
        f=[r for r in frames if r['role']==role];tab=Counter(('zero' if r['zero'] else 'nonzero')+('_coop' if r['coop']>0 else '_no_coop') for r in f)
        byseq={}
        for seq in sorted({r['sequence'] for r in f}):
            rows=sorted([r for r in f if r['sequence']==seq],key=lambda r:r['timestamp_us']);current=[];seqruns=[]
            def finish():
                if current:
                    rr=dict(role=role,sequence=seq,start=current[0]['frame'],end=current[-1]['frame'],messages=len(current),
                        first_to_last_s=(current[-1]['timestamp_us']-current[0]['timestamp_us'])/1e6)
                    runs.append(rr);seqruns.append(rr);current.clear()
            for r in rows:
                if r['zero']:current.append(r)
                else:finish()
            finish();byseq[seq]=dict(messages=len(rows),zero=sum(r['zero'] for r in rows),runs=len(seqruns))
        census[role]=dict(messages=len(f),zero=sum(r['zero'] for r in f),contingency=dict(tab),per_sequence=byseq)
    confirmed=all(summary.get(r+'_inf_zero',{}).get('public_xy_median_m',float('inf'))<.05 for r in ROLES)
    return dict(census=census,direct_projection=summary,zero_runs=runs,
        cal8_zero_coop=dict(frames=len({r['frame'] for r in special}),objects=len(special),rows=special,
            interpretation='Exact zero with coop is not assumed missing; raw vehicle/roadside and cooperative label deltas reported per object. Source fields do not themselves prove offset computability.'),
        missing_lineage=dict(miss),confirmed_zero_placeholder_unadjusted_GT=confirmed,
        scope='Direct label construction relative to public calibration; not physical pose truth. Exact-zero and co-labelled populations are NOT complements.'),records,frames,opened

def execute(review=False):
    name='R6a_raw_labels_independent' if review else NAME;dest=REVIEW if review else OUT;begin=time.monotonic()
    try:
        if review:wait(OUT/'terminal.json')
        record(name,dict(status='running_direct_raw_label_lineage',resource_owner=True))
        result,rows,frames,opened=calculate(review)
        if review:
            gold=load(OUT/'summary.json')
            def equal(a,b):
                if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
                if isinstance(a,list):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
                if isinstance(a,float):return abs(a-b)<1e-8
                return a==b
            assert equal(result,gold),'independent summary mismatch'
            with np.load(OUT/'direct_residuals.npz') as z:
                numeric=np.asarray([r[6:] for r in rows],float);error=float(np.max(np.abs(numeric-z['residual_and_offset']),initial=0))
                assert error<1e-8,error
                assert np.array_equal(np.array(['|'.join(map(str,r[:6])) for r in rows]),z['identity'])
            atomic(dest/'review.json',dict(created_at=now(),max_residual_difference_m=error,counts_exact=True,
                raw_inputs=ev(OUT/'raw_input_manifest.json'),confirmed=result['confirmed_zero_placeholder_unadjusted_GT'],
                independent_calibration_algebra=True,independent_track_join=True,shared_raw_files=True))
        else:
            atomic(dest/'summary.json',result);atomic(dest/'frames.json',frames)
            atomic(dest/'raw_input_manifest.json',dict(created_at=now(),files=opened,metadata=ev(DATA/'cooperative/data_info.json')))
            np.savez_compressed(dest/'direct_residuals.npz',identity=np.asarray(['|'.join(map(str,r[:6])) for r in rows]),residual_and_offset=np.asarray([r[6:] for r in rows],float))
        status='complete_independently_reviewed' if review else 'complete_light_checked'
        atomic(dest/'terminal.json',dict(created_at=now(),status=status,wall_s=time.monotonic()-begin,
            confirmed=result['confirmed_zero_placeholder_unadjusted_GT'],scope=__doc__))
        record(name,dict(status=status,terminal=ev(dest/'terminal.json'),resource_owner=False,owner=None))
    except Exception as e:
        atomic(dest/'terminal.json',dict(created_at=now(),status='failed_retained',error=repr(e),traceback=traceback.format_exc()))
        record(name,dict(status='failed_retained',terminal=ev(dest/'terminal.json'),resource_owner=False,owner=None));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','run','review'));a=p.parse_args()
    prepare() if a.mode=='prepare' else execute(a.mode=='review')
