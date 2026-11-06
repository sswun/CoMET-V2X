"""Finite registration consumers: R1-R4 and quality-conditioned cached AP.

Only reference labels use registration. Saved inference stays untouched.
"""
import argparse,os,time,traceback,json
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
from collections import defaultdict
import numpy as np
from peer14_cycle21_v1_20261102 import ROOT,AUTO,AUDIT,CYCLE,atomic,ev,load,now,register,record
BASE=CYCLE/'registration_read_v1'
INPUTS={'SPD':CYCLE/'raw_registration_v1/producer','V2':CYCLE/'v2_raw_registration_local_v2/producer'}

def prepare():
    BASE.mkdir(exist_ok=False)
    register('registration_read',__file__,'Raw-cloud quality/coverage may be too low for labels. R1 slow>.5 AND jitter<.2 required; bootstrap sequence CIs and all null/negative outcomes retained. Label-anchored K2>K1 on BOTH SPD/V2 val is the only route to reconsider existing K2, not a new variant. V2 K1>=stale-.5 attribution only on exact same quality-conditioned population, never full population by implication.',dict(GPU_forwards=0,CPU_threads=1,wall_h='finite30h dependency wait + 2-8h cached scoring',new_disk_MiB='<40 metrics/transforms; no copied raw/GT/predictions'))
    c=load(CYCLE/'registration_read_contract.json');c.update(inputs={k:str(v.relative_to(ROOT)) for k,v in INPUTS.items()},population='Coverage gate>=.30 by original synchronized dataset/role manifest; score accepted synchronized current-frame conditions ONLY, paired identical original GT/predictions for official and registration protocols. No substitute official geometry for rejected frames, no ROI recrop after shift. Report accepted fraction per saved method population. This conditional subset cannot replace main full-population AP.',label_geometry='SPD official inf center minus vehicle-frame official offset yields public center; then left residual yaw+xy+z; yaw += residual yaw. V2 official remote-only center uses same left residual. Preserve dimensions/local labels/GT order. Roll/pitch excluded from upright-box label transform; gate <=2deg. SE2+z proxy, NOT physical truth.',R1='XY component Pearson and flattened Pearson; compare MA5 registration vs officialMA5 and registration-minus-MA5 vs official-minus-MA5, per sequence. Centered5 accepted chronological frames; require >=5. 1000 sequence bootstrap, fixedseed20261102; report CI and points, no AP selection.',R2_R4='Saved exact current K1 delta and K2 mean/pivot, wait0 used once per frame, on same accepted frame intersections; SE2 left t=pivot+meanXY-Rpivot. Translation/yaw errors vs registration, and MA5/seqmean/jitter decompositions; yaw correlation R4. V2 official input=provided pose, zero residual.',scope='Cached current K1/K2/K3 and stale/CV geometry families. Native/CoAlign registration protocols explicitly remain followups. V2 R3 common difference uses lineage per-label max_pair_center_distance_m for explicit common groups; auxiliary SE2 is label-derived, not truth.')
    atomic(CYCLE/'registration_read_contract.json',c)

def corr(a,b):
    a=np.asarray(a,float).ravel();b=np.asarray(b,float).ravel()
    if len(a)<3 or np.std(a)<1e-12 or np.std(b)<1e-12:return None
    return float(np.corrcoef(a,b)[0,1])

def summary(a):
    a=np.asarray(a,float).ravel();return dict(n=len(a),median=float(np.median(a)) if len(a) else None,p95=float(np.quantile(a,.95)) if len(a) else None,mean=float(a.mean()) if len(a) else None)

def smooth(a):return np.array([a[max(0,i-2):i+3].mean(0) for i in range(len(a))])

def pair_read(items):
    grouped=defaultdict(list)
    for r in items:grouped[r['sequence']].append(r)
    chunks=[];per=[]
    for seq,rs in sorted(grouped.items()):
        rs.sort(key=lambda r:r['frame']);reg=np.array([r['registration_xy'] for r in rs]);official=np.array([r['official_xy'] for r in rs]);rsmooth=smooth(reg);osmooth=smooth(official)
        if len(rs)<5:continue
        chunks.append((rsmooth,osmooth,reg-rsmooth,official-osmooth));per.append(dict(sequence=seq,n=len(rs),slow=corr(rsmooth,osmooth),jitter=corr(reg-rsmooth,official-osmooth),seqmean_registration=reg.mean(0).tolist(),seqmean_official=official.mean(0).tolist()))
    if not chunks:return dict(status='insufficient_accepted_sequences',sequences=per)
    def pooled(ch):return [corr(np.concatenate([v[i] for v in ch]),np.concatenate([v[j] for v in ch])) for i,j in ((0,1),(2,3))]
    point=pooled(chunks);rng=np.random.default_rng(20261102);boot=[pooled([chunks[i] for i in rng.integers(0,len(chunks),len(chunks))]) for _ in range(1000)];ci=[]
    for j in range(2):
        vv=[x[j] for x in boot if x[j] is not None];ci.append(np.quantile(vv,[.025,.975]).tolist() if vv else None)
    means=[r for r in per];return dict(status='complete_light_checked',slow=point[0],jitter=point[1],sequence_bootstrap_CI95=ci,per_sequence=per,seqmean_correlation=corr([r['seqmean_registration'] for r in means],[r['seqmean_official'] for r in means]),preregistered_point_gate=point[0] is not None and point[1] is not None and point[0]>.5 and point[1]<.2,physical_truth_claim_allowed=False,qualification='Geometric proxy, asynchronous +/-50ms SPD pairing, ICP local minima and quality-conditioned population remain limitations. Gate alone does not remove these qualifications.')

def load_registration(dataset):
    p=INPUTS[dataset];term=load(p/'terminal.json');assert term['status']=='complete_light_checked'
    accepted=set(load(p/'accepted_indices.json'));rr=[json.loads(x) for x in (p/'registrations.jsonl').open()];rr=[r for r in rr if r['dataset']==dataset and r['index'] in accepted]
    return term,{(r['sequence'],r['frame']):r for r in rr}

def aligned_errors(items):
    groups=defaultdict(list)
    for r in items:groups[r['sequence']].append(r)
    stats=defaultdict(list)
    for seq,rs in groups.items():
        rs.sort(key=lambda r:r['frame']);reg=np.array([r['registration_xy'] for r in rs]);regyaw=np.unwrap([r['registration_yaw'] for r in rs])
        for method in ('K1','K2','OFFICIAL','PUBLIC'):
            a=np.array([r[method+'_xy'] for r in rs]);yaw=np.unwrap([r[method+'_yaw'] for r in rs])
            for part,x,y,xx,yy in [('all',a,reg,yaw,regyaw),('MA5',smooth(a),smooth(reg),smooth(yaw[:,None]).ravel(),smooth(regyaw[:,None]).ravel()),('seqmean',a.mean(0,keepdims=True),reg.mean(0,keepdims=True),np.array([yaw.mean()]),np.array([regyaw.mean()])),('jitter',a-smooth(a),reg-smooth(reg),yaw-smooth(yaw[:,None]).ravel(),regyaw-smooth(regyaw[:,None]).ravel())]:
                stats[method+'__'+part+'__xy_m'].extend(np.linalg.norm(x-y,axis=1));stats[method+'__'+part+'__yaw_deg'].extend(np.degrees(np.abs(np.arctan2(np.sin(xx-yy),np.cos(xx-yy)))))
    return {k:summary(v) for k,v in stats.items()}

def read_analysis(dataset,accepted,offsets):
    import run_peer13_v2_auxiliary_ap_v1_20261102 as v2
    if dataset=='SPD':
        import run_peer10_fixed_population_aux_v1_20261101 as aux
        c=load(AUDIT/'contract_v1.json');m=aux.core.load_npz(ROOT/c['inputs']['val21_material']['path']);ledger=aux.core.load_jsonl(ROOT/c['inputs']['val_conditions']['path']);rows=aux.core.conditions_for(m,ledger);gate=aux.core.load_npz(ROOT/c['inputs']['val21_gate_material']['path'])['q005_base_delta'];state=aux.core.load_npz(AUDIT/'k2_k3_formal_resume_v3/val21_state.npz')
        role_rows=[('val21',m,rows,gate,state)]
        for role in ('mechanism_holdout14',):
            m=aux.core.load_npz(ROOT/c['inputs'][role+'_material']['path']);rows=aux.core.conditions_for(m,aux.core.load_jsonl(ROOT/c['inputs']['train_conditions']['path']));gate=aux.core.load_npz(ROOT/c['inputs'][role+'_gate_material']['path'])['q005_base_delta'];state=aux.core.load_npz(AUDIT/'k2_k3_formal_resume_v3'/f'{role}_state.npz');role_rows.append((role,m,rows,gate,state))
    else:
        c=load(v2.V2/'contract_recovery_v6.json');role_rows=[]
        for role in ('val21','mechanism_holdout14'):
            ds=v2.fac.PackedDataset(ROOT/c['datasets']['val' if role=='val21' else 'train']);m,rows=v2.transfer.material(c,role,ds);gate=v2.science.core.load_npz(v2.V2/'producer_v1'/f'{role}_q005_state.npz')['q005_delta'];state=v2.science.core.load_npz(v2.TRANSFER/f'{role}_state.npz');role_rows.append((role,m,rows,gate,state))
    items=[]
    for role,m,rows,gate,state in role_rows:
        for i,r in enumerate(rows):
            if int(r['wait_ms'])!=0:continue
            frame=str(r['current_vehicle_frame'] if dataset=='SPD' else r['timestamp']);key=(str(r['sequence']),frame)
            if key not in accepted:continue
            ref=accepted[key];mean=state['K2_infra__mean'][i];pivot=state['K2_infra__pivot'][i];th=mean[2];R=np.array([[np.cos(th),-np.sin(th)],[np.sin(th),np.cos(th)]]);t=pivot+mean[:2]-R@pivot;official=offsets.get(frame,np.zeros(2)) if dataset=='SPD' else np.zeros(2)
            items.append(dict(role=role,sequence=key[0],frame=frame,registration_xy=ref['delta_xy'],registration_yaw=ref['yaw'],official_xy=official.tolist(),K1_xy=gate[i].tolist(),K1_yaw=0.,K2_xy=t.tolist(),K2_yaw=float(th),OFFICIAL_xy=official.tolist(),OFFICIAL_yaw=0.,PUBLIC_xy=[0.,0.],PUBLIC_yaw=0.))
    result=dict(accepted_wait0_frames=len(items),R2=aligned_errors(items),R4=dict(yaw_Pearson=corr([r['registration_yaw'] for r in items],[r['K2_yaw'] for r in items]),registration_yaw_abs_deg=summary(np.degrees(np.abs([r['registration_yaw'] for r in items])))))
    if dataset=='SPD':result['R1']=pair_read(items)
    else:
        frames=v2.frame_map(ROOT/load(v2.BASE/'contract.json')['inputs']['lineage']['path']);trs={(r['sequence'],r['timestamp']):r for r in load(v2.LABELS/'replacement_transforms.json')};data=[]
        for key,r in accepted.items():
            tr=trs[key];f=frames[('val' if r['role']=='val21' else 'train',*key)];common=[v['max_pair_center_distance_m'] for v in f['official_labels'] if v['member_count']>1 and 0 in v['member_agents']]
            if common and tr['registration_pairs']>=3:data.append((float(np.linalg.norm(r['delta_xy'])),float(np.median(common)),float(np.linalg.norm(tr['t'])),r['delta_xy'],tr['t']))
        result['R3']=dict(n=len(data),provided_pose_correction_xy_m=summary([x[0] for x in data]),common_label_difference_correlation=corr([x[0] for x in data],[x[1] for x in data]),auxiliary_magnitude_correlation=corr([x[0] for x in data],[x[2] for x in data]),auxiliary_xy_correlation=corr([x[3] for x in data],[x[4] for x in data]),common_proxy='Per-frame median duplicate common-label maximum 3D center distance, not same as original pooled XY residual; label-derived auxiliary never physical truth.')
    return result

def score(dataset,accepted,term,offsets,out):
    import run_peer10_fixed_population_aux_v1_20261101 as aux
    import run_peer13_v2_auxiliary_ap_v1_20261102 as v2
    from run_peer14_v2_dose_response_v1_20261102 import labels
    from run_peer10_native_source_aux_v1_20261101 import outcome
    from peer07_source_ap_producer_utils_v1_20261031 import metric
    panels=[];pop=[]
    for role in ('val21','mechanism_holdout14'):
        coverage=term['coverage'][dataset+'__'+role]
        if not coverage['reference_labels_allowed']:pop.append(dict(dataset=dataset,role=role,status='coverage_below30_no_labels',coverage=coverage));continue
        if dataset=='SPD':
            h=load(AUDIT/'cycle09_v1/aux_hwl_contract_v3.json');spec=load(aux.verified(h['original_contract']));configs={'current':load(aux.verified(spec['shared_main_contract'])),'geometry':load(aux.verified(spec['baseline_contracts']['geometry']))}
            families=[(f,*loader(role,configs[f])) for f,loader in [('current',aux.current),('geometry',aux.geometry)]]
        else:
            vc=load(v2.V2/'contract_recovery_v6.json');ds=v2.fac.PackedDataset(ROOT/vc['datasets']['val' if role=='val21' else 'train']);m,rows=v2.transfer.material(vc,role,ds);state=v2.science.core.load_npz(v2.TRANSFER/f'{role}_state.npz');pred=v2.science.core.load_npz(v2.TRANSFER/f'{role}_predictions.npz');saved=v2.science.core.load_npz(v2.V2/'producer_v1'/f'{role}_q005_state.npz');boxes={}
            for name in ('K2_infra','K3_K1_expanded','K3_K2_expanded'):boxes[name]=v2.science.geometry(m,{k.split('__',1)[1]:v for k,v in state.items() if k.startswith(name+'__')})[0]
            boxes['K1_exact']=v2.fac.heading_geometry(m,saved['q005_delta'])[0]
            # Build exactly the same frozen stale/CV candidate rank as C15, not a new readout.
            head=v2.V2/'producer_v1/q005_cv_matched39d_head'
            for name in ('stale_raw','stale_same_U','CV_raw','CV_same_U'):
                kind='stale' if name.startswith('stale') else 'cv';boxes[name]=m[kind+'_boxes'];sc=m['raw_scores'] if name.endswith('_raw') else v2.science.core.refit.score_with_head(m,m[kind+'_features'],head);ids=[];ss=[];off=[0]
                for i in range(len(rows)):
                    a,b=map(int,m['condition_offsets'][i:i+2]);ii,s=v2.fac.formal.ordered(m['raw_scores'][a:b],m['source'][a:b],m['protected'][a:b],sc[a:b],None);ids.extend(a+ii);ss.extend(s);off.append(len(ids))
                pred.update({name+'__candidate':np.asarray(ids),name+'__score':np.asarray(ss),name+'__offset':np.asarray(off)})
            families=[('current',m,rows,pred,boxes,np.ones(len(m['gt_boxes']),bool))];frames=v2.frame_map(ROOT/load(v2.BASE/'contract.json')['inputs']['lineage']['path']);trs={(r['split'],r['sequence'],r['timestamp']):r for r in load(v2.LABELS/'replacement_transforms.json')}
        for family,m,rows,pred,boxes,mask in families:
            if dataset=='SPD':codes,mapping=aux.source.map_provenance(dict(m,dataset_index=np.arange(len(rows))),rows,role,aux.LABELROOT);assert mapping['unmatched']==mapping['ambiguous']==0
            selected=[i for i,r in enumerate(rows) if (str(r['sequence']),str(r['current_vehicle_frame'] if dataset=='SPD' else r['timestamp'])) in accepted];pop.append(dict(dataset=dataset,role=role,family=family,accepted_conditions=len(selected),original_conditions=len(rows),coverage=coverage))
            for method,bb in boxes.items():
                stats=defaultdict(lambda:dict(scores=[],tp=[],fp=[],gt=0));groups=('all','veh','coop','inf','veh_coop') if dataset=='SPD' else ('all','local_visible','remote_exclusive')
                for i in selected:
                    r=rows[i];frame=str(r['current_vehicle_frame'] if dataset=='SPD' else r['timestamp']);ref=accepted[(str(r['sequence']),frame)];ga,gb=map(int,m['gt_offsets'][i:i+2]);gt=m['gt_boxes'][ga:gb].copy()
                    if dataset=='SPD':side=codes[ga:gb];remote=side==2
                    else:
                        key=('val' if role=='val21' else 'train',str(r['sequence']),frame);remote,_=labels(frames[key],trs[key],True,gt,[]);side=np.where(remote,2,0)
                    anchored=gt.copy();theta=ref['yaw'];R=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]]);public=gt[remote,:2].astype(float)-(offsets[frame] if dataset=='SPD' else 0);anchored[remote,:2]=public@R.T+ref['delta_xy'];anchored[remote,2]+=ref['delta_z'];anchored[remote,6]+=theta
                    assert np.array_equal(anchored[~remote],gt[~remote]) and np.array_equal(anchored[:,3:6],gt[:,3:6]);a,b=map(int,pred[method+'__offset'][i:i+2]);ii=pred[method+'__candidate'][a:b];ss=pred[method+'__score'][a:b];assert np.all(np.diff(ss)<=0)
                    for protocol,gg in [('official_quality_subset',gt),('registration_SE2_z_quality_subset',anchored)]:
                        overlap=aux.source.iou32(bb[ii],gg[mask[ga:gb]]);cc=side[mask[ga:gb]]
                        for group in groups:
                            g={'local_visible':'veh_coop','remote_exclusive':'inf'}.get(group,group)
                            for th in (50,70):
                                tp,fp,ig,n=outcome(overlap,cc,g,th/100);keep=ig==0;v=stats[protocol,group,int(r['wait_ms']),th];v['scores'].append(ss[keep]);v['tp'].append(tp[keep]);v['fp'].append(fp[keep]);v['gt']+=n
                                if protocol=='official_quality_subset' and group=='all' and method+'__tp'+str(th) in pred:assert np.array_equal(tp,pred[method+'__tp'+str(th)][a:b]),('subset savedTP replay',dataset,method,i,th)
                for (protocol,group,wait,th),v in stats.items():panels.append(dict(dataset=dataset,family=family,role=role,method=method,protocol=protocol,source_group=group,wait_ms=wait,threshold=th,**metric(np.concatenate(v['tp']),np.concatenate(v['fp']),np.concatenate(v['scores']),v['gt'])))
                atomic(out/'panels.json',panels);atomic(out/'status.json',dict(status='registration_subset_AP',dataset=dataset,role=role,family=family,method=method,panels=len(panels),created_at=now()))
    atomic(out/'population_coverage.json',pop)
    return panels

def run(dataset):
    c=load(CYCLE/'registration_read_contract.json');assert c['script']==ev(__file__);out=BASE/dataset;out.mkdir(exist_ok=False);start=time.monotonic();record('registration_read_'+dataset,dict(status='waiting_registration_terminal',contract=ev(CYCLE/'registration_read_contract.json')))
    try:
        while not (INPUTS[dataset]/'terminal.json').exists():
            if time.monotonic()-start>30*3600:raise TimeoutError('finite30h registration wait')
            time.sleep(30)
        term,accepted=load_registration(dataset);atomic(out/'registration_identity.json',dict(terminal=ev(INPUTS[dataset]/'terminal.json'),quality=ev(INPUTS[dataset]/'quality_frozen.json'),accepted=ev(INPUTS[dataset]/'accepted_indices.json')))
        meta=load(ROOT/'artifacts/data/v2x_seq_spd_main_domain_shared_v1/V2X-Seq-SPD/cooperative/data_info.json');offsets={str(r['vehicle_frame']):np.array([r['system_error_offset'][k] for k in ('delta_x','delta_y')]) for r in meta}
        analysis=read_analysis(dataset,accepted,offsets);atomic(out/'R1_R4_analysis.json',analysis);panels=score(dataset,accepted,term,offsets,out)
        atomic(out/'terminal.json',dict(status='complete_light_checked',created_at=now(),analysis=ev(out/'R1_R4_analysis.json'),panels=ev(out/'panels.json') if panels else None,coverage=ev(out/'population_coverage.json'),quality_subset_not_full_population=True,independent_review_pending=True,GPU_forwards=0,wall_s=time.monotonic()-start,no_model_wakeup=True));record('registration_read_'+dataset,dict(status='complete_light_checked',terminal=ev(out/'terminal.json')))
    except Exception as e:
        atomic(out/'terminal.json',dict(status='failed_retained',created_at=now(),error=repr(e),traceback=traceback.format_exc()));record('registration_read_'+dataset,dict(status='failed_retained',terminal=ev(out/'terminal.json')));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','SPD','V2']);a=p.parse_args();prepare() if a.mode=='prepare' else run(a.mode)
