"""A2/A3/A4/A7 with actual data, retained coverage and sequence intervals."""
import argparse,csv,json,time,traceback,subprocess
from collections import defaultdict
import numpy as np
from peer18_v1_20261104 import *

def prepare():
    for name in ('A2','A3','A4','A7'):
        register(name,__file__,
            dict(A2='Source-addressed existing training/test-pose factors only, missing cells explicit; never claim learning to ignore roadside.',
                 A3='External instantaneous errors may be persistent/bias-dominated rather than removable white noise; report both MA5 and fixed-quality ICP, retain failures.',
                 A4='Roadside-exclusive localization CDF under PUBLIC vs final CoMET, fixed GT association shared by both; diagnostic only, no AP inference.',
                 A7='Static local code evidence only. Keyword occurrences are not proof of default inference or a claim that every public result uses offsets.')[name],
            dict(GPU_forwards=0,updates=0,CPU_threads=1,wall_h='finite6h',disk_MiB='<20 compact errors/metrics'),
            baseline=ev(P17/'final_recipe/contract.json'),analysis_only=True,sequence_resamples=2000)

def A2(out):
    refs=[AUDIT/'cycle09_peer12_v1_20261102/Where_seed1_native_full_factor.csv',
          AUDIT/'cycle09_peer12_v1_20261102/TraF_new_OFFICIAL_factor_cells.csv',
          AUDIT/'cycle12_peer14_v1_20261103/CoAlign_completed_composed_graph_all_wait_factors.csv',
          AUDIT/'cycle03_peer14_v1_20261103/CoAlign_all_completed_weight_input_factors.csv']
    tables=[]
    for p in refs:
        if p.exists():tables.append(dict(source=ev(p),rows=list(csv.DictReader(p.open()))))
    # E13 is source protocol x test pose for fixed weights, NOT a training factor.
    for p in (P16/'E13_recovery_v2/all/panels.json',P16/'E13_recovery_v2/vit/panels.json'):
        if p.exists():tables.append(dict(source=ev(p),rows=load(p),scope='fixed training weight; source/timing and input-pose factors, not training effect'))
    assert len(tables)>=4
    atomic(out/'existing_factors.json',dict(tables=tables,
        missing='No fabricated dense3x3. Existing tables retain own model weights, ROI and training budget. '
        'Missing OFFICIAL-trained detector/seed cells remain unmeasured; no new model runs.',
        scope='Existing factor numbers/CI carried with exact source pointers; not independent new experimental/statistical validation.',
        forbidden_claim='Networks learned to ignore roadside evidence: unsupported by peer13 ego-only controls.'))

def stats(seq,t,e):
    groups=[]
    for s in np.unique(seq):
        ii=np.flatnonzero(seq==s);ii=ii[np.argsort(t[ii],kind='stable')];a=e[ii]
        mu=a.mean(0);center=a-mu;pair=(np.diff(t[ii])>0)&(np.diff(t[ii])<=350000)
        x,y=center[:-1][pair],center[1:][pair]
        groups.append(np.array([len(a),np.sum(a*a),len(a)*np.sum(mu*mu),np.sum(x*y),np.sum(x*x),np.sum(y*y),np.sum(np.linalg.norm(a,axis=1)>1),np.sum((y-x)**2),len(x)],float))
    g=np.stack(groups)
    def measure(a):
        den=max(a[1],1e-15);rho=a[3]/max(np.sqrt(a[4]*a[5]),1e-15)
        return np.array([np.sqrt(a[1]/max(a[0],1)),a[2]/den,rho,a[6]/max(a[0],1),a[7]/max(2*a[8],1)])
    point=measure(g.sum(0));rng=np.random.default_rng(2026110403)
    boot=np.array([measure(g[rng.integers(len(g),size=len(g))].sum(0)) for _ in range(2000)])
    return dict(zip(('radial_RMSE_m','sequence_bias_fraction','demeaned_lag1_correlation','failure_gt1m_fraction','difference_white_variance_m2'),
        [dict(value=float(v),CI95=np.quantile(boot[:,j],[.025,.975]).tolist()) for j,v in enumerate(point)]))

def references(rows):
    from run_peer12_cached_ceiling_v1_20261101 import reference_deltas
    path=ROOT/'artifacts/data/v2x_seq_spd_traf_align_author21_v1/V2X-Seq-SPD/cooperative/data_info.json'
    official={str(r['vehicle_frame']):r for r in load(path)}
    ma=reference_deltas(rows,official)['OFFICIAL_MA5']
    from review_peer16_E1_physical_velocity_v1_20261103 import REG
    accepted=set(load(REG/'accepted_indices.json'));icp={}
    for line in (REG/'registrations.jsonl').open():
        r=json.loads(line)
        if r['dataset']=='SPD' and r['role']=='val21' and not r.get('error') and int(r['index']) in accepted:
            icp[str(r['sequence']),str(r['frame'])]=np.asarray(r['delta_xy'],float)
    mask=np.array([(str(r['sequence']),str(r['current_vehicle_frame'])) in icp for r in rows])
    physical=np.array([icp.get((str(r['sequence']),str(r['current_vehicle_frame'])),np.zeros(2)) for r in rows])
    return ma,physical,mask,dict(MA5=ev(path),ICP=ev(REG/'terminal.json'),quality=ev(REG/'quality_frozen.json'),
        interpretation='MA5 is annotation-assisted; synchronous ICP quality subset is a selected registration proxy, not physical ground truth. ICP-vs-itself omitted.')

def A3(out):
    from run_peer16_cached_methods_v1_20261103 import materials,observation
    m,rows=materials('author')[0]['val21'];obs=observation(m)
    estimates={'single_message_anchor_mean':np.array([z.mean(0) if len(z) else np.zeros(2) for _,z in obs])}
    valid={'single_message_anchor_mean':np.array([len(z)>0 for _,z in obs])}
    # These old module arrays have the same message order, verified by source identity.
    from bootstrap_peer15_paper_v1_20261103 import configs
    from run_peer10_aux_remaining_v2_20261101 import modules
    _,cs=configs();mm,mr,_,_,_=modules('val21',cs['modules'])
    for key in ('sequence','wait_ms','receiver_timestamp_us'):assert np.array_equal(m[key],mm[key])
    with np.load(P16/'E3_external_temporal_v3/val21_states.npz') as z:
        for name in ('freealign','roco','cbm'):
            estimates[name]=z[name+'__observation'];valid[name]=z[name+'__valid'].astype(bool)
    ma,icp,mask,identity=references(rows);estimates['ICP']=icp;valid['ICP']=mask
    output=[];arrays={}
    for name,value in estimates.items():
        for ref,target,selected in [('MA5',ma,np.ones(len(rows),bool)),('ICP_quality_subset',icp,mask)]:
            if name=='ICP' and ref=='ICP_quality_subset':continue
            for wait in (0,100,200,400):
                pop=m['wait_ms']==wait;use=pop&selected&valid[name]
                if not use.any():continue
                seq=m['sequence'][use].astype(str);ts=m['receiver_timestamp_us'][use];err=value[use]-target[use]
                result=stats(seq,ts,err);key=name+'__'+ref+'__'+str(wait)
                arrays[key+'_sequence']=seq;arrays[key+'_timestamp']=ts;arrays[key+'_error']=err
                persistent=result['demeaned_lag1_correlation']['value']>.5 or result['sequence_bias_fraction']['value']>.5
                output.append(dict(method=name,reference=ref,wait_ms=wait,available=int(use.sum()),
                    reference_population=int((pop&selected).sum()),availability=float(use.sum()/max((pop&selected).sum(),1)),
                    metrics=result,persistent_or_bias_dominated=persistent,
                    geometry='module estimate is mean XY displacement of cached rigid/3D-tilted boxes relative to stale; rotation effects remain in centroid'))
                atomic(out/'panels.json',output)
    np.savez_compressed(out/'errors.npz',**arrays);atomic(out/'identity.json',identity)

def A4(out):
    from run_peer16_cached_methods_v1_20261103 import materials,observation
    import run_peer17_final_recipe_v1_20261104 as final
    from run_peer07_spd_gt_source_split_ap_v1_20261031 import map_provenance
    from scipy.optimize import linear_sum_assignment
    m,rows=materials('author')[0]['val21'];allrows=[None]*(max(map(int,m['dataset_index']))+1)
    for i,r in zip(m['dataset_index'],rows):allrows[int(i)]=r
    codes,join=map_provenance(m,allrows,'val21',ROOT/'artifacts/data/v2x_seq_spd_main_domain_shared_v1/V2X-Seq-SPD/cooperative/label')
    assert join['unmatched']==join['ambiguous']==0
    obs=observation(m);n=np.array([len(z) for _,z in obs]);z=np.array([a.mean(0) if len(a) else np.zeros(2) for _,a in obs]);d=final.old.author.kf(m,n,z,.15,.6)[0]
    before,_=final.old.author.heading.build_heading(m,np.zeros_like(d));after,_=final.old.author.heading.build_heading(m,d)
    _,_,mask,identity=references(rows);values=defaultdict(list);pop=defaultdict(int)
    for i,r in enumerate(rows):
        a,b=map(int,m['condition_offsets'][i:i+2]);ga,gb=map(int,m['gt_offsets'][i:i+2])
        rr=a+np.flatnonzero(m['source'][a:b]==1);gg=ga+np.flatnonzero((codes[ga:gb]==2)&final.old.author.common_mask(m['gt_boxes'][ga:gb]))
        for subset in ('all','ICP_quality_subset'):
            if subset=='ICP_quality_subset' and not mask[i]:continue
            pop[int(m['wait_ms'][i]),subset]+=len(gg)
        if not len(rr) or not len(gg):continue
        distance=np.linalg.norm(before[rr,None,:2]-m['gt_boxes'][None,gg,:2],axis=-1)
        x,y=linear_sum_assignment(distance);good=distance[x,y]<=3.;ri,gi=rr[x[good]],gg[y[good]]
        for method,box in [('PUBLIC',before),('CoMET',after)]:
            err=np.linalg.norm(box[ri,:2]-m['gt_boxes'][gi,:2],axis=1)
            for subset in ('all','ICP_quality_subset'):
                if subset=='ICP_quality_subset' and not mask[i]:continue
                values[int(m['wait_ms'][i]),subset,method].extend((str(r['sequence']),float(e)) for e in err)
    arrays={};panels=[];rng=np.random.default_rng(2026110403)
    for (w,subset,method),items in sorted(values.items()):
        seq=np.array([s for s,e in items]);e=np.array([e for s,e in items]);names=np.unique(seq)
        def stat(x):return [float((x>.5).mean()),float((x>1.).mean()),float(np.sqrt(np.mean(x*x)))]
        groups=[e[seq==s] for s in names];boots=np.array([stat(np.concatenate([groups[j] for j in rng.integers(len(groups),size=len(groups))])) for _ in range(2000)])
        key=str(w)+'__'+subset+'__'+method;arrays[key+'_sequence']=seq;arrays[key+'_error']=e
        panels.append(dict(wait_ms=w,subset=subset,method=method,matched=len(e),GT_population=pop[w,subset],
            metrics=dict(zip(('fraction_gt0p5m','fraction_gt1m','RMSE_m'),[dict(value=v,CI95=np.quantile(boots[:,j],[.025,.975]).tolist()) for j,v in enumerate(stat(e))])),
            CDF=[dict(meters=float(t),fraction=float((e<=t).mean())) for t in np.arange(0,3.001,.05)]))
    np.savez_compressed(out/'errors.npz',**arrays);atomic(out/'panels.json',panels)
    atomic(out/'identity.json',dict(source_join=join,references=identity,
        matching='One-to-one Hungarian XY centre association of uncorrected heading PUBLIC remote boxes to inf GT, gate3m, fixed pair IDs for both arms; unmatched GT counted. Conditional localization diagnostic, not AP or all-object error; no outcome-based matching changes.'))

def A7(out):
    roots=[ROOT/'vendor/DAIR-V2X',ROOT/'vendor/OpenCOOD-main',ROOT/'third_party/third_party/related_works/TraF-Align',
        ROOT/'third_party/external/FFNet_VIC3D_20261030',ROOT/'third_party/external/CoAlign_64d1987affe4fb881c439cc498c30ab01bec67e8']
    rows=[]
    for root in roots:
        if not root.exists():continue
        result=subprocess.run(['rg','-n','--glob','*.py','--glob','!build/**','--glob','!data/**',
            'system_error_offset|delta_x|delta_y|relative_error',str(root)],capture_output=True,text=True)
        assert result.returncode in (0,1)
        for line in result.stdout.splitlines():
            path,num,text=line.split(':',2);p=__import__('pathlib').Path(path);n=int(num);code=p.read_text(errors='replace').splitlines()
            rows.append(dict(file=str(p.relative_to(ROOT)),line=n,text=text,context=code[max(0,n-4):n+3],
                source=ev(p),classification='static_occurrence_requires_call_path_interpretation'))
    assert rows;atomic(out/'code_evidence.json',rows)
    atomic(out/'interpretation.json',dict(scope='Static occurrences with file/line/context/SHA; not proof of default settings or all published results.',
        absent_roots=[str(p.relative_to(ROOT)) for p in roots if not p.exists()],
        warning='Inference assignment, disabled options, dataset preprocessing and annotation construction must be distinguished before prose claims.'))

def run(name):
    out=BASE/name;assert load(out/'contract.json')['script']==ev(__file__);start=time.monotonic()
    record(name,dict(status='running_analysis',resource_owner=True))
    try:
        globals()[name](out)
        atomic(out/'terminal.json',dict(created_at=now(),status='complete_light_checked',GPU_forwards=0,updates=0,
            wall_s=time.monotonic()-start,independent_pending=True,script=ev(__file__)))
        record(name,dict(status='complete_light_checked',terminal=ev(out/'terminal.json'),owner=None))
    except Exception as e:
        atomic(out/'terminal.json',dict(created_at=now(),status='failed_retained',error=repr(e),traceback=traceback.format_exc()))
        record(name,dict(status='failed_retained',terminal=ev(out/'terminal.json'),owner=None));raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','A2','A3','A4','A7']);a=p.parse_args();prepare() if a.mode=='prepare' else run(a.mode)
