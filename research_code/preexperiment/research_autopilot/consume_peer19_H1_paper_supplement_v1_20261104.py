"""R2 concrete anchor/trajectory assets, then versioned unified H1 tables.

No fitting, detector forwards, hypothesis changes, or formal-role selection.
Existing frozen generator stays unchanged; integration waits for its terminal,
archives its small table files, and independently checks the new numeric ledger.
"""
import argparse,csv,io,json,math,shutil,time,traceback
from collections import defaultdict
import numpy as np
from scipy.optimize import linear_sum_assignment
from peer19_v1_20261104 import *
import run_peer19_H1_assets_v1_20261104 as h1

OUT=BASE/'H1_paper_supplement'
TABLE=ROOT/'docs/paper_draft_v0_20261103/final_tables_v3'
JOBS=('H1_paper_supplement','final_tables_v3_integration','final_tables_v3_CoAlign_attachment')

def prepare():
    for name in JOBS:
        register(name,__file__,__doc__,dict(GPU_forwards=0,updates=0,CPU_threads=1,
            disk_MiB='<60 metadata/trajectory/CSV; no copied candidate/GT caches',wall_h='bounded36h dependencies'),
            deadline='2026-11-05T18:00:00+08:00' if name!=JOBS[2] else '2026-11-06T06:00:00+08:00',
            dependencies=['Actual H1 saved states/weights','R1 independent reviews','frozen v3 generator numeric audit'],
            old_tables_preserved='final_tables_v3/versions/initial_generator_v1',
            R3_separate=True,selection='No new fit/cal or formal-role selection')

def dump_csv(path,rows):
    assert rows
    stream=io.StringIO();writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');tmp.write_text(stream.getvalue());tmp.replace(path)

def independent_observations(m):
    counts=[];means=[]
    for lo,hi in zip(m['condition_offsets'][:-1],m['condition_offsets'][1:]):
        lo,hi=int(lo),int(hi);remote=lo+np.where(m['source'][lo:hi]==1)[0];ego=lo+np.where(m['source'][lo:hi]==0)[0]
        if not len(remote) or not len(ego):counts.append(0);means.append([0.,0.]);continue
        # Same published float32 geometry, separate implementation/call site.
        r=m['cv_boxes'][remote];e=m['cv_boxes'][ego];distance=np.linalg.norm(r[:,None,:2]-e[None,:,:2],axis=2)
        ii,jj=linear_sum_assignment(distance);keep=(distance[ii,jj]<=2.5)&(np.argmin(distance,axis=1)[ii]==jj)&(np.argmin(distance,axis=0)[jj]==ii)
        ii,jj=ii[keep],jj[keep];counts.append(len(ii))
        residual=e[jj,:2].astype(np.float64)-r[ii,:2].astype(np.float64)
        means.append(residual.mean(axis=0).tolist() if len(ii) else [0.,0.])
    return np.asarray(counts),np.asarray(means)

def independently_recur(m,n,z,q):
    history={};states=[];variances=[];steps=[];gains=[];innovations=[]
    for seq,w,count,measurement in zip(m['sequence'],m['wait_ms'],n,z):
        key=str(seq),int(w);x,p,k=history.get(key,(np.zeros(2),1.,0));p0=p+q**2
        gain=p0/(p0+.36/count) if count else 0.;innovation=measurement-x
        xx=x+gain*innovation;pp=(1-gain)*p0
        states.append(xx);variances.append(pp);steps.append(k);gains.append(gain);innovations.append(innovation)
        history[key]=(xx,pp,k+1)
    return tuple(map(np.asarray,(states,variances,steps,gains,innovations)))

def assets():
    begin=time.monotonic();record(JOBS[0],dict(status='running_concrete_H1_anchor_trajectory_assets',resource_owner=True))
    OUT.mkdir(parents=True,exist_ok=True);panels=[];checks=[];identities={};trajectories=[]
    try:
        kind,_,_,_=h1.context()
        for domain in h1.recipe.DOMAINS:
            data,_,_=h1.recipe.materials(domain);dest=h1.REC/domain;cal=load(dest/'calibration.json');q=cal['q']
            identities[domain]=dict(calibration=ev(dest/'calibration.json'),q=q,tau=cal['choices'][kind]['tau'],kind=kind,
                heads={str(s):dict(weights=ev(h1.recipe.head(domain,kind,s)/'direct_u.pth'),maps=ev(h1.recipe.head(domain,kind,s)/'source_maps.json')) for s in h1.recipe.SEEDS},
                source_loader=ev(Path(h1.recipe.materials.__code__.co_filename)),states={},
                anchor='ALL source==0, Hungarian + mutual NN <=2.5m; no prediction gating',
                geometry='heading-projected CV + causal KF translation',readout='no-position; prefix separate from anchors',inference_GT=False)
            for role,(m,rows) in data.items():
                statepath=dest/(role+'_state.npz');start=time.monotonic()
                while not statepath.exists():assert time.monotonic()-start<30*3600;time.sleep(30)
                with np.load(statepath) as z:saved={k:z[k] for k in z.files}
                n,z=independent_observations(m);d,v,step,gain,innovation=independently_recur(m,n,z,q)
                assert np.array_equal(n,saved['anchors']) and np.array_equal(z,saved['observation'])
                err=float(np.max(abs(d-saved['delta'])));varerr=float(np.max(abs(v-saved['variance'])))
                assert err<1e-12 and varerr<1e-12 and np.array_equal(step,saved['step'])
                checks.append(dict(domain=domain,role=role,conditions=len(n),state_max_error_m=err,variance_max_error=varerr,anchors_exact=True,observations_exact=True))
                identities[domain]['states'][role]=ev(statepath)
                for w in sorted(set(map(int,m['wait_ms']))):
                    use=m['wait_ms']==w;nn=n[use];seq=m['sequence'][use];groups=sorted(set(map(str,seq)))
                    sums=np.asarray([nn[seq==s].sum() for s in groups]);sizes=np.asarray([(seq==s).sum() for s in groups]);nonzero=np.asarray([(nn[seq==s]>0).sum() for s in groups])
                    draw=np.random.default_rng(20261104).integers(0,len(groups),(2000,len(groups)))
                    den=sizes[draw].sum(1);mean=sums[draw].sum(1)/den;coverage=nonzero[draw].sum(1)/den
                    panels.append(dict(domain=domain,role=role,wait_ms=w,conditions=len(nn),sequences=len(groups),
                        total_anchors=int(nn.sum()),mean_anchors=float(nn.mean()),anchored_message_fraction=float(np.mean(nn>0)),
                        no_anchor_fraction=float(np.mean(nn==0)),mean_anchors_given_observed=float(nn[nn>0].mean()) if np.any(nn>0) else None,
                        mean_anchors_CI95=np.quantile(mean,[.025,.975]).tolist(),anchored_fraction_CI95=np.quantile(coverage,[.025,.975]).tolist()))
                if domain=='author' and role in ('val21','mechanism_holdout14'):
                    for i,(seq,t,w) in enumerate(zip(m['sequence'],m['receiver_timestamp_us'],m['wait_ms'])):
                        trajectories.append(dict(role=role,sequence=str(seq),receiver_timestamp_us=int(t),wait_ms=int(w),state_row=i,step=int(step[i]),
                            anchors=int(n[i]),observed=bool(n[i]),observation_x_m=float(z[i,0]),observation_y_m=float(z[i,1]),
                            delta_x_m=float(d[i,0]),delta_y_m=float(d[i,1]),posterior_variance_m2=float(v[i]),
                            gain=float(gain[i]),innovation_x_m=float(innovation[i,0]) if n[i] else None,innovation_y_m=float(innovation[i,1]) if n[i] else None))
                atomic(OUT/'anchor_statistics.json',panels);atomic(OUT/'independent_recurrence_checks.json',checks)
            del data
        dump_csv(OUT/'KF_trajectories_author.csv',trajectories);atomic(OUT/'method_identity.json',dict(created_at=now(),registry=ev(h1.REC/'method_registry.json'),domains=identities,
            five_checkpoint_identities_not_five_architectures=True,three_readout_seeds_not_detector_seeds=True,no_target_fit_boundaries=True))
        atomic(OUT/'terminal.json',dict(status='complete_independently_reviewed',created_at=now(),anchor_statistics=ev(OUT/'anchor_statistics.json'),
            trajectory=ev(OUT/'KF_trajectories_author.csv'),identity=ev(OUT/'method_identity.json'),checks=ev(OUT/'independent_recurrence_checks.json'),
            scope='Independent all-ego observation matching and KF recurrence; cluster CI uses saved same observations, not independent detection/training. No AP review claimed.',wall_s=time.monotonic()-begin))
        record(JOBS[0],dict(status='complete_independently_reviewed',terminal=ev(OUT/'terminal.json'),resource_owner=False,owner=None))
    except Exception as e:fail(JOBS[0],OUT,e);raise

def fail(name,dest,e):
    atomic(dest/'terminal.json',dict(status='failed_retained',created_at=now(),error=repr(e),traceback=traceback.format_exc()))
    record(name,dict(status='failed_retained',terminal=ev(dest/'terminal.json'),resource_owner=False,owner=None))

def integrate():
    name=JOBS[1];dest=BASE/name;record(name,dict(status='waiting_actual_v3_numeric_review_and_H1_assets',resource_owner=False,owner=None))
    try:
        wait(BASE/'final_tables_v3/terminal.json',36,accepted=('complete_light_checked',));wait(OUT/'terminal.json',36,accepted=('complete_independently_reviewed',))
        record(name,dict(status='running_versioned_unified_table1_assets',resource_owner=True))
        archive=TABLE/'versions/initial_generator_v1';archive.mkdir(parents=True,exist_ok=False)
        for p in list(TABLE.iterdir()):
            if p.is_file():shutil.move(str(p),archive/p.name)
        cells=load(archive/'numeric_ledger.json')['cells'];manifest=load(archive/'manifest.json')
        baseline=[dict(c,table='table1') for c in cells if c['table']=='legacy_unchanged_table1']
        assert baseline and all(not c['row'].startswith(('CoMET_','FFNet')) for c in baseline)
        cells.extend(baseline)
        src=OUT/'anchor_statistics.json';ref=ev(src);review=ev(OUT/'terminal.json')
        for i,row in enumerate(load(src)):
            for key,value in row.items():
                if key in ('domain','role','wait_ms') or value is None:continue
                values=enumerate(value) if isinstance(value,list) else [(None,value)]
                for j,v in values:
                    cells.append(dict(table='H1_anchor_statistics',row='|'.join(str(row[k]) for k in ('domain','role','wait_ms')),
                        metric=key+('boundary'+str(j) if j is not None else ''),value=float(v),operation='identity',factor=1.,
                        operands=[dict(source=ref,pointer=[i,key]+([j] if j is not None else []),format='json')],
                        review_status='independent_observations_and_recurrence',review_evidence=review,
                        scope='All-ego H1 actual population. Cluster CI2000 sequences; no physical-reference or detector-training independence claim.'))
        for table in sorted({c['table'] for c in cells}):
            dump_csv(TABLE/(table+'.csv'),[dict(row=c['row'],metric=c['metric'],value=c['value'],review_status=c['review_status'],scope=c['scope'],source_operands=json.dumps(c['operands'],ensure_ascii=False)) for c in cells if c['table']==table])
        atomic(TABLE/'numeric_ledger.json',dict(created_at=now(),cells=cells))
        atomic(TABLE/'trajectory_asset.json',dict(data=ev(OUT/'KF_trajectories_author.csv'),review=review,duplicate_state_cache=False))
        atomic(TABLE/'whole_method_identity.json',load(OUT/'method_identity.json'))
        unchanged=dict(A2=P18/'A2/existing_factors.json',A7=P18/'A7_callpaths/callpath_evidence.json')
        # Resolve real existing artifacts explicitly; a missing optional source is open, not fabricated.
        references={k:(ev(p) if p.exists() else dict(status='missing_exact_expected_path',requested_path=str(p.relative_to(ROOT)))) for k,p in unchanged.items()}
        manifest.update(created_at=now(),generator_integration=ev(__file__),previous_generation=ev(archive/'manifest.json'),
            unified_table1_includes_actual_H1_and_unchanged_baselines=True,anchor_statistics=ev(src),trajectory=ev(OUT/'KF_trajectories_author.csv'),
            whole_method_identity=ev(OUT/'method_identity.json'),unchanged_A2_A7=references,numeric_ledger=ev(TABLE/'numeric_ledger.json'),
            status='integrated_pending_numeric_review',R3_independent_attachment='P1 waits separately, does not hold up P0')
        atomic(TABLE/'manifest.json',manifest)
        import review_peer17_final_tables_v2_20261104 as audit
        audit.OUT=TABLE;result=audit.run();assert result['status']=='passed',result
        atomic(dest/'terminal.json',dict(status='complete_light_checked',created_at=now(),numeric_audit=ev(TABLE/'independent_numeric_audit.json'),
            cells=len(cells),scientific_review='Each cell scoped, not whole-paper closure',main_tex_unchanged=True))
        record(name,dict(status='complete_light_checked',terminal=ev(dest/'terminal.json'),resource_owner=False,owner=None))
    except Exception as e:fail(name,dest,e);raise

def coalign():
    name=JOBS[2];dest=BASE/name;record(name,dict(status='waiting_independent_R3_separate_from_P0',resource_owner=False,owner=None))
    try:
        wait(BASE/'coalign_H1_independent/terminal.json',36,accepted=('complete_independently_reviewed',))
        target=TABLE/'CoAlign_R3_independent';target.mkdir(parents=True,exist_ok=False)
        for filename in ('AP.json','bootstrap.json'):
            source=BASE/'coalign_H1_independent'/filename;rows=load(source);dump_csv(target/(filename[:-5]+'.csv'),[dict(source_row=i,values=json.dumps(r,ensure_ascii=False)) for i,r in enumerate(rows)])
        atomic(target/'manifest.json',dict(created_at=now(),sources={f:ev(BASE/'coalign_H1_independent'/f) for f in ('AP.json','bootstrap.json','terminal.json')},
            system_not_module=True,three_existing_fusion_weights_no_training=True,main_P0_tables_not_mutated=True))
        atomic(dest/'terminal.json',dict(status='complete_light_checked',created_at=now(),manifest=ev(target/'manifest.json')))
        record(name,dict(status='complete_light_checked',terminal=ev(dest/'terminal.json'),resource_owner=False,owner=None))
    except Exception as e:fail(name,dest,e);raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','assets','integrate','coalign']);globals()[p.parse_args().mode]()
