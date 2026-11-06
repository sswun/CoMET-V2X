"""Finite independent point/AP and source-span reviewers; scope is per asset.

A1 shares reference construction and saved readout scores; matches geometry
independently. A5 independently checks reported point equations (not all CI).
A7 source spans are checked independently, not dynamic published-runtime proof.
"""
import argparse,hashlib,time,traceback
import numpy as np
from peer18_v1_20261104 import *
OUT=BASE/'A1_A5_A7_independent'
def prepare(mode):
    register('review_'+mode,__file__,__doc__,dict(detector_forwards=0,updates=0,CPU_threads=1,disk_MiB='<5'),asset=mode,
        actual_source=ev(BASE/mode/'contract.json') if mode!='A7' else ev(BASE/'A7_callpaths/contract.json'))
def ready(path):
    starttime=time.monotonic()
    while not path.exists():
        assert time.monotonic()-starttime<20*3600;time.sleep(30)
    assert load(path)['status'] in ('complete_light_checked','partial_explicit_pending')
def A7(out):
    ready(BASE/'A7_callpaths/terminal.json');rows=load(BASE/'A7_callpaths/callpath_evidence.json')['rows'];checked=[]
    for row in rows:
        for entry in row['evidence']:
            p=ROOT/entry['source']['path'];assert hashlib.sha256(p.read_bytes()).hexdigest()==entry['source']['sha256']
            fragment='\n'.join(p.read_text().splitlines()[entry['line_start']-1:entry['line_end']]);assert fragment==entry['lines']
            checked.append(dict(source=entry['source'],line=entry['line_start'],end=entry['line_end'],exact=True))
    assert len(checked)>=8;atomic(out/'source_spans.json',checked)
    return dict(scope='Independent SHA/span verification of source-level interpreted call paths; no universal default/publication claim and no dynamic tracing.')
def A5(out):
    ready(BASE/'A5/terminal.json');checked=[]
    with np.load(BASE/'A5/sequence_statistics.npz') as z:
        for r in load(BASE/'A5/panels.json'):
            wait=r['wait_ms'];g=z['wait'+str(wait)+'_sufficient'];s=g.sum(0);p=s[3]/s[2];R=.36/(s[4]/s[3]);Q=.15**2;P=r['mean_field_P_per_axis_m2'];X=P+Q
            assert abs(p*X*X-Q*(X+R))<1e-10
            values=dict(message_rate_Hz=s[2]/s[1],informative_arrival_lambda_Hz=s[3]/s[1],
                mean_anchors_given_informative=s[4]/s[3],informative_fraction=p,
                measured_CoMET_to_MA5_RMSE_m=np.sqrt(s[5]/s[6]),measured_CoMET_to_ICP_quality_RMSE_m=np.sqrt(s[7]/s[8]),
                predicted_radial_RMSE_m=np.sqrt(2*P),N_eff_tau_times_lambda=r['tau_c_seconds']*s[3]/s[1])
            assert all(abs(float(v)-r[k])<1e-10 for k,v in values.items())
            a=s[9:].reshape(100,5);rho=a[:,0]/np.maximum(np.sqrt(a[:,1]*a[:,2]),1e-15);lag=a[:,3]/np.maximum(a[:,4],1)
            assert np.max(np.abs(rho-r['ACF']))<1e-10 and np.max(np.abs(lag-r['lag_seconds']))<1e-10
            crossed=np.flatnonzero((a[:,4]>0)&(rho<=1/np.e))
            assert bool(r['tau_c_right_censored'])==bool(not len(crossed))
            if len(crossed):assert abs(r['tau_c_seconds']-lag[crossed[0]])<1e-10
            checked.append(dict(wait_ms=wait,equation_residual=float(p*X*X-Q*(X+R)),points_exact=True))
    atomic(out/'point_checks.json',checked)
    return dict(scope='Independent sufficient-statistic point equations/ACF/arrival/Riccati residual only. Reference construction and sequence bootstrap CI not independently recomputed here; mean-field not a physical bound.')
def A1(out):
    ready(BASE/'A1/terminal.json')
    import run_peer17_final_recipe_v1_20261104 as final
    from run_peer16_cached_methods_v1_20261103 import materials,observation
    from run_peer12_cached_ceiling_v1_20261101 import reference_deltas
    from review_peer16_E4_E7_AP_v1_20261103 import check,geometry
    import review_traf_v2xseq_adapted_fixed60_raw_20261006 as ind
    m,rows=materials('author')[0]['val21'];official={str(r['vehicle_frame']):r for r in load(ROOT/'artifacts/data/v2x_seq_spd_traf_align_author21_v1/V2X-Seq-SPD/cooperative/data_info.json')};refs=reference_deltas(rows,official)
    obs=observation(m);n=np.array([len(z) for _,z in obs]);z=np.array([a.mean(0) if len(a) else np.zeros(2) for _,a in obs]);d=final.old.author.kf(m,n,z,.15,.6)[0]
    delta={'PUBLIC':np.zeros_like(d),'OFFICIAL':refs['OFFICIAL'],'MA5':refs['OFFICIAL_MA5'],'CoMET':d};checks=[];point={}
    for seed in final.SEEDS:
        gold=[r for r in load(BASE/'A1/reference_panels.json') if r['seed']==seed]
        gold += [dict(r,method='CoMET') for r in load(final.OUT/'author/panels.json') if r['role']=='val21' and r['seed']==seed]
        with np.load(BASE/'A1'/('seed'+str(seed)+'_predictions.npz')) as p:
            for method,dv in delta.items():
                cc=check(m,geometry(m,dv),p,method,gold);checks.extend(dict(r,seed=seed) for r in cc)
                for r in cc:
                    for th in (50,70):point[seed,method,th,r['wait_ms']]=r['AP'+str(th)]
            for w in (0,100,200,400):
                indices=np.flatnonzero(m['wait_ms']==w);scores=[];tp={50:[],70:[]};GT=0
                for i in indices:
                    a,b=map(int,p['EGO__offset'][i:i+2]);scores.extend(p['EGO__score'][a:b]);ga,gb=map(int,m['gt_offsets'][i:i+2]);GT+=int(final.old.author.common_mask(m['gt_boxes'][ga:gb]).sum())
                    for th in (50,70):tp[th].extend(p['EGO__tp'+str(th)][a:b])
                for th in (50,70):point[seed,'EGO',th,w]=100*ind.voc_ap(tp[th],(1-np.asarray(tp[th])).tolist(),scores,GT,True)['ap']
        atomic(out/'AP.json',checks)
    assert all(abs(r['error'+str(th)])<1e-8 and r['TP_mismatch'+str(th)]==0 for r in checks for th in (50,70))
    for r in load(BASE/'A1/retention.json'):
        s,th,w=r['seed'],r['threshold'],r['wait_ms']
        def val(method):return float(np.mean([point[s,method,th,wait] for wait in (0,100,200,400)])) if w=='four_wait_mean' else point[s,method,th,w]
        den=val(r['reference'])-val('EGO');numer=val(r['method'])-val('EGO')
        if abs(den)>1e-12:assert abs(numer/den-r['retention'])<1e-8
    return dict(scope='Independent polygon matching/AP for final late references + point-ratio arithmetic; saved ego TP, reference construction and weighted bootstrap shared. Native coverage remains incomplete, no median-backbone claim.')
def run(mode):
    dest=OUT/mode;dest.mkdir(parents=True,exist_ok=False)
    try:
        scope=globals()[mode](dest);atomic(dest/'terminal.json',dict(created_at=now(),status='complete_independently_reviewed',**scope))
        record('review_'+mode,dict(status='complete_independently_reviewed',terminal=ev(dest/'terminal.json'),owner=None,resource_owner=False))
    except Exception as e:
        atomic(dest/'terminal.json',dict(created_at=now(),status='failed_retained',error=repr(e),traceback=traceback.format_exc()));raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run']);p.add_argument('--mode',choices=['A1','A5','A7'],required=True);a=p.parse_args();globals()[a.action](a.mode)
