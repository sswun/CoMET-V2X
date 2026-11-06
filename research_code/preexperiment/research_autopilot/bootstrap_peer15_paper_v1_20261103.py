"""Paired whole-sequence pooled-AP bootstrap, identities never pooled across seeds."""
import argparse, time, traceback
from collections import defaultdict
from pathlib import Path
import numpy as np
from peer15_v1_20261103 import ROOT,BASE,AUTO,AUDIT,C12,atomic,ev,load,now,register,record
from peer14_cycle12_v1_20261103 import C15,C18
import run_peer10_fixed_population_aux_v1_20261101 as aux
from run_peer10_aux_remaining_v2_20261101 import modules
from object_history_selector_core_v1_20261021 import common_mask
from analyze_v2xvit_current_geometry_sequence_bootstrap_v1_20261025 import weighted_ap_batch,voc_ap
REPS=2000

def prepare():
    for mode in ('late','cross','coalign','recipe'):
        register('bootstrap_'+mode,Path(__file__).resolve(),'Paired whole-sequence pooled-AP CI may cross zero despite positive point AP; report positive/zero/negative sequence counts and all waits, never filter signs. Existing recipe/source/head identities retained separately.',dict(GPU_forwards=0,updates=0,CPU_threads=1,replicates=REPS,wall_h='1-8 depending frozen family; recipe finite16h dependency',disk_MiB='<30 bootstrap draws/deltas only; original predictions/GT zero-copy'),mode=mode,method='Weighted empirical cluster resampling of whole sequences, same count draws for both methods/all waits, original accepted score-tie order preserved. Percentile95% pooled AP delta, NOT bootstrap of mean per-sequence AP.',seed=20261103,paired_population='Exact sequence/wait/current-frame/GT-count join for cross-family contrasts; no cross-paradigm absolute ranking',primary_inputs=ev(AUDIT/'contract_v1.json'))

def counts(seqs):
    rng=np.random.default_rng(20261103);draw=rng.integers(len(seqs),size=(REPS,len(seqs)));c=np.zeros((REPS,len(seqs)),np.int16);np.add.at(c,(np.arange(REPS)[:,None],draw),1);return c

def make(seq,wait,gt,offset,score,tp):
    seq=np.asarray(seq).astype(str);wait=np.asarray(wait);gt=np.asarray(gt,np.int64);offset=np.asarray(offset,np.int64)
    assert len(seq)==len(wait)==len(gt)==len(offset)-1 and offset[-1]==len(score)==len(tp)
    return dict(sequence=seq,wait=wait,gt=gt,offset=offset,score=np.asarray(score),tp=np.asarray(tp))

def from_pred(m,p,name,th):
    gt=np.asarray([int(common_mask(m['gt_boxes'][int(a):int(b)]).sum()) for a,b in zip(m['gt_offsets'][:-1],m['gt_offsets'][1:])])
    return make(m['sequence'],m['wait_ms'],gt,p[name+'__offset'],p[name+'__score'],p[name+'__tp'+str(th)])

def panel(d,wait,seqs):
    ids=np.flatnonzero(d['wait']==wait);index={s:i for i,s in enumerate(seqs)};select=np.concatenate([np.arange(int(d['offset'][i]),int(d['offset'][i+1])) for i in ids]);cluster=np.concatenate([np.full(int(d['offset'][i+1]-d['offset'][i]),index[d['sequence'][i]],np.int16) for i in ids]);gt=np.zeros(len(seqs),np.int64)
    for i in ids:gt[index[d['sequence'][i]]]+=d['gt'][i]
    return d['score'][select],d['tp'][select],cluster,gt

def contrast(out,label,a,b,threshold,identity):
    seqs=sorted(set(a['sequence']));assert seqs==sorted(set(b['sequence']))
    # Compare complete sequence/wait/frame-ordinal GT, allowing source layouts.
    def population(d):
        result={}
        for seq in seqs:
            for wait in sorted(set(map(int,d['wait']))):result[seq,wait]=d['gt'][(d['sequence']==seq)&(d['wait']==wait)].tolist()
        return result
    assert population(a)==population(b),('paired_GT_population',label)
    draws=counts(seqs);rows=[];rep={};seqrows=[]
    for wait in sorted(set(map(int,a['wait']))):
        x=panel(a,wait,seqs);y=panel(b,wait,seqs);assert np.array_equal(x[3],y[3])
        aa=weighted_ap_batch(*x,draws,batch_size=8);bb=weighted_ap_batch(*y,draws,batch_size=8);delta=100*(aa-bb)
        # Weight-one replay independently checks point statistic against VOC.
        check=[]
        for p in (x,y):check.append(abs(weighted_ap_batch(*p,np.ones((1,len(seqs)),np.int16),batch_size=1)[0]-voc_ap(p[1],p[0],int(p[3].sum())))<1e-12)
        assert all(check)
        point=100*(voc_ap(x[1],x[0],int(x[3].sum()))-voc_ap(y[1],y[0],int(y[3].sum())))
        per=[]
        for j,s in enumerate(seqs):
            xm=x[2]==j;ym=y[2]==j;value=100*(voc_ap(x[1][xm],x[0][xm],int(x[3][j]))-voc_ap(y[1][ym],y[0][ym],int(y[3][j])));per.append(value);seqrows.append(dict(comparison=label,wait_ms=wait,threshold=threshold,sequence=s,delta_pp=value))
        rows.append(dict(comparison=label,threshold=threshold,wait_ms=wait,delta_pp=point,CI95_low=float(np.quantile(delta,.025)),CI95_high=float(np.quantile(delta,.975)),positive_sequences=sum(v>0 for v in per),negative_sequences=sum(v<0 for v in per),zero_sequences=sum(v==0 for v in per),sequences=len(seqs),replicates=REPS,GT=int(x[3].sum()),identity=identity,weighted_one_replay_exact=True));rep['wait'+str(wait)]=delta
    # Four-wait mean is paired over the same resampled whole sequences.
    average=np.mean(list(rep.values()),axis=0);rows.append(dict(comparison=label,threshold=threshold,wait_ms='four_wait_mean',delta_pp=float(np.mean([r['delta_pp'] for r in rows])),CI95_low=float(np.quantile(average,.025)),CI95_high=float(np.quantile(average,.975)),sequences=len(seqs),replicates=REPS,identity=identity))
    p=out/(label+'_AP'+str(threshold));atomic(p.with_suffix('.json'),dict(panels=rows,per_sequence=seqrows));np.savez_compressed(p.with_suffix('.npz'),cluster_counts=draws,sequence=np.asarray(seqs),**rep)
    return rows

def configs():
    v=load(AUDIT/'cycle09_v1/aux_hwl_contract_v3.json');s=load(aux.verified(v['original_contract']));return s,{k:load(aux.verified(r)) for k,r in s['baseline_contracts'].items()}

def late(out):
    spec,c=configs();families={};rowsout=[]
    for fam,loader in [('geometry',aux.geometry),('modules',modules),('official_late',aux.official_late)]:
        m,rows,p,boxes,mask=loader('val21',c[fam]);families[fam]=(m,rows,p)
    gm,gr,gp=families['geometry'];mm,mr,mp=families['modules'];lm,lr,lp=families['official_late']
    pairs=[('author_kf_minus_stale','geometry','kf__matched_head','geometry','stale__matched_head'),('author_kf_minus_CV','geometry','kf__matched_head','geometry','cv__matched_head'),('author_kf_minus_FreeAlign_matched','geometry','kf__matched_head','geometry','freealign__matched_head'),('author_kf_minus_official_late','geometry','kf__matched_head','official_late','official_late_no_comp'),('author_kf_minus_official_timecomp','geometry','kf__matched_head','official_late','official_late_time_comp')]
    for b in ('stale','freealign','roco','cbm'):pairs.append(('freshU_KF_minus_'+b,'modules','causal_kf__fresh39d','modules',b+'__fresh39d'))
    for label,af,an,bf,bn in pairs:
        am,ar,ap=families[af];bm,br,bp=families[bf]
        def keys(rr):return sorted((str(r['sequence']),int(r['wait_ms']),str(r['current_vehicle_frame'])) for r in rr)
        assert keys(ar)==keys(br)
        for th in (50,70):rowsout.extend(contrast(out,label,from_pred(am,ap,an,th),from_pred(bm,bp,bn,th),th,dict(a=an,a_family=af,b=bn,b_family=bf,role='val21',contracts={af:ev(AUTO/'status_audit_20261031_peer07_geometry_matched_source_ap_v1/contract_v1.json')})))
        atomic(out/'panels.json',rowsout)
    # New raw author compact outcomes are the sole independently recomputed native source.
    native=BASE/'external_raw_traf';start=time.monotonic()
    while not (native/'terminal.json').exists():
        assert time.monotonic()-start<3600,'finite1h raw collector wait';time.sleep(30)
    for th in (50,70):
        parts=[];offset=[0];seq=[];waits=[];gt=[];tp=[];score=[]
        for w in (0,100,200,400):
            with np.load(native/('wait'+str(w).zfill(3)+'_matches.npz')) as z:
                key='iou'+str(th);seq.extend(z['sequence'].tolist());waits.extend([w]*len(z['sequence']));gt.extend(z[key+'_frame_gt']);tp.extend(z[key+'_tp']);score.extend(z[key+'_scores']);offset.extend((z[key+'_offset'][1:]+offset[-1]).tolist())
        b=make(seq,waits,gt,offset,score,tp);rowsout.extend(contrast(out,'author_kf_minus_native_fusion',from_pred(gm,gp,'kf__matched_head',th),b,th,dict(a='author kf__matched_head',b='same author checkpoint native fusion',raw_review=ev(native/'terminal.json'))));atomic(out/'panels.json',rowsout)
    return rowsout

def cross(out):
    import run_crosscheckpoint_temporal_matched39d_v1_20261029 as matched
    summary=load(AUTO/'status_audit_20261031_crosscheckpoint_motion_direction_information_state_current_inputs_v1/producer_v1/summary.json');ec=load(ROOT/summary['contract']['path']);man=load(aux.verified(ec['inputs']['input_identity_manifest']));sc=load(aux.verified(man['source_contract']));rows=[]
    for domain in ('v2xvit_seed2024','v2xvit_seed2025','where2comm'):
        materials,_,_=matched.prepare_domain(sc['domains'][domain]);m=materials['val21'];p=aux.core.load_npz(aux.verified(summary['domains'][domain]['outputs']['val21']['predictions']))
        for th in (50,70):rows.extend(contrast(out,domain+'_heading_minus_CV',from_pred(m,p,'heading_original_q005_kf',th),from_pred(m,p,'original_cv_q005_kf',th),th,dict(domain=domain,head=man['domains'][domain]['matched_q005_head'],not_native_comparison=True)))
        if domain.startswith('v2xvit_'):
            seed=domain[-4:];path=AUTO/('status_audit_20261025_evening/v2xvit_current_geometry_sequence_bootstrap_v1/native_seed'+seed+'_v2/native_sequence_arrays_v2.npz');z=aux.core.load_npz(path);a=from_pred(m,p,'heading_original_q005_kf',70);seqs=sorted(set(a['sequence']));assert list(z['sequences'].astype(str))==seqs
            # Native compact arrays already have whole-sequence clusters, no frame offsets.
            # Expand the condition-free representation into one record per sequence/wait,
            # and identically aggregate candidate condition GT to that same population.
            def grouped_data(d):
                ss=[];ww=[];gg=[];scores=[];tp=[];off=[0]
                for w in (0,100,200,400):
                    x=panel(d,w,seqs)
                    for j,s in enumerate(seqs):
                        use=x[2]==j;ss.append(s);ww.append(w);gg.append(x[3][j]);scores.extend(x[0][use]);tp.extend(x[1][use]);off.append(len(tp))
                return make(ss,ww,gg,off,scores,tp)
            ss=[];ww=[];gg=[];scores=[];tp=[];off=[0]
            for w in (0,100,200,400):
                key='wait'+str(w).zfill(3)
                for j,s in enumerate(seqs):
                    use=z[key+'__cluster']==j;ss.append(s);ww.append(w);gg.append(z[key+'__gt_by_cluster'][j]);scores.extend(z[key+'__score'][use]);tp.extend(z[key+'__tp70'][use]);off.append(len(tp))
            rows.extend(contrast(out,domain+'_heading_minus_native',grouped_data(a),make(ss,ww,gg,off,scores,tp),70,dict(domain=domain,native=ev(path),scope='same detector native input/GT; AP50 native ledger absent here, remains explicit followup')))
        atomic(out/'panels.json',rows)
    return rows

def coalign(out):
    import run_peer14_coalign_composed_graph_v1_20261103 as composed
    import review_traf_v2xseq_adapted_fixed60_raw_20261006 as independent
    rows=[]
    for lane in ('gpu0','gpu1','remote'):
        seed=composed.plans(lane)[0]
        for train in ('public','online'):
            ref=C15/'coalign_full_v1'/('adapted_evaluation' if train=='public' else 'online_adapted_evaluation') if seed==20261102 else C18/'coalign_confirmation_v1'/('seed'+str(seed))/(train+'_reference/evaluation')
            graph=ref if seed==20261102 else C18/'coalign_confirmation_v1'/('seed'+str(seed))/(train+'_graph/evaluation')
            ds={}
            for name,path,arm in [('K1_INPUT',ref,'K1_INPUT'),('PUBLIC_GRAPH',graph,'COALIGN_FULL')]:
                records=load(path/'records.json');assert len(records)==13264;d=defaultdict(list);offset=[0];tps={50:[],70:[]};scores=[]
                for w in (0,100,200,400):
                    rr=[r for r in records if r['wait_ms']==w]
                    with np.load(path/f'wait{w:03d}.npz') as z,np.load(ROOT/rr[0]['GT_shard']) as gz:
                        for r in rr:
                            i=r['shard_case'];gt=gz[f"c{r['GT_case']}_gt"];gt=gt[gt[:,7]==0,:7];b=z[f'c{i}_{arm}_box3d_lidar'];s=z[f'c{i}_{arm}_scores'];label=z[f'c{i}_{arm}_label_preds'];keep=(label==0)&(b[:,0]>=-100.8)&(b[:,0]<=100.8)&(abs(b[:,1])<=40);b,s=b[keep],s[keep];n=len(s)
                            for th in (50,70):
                                ss,tp,fp,ng=independent.match_frame(b,s,np.zeros(n),gt,np.zeros(len(gt)),th/100);tps[th].extend(tp)
                            scores.extend(ss);offset.append(len(scores));d['sequence'].append(str(r['sequence']));d['wait'].append(w);d['gt'].append(len(gt))
                ds[name]={th:make(d['sequence'],d['wait'],d['gt'],offset,scores,tp) for th,tp in tps.items()}
            for th in (50,70):rows.extend(contrast(out,'CoAlign_'+str(seed)+'_'+train+'_K1INPUT_minus_GRAPH',ds['K1_INPUT'][th],ds['PUBLIC_GRAPH'][th],th,dict(seed=seed,training=train,K1_INPUT=ev(ref/'terminal.json'),PUBLIC_GRAPH=ev(graph/'terminal.json'),scope='same full CoAlign weights, own native ROI; independently reconstructed raw matching, not common-ROI absolute ranking')))
            atomic(out/'panels.json',rows)
    return rows

def recipe(out):
    p=BASE/'author_recipe_v2';start=time.monotonic()
    while not (p/'primary_predictions.npz').exists():
        if (p/'terminal.json').exists():assert load(p/'terminal.json')['status']!='failed_retained',load(p/'terminal.json')
        assert time.monotonic()-start<16*3600,'finite16h current author recipe wait';time.sleep(30)
    pred=aux.core.load_npz(p/'primary_predictions.npz');rows=[]
    def arm(name,t):return make(pred['condition_sequence'],pred['condition_wait_ms'],pred['condition_GT_count'],pred[name+'__offset'],pred[name+'__score'],pred[name+'__tp'+str(t)])
    for t in (50,70):rows.extend(contrast(out,'author_current_heading_minus_v1',arm('current_heading_calq_matchedU',t),arm('v1_CV_q015_matchedU',t),t,dict(recipe=ev(p/'contract.json'),calibration=ev(p/'calibration.json'),head='same author matched U both arms, no refit')))
    atomic(out/'panels.json',rows);return rows

def run(mode):
    out=BASE/('bootstrap_'+mode);start=time.monotonic();cpu=time.process_time();c=load(out/'contract.json');assert c['script']==ev(__file__);record('bootstrap_'+mode,dict(status='running_paired_whole_sequence_pooled_AP'))
    try:
        rows=globals()[mode](out);atomic(out/'terminal.json',dict(created_at=now(),status='complete_light_checked',panels=ev(out/'panels.json'),panels_count=len(rows),GPU_forwards=0,updates=0,replicates=REPS,wall_s=time.monotonic()-start,CPU_s=time.process_time()-cpu,independent_statistic_review_pending=True,no_model_wakeup=True));record('bootstrap_'+mode,dict(status='complete_light_checked',terminal=ev(out/'terminal.json')))
    except Exception as e:
        atomic(out/'terminal.json',dict(created_at=now(),status='failed_retained',error=repr(e),traceback=traceback.format_exc(),wall_s=time.monotonic()-start));record('bootstrap_'+mode,dict(status='failed_retained',terminal=ev(out/'terminal.json')));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','late','cross','coalign','recipe']);a=p.parse_args();prepare() if a.mode=='prepare' else run(a.mode)
