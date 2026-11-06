"""All registered cached families, fixed official population, zero forwards.

This is not the older auxiliary-centre ROI protocol. Never move, rerank or
reselect predictions; keep each original method/checkpoint population distinct.
"""
import os,json,time,traceback,resource
from collections import defaultdict
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import numpy as np
from peer10_tournament_v1_20261101 import ROOT,AUTO,AUDIT,atomic,ev,load,now,sha
from peer10_cycle06_v1_20261101 import CYCLE,update
import run_peer10_se2_expanded_anchor_v1_20261101 as science
from run_peer10_native_source_aux_v1_20261101 import outcome
core=science.core;source=science.source
GROUPS=('all','veh','coop','inf','veh_coop')
PROTOCOLS=('official','interpolated','no_interpolation')
LABELROOT=ROOT/'artifacts/data/v2x_seq_spd_main_domain_shared_v1/V2X-Seq-SPD/cooperative/label'

def verified(spec):
    path=ROOT/spec['path'];assert sha(path)==spec['sha256'],('input SHA',str(path));return path

def auxiliary(m,rows,codes,replacements):
    result={p:m['gt_boxes'].copy() for p in PROTOCOLS};cache={};maximum=0.;labelhash={}
    for i,row in enumerate(rows):
        frame=str(row['current_vehicle_frame']);ga,gb=map(int,m['gt_offsets'][i:i+2])
        if frame not in cache:
            path=LABELROOT/(frame+'.json');objects=load(path);labelhash[frame]=sha(path)
            cache[frame]=[(np.asarray([o['3d_location']['x'],o['3d_location']['y']]),str(o['token'])) for o in objects if o['from_side']=='inf']
        for idx in ga+np.flatnonzero(codes[ga:gb]==2):
            candidates=[(float(np.linalg.norm(x-m['gt_boxes'][idx,:2])),token) for x,token in cache[frame]]
            matched=[(d,t) for d,t in candidates if d<.001];assert len(matched)==1,('inf join',frame,int(idx),matched)
            distance,token=matched[0];maximum=max(maximum,distance)
            for protocol in PROTOCOLS[1:]:result[protocol][idx]=replacements[protocol][frame][token]
    for p in PROTOCOLS:assert np.array_equal(result[p][codes!=2],m['gt_boxes'][codes!=2])
    return result,{'source_max_error_m':maximum,'original_label_SHA':labelhash,'veh_coop_unchanged':True}

def evaluate(m,pred,boxes,codes,role,auxgt,gt_roi,method,out):
    # Pre-filtered raw/module pools have no new ROI. Canonical/current materials
    # pass their ORIGINAL official common ROI mask; never mask auxiliary centres.
    offsets=pred[method+'__offset'];ids=pred[method+'__candidate'];scores=pred[method+'__score'];stats=defaultdict(lambda:{'gt':0,'scores':[],**{k:[] for k in ('tp50','fp50','ignore50','tp70','fp70','ignore70')}});seqstats=defaultdict(lambda:{'gt':0,'scores':[],**{k:[] for k in ('tp50','fp50','ignore50','tp70','fp70','ignore70')}})
    arrays={};replay={t:np.zeros(len(ids),np.int8) for t in (50,70)}
    for protocol in PROTOCOLS:
        for group in GROUPS:
            for t in (50,70):
                for field in ('tp','fp','ignore'):arrays[f'{protocol}__{group}__{field}{t}']=np.zeros(len(ids),np.int8)
    for i,wait in enumerate(m['wait_ms']):
        a,b=map(int,m['condition_offsets'][i:i+2]);pa,pb=map(int,offsets[i:i+2]);ga,gb=map(int,m['gt_offsets'][i:i+2]);chosen=ids[pa:pb];assert not len(chosen) or (chosen.min()>=a and chosen.max()<b and len(np.unique(chosen))==len(chosen))
        pp=boxes[chosen];ss=scores[pa:pb];assert np.isfinite(pp).all() and np.isfinite(ss).all();assert np.all(np.diff(ss)<=0),('frozen rank',method,i)
        code=codes[ga:gb][gt_roi[ga:gb]];sequence=str(m['sequence'][i]);gtcount=int(gt_roi[ga:gb].sum())
        for protocol in PROTOCOLS:
            overlap=source.iou32(pp,auxgt[protocol][ga:gb][gt_roi[ga:gb]])
            for group in GROUPS:
                for t in (50,70):
                    tp,fp,ig,count=outcome(overlap,code,group,t/100)
                    if protocol=='official' and group=='all':replay[t][pa:pb]=tp
                    for field,value in [('tp',tp),('fp',fp),('ignore',ig)]:arrays[f'{protocol}__{group}__{field}{t}'][pa:pb]=value
                    for st in (stats[(protocol,int(wait),group)],seqstats[(sequence,protocol,int(wait),group)]):
                        if t==50:st['gt']+=count;st['scores'].append(ss)
                        st['tp'+str(t)].append(tp);st['fp'+str(t)].append(fp);st['ignore'+str(t)].append(ig)
        if i%256==0:atomic(out/'status.json',{'status':'scoring','method':method,'role':role,'conditions_complete':i+1,'total_conditions':len(m['wait_ms'])})
    for t in (50,70):
        if method+f'__tp{t}' in pred:assert np.array_equal(replay[t],pred[method+f'__tp{t}']),('official TP drift',method,role,t)
    result=[]
    for isseq,table in ((False,stats),(True,seqstats)):
        rows=[]
        for key,v in sorted(table.items()):
            protocol,wait,group=key[-3:];row={'role':role,'method':method,'protocol':protocol,'wait_ms':wait,'source_group':group,'gt':v['gt']}
            if isseq:row['sequence']=key[0]
            ss=np.concatenate(v['scores'])
            for t in (50,70):
                tp,fp,ig=[np.concatenate(v[k+str(t)]) for k in ('tp','fp','ignore')];keep=ig==0;metric=source.metric(tp[keep],fp[keep],ss[keep],v['gt']);row.update({f'ap{t}':metric['ap'],f'tp{t}':metric['tp'],f'fp{t}':metric['fp'],f'ignored{t}':int(ig.sum())})
            if not isseq and protocol=='official' and group=='all':
                keep_cases=m['wait_ms']==wait;chunks=[np.arange(int(offsets[i]),int(offsets[i+1])) for i in np.flatnonzero(keep_cases)];indices=np.concatenate(chunks)
                for t in (50,70):
                    if method+f'__tp{t}' in pred:
                        gold=pred[method+f'__tp{t}'][indices];expected=source.metric(gold,1-gold,scores[indices],v['gt']);assert abs(row[f'ap{t}']-expected['ap'])<1e-10,('official AP drift',method,role,wait,t)
            rows.append(row)
        result.append(rows)
    return *result,arrays

def current(role,c):
    m=core.load_npz(verified(c['inputs'][role+'_material']));ledger=core.load_jsonl(verified(c['inputs']['train_conditions' if role=='mechanism_holdout14' else 'val_conditions']));rows=core.conditions_for(m,ledger);p=AUDIT/'k2_k3_formal_resume_v3';saved=core.load_npz(p/(role+'_state.npz'));methods=sorted({key.split('__')[0] for key in saved});boxes={}
    for method in methods:boxes[method]=science.geometry(m,{k:saved[method+'__'+k] for k in ('mean','pivot')})[0]
    gate=core.load_npz(verified(c['inputs'][role+'_gate_material']));boxes['K1_exact']=core.build_heading(m,gate['q005_base_delta'])[0];pred=core.load_npz(p/(role+'_predictions.npz'));pred.update(core.load_npz(p/(role+'_matched_predictions.npz')));boxes['K2_own_matched_U']=boxes['K2_infra']
    return m,rows,pred,boxes,source.common_mask(m['gt_boxes'])

def geometry(role,c):
    from object_history_selector_core_v1_20261021 import PackedDataset
    dataset=PackedDataset(ROOT/c['datasets'][role]);geometries={g:core.load_npz(verified(c['inputs'][role+'_'+g])) for g in c['geometries']};m=dict(geometries['stale']);gts=[]
    for i,index in enumerate(m['dataset_index']):
        pool=dataset.pool(int(index));ga,gb=map(int,m['gt_offsets'][i:i+2]);assert np.array_equal(m['gt_original_indices'][ga:gb],pool['gt_original_indices']);gts.append(pool['gt_boxes'])
    m['gt_boxes']=np.concatenate(gts);rows=[dataset.meta[int(i)] for i in m['dataset_index']]
    for data in geometries.values():
        for k in ('dataset_index','sequence','wait_ms','receiver_timestamp_us','condition_offsets','gt_offsets','gt_original_indices'):assert np.array_equal(m[k],data[k])
    return m,rows,core.load_npz(verified(c['inputs'][role+'_predictions'])),{n:geometries[n.split('__')[0]]['boxes'] for n in c['methods']},source.common_mask(m['gt_boxes'])

def modules(role,c):
    m=core.load_npz(verified(c['inputs'][role+'_common']));meta=core.load_jsonl(verified(c['inputs'][role+'_conditions']));pool=core.load_npz(verified(c['inputs'][role+'_pools']));rows=[meta[int(i)] for i in m['dataset_index']];m['gt_boxes']=np.concatenate([pool['gt_boxes'][int(r['gt_start']):int(r['gt_stop'])] for r in rows]);assert len(m['gt_boxes'])==m['gt_offsets'][-1]
    fa=core.load_npz(verified(c['inputs'][role+'_freealign']));mod=core.load_npz(verified(c['inputs'][role+'_module']));gs={'stale':m['stale_boxes'],'causal_kf':m['kf_boxes'],'freealign':fa['freealign_boxes'],'roco':mod['roco_boxes'],'cbm':mod['cbm_boxes']}
    for data in (fa,mod):
        for k in ('dataset_index','condition_offsets'):assert np.array_equal(m[k],data[k])
    return m,rows,core.load_npz(verified(c['inputs'][role+'_predictions'])),{n:gs[n.split('__')[0]] for n in c['methods']},np.ones(len(m['gt_boxes']),bool)

def ego(role,c):
    from run_peer07_ego_original_source_ap_v1_20261031 import compose
    meta=core.load_jsonl(verified(c['inputs'][role+'_conditions']));pool=core.load_npz(verified(c['inputs'][role+'_pools']));saved=core.load_npz(verified(c['inputs'][role+'_predictions']));m,p,b,_,_=compose(meta,pool,saved,role);rows=[meta[int(i)] for i in m['dataset_index']]
    return m,rows,p,{'ego':b},np.ones(len(m['gt_boxes']),bool)

def official_late(role,c):
    # The two segments have disjoint waits. Preserve original per-frame stable
    # score sorting, no new ROI/label eligibility or output truncation.
    meta=core.load_jsonl(verified(c['inputs'][role+'_conditions']));pool=core.load_npz(verified(c['inputs'][role+'_pools']));items=[]
    for segment in c['segments']:
        p=core.load_npz(verified(c['inputs'][segment+'_predictions']));saved=core.load_npz(verified(c['inputs'][segment+'_evaluation_arrays']))
        for i,index in enumerate(p['dataset_index']):
            actual=str(p['role_names'][p['role_code'][i]])
            if actual==role:items.append((meta[int(index)],p,i,saved))
    items.sort(key=lambda x:(int(x[0]['wait_ms']),str(x[0]['sequence']),int(x[0]['receiver_timestamp_us'])))
    rows=[x[0] for x in items];m={k:np.asarray([r[k] for r in rows]) for k in ('sequence','wait_ms','receiver_timestamp_us')};m['dataset_index']=np.asarray([r['condition_index'] for r in rows]);gtparts=[pool['gt_boxes'][int(r['gt_start']):int(r['gt_stop'])] for r in rows];m['gt_boxes']=np.concatenate(gtparts);m['gt_offsets']=np.r_[0,np.cumsum([len(g) for g in gtparts])];pred={};boxes={};nmax=[]
    # Different arms may contain different numbers of predictions. Pad their
    # *candidate address space* per frame, not the actual frozen output lists.
    for r,p,i,_ in items:nmax.append(max(int(p[a+'_offsets'][i+1]-p[a+'_offsets'][i]) for a in ('no_comp','time_comp')))
    m['condition_offsets']=np.r_[0,np.cumsum(nmax)]
    for arm in ('no_comp','time_comp'):
        name='official_late_'+arm;bb=np.zeros((sum(nmax),7),np.float32);ids=[];ss=[];offset=[0];tps={50:[],70:[]}
        for j,(r,p,i,saved) in enumerate(items):
            a,b=map(int,p[arm+'_offsets'][i:i+2]);order=np.argsort(-p[arm+'_scores'][a:b],kind='stable');chosen=np.arange(m['condition_offsets'][j],m['condition_offsets'][j]+b-a);bb[chosen]=p[arm+'_boxes'][a:b][order];ids.extend(chosen);ss.extend(p[arm+'_scores'][a:b][order]);offset.append(len(ids))
            for t in (50,70):tps[t].append(saved[f'{arm}__tp{t}'][a:b])
        boxes[name]=bb;pred.update({name+'__candidate':np.asarray(ids,np.int64),name+'__offset':np.asarray(offset,np.int64),name+'__score':np.asarray(ss),**{name+f'__tp{t}':np.concatenate(tps[t]) for t in (50,70)}})
    return m,rows,pred,boxes,np.ones(len(m['gt_boxes']),bool)

def main():
    out=CYCLE/'fixed_population_aux';out.mkdir(exist_ok=False);start=time.monotonic();cpu=time.process_time();allpanels=[];allseq=[]
    try:
        spec=load(CYCLE/'contract.json');c=load(verified(spec['shared_main_contract']));lt=load(verified(spec['label_terminal']));replacements={p:{r['vehicle_frame']:{v['cooperative_token']:v['box'] for v in r['replacements']} for r in map(json.loads,verified(lt['outputs'][p]).read_text().splitlines())} for p in PROTOCOLS[1:]}
        configs={'current':c,**{family:load(verified(ref)) for family,ref in spec['baseline_contracts'].items()}}
        for family,loader in [('current',current),('geometry',geometry),('modules',modules),('ego',ego),('official_late',official_late)]:
            for role in ('mechanism_holdout14','val21'):
                sub=out/family/role;sub.mkdir(parents=True);m,rows,pred,boxes,gtmask=loader(role,configs[family]);assert all(str(r['role'])==role for r in rows)
                # Map original provenance before any GT geometry replacement.
                ledger=rows;temporary=dict(m,dataset_index=np.arange(len(rows)));codes,mapping=source.map_provenance(temporary,ledger,role,LABELROOT);assert mapping['unmatched']==mapping['ambiguous']==0
                auxgt,audit=auxiliary(m,rows,codes,replacements);atomic(sub/'population_identity.json',{'mapping':mapping,'label_join':audit,'fixed_original_GT_in_ROI':int(gtmask.sum()),'fixed_predictions':{k:int(len(pred[k+'__candidate'])) for k in boxes},'no_new_ROI_filter':True,'family_contract':spec['shared_main_contract'] if family=='current' else spec['baseline_contracts'][family]})
                for method,b in boxes.items():
                    mp,ms,arrays=evaluate(m,pred,b,codes,role,auxgt,gtmask,method,sub)
                    core.save_npz(sub/(method+'_outcomes.npz'),**arrays)
                    allpanels.extend(dict(r,family=family) for r in mp);allseq.extend(dict(r,family=family) for r in ms);atomic(out/'panels.json',allpanels);atomic(out/'per_sequence_panels.json',allseq);atomic(out/'status.json',{'status':'running','family':family,'role':role,'completed_method':method,'panels':len(allpanels),'wall_s':time.monotonic()-start})
                assert all(sha(LABELROOT/(f+'.json'))==s for f,s in audit['original_label_SHA'].items())
        result={'status':'complete_light_checked','created_at':now(),'contract':ev(CYCLE/'contract.json'),'panels':ev(out/'panels.json'),'per_sequence':ev(out/'per_sequence_panels.json'),'checks':{'all_registered_families_covered':True,'official_TP_replayed':True,'no_prediction_or_population_selection':True,'original_GT_unchanged':True},'cost':{'wall_s':time.monotonic()-start,'cpu_s':time.process_time()-cpu,'peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,'GPU_forwards':0,'updates':0},'not_complete_K5_release':['CoAlign full-system','native lag1/MA','FFNet auxiliary','all historical native controls'],'independent_review_if_top3_pending':True};atomic(out/'terminal.json',result);update('fixed_population_aux',{'status':'complete_light_checked','terminal':ev(out/'terminal.json')})
    except Exception as e:atomic(out/'terminal.json',{'status':'failed_retained','error':repr(e),'traceback':traceback.format_exc(),'partial_panels_retained':len(allpanels)});update('fixed_population_aux',{'status':'failed_retained','terminal':ev(out/'terminal.json')});raise
if __name__=='__main__':main()
