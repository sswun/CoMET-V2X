"""Lowest-priority preregistered four-arm CPU tier ensemble, no estimator change.

Fixed implementation details before R8e cal AP: score-desc deterministic
maximum-IoU one-to-one pairs, .15 BEV NMS, pooled cal TP70 isotonic mapping,
unmatched scalar = clipped least-squares calibration coefficient. Calibration
fitting uses only the matching paired seed's cal8; one four-arm choice averages
all three seeds/four waits. No GT/labels enter deployed combination.
"""
import argparse,csv,time,traceback
import numpy as np
from peer26_v1_20261106 import *
from peer20_population_core_v1_20261105 import data,packed,statistic
import peer20_saved_native_v1_20261105 as native
from e1_receiver_value_core_v1_20261020 import iou32,greedy
import review_traf_v2xseq_adapted_fixed60_raw_20261006 as independent
from run_peer26_R8b_v1_20261106 import context
from review_peer16_E4_E7_AP_v1_20261103 import geometry
OUT=BASE/'R8e';TABLE=ROOT/'docs/research_v2/R8e'
SEEDS=(20261101,20261102,20261103);ARMS=((.3,False),(.3,True),(.5,False),(.5,True))

def tag(arm):return 'iou'+str(arm[0])+'_cal'+str(int(arm[1]))
def npz(p):
    with np.load(p) as z:return {k:z[k] for k in z.files}
def ident(r):return str(r['sequence']),int(r['receiver_timestamp_us']),int(r['wait_ms'])

def prepare():
    register('R8e',__file__,__doc__,dict(CPU_threads=1,GPU_forwards=0,updates=0,replicates=2000,wall_h='estimated12-60 including independent review, before Nov14',
        disk_MiB='<450 compact selected scores/TP/provenance/draws, shared geometry/raw; no whole cache copies'),
        scientific_contract='peer26 R8e exact four arms, cal-only, three seeds, H1 position, max calibrated score, unmatched coefficient, BEV NMS',
        arms=list(ARMS),calibration='per paired seed, pooled cal8 TP70 isotonic (PAVA vs sklearn independent), no val/hold fitting',
        unmatched_coefficient='min(1,max(0,sum(score*calTP70)/sum(score^2))) after selected calibration; no tuning',
        pairing='descending BEV IoU globally one-to-one, deterministic i,j tie; independent polygon construction',
        nms=.15,gate='val meanAP70 >= each baseline+.50 and both paired CI low>0; val AP50>=tier2-.20; hold AP70 higher than both',
        tier2='R7b A2000 seeds20261101/02/03',tier1='author frozen H1 paired heads2023/24/25',
        source_reviews=[ev(P21/'R7b_gate_postcheck/terminal.json'),ev(P20/'R6b_author_independent/terminal.json')],
        no_new_pose_estimator=True,no_paper_changes=True)
    register('R8e_chain',__file__,'Finite chain, immediately after higher-priority registered CPU tasks.',dict(CPU_threads=1,GPU_forwards=0,updates=0,disk_MiB='<2'),output=str(OUT/'terminal.json'))

def fit_iso(score,y,review=False):
    if review:
        from sklearn.isotonic import IsotonicRegression
        model=IsotonicRegression(out_of_bounds='clip').fit(score,y)
        return dict(x=model.X_thresholds_.tolist(),y=model.y_thresholds_.tolist())
    x,inverse,count=np.unique(score,return_inverse=True,return_counts=True)
    sums=np.bincount(inverse,weights=y);blocks=[]
    for i,(w,s) in enumerate(zip(count,sums)):
        blocks.append([i,i,int(w),float(s)])
        while len(blocks)>1 and blocks[-2][3]/blocks[-2][2]>blocks[-1][3]/blocks[-1][2]:
            right=blocks.pop();left=blocks.pop();blocks.append([left[0],right[1],left[2]+right[2],left[3]+right[3]])
    out=np.empty(len(x))
    for a,b,w,s in blocks:out[a:b+1]=s/w
    return dict(x=x.tolist(),y=out.tolist())

def score_map(score,model):return np.interp(score,model['x'],model['y'])
def calibration(score,y):
    a,b=fit_iso(score,y),fit_iso(score,y,True)
    assert np.max(abs(score_map(score,a)-score_map(score,b)))<1e-12
    coeff={}
    for mode in (False,True):
        s=score_map(score,a) if mode else score
        coeff[str(mode)]=float(np.clip(np.dot(s,y)/max(np.dot(s,s),1e-30),0,1))
    return dict(isotonic=a,coeff=coeff,independent_fitting=True,role='calibration8')

def matrix(a,b,review=False):
    if not review:return iou32(a,b)
    # Independent corner/polygon implementation, rounded to accepted float32.
    from shapely.geometry import Polygon
    def poly(x):
        c,s=np.cos(x[6]),np.sin(x[6]);rot=np.array([[c,-s],[s,c]])
        corners=np.array([[1,1],[1,-1],[-1,-1],[-1,1]])*np.array([x[5],x[4]])/2
        return Polygon(corners@rot.T+x[:2])
    pa,pb=[poly(x) for x in a],[poly(x) for x in b]
    return np.asarray([[p.intersection(q).area/p.union(q).area if p.intersects(q) else 0. for q in pb] for p in pa],np.float32).reshape(len(a),len(b))

def combine(a,sa,b,sb,arm,models,review=False):
    threshold,cal=arm;sa=score_map(sa,models[0]['isotonic']) if cal else sa.copy();sb=score_map(sb,models[1]['isotonic']) if cal else sb.copy()
    overlap=matrix(a,b,review);matches=[];useda=set();usedb=set()
    if review:
        candidates=sorted((-float(overlap[i,j]),i,j) for i in range(len(a)) for j in range(len(b)) if overlap[i,j]>=threshold)
    else:
        ii,jj=np.nonzero(overlap>=threshold);order=np.lexsort((jj,ii,-overlap[ii,jj]));candidates=[(-float(overlap[ii[k],jj[k]]),int(ii[k]),int(jj[k])) for k in order]
    for _,i,j in candidates:
        if i in useda or j in usedb:continue
        useda.add(i);usedb.add(j);matches.append((i,j))
    boxes=[];scores=[];origin=[]
    for i,j in matches:boxes.append(a[i]);scores.append(max(sa[i],sb[j]));origin.append((0,i))
    for k,(bb,ss,used) in enumerate(((a,sa,useda),(b,sb,usedb))):
        for i in range(len(bb)):
            if i not in used:boxes.append(bb[i]);scores.append(ss[i]*models[k]['coeff'][str(cal)]);origin.append((k,i))
    boxes=np.asarray(boxes,float).reshape(-1,7);scores=np.asarray(scores,float);origin=np.asarray(origin,np.int64).reshape(-1,2)
    order=np.lexsort((np.arange(len(scores)),-scores));keep=[]
    if review:
        for i in order:
            if not keep or np.max(matrix(boxes[i:i+1],boxes[keep],True))<=.15:keep.append(int(i))
    else:
        while len(order):
            i=int(order[0]);keep.append(i);rest=order[1:]
            order=rest[matrix(boxes[i:i+1],boxes[rest])[0]<=.15] if len(rest) else rest
    return boxes[keep],scores[keep],origin[keep]

def sources(role,seed,review=False):
    import run_peer19_recipe_v3_20261104 as recipe
    m0,rows0=data('author')[0][role];i=SEEDS.index(seed);hs=2023+i;label='H1_seed'+str(hs)
    state=npz(P20/'R6b_author'/(role+'_H1_state.npz'))['delta']
    bb=geometry(m0,state) if review else recipe.geometry(m0,state)[0]
    late=npz(P20/'R6b_author'/(role+'_'+label+'_predictions.npz'))
    root=P21/'R7b/A'/('seed'+str(seed))/role
    m,rows,p,b=native.assemble('author',role,native.shard_cases(root,'COMET_H1'),'native',review)
    by={ident(r):j for j,r in enumerate(rows0)};frames=[];scores=[[],[]];truth=[[],[]]
    for j,r in enumerate(rows):
        k=by[ident(r)];x,y=map(int,late[label+'__offset'][k:k+2]);aa=late[label+'__candidate'][x:y]
        u,v=map(int,p['native__offset'][j:j+2]);ga,gb=map(int,m['gt_offsets'][j:j+2])
        frames.append((bb[aa],late[label+'__score'][x:y],b[u:v],p['native__score'][u:v],m['gt_boxes'][ga:gb],aa,np.arange(u,v)))
        scores[0].append(late[label+'__score'][x:y]);truth[0].append(late[label+'__tp70'][x:y]);scores[1].append(p['native__score'][u:v]);truth[1].append(p['native__tp70'][u:v])
    return m,rows,frames,[np.concatenate(x) for x in scores],[np.concatenate(x) for x in truth]

def evaluate(role,seed,arm,models):
    dest=OUT/(role+'_'+str(seed)+'_'+tag(arm));dest.mkdir(parents=True,exist_ok=False)
    raw=P21/'R7b/A'/('seed'+str(seed))/role;hs=2023+SEEDS.index(seed)
    atomic(dest/'identity.json',dict(raw=ev(raw/'terminal.json'),records=ev(raw/'records.json'),
        tier2_weight=ev(raw.parent/'training/final.pth'),
        tier1_state=ev(P20/'R6b_author'/(role+'_H1_state.npz')),
        tier1_selected_prediction=ev(P20/'R6b_author'/(role+'_H1_seed'+str(hs)+'_predictions.npz')),
        source_GT_review=ev(P21/'R7b_analysis'/('A_'+str(seed)+'_'+role+'_independent')/'terminal.json'),
        cal_mapping=ev(OUT/'score_calibration.json'),frozen_arm=arm,shared_boxes_no_copy=True))
    gold=None;goldstats=None;golddraw=None
    for review in (False,True):
        m,rows,frames,_,_=sources(role,seed,review);scores=[];tps={50:[],70:[]};offset=[0];origins=[];ids=[]
        for a,sa,b,sb,gt,ai,bi in frames:
            boxes,ss,origin=combine(a,sa,b,sb,arm,models,review);scores.extend(ss);offset.append(len(scores));origins.extend(origin[:,0])
            ids.extend(ai[j] if k==0 else bi[j] for k,j in origin)
            if review:
                for th in (50,70):
                    _,tp,_,n=independent.match_frame(boxes,-np.arange(len(boxes),dtype=float),np.zeros(len(boxes)),gt,np.zeros(len(gt)),th/100)
                    assert n==len(gt);tps[th].extend(tp)
            else:
                overlap=iou32(boxes,gt)
                for th in (50,70):tps[th].extend((greedy(overlap,np.arange(len(boxes)),th/100)>=0).astype(np.int8))
        p={'combo__score':np.asarray(scores),'combo__offset':np.asarray(offset),'combo__origin':np.asarray(origins),'combo__candidate':np.asarray(ids),
            **{'combo__tp'+str(th):np.asarray(v,np.int8) for th,v in tps.items()}}
        if review:assert all(np.array_equal(p[k],gold[k]) for k in p),'independent combination or matching mismatch'
        else:gold=p;np.savez_compressed(dest/'predictions.npz',**p)
        stats=[];draws={}
        for th in (50,70):
            rr,dd=statistic(packed(m,p,'combo',th),np.ones(len(rows),bool),review);stats.extend(dict(r,threshold=th) for r in rr);draws['AP'+str(th)]=dd
        if review:
            for x,y in zip(stats,goldstats):assert abs(x['AP']-y['AP'])<1e-8 and np.max(abs(np.asarray(x['CI95'])-y['CI95']))<1e-8
            assert all(np.max(abs(draws[k]-golddraw[k]))<1e-8 for k in draws)
        else:goldstats=stats;golddraw=draws;atomic(dest/'statistics.json',stats);np.savez_compressed(dest/'draws.npz',**draws)
    atomic(dest/'terminal.json',dict(created_at=now(),status='complete_independently_reviewed',statistics=ev(dest/'statistics.json'),draws=ev(dest/'draws.npz'),
        predictions=ev(dest/'predictions.npz'),canonical_GT_identity_checked=True,independent_combination_matching_AP_CI=True))
    return dest

def summaries(role,arm):
    result=[]
    for th in (50,70):
        values={};samples={}
        for method in ('combo','tier1','tier2'):
            pts=[];draws=[]
            for i,seed in enumerate(SEEDS):
                if method=='combo':src=OUT/(role+'_'+str(seed)+'_'+tag(arm));stat=src/'statistics.json';draw=src/'draws.npz';prefix=''
                elif method=='tier1':src=P20/'R6b_author';stat=src/(role+'_H1_seed'+str(2023+i)+'_statistics.json');draw=src/(role+'_H1_seed'+str(2023+i)+'_draws.npz');prefix='all_'
                else:src=P21/'R7b_analysis'/('A_'+str(seed)+'_'+role);stat=src/'result_statistics.json';draw=src/'result_draws.npz';prefix='all_'
                rr=load(stat);pts.append(next(r['AP'] for r in rr if r['threshold']==th and r['wait_ms']=='four_wait_mean' and r.get('population','all')=='all'))
                with np.load(draw) as z:draws.append(z[prefix+'AP'+str(th)].mean(0))
            values[method]=float(np.mean(pts));samples[method]=np.mean(draws,axis=0)
        for base in ('tier1','tier2'):
            result.append(dict(role=role,threshold=th,reference=base,AP=values['combo'],baseline_AP=values[base],delta=values['combo']-values[base],
                paired_CI95=np.quantile(samples['combo']-samples[base],[.025,.975]).tolist(),replicates=2000,seeds=3,waits=4))
    return result

def chain():
    begin=time.monotonic();completed=[]
    try:
        assert now()<DEADLINE;models={};points={}
        for seed in SEEDS:
            _,_,_,scores,labels=sources('calibration8',seed)
            models[seed]=[calibration(s,y) for s,y in zip(scores,labels)]
        atomic(OUT/'score_calibration.json',dict(created_at=now(),models=models,fit_role='calibration8',independent_fitting=True))
        for arm in ARMS:
            values=[]
            for seed in SEEDS:
                dest=evaluate('calibration8',seed,arm,models[seed]);completed.append(dest.name)
                values.append(next(r['AP'] for r in load(dest/'statistics.json') if r['threshold']==70 and r['wait_ms']=='four_wait_mean'))
            points[tag(arm)]=float(np.mean(values))
        chosen=max(ARMS,key=lambda arm:(points[tag(arm)],-ARMS.index(arm)))
        atomic(OUT/'cal_choice.json',dict(created_at=now(),selected=chosen,cal_AP70=points,formal_not_read=True,independent_all_cal=True,roles=['calibration8']))
        rows=[]
        for role in ('val21','mechanism_holdout14'):
            for seed in SEEDS:
                dest=evaluate(role,seed,chosen,models[seed]);completed.append(dest.name)
            rows.extend(summaries(role,chosen))
        checks={}
        for base in ('tier1','tier2'):
            r=next(r for r in rows if r['role']=='val21' and r['threshold']==70 and r['reference']==base)
            checks[base+'_val_gain_ge_point50']=r['delta']>=.50;checks[base+'_val_CI_positive']=r['paired_CI95'][0]>0
            r=next(r for r in rows if r['role']=='mechanism_holdout14' and r['threshold']==70 and r['reference']==base);checks[base+'_hold_same_sign']=r['delta']>0
        checks['val_AP50_ge_tier2_minus_point20']=next(r['delta'] for r in rows if r['role']=='val21' and r['threshold']==50 and r['reference']=='tier2')>=-.20
        atomic(OUT/'adoption_gate.json',dict(created_at=now(),accepted_extension=all(checks.values()),checks=checks,panels=rows,chosen=chosen))
        TABLE.mkdir(parents=True,exist_ok=True)
        with (TABLE/'results.csv').open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        with (TABLE/'README.md').open('x') as f:f.write('# R8e 两档框级组合\n\n'+('预注册门通过。' if all(checks.values()) else '预注册门未通过，保留负结果，不递归调参。')+'仅为系统组合；四候选只用cal8选。GT逐条件核同，生产与独立组合/匹配/AP/2000整序列CI。论文表不改。\n')
        atomic(OUT/'terminal.json',dict(created_at=now(),status='complete_independently_reviewed',gate=ev(OUT/'adoption_gate.json'),table=ev(TABLE/'results.csv'),wall_s=time.monotonic()-begin))
        atomic(BASE/'R8e_chain/terminal.json',dict(created_at=now(),status='complete_independently_reviewed',result=ev(OUT/'terminal.json')))
        for n in ('R8e','R8e_chain'):record(n,dict(status='complete_independently_reviewed',owner=None,resource_owner=False,terminal=ev(BASE/n/'terminal.json')))
    except Exception as exc:
        atomic(BASE/'R8e_chain/terminal.json',dict(created_at=now(),status='failed_retained',error=repr(exc),traceback=traceback.format_exc(),completed=completed))
        record('R8e_chain',dict(status='failed_retained',owner=None,resource_owner=False,terminal=ev(BASE/'R8e_chain/terminal.json')));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','chain'));a=p.parse_args();prepare() if a.mode=='prepare' else chain()
