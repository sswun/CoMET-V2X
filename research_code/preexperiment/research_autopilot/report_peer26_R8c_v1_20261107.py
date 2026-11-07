"""Complete descriptive capture/T3/retention reporting from fixed R8 outputs.

No AP-based selection or new detector forward. Stored AP/CI reused. Capture
means at least one accepted gate anchor; report both entire-stream failure and
no anchor in the first five messages, with eligible-message counts explicit.
T3 incremental error is relative to clean H1, not physical truth.
"""
import argparse,csv,traceback
import numpy as np
from peer26_v1_20261106 import *
import recover_peer26_material_v2_20261106 as material
import run_peer26_R8c_late_v1_20261106 as late
from run_peer26_R8b_v1_20261106 import context,DOMAINS
from peer20_population_core_v1_20261105 import data
OUT=BASE/'R8c_reporting';TABLE=ROOT/'docs/research_v2/R8c/reporting'

def prepare():
    register('R8c_reporting',__file__,__doc__,dict(CPU_threads=1,GPU_forwards=0,updates=0,disk_MiB='<20 metrics, no shared-prediction copy',wall_h='estimated2-8 after late outputs; before Nov14'),
        dependencies=[ev(BASE/('R8c_late_'+d)/'contract.json') for d in DOMAINS],
        capture_definition='all-stream never gate anchor + first-five-message no gate anchor; eligible count >=3ego and3remote; no method selection',
        methods='PUBLIC gate-reference, single, H1, M share uncorrected fixed gate; A uses fixed-acquisition gate; absent M not evaluated',
        recovery='first post-step norm(state_injected+shift-clean_H1)<=.5; right censored; all methods incl M; independent crossing rebuild',
        AP_and_retention='reuse already independently accepted panels; no new AP/bootstrap')

def crossing(state,shift,clean,ids,review=False):
    ids=[int(i) for i in ids if np.linalg.norm(shift[i])>0]
    if review:
        hit=next((j+1 for j,i in enumerate(ids) if sum(float(v)**2 for v in state[i]+shift[i]-clean[i])<=.25),None)
    else:
        a=np.asarray(ids,int);hits=np.flatnonzero(np.linalg.norm(state[a]+shift[a]-clean[a],axis=1)<=.5);hit=int(hits[0])+1 if len(hits) else None
    return hit,len(ids)

def run():
    try:
        capture=[];recovery=[];sources=[]
        for domain in DOMAINS:
            assert load(BASE/('R8c_late_'+domain)/'terminal.json')['status']=='complete_independently_reviewed'
            dataset,_,_=data(domain);q,_,_=context(domain)
            for role,kind,amp in late.scenarios(domain):
                name=domain+'_'+role+'_'+kind+'_'+str(amp);motion=late.OUT/'motion'/name
                ref=load(motion/'base_case.json');case=Path(ref['directory']);assert ev(case/'terminal.json')==ref['terminal']
                m0,rows=dataset[role];m,shift=material.inject(m0,kind,amp);model=material.adapter()
                n,z=model.observations(m);nr,zr=model.observations(m,True);assert np.array_equal(n,nr) and np.array_equal(z,zr)
                state,info=model.acquisition_state(m,q);other,review=model.acquisition_state(m,q,review=True)
                assert np.array_equal(state,other) and info==review
                with np.load(case/'states.npz') as zz:states={k:zz[k] for k in zz.files}
                assert np.array_equal(states['CoMET_A'],state)
                if (motion/'CoMET_M_state.npz').exists():
                    with np.load(motion/'CoMET_M_state.npz') as zz:states['CoMET_M']=zz['delta']
                with np.load(P20/('R6b_'+domain)/(role+'_H1_state.npz')) as zz:clean=zz['delta']
                groups={}
                for i,(seq,w) in enumerate(zip(m['sequence'],m['wait_ms'])):groups.setdefault((str(seq),int(w)),[]).append(i)
                acq=np.asarray(info['anchors'])
                for method in states:
                    counts=acq if method=='CoMET_A' else n
                    for (seq,w),ids0 in groups.items():
                        ids=np.asarray(ids0);eligible=[]
                        for i in ids:
                            a,b=map(int,m['condition_offsets'][i:i+2]);src=m['source'][a:b];eligible.append(int(np.sum(src==0)>=3 and np.sum(src==1)>=3))
                        first=ids[:5];noanchor=not bool(np.any(counts[ids]>0));cold=not bool(np.any(counts[first]>0))
                        # Separate simple Python reduction independently checks gate counts.
                        assert noanchor==all(int(counts[i])==0 for i in ids0)
                        capture.append(dict(domain=domain,role=role,kind=kind,amplitude=amp,method=method,sequence=seq,wait_ms=w,
                            never_gate_anchor=noanchor,first5_no_gate_anchor=cold,messages=len(ids),eligible_messages=sum(eligible),
                            PUBLIC_is_gate_reference_not_estimator=method=='PUBLIC'))
                        if kind=='T3':
                            hit,num=crossing(states[method],shift,clean,ids);assert (hit,num)==crossing(states[method],shift,clean,ids,True)
                            recovery.append(dict(domain=domain,role=role,method=method,sequence=seq,wait_ms=w,
                                messages_to_first_half_meter=hit,post_step_messages=num,right_censored=hit is None,reference='incremental error vs clean H1, not physical GT'))
                sources.append(dict(case=name,base=ref,states=ev(case/'states.npz'),motion_state=ev(motion/'CoMET_M_state.npz') if 'CoMET_M' in states else None))
                atomic(OUT/'progress.json',dict(created_at=now(),completed=len(sources),last=name))
        TABLE.mkdir(parents=True,exist_ok=True)
        for name,rows in [('capture_streams.csv',capture),('T3_recovery_all_methods.csv',recovery)]:
            with (TABLE/name).open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        groups={}
        for r in capture:
            key=tuple(r[k] for k in ('domain','role','kind','amplitude','method'));groups.setdefault(key,[]).append(r)
        aggregate=[dict(zip(('domain','role','kind','amplitude','method'),key),streams=len(rr),
            never_gate_anchor_fraction=sum(r['never_gate_anchor'] for r in rr)/len(rr),first5_no_gate_anchor_fraction=sum(r['first5_no_gate_anchor'] for r in rr)/len(rr),
            streams_with_eligible_messages=sum(r['eligible_messages']>0 for r in rr)) for key,rr in groups.items()]
        atomic(TABLE/'capture_summary.json',aggregate);atomic(TABLE/'sources.json',sources)
        with (TABLE/'README.md').open('x') as f:f.write('# R8c 描述性补齐\n\n捕获失败同时报全流无门内锚点与前5消息无锚点比例；PUBLIC列仅为其输入几何的门参照，不声称它估计位姿。T3含CoMET-M，误差是相对干净H1的增量扰动误差，非物理真值；首次回到0.5m，未恢复右删失。生产与独立状态、匹配及首达索引核验。AP/CI沿用各域已独立接受结果，不选方法、不改论文表。\n')
        atomic(OUT/'terminal.json',dict(created_at=now(),status='complete_independently_reviewed',sources=ev(TABLE/'sources.json'),capture=ev(TABLE/'capture_streams.csv'),T3=ev(TABLE/'T3_recovery_all_methods.csv'),descriptive_only=True))
        record('R8c_reporting',dict(status='complete_independently_reviewed',owner=None,resource_owner=False,terminal=ev(OUT/'terminal.json')))
    except Exception as exc:
        atomic(OUT/'terminal.json',dict(created_at=now(),status='failed_retained',error=repr(exc),traceback=traceback.format_exc()))
        record('R8c_reporting',dict(status='failed_retained',owner=None,resource_owner=False,terminal=ev(OUT/'terminal.json')));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','run'));a=p.parse_args();prepare() if a.mode=='prepare' else run()
