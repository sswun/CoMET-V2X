"""Recover only unscored fixed-population families; reuse 2040 old panels."""
import time,traceback,resource
import numpy as np
import run_peer10_fixed_population_aux_v1_20261101 as aux
from peer10_tournament_v1_20261101 import ROOT,atomic,ev,load,now
from peer10_cycle09_v1_20261101 import CYCLE,update
core=aux.core

def modules(role,c):
    m=core.load_npz(aux.verified(c['inputs'][role+'_common']))
    meta=core.load_jsonl(aux.verified(c['inputs'][role+'_conditions']))
    pool=core.load_npz(aux.verified(c['inputs'][role+'_pools']))
    rows=[meta[int(i)] for i in m['dataset_index']]
    parts=[pool['gt_boxes'][int(r['gt_start']):int(r['gt_stop'])] for r in rows]
    offsets=np.r_[0,np.cumsum([len(x) for x in parts])].astype(np.int64)
    if 'gt_offsets' in m:assert np.array_equal(m['gt_offsets'],offsets)
    m['gt_offsets']=offsets;m['gt_boxes']=np.concatenate(parts)
    assert len(rows)==len(m['wait_ms']) and len(m['gt_boxes'])==offsets[-1]
    for k in ('sequence','wait_ms','receiver_timestamp_us'):
        if k in m:assert np.array_equal(m[k],np.asarray([r[k] for r in rows])),k
    fa=core.load_npz(aux.verified(c['inputs'][role+'_freealign']))
    mod=core.load_npz(aux.verified(c['inputs'][role+'_module']))
    gs={'stale':m['stale_boxes'],'causal_kf':m['kf_boxes'],'freealign':fa['freealign_boxes'],'roco':mod['roco_boxes'],'cbm':mod['cbm_boxes']}
    for data in (fa,mod):
        for k in ('dataset_index','condition_offsets'):assert np.array_equal(m[k],data[k])
    return m,rows,core.load_npz(aux.verified(c['inputs'][role+'_predictions'])),{n:gs[n.split('__')[0]] for n in c['methods']},np.ones(len(m['gt_boxes']),bool)

def probe(family,role,m,pred,boxes,mask):
    for i in (0,len(m['wait_ms'])-1):
        ga,gb=map(int,m['gt_offsets'][i:i+2]);gt=m['gt_boxes'][ga:gb][mask[ga:gb]]
        for method,b in boxes.items():
            a,z=map(int,pred[method+'__offset'][i:i+2]);ids=pred[method+'__candidate'][a:z];pp=b[ids]
            for t in (50,70):
                left=aux.outcome(aux.source.iou32(pp,gt),np.zeros(len(gt),np.int8),'all',t/100)[0]
                repeat=aux.outcome(aux.source.iou32(pp,gt),np.zeros(len(gt),np.int8),'all',t/100)[0]
                assert np.array_equal(left,repeat)
                assert np.array_equal(left,pred[method+f'__tp{t}'][a:z]),('official TP',family,role,i,method,t)

def main():
    out=CYCLE/'aux_remaining';out.mkdir(exist_ok=False);start=time.monotonic();cpu=time.process_time();panels=[];seq=[];checks=[]
    try:
        spec=load(CYCLE/'contract.json');old=load(aux.verified(spec['original_aux_contract']))
        assert ev(ROOT/spec['completed_aux_panels']['path'])==spec['completed_aux_panels']
        previous=load(ROOT/spec['completed_aux_panels']['path']);assert len(previous)==2040
        lt=load(aux.verified(old['label_terminal']))
        replacements={p:{r['vehicle_frame']:{v['cooperative_token']:v['box'] for v in r['replacements']} for r in map(__import__('json').loads,aux.verified(lt['outputs'][p]).read_text().splitlines())} for p in aux.PROTOCOLS[1:]}
        # First probe every missing family before any new long AP evaluation.
        for family,loader in [('modules',modules),('ego',aux.ego),('official_late',aux.official_late)]:
            c=load(aux.verified(old['baseline_contracts'][family]))
            for role in ('mechanism_holdout14','val21'):
                m,rows,pred,boxes,mask=loader(role,c);probe(family,role,m,pred,boxes,mask)
                checks.append({'family':family,'role':role,'conditions':len(rows),'first_last_official_TP_same_env':True})
                atomic(out/'loader_probe.json',checks)
                del m,rows,pred,boxes,mask
        update('aux_remaining',{'status':'running_probes_passed','contract':ev(CYCLE/'contract.json'),'probe':ev(out/'loader_probe.json')})
        for family,loader in [('modules',modules),('ego',aux.ego),('official_late',aux.official_late)]:
            c=load(aux.verified(old['baseline_contracts'][family]))
            for role in ('mechanism_holdout14','val21'):
                sub=out/family/role;sub.mkdir(parents=True);m,rows,pred,boxes,mask=loader(role,c)
                assert all(str(r['role'])==role for r in rows)
                codes,mapping=aux.source.map_provenance(dict(m,dataset_index=np.arange(len(rows))),rows,role,aux.LABELROOT)
                assert mapping['unmatched']==mapping['ambiguous']==0
                gt,audit=aux.auxiliary(m,rows,codes,replacements)
                atomic(sub/'population_identity.json',{'mapping':mapping,'label_join':audit,'fixed_GT':int(mask.sum()),'family_contract':old['baseline_contracts'][family],'no_new_ROI_filter':True})
                for method,b in boxes.items():
                    pp,ss,arrays=aux.evaluate(m,pred,b,codes,role,gt,mask,method,sub)
                    core.save_npz(sub/(method+'_outcomes.npz'),**arrays)
                    panels.extend(dict(r,family=family) for r in pp);seq.extend(dict(r,family=family) for r in ss)
                    atomic(out/'panels.json',panels);atomic(out/'per_sequence_panels.json',seq)
                    atomic(out/'status.json',{'status':'scoring','family':family,'role':role,'method':method,'new_panels':len(panels),'referenced_panels':2040,'wall_s':time.monotonic()-start})
                assert all(aux.sha(aux.LABELROOT/(f+'.json'))==s for f,s in audit['original_label_SHA'].items())
        terminal={'status':'complete_light_checked','time':now(),'contract':ev(CYCLE/'contract.json'),'referenced_complete_prefix':spec['completed_aux_panels'],'new_panels':ev(out/'panels.json'),'per_sequence':ev(out/'per_sequence_panels.json'),'loader_probe':ev(out/'loader_probe.json'),'checks':{'official_TP_AP_replayed':True,'original_labels_unchanged':True,'remaining_three_families_covered':True,'no_auxiliary_selection':True},'cost':{'wall_s':time.monotonic()-start,'cpu_s':time.process_time()-cpu,'peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,'GPU_forwards':0,'updates':0},'K5_other_release_gaps_retained':['FFNet','CoAlign full-system','lag1/MA','historical native controls']}
        atomic(out/'terminal.json',terminal);update('aux_remaining',{'status':terminal['status'],'terminal':ev(out/'terminal.json')})
    except Exception as e:
        atomic(out/'terminal.json',{'status':'failed_retained','error':repr(e),'traceback':traceback.format_exc(),'new_panels_retained':len(panels),'probe_checks':checks,'cost':{'wall_s':time.monotonic()-start,'cpu_s':time.process_time()-cpu}})
        update('aux_remaining',{'status':'failed_retained','terminal':ev(out/'terminal.json')});raise

if __name__=='__main__':main()
