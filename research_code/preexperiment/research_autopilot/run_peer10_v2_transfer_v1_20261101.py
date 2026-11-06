"""Current K2/K3 on their real V2 inputs, moving peer pivot and own readout."""
import os,time,traceback,resource
from pathlib import Path
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import numpy as np
from peer10_tournament_v1_20261101 import ROOT,AUDIT,atomic,ev,load,now,sha
from peer10_cycle03_registration_v1_20261101 import CYCLE,update
import run_peer10_se2_expanded_anchor_v1_20261101 as science
import run_v2v4real_current_q005_heading_anchor_gate_factorial_v1_20261030 as fac

def material(c,role,dataset):
    paths=c['materials'][role];z={name:science.core.load_npz(ROOT/path) for name,path in paths.items()};cv=z['cv'];common=('raw_scores','source','protected','anchor','condition_offsets','dataset_index','sequence','wait_ms','receiver_timestamp_us')
    assert cv['sequence'].dtype.str.endswith('U96')
    for other in z.values():
        assert all(np.array_equal(cv[k],other[k]) for k in common)
    rows=[];gt=[];offsets=[0]
    for i,index in enumerate(cv['dataset_index']):
        row=dict(dataset.meta[int(index)]);assert (str(row['sequence']),int(row['wait_ms']),int(row['receiver_timestamp_us']))==(str(cv['sequence'][i]),int(cv['wait_ms'][i]),int(cv['receiver_timestamp_us'][i]))
        if row.get('source'):
            s=dict(row['source']);s['source_to_current_ego']=np.linalg.solve(np.asarray(row['current_ego_pose_lidar_to_world']),np.asarray(s['source_pose_lidar_to_world'])).tolist();row['source']=s
            assert int(s['source_timestamp_us'])<=int(row['receiver_timestamp_us'])
        rows.append(row);g=dataset.pool(int(index))['gt_boxes'];gt.append(g);offsets.append(offsets[-1]+len(g))
    m={k:cv[k] for k in common};m.update(cv_boxes=cv['boxes'],cv_features=cv['features'],stale_boxes=z['stale']['boxes'],stale_features=z['stale']['features'],gt_boxes=np.concatenate(gt),gt_offsets=np.asarray(offsets))
    expected=c['expected_roles'][role];assert len(rows)==expected['conditions'] and len(m['source'])==expected['candidates'] and len(m['gt_boxes'])==expected['gt']
    return m,rows

def main():
    out=CYCLE/'v2_transfer';out.mkdir(exist_ok=False);start=time.monotonic();cpu=time.process_time()
    try:
        if 'embodied-research.slice' not in Path('/proc/self/cgroup').read_text():raise RuntimeError('uncontrolled worker')
        contract=load(CYCLE/'contract.json');spec=contract['v2_transfer']
        for f in spec['bound_inputs']:assert sha(ROOT/f['path'])==f['sha256'],f['path']
        c=load(ROOT/spec['base_contract']['path']);old=ROOT/spec['base_producer'];head=old/'q005_cv_matched39d_head'
        # PackedDataset is authoritative for GT and full metadata; do not regenerate raw detectors/caches.
        train=fac.PackedDataset(ROOT/c['datasets']['train']);val=fac.PackedDataset(ROOT/c['datasets']['val']);panels=[];fallback={};identities={}
        for role in ('fit24','calibration8','mechanism_holdout14','val21'):
            ds=val if role=='val21' else train;m,rows=material(c,role,ds);saved=science.core.load_npz(old/f'{role}_q005_state.npz');base=saved['q005_delta']
            assert np.array_equal(saved['state_key_sequence'],m['sequence']) and np.array_equal(saved['wait_ms'],m['wait_ms']) and np.array_equal(saved['dataset_index'],m['dataset_index']) and np.array_equal(saved['receiver_timestamp_us'],m['receiver_timestamp_us'])
            identity={'role':role,'conditions':len(rows),'full_U96_sequences':len(np.unique(m['sequence'])),'GT':len(m['gt_boxes']),'detector_checkpoint':c['inputs']['input_04_net_epoch60'],'own_readout':ev(head/'direct_u.pth'),'state':ev(old/f'{role}_q005_state.npz'),'public_moving_peer_pivot':True,'source_not_future':True,'SPD_cache_not_used':True,'GT_in_state_or_inference':False}
            atomic(out/f'{role}_actual_input_identity_before_evaluation.json',identity);identities[role]=identity
            states={'K2_infra':science.state(m,rows,spec['prior_degrees']),'K3_K1_expanded':science.state(m,rows,spec['prior_degrees'],expanded=True,score_floor=spec['K3_score_floor'],translation_only=True),'K3_K2_expanded':science.state(m,rows,spec['prior_degrees'],expanded=True,score_floor=spec['K3_score_floor'])}
            s=states['K3_K1_expanded'];s['mean'][s['protected_pair_count']>0,:2]=base[s['protected_pair_count']>0]
            if role=='fit24':
                replay=science.state(m,rows,spec['prior_degrees']);assert np.array_equal(replay['mean'],states['K2_infra']['mean'])
                atomic(out/'light_replay.json',{'status':'passed','same_environment_state_exact':True,'prior_transferred_not_selected_on_V2':True})
            boxes={};scores={}
            for name,state in states.items():boxes[name],features,_=science.geometry(m,state);scores[name]=science.core.refit.score_with_head(m,features,head)
            boxes['K1_exact'],features=fac.heading_geometry(m,base);scores['K1_exact']=science.core.refit.score_with_head(m,features,head)
            # Original V2 protocol: no post-geometry ROI or candidate re-selection.
            fac.METHODS=tuple(boxes);p,pred=fac.evaluate_v2(ds,m,boxes,scores);panels.extend(dict(r,role=role) for r in p)
            if role in fac.FORMAL:
                previous=science.core.load_npz(old/f'{role}_factorial_predictions.npz')
                checks={k:np.array_equal(pred['K1_exact__'+k],previous['heading_q005_base__'+k]) for k in ('candidate','score','tp50','tp70','offset')}
                assert all(checks.values()),checks
                atomic(out/f'{role}_K1_exact_replay.json',checks)
                science.core.save_npz(out/f'{role}_predictions.npz',**pred)
                science.core.save_npz(out/f'{role}_state.npz',**{name+'__'+k:v for name,s in states.items() for k,v in s.items()})
            if role=='calibration8':
                k1=float(np.mean([r['ap70'] for r in p if r['method']=='K1_exact']))
                fallback={name:name if float(np.mean([r['ap70'] for r in p if r['method']==name]))>=k1 else 'K1_exact' for name in states}
                atomic(out/'fallback_frozen_before_formal.json',{'calibration_mean_K1_AP70':k1,'global_choice_by_candidate':fallback,'no_formal_cell_selection':True})
            atomic(out/'panels.json',panels);atomic(out/'status.json',{'status':'running','completed_role':role,'wall_s':time.monotonic()-start})
            del m,rows,states,boxes,scores
        fallbackpanels=[dict(r,raw_candidate=name,method=name+'_cal_fallback',selected_method=selected) for name,selected in fallback.items() for r in panels if r['role'] in fac.FORMAL and r['method']==selected]
        atomic(out/'fallback_panels.json',fallbackpanels)
        result={'status':'complete_light_checked','created_at':now(),'panels':ev(out/'panels.json'),'fallback_panels':ev(out/'fallback_panels.json'),'actual_inputs':identities,'cost':{'wall_s':time.monotonic()-start,'cpu_s':time.process_time()-cpu,'peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,'detector_forwards':0,'new_updates':0},'review':'same-environment fit replay and exact previous K1 formal replay; full independent top3 review pending','remaining':['V2 ego-visible/remote-exclusive source breakdown','K4 actual native transfer; not replaced by this cached K2/K3 task','historical stronger stale-system reference'],'official_SPD_protocol_unchanged':True}
        atomic(out/'terminal.json',result);update('v2_transfer',{'status':'complete_light_checked','terminal':ev(out/'terminal.json'),'next_action':result['remaining']})
    except Exception as exc:
        atomic(out/'terminal.json',{'status':'failed_retained','error':repr(exc),'traceback':traceback.format_exc(),'wall_s':time.monotonic()-start});update('v2_transfer',{'status':'failed_retained','terminal':ev(out/'terminal.json'),'next_action':'inspect actual failure, preserve old output'});raise
if __name__=='__main__':main()
