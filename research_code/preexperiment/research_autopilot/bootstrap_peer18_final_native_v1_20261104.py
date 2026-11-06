"""P0 actual final recipe/native paired CI; exact detector and current-GT joins.

strict60 is TraF strict-trained SHA d681..., NOT ViT2023. Reuse accepted ViT
per-frame compact outcomes, rematch only missing strict60/Where compacts.
System comparison across fusion/readout paths is not module attribution.
"""
import argparse,json,time,traceback
import numpy as np
from peer18_v1_20261104 import *
import run_peer17_final_recipe_v1_20261104 as final
import bootstrap_peer15_paper_v1_20261103 as boot
from run_peer16_cached_methods_v1_20261103 import materials,cross_specs
from run_peer16_E7_native_readout_v1_20261103 import native_root
from review_peer15_external_raw_AP_v1_20261103 import raw_path
import review_traf_v2xseq_adapted_fixed60_raw_20261006 as ind
from object_history_selector_core_v1_20261021 import common_mask
OUT=BASE/'final_native_contrasts'
DOMAINS=('strict60','v2xvit_seed2024','v2xvit_seed2025','where2comm')
STRICT=AUTO/'status_audit_20261023_acceleration/traf_strict_training_fairness_v1/evaluation_v1'
def prepare():
    sc,man=cross_specs()
    for domain in DOMAINS:
        weight=(load(AUTO/'status_audit_20261026_2155/traf_strict60_shared_offset_successor_v1/contract_recovery_v3.json')['current_method_dispatch']
                if domain=='strict60' else man['domains'][domain]['detector_checkpoint'])
        register('final_native_'+domain,__file__,__doc__,dict(GPU_forwards=0,updates=0,CPU_threads=1,replicates=2000,disk_MiB='<30 compact outcomes/draws per domain'),
            domain=domain,detector=weight,final_identity=ev(final.OUT/domain/'identity.json'),
            scope='val21 only, official main commonROI,3headseeds. Real raw currentGT equality/frame/weight joins before pairedCI; native != protected native_prefix_raw.',
            statistical_independence=False)
def canon(a):
    a=np.asarray(a,np.float64).reshape(-1,7)
    return a[np.lexsort(tuple(a[:,j] for j in reversed(range(7))))] if len(a) else a
def run(domain):
    name='final_native_'+domain;dest=OUT/domain;dest.mkdir(parents=True,exist_ok=False)
    c=load(BASE/name/'contract.json');assert c['script']==ev(__file__);begin=time.monotonic()
    record(name,dict(status='running_actual_weight_GT_ROI_join',resource_owner=True))
    try:
        m,mr=materials(domain)[0]['val21'];by={(str(r['sequence']),int(r['wait_ms']),int(r['global_index'])):i for i,r in enumerate(mr)}
        assert len(by)==13264
        data={th:dict(seq=[],wait=[],gt=[],scores=[],tp=[],off=[0]) for th in (50,70)};sources=[]
        for w in (0,100,200,400):
            method='where' if domain=='where2comm' else 'vit'+domain[-4:]
            root=STRICT/f'strict{w:03d}/run' if domain=='strict60' else native_root(method)/f'delay_{w:03d}ms'
            if not (root/'frame_manifest.jsonl').exists():root=native_root(method)/f'strict{w:03d}'
            manifest=root/'frame_manifest.jsonl';rows=[json.loads(l) for l in manifest.read_text().splitlines() if l.strip()];assert len(rows)==3316
            sources.append(ev(manifest));status=load(root/'status.json')
            assert status['checkpoint_sha256']==c['detector']['sha256' if domain!='strict60' else 'checkpoint_sha256'],('detector_checkpoint',domain)
            accepted=None
            if domain.startswith('v2xvit'):
                accepted=np.load(P16/'E5_native_seed_AP50'/f'seed{domain[-4:]}_wait{w}_matches.npz')
                assert np.array_equal(accepted['global_index'],[r['global_index'] for r in rows])
            for ordinal,r in enumerate(rows):
                key=(str(r['sequence']),w,int(r['global_index']));i=by[key]
                assert str(r.get('effective_vehicle_frame',r.get('effective_label_frame')))==str(mr[i]['current_vehicle_frame']),('current_frame',domain,key)
                p=raw_path(r,root);assert ind.sha256(p)==r['raw_sha256']
                with np.load(p) as z:
                    gt=z['gt_boxes'];gl=z['gt_labels'];gg=gt[common_mask(gt)&(gl==0)]
                    a,b=map(int,m['gt_offsets'][i:i+2]);expected=m['gt_boxes'][a:b];expected=expected[common_mask(expected)]
                    assert gg.shape==expected.shape and np.allclose(canon(gg),canon(expected),rtol=0,atol=2e-4),('GT_geometry',domain,key)
                    for th in (50,70):
                        if accepted is not None:
                            k='iou'+str(th);lo,hi=map(int,accepted[k+'_offset'][ordinal:ordinal+2]);ss=accepted[k+'_scores'][lo:hi];tp=accepted[k+'_tp'][lo:hi]
                            assert int(accepted[k+'_frame_gt'][ordinal])==len(gg)
                        else:
                            pb=z['pred_boxes'];ps=z['pred_scores'];pl=z['pred_labels'];keep=common_mask(pb)
                            ss,tp,_,ng=ind.match_frame(pb[keep],ps[keep],pl[keep],gg,np.zeros(len(gg)),th/100);assert ng==len(gg)
                        d=data[th];d['seq'].append(key[0]);d['wait'].append(w);d['gt'].append(len(gg));d['scores'].extend(ss);d['tp'].extend(tp);d['off'].append(len(d['scores']))
                if ordinal%256==0:atomic(dest/'status.json',dict(created_at=now(),stage='actual_raw_weight_GT_check',domain=domain,wait=w,frames=ordinal+1,wall_s=time.monotonic()-begin))
            if accepted is not None:accepted.close()
        atomic(dest/'identity_join.json',dict(checkpoint=c['detector'],manifests=sources,current_frame_equal=True,common_GT_geometry_equal=True,geometry_atol_m=2e-4,role='val21',recipe=c['final_identity'],shared_statistical_kernel=True))
        panels=[]
        for th,d in data.items():
            b=boot.make(d['seq'],d['wait'],d['gt'],d['off'],d['scores'],d['tp'])
            np.savez_compressed(dest/f'native_AP{th}_compact.npz',**b)
            for seed in final.SEEDS:
                with np.load(final.OUT/domain/f'val21_seed{seed}_predictions.npz') as pred:a=boot.from_pred(m,pred,'CoMET_seed'+str(seed),th)
                panels.extend(boot.contrast(dest,domain+'_final_seed'+str(seed)+'_minus_actual_native',a,b,th,dict(domain=domain,role='val21',head_seed=seed,identity=ev(dest/'identity_join.json'),system_comparison_not_module_attribution=True)))
                atomic(dest/'panels.json',panels)
        atomic(dest/'terminal.json',dict(created_at=now(),status='complete_light_checked',panels=ev(dest/'panels.json'),wall_s=time.monotonic()-begin,independent_statistics=False,raw_rematch=domain in ('strict60','where2comm'),independent_numeric_consumer_pending=True))
        record(name,dict(status='complete_light_checked',terminal=ev(dest/'terminal.json'),owner=None,resource_owner=False))
    except Exception as e:
        atomic(dest/'terminal.json',dict(created_at=now(),status='failed_retained',error=repr(e),traceback=traceback.format_exc(),wall_s=time.monotonic()-begin));record(name,dict(status='failed_retained',terminal=ev(dest/'terminal.json'),owner=None,resource_owner=False));raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare',*DOMAINS]);a=p.parse_args();prepare() if a.action=='prepare' else run(a.action)
