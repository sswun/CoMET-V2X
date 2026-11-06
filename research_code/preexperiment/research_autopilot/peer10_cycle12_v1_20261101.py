"""12h immutable registration and finite workers; no model/event dispatch."""
from peer10_tournament_v1_20261101 import ROOT,AUTO,AUDIT,atomic,ev,load,now,task
import peer10_cycle09_v1_20261101 as registry

CYCLE=AUDIT/'cycle12_v1'
JOBS={
 'independent_aux':('peer10_cached_hwl_full_independent_review_v1_20261101',task('K5')),
 'ffnet_error_slot':('peer10_ffnet_explicit_cuda_error_slot_v3_20261101','autonomous_research_continuation_20261024'),
 'gpu1_hold_online':('peer10_k4_seed2_public_hold14_native_v1_20261101',task('K4')),
}
registry.JOBS.update(JOBS)
def update(name,values):registry.update(name,values)

def prepare():
    CYCLE.mkdir(exist_ok=False)
    instructions={n:ev(AUTO/n) for n in ('current_handoff.md','main_chat_directive.md')}
    for n in instructions:(AUTO/n).read_text()
    atomic(CYCLE/'storage_entry.json',load(ROOT/'artifacts/maintenance/storage_status/latest.json'))
    c={'created_at':now(),'latest_instructions':instructions,
       'official_protocol_split_ROI_unchanged':True,'model_wakeup':'3h calendar only; no model/event calls',
       'shared_contract':ev(AUDIT/'contract_v1.json'),
       'corrected_auxiliary_contract':ev(AUDIT/'cycle09_v1/aux_hwl_contract_v3.json'),
       'corrected_auxiliary_terminal':ev(AUDIT/'cycle09_v1/fixed_population_hwl_v3/terminal.json'),
       'previous_ffnet_diagnostic':ev(AUDIT/'cycle09_v1/ffnet_resume/diagnostic/manifest_v1.json'),
       'previous_ffnet_failed_terminal':ev(AUDIT/'cycle09_v1/ffnet_resume/terminal.json'),
       'original_ffnet_contract':ev(AUDIT/'cycle09_v1/gpu_contract_recovery_v2.json'),
       'ffnet_prefix':ev(AUDIT/'cycle03_v1/ffnet_formal/producer/manifest_v1.json'),
       'seed2_public_weight':ev(AUDIT/'seed2_public/training/final.pth'),
       'jobs':{
        'independent_aux':{'prediction':'An independently coded primary-label join, HWL check, polygon intersection, target-first ignore and VOC AP should reproduce all 3600 corrected panels. Any discrepancy is retained, not selected away.',
          'estimated_cost':{'CPU_threads':1,'wall_hours':'3-8 under shared thermal quota','GPU_forwards':0,'updates':0,'new_disk_MiB':30},
          'boundary':'Reuse frozen layout/box reconstruction only; do not call producer auxiliary/provenance/IoU/matching/AP. Compare saved outcomes, save only metrics and mismatch summaries; raw labels and outcomes remain shared.'},
        'ffnet_error_slot':{'prediction':'Zero-grid IoU may leave cudaErrorInvalidConfiguration in the host error slot until a subsequent torch allocation; blocking alone may not localize it.',
          'estimated_cost':{'operator_calls':4,'safe_probe_conditions':8,'formal_remaining_conditions':42020,'CPU_threads':1,'wall_hours':'bounded probe then author full remaining, multi-hour'},
          'gate':'Fresh process, loaded author compiled IoU: nonempty error0, empty-side explicit cudaGetLastError=9, mathematically guarded empty calls error0, nonempty equal. Then safe probe must reproduce the 5 saved nonempty outputs/TP and complete all 8; only then resume existing 1024 prefix. No author model/input/weight edits.',
          'on_failure':'retain failure and run already registered GPU K4 successor; no model wakeup'},
        'gpu1_hold_online':{'prediction':'Independent seed2 PUBLIC hold14 three-input evaluation measures training randomness without choosing on val/hold results.',
          'estimated_cost':{'cases':9744,'GPU_forwards':29244,'updates':0,'new_disk_GiB':'<2'},
          'weight':ev(AUDIT/'seed2_public/training/final.pth'),'inputs':'exact frozen native hold14 adapter, same three inputs and four waits; no new readout'},
       },
       'gpu1_predecessor':'embodied-peer10-cycle09-gpu1-successors-v1-20261101.service',
       'GPU0_and_4090':'Existing independent chains retained; no duplicate model process',
       'storage':'Shared pointcloud/GT/labels/weights/prefix are referenced, not copied. Formal predictions retained; no cleanup of active consumers. Soft budget nonblocking.',
       'scripts':{n:ev(ROOT/'preexperiment/research_autopilot'/n) for n in ('peer10_cycle12_v1_20261101.py','review_peer10_cached_hwl_independent_v1_20261101.py','run_peer10_ffnet_error_slot_v3_20261101.py','run_peer10_cycle12_gpu1_chain_v1_20261101.py')}}
    atomic(CYCLE/'contract.json',c)
    for name,j in c['jobs'].items():update(name,{'status':'registered_ready','contract':ev(CYCLE/'contract.json'),'prediction_before_result':j['prediction'],'estimated_cost':j['estimated_cost']})
if __name__=='__main__':prepare()
