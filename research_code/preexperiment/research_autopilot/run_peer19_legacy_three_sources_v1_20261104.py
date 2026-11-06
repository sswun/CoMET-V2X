"""R5: recompute three legacy numbers from exact67-sequence metadata/choices.

Do not overwrite the original number ledger or audits. Explicit exact values,
rounding, population and adopt/delete decisions are new paper inputs.
"""
import argparse,json,statistics,traceback
from collections import defaultdict
from peer19_v1_20261104 import *
OUT=BASE/'legacy_three_sources'
META=ROOT/'artifacts/data/v2x_seq_spd_main_domain_shared_v1/V2X-Seq-SPD/cooperative/data_info.json'
CHOICES=AUTO/'status_audit_20261009_1210/strict_loader_qualification_v4'
INVENTORY=AUTO/'status_audit_20261019_1200/strict_source_cost_inventory_v1/strict_conditions.jsonl'

def prepare():
    register('legacy_three_sources',__file__,__doc__,dict(GPU_forwards=0,CPU_threads=1,disk_MiB='<2 compact persequence counts, no copied raw data'),metadata=ev(META),choices={str(w):ev(CHOICES/f'delay_{w:03d}ms_choices.jsonl') for w in (0,100,200,400)},strict_inventory=ev(INVENTORY),original_ledger=ev(ROOT/'docs/paper_draft_v0_20261103/number_ledger.csv'),unsupported='Explicit delete request if exact population/value cannot be reconstructed, not silent substitution.')

def run():
    try:
        inv=[json.loads(l) for l in INVENTORY.open()];lookup={(str(r['sequence']),int(r['decision_us']),int(r['wait_ms'])):r for r in inv}
        assert len(inv)==len(lookup)==13264;panels=[];all_rows=[]
        for w in (0,100,200,400):
            p=CHOICES/f'delay_{w:03d}ms_choices.jsonl';rows=[json.loads(l) for l in p.open()];assert len(rows)==3316
            ages=[];without=0
            for r in rows:
                i=lookup[str(r['sequence']),int(r['vehicle_pointcloud_timestamp_us']),w]
                assert i['current_vehicle_frame']==r['effective_vehicle_frame'] and i['no_packet_fallback']==r['no_packet_fallback']
                if r['no_packet_fallback']:without+=1;assert i['source'] is None;continue
                age=(int(r['vehicle_pointcloud_timestamp_us'])-int(r['infrastructure_pointcloud_timestamp_us']))/1000
                assert age==r['actual_age_ms']==i['source']['actual_age_ms'] and age>=w
                assert i['source']['source_frame']==r['effective_infrastructure_frame'];ages.append(age)
            panels.append(dict(wait_ms=w,frames=len(rows),without_message=without,median_age_ms=statistics.median(ages),min_age_ms=min(ages),max_age_ms=max(ages),source=ev(p)))
        groups=defaultdict(set);serialized=defaultdict(set)
        for r in load(META):
            s=str(r['vehicle_sequence']);d=r['system_error_offset'];groups[s].add((float(d['delta_x']),float(d['delta_y'])));serialized[s].add(json.dumps(d,sort_keys=True))
        assert {k:len(v) for k,v in groups.items()}=={k:len(v) for k,v in serialized.items()}
        counts={s:len(v) for s,v in sorted(groups.items())};total=len(counts);varying=sum(v>1 for v in counts.values())
        found=dict(strict_age_median_ms={str(r['wait_ms']):r['median_age_ms'] for r in panels if r['wait_ms'] in (0,400)},frames_without_message=[r['without_message'] for r in panels],pair_offset_varying_sequences=str(varying)+'/'+str(total))
        declared=dict(strict_age_median_ms={'0':45,'400':446},frames_without_message=[6,27,48,90],pair_offset_varying_sequences='63/67')
        decisions=[]
        for key,value in found.items():
            exact=value==declared[key]
            rounded=key=='strict_age_median_ms' and all(round(v)==declared[key][k] for k,v in value.items())
            decisions.append(dict(key=key,declared=declared[key],computed=value,status='resolved_exact' if exact else ('resolved_explicit_rounding' if rounded else 'delete_original_number'),adopt_in_new_table=bool(exact or rounded),original_ledger_unchanged=True))
        atomic(OUT/'strict_panels.json',panels);atomic(OUT/'per_sequence_distinct_offset_counts.json',counts);atomic(OUT/'decisions.json',decisions)
        atomic(OUT/'terminal.json',dict(status='complete_independently_reviewed',created_at=now(),decisions=ev(OUT/'decisions.json'),independence='Source timestamp arithmetic independently cross-joined to separate full strict inventory; numeric tuple and original serialized offset cardinalities independently agree. No AP or whole-ledger closure.',metadata=ev(META),actual_population_sequences=total,original_audit_differences_not_erased=True))
        record('legacy_three_sources',dict(status='complete_independently_reviewed',terminal=ev(OUT/'terminal.json'),resource_owner=False,owner=None));print(decisions)
    except Exception as e:
        atomic(OUT/'terminal.json',dict(status='failed_retained',created_at=now(),error=repr(e),traceback=traceback.format_exc()));record('legacy_three_sources',dict(status='failed_retained',terminal=ev(OUT/'terminal.json'),resource_owner=False,owner=None));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run']);globals()[p.parse_args().mode]()
