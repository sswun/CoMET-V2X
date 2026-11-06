"""Expose all accepted R1 ablation/primary CIs; no repeated AP/bootstrap.

Five immutable reviewer bootstrap JSON sources become row-addressed CSV, with
an independent stdlib round-trip check. Systematic controls/calibration roles
are never selected as main methods. This is a data-asset consumer only.
"""
import argparse,csv,io,json,time,traceback
from peer19_v1_20261104 import *
import run_peer19_H1_assets_v1_20261104 as h1
NAME='R1_independent_CI_attachment'
OUT=BASE/NAME
TARGET=ROOT/'docs/paper_draft_v0_20261103/final_tables_v3/R1_independent_CI'

def prepare():
    register(NAME,__file__,__doc__,dict(GPU_forwards=0,updates=0,CPU_threads=1,disk_MiB='<12 CSV/provenance, no duplicate draws',wall_h='bounded36h dependencies'),
        deadline='2026-11-05T12:00:00+08:00',dependencies=['All five R1 independent terminal/bootstrap.json'],
        interpretation='Cal8 diagnostic CIs descriptive; selected main H1 frozen by cal8, never per-formal-cell selection.')

def run():
    record(NAME,dict(status='waiting_five_R1_independent_CI_sources',resource_owner=False,owner=None));checks=[];refs={}
    try:
        # Wait for integration too, so its small-file versioning cannot move this asset.
        wait(BASE/'final_tables_v3_integration/terminal.json',36,accepted=('complete_light_checked',))
        TARGET.mkdir(parents=True,exist_ok=False)
        for domain in h1.recipe.DOMAINS:
            root=BASE/('recipe_independent_interface_recovery_v2' if domain=='strict60' else 'recipe_independent')/domain
            wait(root/'terminal.json',36,accepted=('complete_independently_reviewed',))
            path=root/'bootstrap.json';rows=load(path);ref=ev(path);review=ev(root/'terminal.json');refs[domain]=dict(source=ref,review=review)
            stream=io.StringIO();writer=csv.DictWriter(stream,fieldnames=['source_row','comparison','threshold','wait_ms','delta_pp','CI95_low','CI95_high','positive_sequences','negative_sequences','zero_sequences','identity','source','review'])
            writer.writeheader()
            for i,row in enumerate(rows):
                writer.writerow(dict(source_row=i,**{k:row[k] for k in ('comparison','threshold','wait_ms','delta_pp','CI95_low','CI95_high','positive_sequences','negative_sequences','zero_sequences')},
                    identity=json.dumps(row['identity'],ensure_ascii=False),source=json.dumps(ref),review=json.dumps(review)))
            target=TARGET/(domain+'.csv');target.write_text(stream.getvalue())
            recovered=list(csv.DictReader(target.open()));assert len(recovered)==len(rows)
            for i,(a,b) in enumerate(zip(rows,recovered)):
                assert int(b['source_row'])==i and b['comparison']==a['comparison'] and json.loads(b['identity'])==a['identity']
                for k in ('threshold','delta_pp','CI95_low','CI95_high','positive_sequences','negative_sequences','zero_sequences'):assert float(b[k])==float(a[k])
            checks.append(dict(domain=domain,rows=len(rows),source=ref,review=review,CSV=ev(target),roundtrip_exact=True))
        atomic(TARGET/'manifest.json',dict(created_at=now(),consumer=ev(__file__),sources=refs,checks=checks,
            no_new_AP_or_bootstrap=True,not_independent_detection_training=True,root_numeric_ledger_not_modified=True))
        atomic(OUT/'terminal.json',dict(status='complete_light_checked',created_at=now(),manifest=ev(TARGET/'manifest.json'),
            independent_numeric_serialization_check='stdlib csv decode against every immutable JSON source row; scientific CI independence belongs to referenced R1 reviewers'))
        record(NAME,dict(status='complete_light_checked',terminal=ev(OUT/'terminal.json'),resource_owner=False,owner=None))
    except Exception as e:
        atomic(OUT/'terminal.json',dict(status='failed_retained',created_at=now(),error=repr(e),traceback=traceback.format_exc()))
        record(NAME,dict(status='failed_retained',terminal=ev(OUT/'terminal.json'),resource_owner=False,owner=None));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run']);globals()[p.parse_args().mode]()
