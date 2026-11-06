"""Recover only v3 table row identities; retain failed build and all science.

No detector/training/AP/bootstrap rerun. Namespace generated panel rows by
their immutable source path. Keep the failed root files in a version directory,
then run the frozen generator/integration with explicit versioned bindings.
The previously unproduced canonical integration receipt unblocks the old DAG;
the original failed builder receipt is never overwritten or relabeled success.
"""
import argparse
import fcntl
import shutil
import time
from pathlib import Path
from peer21_v1_20261105 import *

NAME = 'v3_row_identity_recovery'
OUT = BASE / NAME
TABLE = ROOT / 'docs/paper_draft_v0_20261103/final_tables_v3'
BUILD = ROOT / 'preexperiment/research_autopilot/build_peer19_final_tables_v3_20261104.py'
INTEGRATE = ROOT / 'preexperiment/research_autopilot/consume_peer19_H1_paper_supplement_v1_20261104.py'
ARCHIVE = TABLE / 'versions/failed_generator_20261105_095657'


def prepare():
    register(NAME, __file__, __doc__, dict(GPU_forwards=0, updates=0,
        CPU_threads=1, new_AP=0, new_bootstrap=0,
        disk_MiB='<45 corrected numeric ledger/CSV; failed small tables retained, no raw cache copies'),
        builder=ev(BUILD), integration=ev(INTEGRATE),
        failed_terminal=ev(P19/'final_tables_v3/terminal.json'),
        correction='Panel row key = source relative path + old source row/index/identity',
        execution_class='lightweight JSON/CSV asset consumer; no materialized dataset',
        original_failed_builder_preserved=True,
        downstream='Old DAG observes canonical integration, then R1 CI and strict cal4 consumers')


def run():
    lock=(OUT/'owner.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    begin=time.monotonic()
    original_failure=ev(P19/'final_tables_v3/terminal.json')
    try:
        assert load(P19/'final_tables_v3/terminal.json')['status']=='failed_retained'
        assert not (P19/'final_tables_v3_integration/terminal.json').exists()
        audit=load(TABLE/'independent_numeric_audit.json')
        issues=audit['issues']
        assert issues and all(i.get('error')=='CSV mismatch or duplicate key' and i.get('matches',0)>1 for i in issues)
        # Verify every collision is from distinct source panel files. A numeric
        # discrepancy or duplicated source pointer is not excused by this fix.
        groups={}
        for c in load(TABLE/'numeric_ledger.json')['cells']:
            groups.setdefault((c['table'],c['row'],c['metric']),[]).append(c)
        collisions=[]
        for key,cells in groups.items():
            if len(cells)<2:continue
            paths=[c['operands'][0]['source']['path'] for c in cells]
            assert len(paths)==len(set(paths)),(key,paths)
            collisions.append(dict(table=key[0],row=key[1],metric=key[2],sources=paths))
        assert collisions
        atomic(OUT/'diagnosis.json',dict(created_at=now(),old_audit=ev(TABLE/'independent_numeric_audit.json'),
            cells=audit['cells'],colliding_keys=len(collisions),issues=len(issues),
            all_issues_cross_source_row_collisions=True,examples=collisions[:8],
            no_AP_or_CI_recomputation=True))
        record(NAME,dict(status='running_versioned_table_identity_recovery',resource_owner=True,
            owner='embodied-peer21-v3-row-identity-recovery-v1-20261105.service'))
        ARCHIVE.mkdir(parents=True,exist_ok=False)
        originals={}
        for p in sorted(TABLE.iterdir()):
            if not p.is_file():continue
            before=ev(p);target=ARCHIVE/p.name;shutil.move(str(p),target)
            originals[p.name]=dict(before=before,archived=ev(target))
        atomic(OUT/'failed_table_archive.json',dict(created_at=now(),files=originals,
            failed_terminal=original_failure,no_deletion=True))
        source=BUILD.read_text()
        old="row=str(i)+'|'+tag"
        assert source.count(old)==1
        source=source.replace(old,"row=str(path.relative_to(ROOT))+'|'+str(i)+'|'+tag")
        source=source.replace('recipe_independent_interface_recovery_v2','recipe_independent_resource_recovery_v5')
        ns=dict(__file__=str(BUILD),__name__='peer21_versioned_v3_row_recovery')
        exec(compile(source,str(BUILD)+'::peer21_row_identity_v1','exec'),ns)
        ns['JOB']=OUT/'generation'
        ns['record']=lambda unused,values: record(NAME,dict(values,stage='generation'))
        ns['run']()
        assert load(OUT/'generation/terminal.json')['status']=='complete_light_checked'
        # The old integration has never run. Only its failed-builder dependency
        # is rebound to the new generation; all scientific operands stay fixed.
        source=INTEGRATE.read_text()
        old="wait(BASE/'final_tables_v3/terminal.json',36,accepted=('complete_light_checked',))"
        assert source.count(old)==1
        source=source.replace(old,"wait(Path("+repr(str(OUT/'generation/terminal.json'))+"),36,accepted=('complete_light_checked',))")
        ns=dict(__file__=str(INTEGRATE),__name__='peer21_versioned_v3_integration')
        exec(compile(source,str(INTEGRATE)+'::peer21_builder_dependency_v1','exec'),ns)
        ns['record']=lambda unused,values: record(NAME,dict(values,stage='integration'))
        ns['integrate']()
        terminal=P19/'final_tables_v3_integration/terminal.json'
        assert load(terminal)['status']=='complete_light_checked'
        assert ev(P19/'final_tables_v3/terminal.json')==original_failure
        atomic(TABLE/'row_identity_recovery.json',dict(created_at=now(),adapter=ev(__file__),
            contract=ev(OUT/'contract.json'),diagnosis=ev(OUT/'diagnosis.json'),
            archive=ev(OUT/'failed_table_archive.json'),generation=ev(OUT/'generation/terminal.json'),
            integration=ev(terminal),source_pointer_namespace=True,
            original_failed_terminal_unchanged=original_failure,
            no_new_science_or_recipe_selection=True))
        atomic(OUT/'terminal.json',dict(created_at=now(),status='complete_light_checked',
            generation=ev(OUT/'generation/terminal.json'),integration=ev(terminal),
            numeric_audit=ev(TABLE/'independent_numeric_audit.json'),
            recovery=ev(TABLE/'row_identity_recovery.json'),wall_s=time.monotonic()-begin,
            original_failed_builder_preserved=True,whole_release_closed=False))
        record(NAME,dict(status='complete_light_checked',terminal=ev(OUT/'terminal.json'),owner=None,resource_owner=False))
    except Exception as error:
        import traceback
        atomic(OUT/'terminal.json',dict(created_at=now(),status='failed_retained',error=repr(error),traceback=traceback.format_exc()))
        record(NAME,dict(status='failed_retained',terminal=ev(OUT/'terminal.json'),owner=None,resource_owner=False))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','run'));globals()[p.parse_args().mode]()
