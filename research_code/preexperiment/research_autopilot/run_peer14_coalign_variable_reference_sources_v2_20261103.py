"""Actual-wait shared-GT source scoring; fixes ego wait0, no GPU repeat."""
import argparse,inspect,os,time,traceback,subprocess
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
from peer14_cycle00_v1_20261103 import ROOT,AUDIT,CYCLE,C21,REMOTE_ROOT,atomic,ev,load,now,register,record
import run_peer13_coalign_confirmation_sources_v1_20261102 as old
import run_peer14_coalign_upper_parent_recovery_v2_20261103 as upper
BASE=CYCLE/'coalign_variable_reference_sources_v2'

def namespace():
    ns,c=old.compute_namespace();s=inspect.getsource(old.scorer.compute)
    changes=[("frozen={'producer_terminal'","waits=sorted({r['wait_ms'] for r in load(producer/'records.json')});frozen={'producer_terminal'"),
    ('assert len(records)==13264 and len({(r[\'wait_ms\'],r[\'global_index\']) for r in records})==13264',"assert len({(r['wait_ms'],r['global_index']) for r in records})==len(records)"),
    ('assert len(selected)==3316',"assert len(selected)==expected[wait]['cases']"),
    ('assert len(panels)==4*3*2*len(arms)*5*2','assert len(panels)==len(waits)*3*2*len(arms)*5*2'),
    ("gt=shard[f'c{idx}_gt']","gt=gtshard[f\"c{r['GT_case']}_gt\"]"),
    ("with np.load(producer/f'wait{wait:03d}.npz',allow_pickle=False) as shard:","with np.load(producer/f'wait{wait:03d}.npz',allow_pickle=False) as shard, np.load(ROOT/next(r['GT_shard'] for r in records if r['wait_ms']==wait),allow_pickle=False) as gtshard:")]
    for a,b in changes:assert s.count(a)==1,a;s=s.replace(a,b)
    assert s.count('(0,100,200,400)')==2;s=s.replace('(0,100,200,400)','waits')
    ns['CYCLE']=BASE;exec(compile(s,__file__+'::actual_wait_shared_GT_source','exec'),ns)
    import copy
    c=copy.deepcopy(c);c['native_source_aux']['native_eval_range']=[-100.8,-40,-3.5,100.8,40,1.5]
    return ns,c

def prepare():
    BASE.mkdir(exist_ok=False);namespace();targets=[str((C21/'coalign_reference_completion_v1'/t/m/'evaluation').relative_to(AUDIT)) for t in ('public','online') for m in ('ego','slow')]
    targets.append(str((C21/'coalign_author_reference_v1/evaluation').relative_to(AUDIT)))
    c=load(CYCLE/'coalign_upper_contract.json')
    for t in ('public','online'):
        for p in c['plans'][t]:
            for mode in p['modes']:targets.append(str((upper.BASE/t/p['name']/mode/'evaluation').relative_to(AUDIT)))
    targets.append(str((upper.BASE/'author_official/evaluation').relative_to(AUDIT)))
    targets +=[str((upper.BASE/'official_trained'/m/'evaluation').relative_to(AUDIT)) for m in ('factor','ego','slow')]
    register('variable_reference_sources',__file__,'Repair source consumer hard-coded wait shards for ego-only wait0. Preserve failed oldCPU output; score actual records/waits only with exact native AP replay. Slow/full/ego references nondeployable, no new model forward.',dict(GPU_forwards=0,CPU_threads=1,wall_h='finite36h known reference producer wait + sequential source scoring',new_disk_GiB='<1 compact outcomes; no GT/pred copies'),original_scorer=ev(old.scorer.__file__),source_adapter=ev(old.__file__),targets=targets,old_failure=ev(C21/'coalign_reference_completion_v1/sources/terminal.json'))
    register('upper_collection',__file__,'Finite remote terminal collector, no model wake, same relative path and no different-main overwrite.',dict(GPU_forwards=0,CPU_threads=1,wall_h='finite36h',new_disk='unique remote author/officialtraining predictions only'))

def run():
    c=load(CYCLE/'variable_reference_sources_contract.json');assert c['script']==ev(__file__);out=BASE/'producer';out.mkdir();ns,spec=namespace();start=time.monotonic();results=[];record('variable_reference_sources',dict(status='running_actual_wait_source_chain'))
    for name in c['targets']:
        try:
            p=AUDIT/name
            while not (p/'terminal.json').exists():
                if time.monotonic()-start>36*3600:raise TimeoutError('finite36h reference dependency')
                atomic(out/'status.json',dict(status='waiting_frozen_reference',producer=name,created_at=now(),completed=len(results)));time.sleep(30)
            ns['compute'](name,spec);results.append(dict(producer=name,terminal=ev(BASE/'native_source_aux'/name/'terminal.json')))
        except Exception as e:results.append(dict(producer=name,error=repr(e),traceback=traceback.format_exc()))
        atomic(out/'results.json',results)
    status='complete_light_checked' if all('terminal' in r for r in results) else 'failed_retained';atomic(out/'terminal.json',dict(status=status,created_at=now(),results=ev(out/'results.json'),GPU_forwards=0,wall_s=time.monotonic()-start,no_model_wakeup=True));record('variable_reference_sources',dict(status=status,terminal=ev(out/'terminal.json')))

def collect():
    out=CYCLE/'upper_collection_v1';out.mkdir();start=time.monotonic();done={};record('upper_collection',dict(status='running_finite_remote_collection'))
    paths=[C21/'coalign_author_reference_v1',upper.BASE/'remote']
    while time.monotonic()-start<36*3600:
        for p in paths:
            key=str(p.relative_to(ROOT));terminal=p/('producer/terminal.json' if p==paths[0] else 'terminal.json')
            if key in done:continue
            remote_terminal=REMOTE_ROOT/terminal.relative_to(ROOT)
            if subprocess.run(['ssh','remote-gpu','test','-f',str(remote_terminal)],capture_output=True).returncode:continue
            source=p if p==paths[0] else upper.BASE;source.mkdir(parents=True,exist_ok=True)
            subprocess.run(['rsync','-a','--ignore-existing','remote-gpu:'+str(REMOTE_ROOT/source.relative_to(ROOT))+'/',str(source)+'/'],check=True)
            done[key]=dict(collected_at=now(),terminal=ev(terminal),different_existing_never_overwritten=True);atomic(out/'receipts.json',done)
        if len(done)==len(paths):break
        atomic(out/'status.json',dict(created_at=now(),collected=len(done),planned=len(paths),no_model_wakeup=True));time.sleep(60)
    status='complete_light_checked' if len(done)==len(paths) else 'failed_retained';atomic(out/'terminal.json',dict(status=status,created_at=now(),receipts=done,no_model_wakeup=True));record('upper_collection',dict(status=status,terminal=ev(out/'terminal.json')))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run','collect']);a=p.parse_args();globals()[a.mode]()
