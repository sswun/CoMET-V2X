"""Stdlib-only finite experiment DAG, max two heavy CPU jobs plus GPU feeder.

No model invocation, timer, event wakeup, retries, or watchdog. A fixed list of
already-authorized steps is launched only after file dependencies are ready.
Failed jobs remain failed for the next calendar judgment, unrelated jobs proceed.
"""
import argparse, fcntl, hashlib, json, os, subprocess, time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
AUTO=ROOT/'artifacts/research_autopilot';BASE=AUTO/'status_audit_20261105_peer20'
P19=AUTO/'status_audit_20261104_peer19';OUT=BASE/'finite_resource_DAG_v1'
HERE=Path(__file__).resolve()
DOMAINS=('author','strict60','v2xvit_seed2024','v2xvit_seed2025','where2comm')
OK=('complete_light_checked','complete_independently_reviewed','complete_recovery_execution','native_AP_complete_light_checked')

def load(p):return json.loads(Path(p).read_text())
def stamp():return time.strftime('%Y-%m-%dT%H:%M:%S%z')
def atomic(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2));tmp.replace(p)
def ref(p):
    p=Path(p);return dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size)
def scientific_ready(p):
    if not Path(p).exists():return False
    value=load(p);return 'status' not in value or value['status'] in OK
def active(unit):
    r=subprocess.run(['systemctl','--user','show',unit,'-p','ActiveState','--value'],capture_output=True,text=True)
    return r.stdout.strip() in ('active','activating')
def name_for(module,mode,domain=None,family=None,seed=None,phase=None):
    if 'peer19' in module and module!='recover_peer19_committed_review_v4_20261105':return 'resource_recovery_'+module+'_'+mode
    n='resource_recovery_'+module.removesuffix('_20261105').removesuffix('_20261104')+'_'+mode
    if module=='consume_peer20_R7_v1_20261105':n='R7_'+n
    for x in (domain,family,str(seed) if seed else None,phase):
        if x:n+='_'+x
    return n

def plan():
    jobs=[]
    def add(key,module,mode,output,deps=(),domain=None,family=None,seed=None,phase=None,existing=None):
        name=name_for(module,mode,domain,family,seed,phase)
        if module=='recover_peer19_committed_review_v4_20261105':
            name='R1_review_recovery_'+domain;script=module+'.py';args=[mode,'--domain',domain]
        elif module=='recover_peer19_strict_review_v5_20261105':
            name='R1_review_recovery_v5_strict60';script=module+'.py';args=[mode]
        else:
            script='recover_peer19_consumers_v4_20261105.py' if 'peer19' in module else 'recover_peer20_cpu_prefix_v4_20261105.py'
            args=['--module',module,'--mode',mode]
            for k,v in (('domain',domain),('family',family),('seed',seed),('phase',phase)):
                if v is not None:args+=['--'+k,str(v)]
        jobs.append(dict(key=key,name=name,script=script,args=args,output=str(output),deps=list(map(str,deps)),
            unit=existing or 'embodied-peer20-dag-'+key+'-v1-20261105.service',status='pending',heavy=True))
    reviews=[]
    for d in DOMAINS:
        rd=P19/('recipe_independent_resource_recovery_v5' if d=='strict60' else 'recipe_independent')/d/'terminal.json';reviews.append(rd)
        module='recover_peer19_strict_review_v5_20261105' if d=='strict60' else 'recover_peer19_committed_review_v4_20261105'
        existing='embodied-peer20-R1-review-recovery-'+('v5-strict60' if d=='strict60' else d)+'-v1-20261105.service' if d in ('author','strict60') else None
        add('R1-'+d,module,'run',rd,domain=d if d!='strict60' else None,existing=existing)
    add('R4-read','consume_peer19_R4_independent_read_v1_20261104','run',P19/'R4_independent_read/terminal.json',reviews[:2]+[P19/'robust_external_independent/terminal.json'])
    add('direct390','compare_peer19_H1_direct_v1_20261104','run',P19/'H1_direct_contrasts/terminal.json',reviews)
    add('v3-build','build_peer19_final_tables_v3_20261104','run',P19/'final_tables_v3/terminal.json',reviews+[P19/(n+'/terminal.json') for n in ('H1_direct_contrasts','H1_assets_independent','H1_boundaries/opv_independent','H1_boundaries/v2_independent','robust_external_independent','legacy_three_sources')])
    add('v3-integrate','consume_peer19_H1_paper_supplement_v1_20261104','integrate',P19/'final_tables_v3_integration/terminal.json',[P19/'final_tables_v3/terminal.json',P19/'H1_paper_supplement/terminal.json'])
    add('R1-CI-attach','consume_peer19_R1_independent_CI_v1_20261104','run',P19/'R1_independent_CI_attachment/terminal.json',[P19/'final_tables_v3_integration/terminal.json',*reviews])
    add('strict-cal4-attach','consume_peer19_strict_cal4_tables_v1_20261104','run',P19/'strict_cal4_H1_table_completion/terminal.json',[P19/'strict_cal4_H1_independent/terminal.json',P19/'final_tables_v3_integration/terminal.json'])
    # R6e first finish the only missing cal producer without holding datasets.
    for d in DOMAINS:
        src=BASE/('R6e_namespace_recovery_v3' if d=='strict60' else 'R6e')/d
        add('clean-cal-'+d,'run_peer20_clean_head_v1_20261105','run',src/'calibration_ready.json',domain=d,phase='cal')
    ready=[BASE/('R6e_namespace_recovery_v3' if d=='strict60' else 'R6e')/d/'calibration_ready.json' for d in DOMAINS]
    choice=BASE/'R6e/global_choice/terminal.json'
    add('clean-choice','recover_peer20_clean_head_namespace_v3_20261105','choose',choice,ready)
    cleanreviews=[]
    for d in DOMAINS:
        src=BASE/('R6e_namespace_recovery_v3' if d=='strict60' else 'R6e')/d
        r=BASE/('R6e_namespace_v3_strict60_independent' if d=='strict60' else 'R6e_'+d+'_independent')/'terminal.json';cleanreviews.append(r)
        add('clean-formal-'+d,'run_peer20_clean_head_v1_20261105','run',src/'terminal.json',[choice],domain=d,phase='formal')
        add('clean-review-'+d,'run_peer20_clean_head_v1_20261105','review',r,[src/'terminal.json'],domain=d,phase='review')
    add('clean-independent-choice','consume_peer20_clean_choice_v1_20261105','accept',BASE/'R6e_choice_independent/terminal.json',[choice,*cleanreviews])
    # Core comparisons interleaved with independent review, not many waiters.
    for d in DOMAINS:
        b=BASE/('R6b_'+d);br=BASE/('R6b_'+d+'_independent');aux=BASE/('R6d_'+d)
        add('core-'+d,'run_peer20_population_v1_20261105','run',b/'terminal.json',domain=d)
        add('core-review-'+d,'run_peer20_population_v1_20261105','review',br/'terminal.json',[b/'terminal.json'],domain=d)
        add('core-paired-'+d,'consume_peer20_core_statistics_v1_20261105','run',BASE/('R6bc_paired_'+d)/'terminal.json',[br/'terminal.json'],domain=d)
        add('aux-'+d,'run_peer20_auxiliary_GT_v1_20261105','run',aux/'terminal.json',[b/'terminal.json'],domain=d)
        add('aux-review-'+d,'run_peer20_auxiliary_GT_v1_20261105','review',BASE/('R6d_'+d+'_independent')/'terminal.json',[aux/'terminal.json'],domain=d)
        for review in (False,True):
            out=BASE/('R6e_clean_assets_'+d+('_independent' if review else ''))/'terminal.json'
            deps=[BASE/'R6e_choice_independent/terminal.json',b/'terminal.json']
            if review:deps.append(BASE/('R6e_clean_assets_'+d)/'terminal.json')
            add('clean-assets-'+d+('-review' if review else ''),'consume_peer20_clean_choice_v1_20261105','review' if review else 'assets',out,deps,domain=d)
    for family in ('modules','late','native_author','native_strict60','native_vit','native_vit2024','native_vit2025','native_fcooper','native_where'):
        recovery=family in ('native_author','native_strict60','native_vit2025')
        module='recover_peer20_external_sources_v2_20261105' if recovery else 'run_peer20_external_population_v1_20261105'
        prefix='R6bd_external_v2_' if recovery else 'R6bd_external_'
        src=BASE/(prefix+family)
        add('external-'+family,module,'run',src/'terminal.json',family=family)
        add('external-review-'+family,module,'review',BASE/(prefix+family+'_independent')/'terminal.json',[src/'terminal.json'],family=family)
    a5=BASE/'R6c_A5_cluster'
    add('ACF','run_peer20_ACF_cluster_v1_20261105','run',a5/'terminal.json')
    add('ACF-review','run_peer20_ACF_cluster_v1_20261105','review',BASE/'R6c_A5_cluster_independent/terminal.json',[a5/'terminal.json'])
    # Collector's only dependency is remotely polled by this stdlib coordinator.
    add('R7-collector','consume_peer20_R7_v1_20261105','collect',BASE/'R7_collector/terminal.json',[OUT/'remote_R7_ready.json'])
    for s in (20261101,20261102,20261103):
        train=BASE/('R7_environment_v2' if s==20261102 else 'R7_resource_recovery_v3')/('seed'+str(s))
        deps=[train/r/'terminal.json' for r in ('val21','mechanism_holdout14')]
        if s==20261102:deps+=[BASE/'R7_collector/terminal.json']
        src=BASE/('R7_analysis_seed'+str(s))
        add('R7-AP-'+str(s),'consume_peer20_R7_v1_20261105','run',src/'terminal.json',deps,seed=s)
        add('R7-review-'+str(s),'consume_peer20_R7_v1_20261105','review',BASE/('R7_analysis_seed'+str(s)+'_independent')/'terminal.json',[src/'terminal.json'],seed=s)
    add('R7-read','consume_peer20_R7_v1_20261105','read',BASE/'R7_preregistered_read/terminal.json',[BASE/('R7_analysis_seed'+str(s)+'_independent')/'terminal.json' for s in (20261101,20261102,20261103)])
    return jobs

def prepare():
    from peer20_v1_20261105 import register,record
    jobs=plan()
    register('finite_resource_DAG_v1',__file__,__doc__,dict(CPU_threads=1,GPU_forwards=0,updates=0,disk_MiB='<5 runtime state/logs'),
        max_heavy_CPU=2,extra_GPU_feeders=1,launch_available_memory_GiB=28,
        no_infinite_retry=True,no_model_events=True,no_guard_threshold_changes=True,finite_jobs=len(jobs))
    for j in jobs:
        if (BASE/j['name']/'contract.json').exists():continue
        source=ROOT/'preexperiment/research_autopilot'/j['script']
        register(j['name'],source,'Recover existing authorized finite step; no scientific expansion or deadline.',
            dict(CPU_threads=1,GPU_forwards=0,updates=4000 if j['key']=='clean-cal-strict60' else 0,replicates=2000 if any(x in j['key'] for x in ('core','aux','external','AP','review','ACF')) else 0,disk_MiB='<250 compact outcomes; shared raw predictions'),
            resource_recovery=True,command=j['args'],scientific_output=j['output'],dependencies=j['deps'])
        record(j['name'],dict(status='queued_finite_DAG',owner=None,resource_owner=False,dispatcher='finite_resource_DAG_v1'))
    atomic(OUT/'plan.json',dict(created_at=stamp(),jobs=jobs,max_heavy_CPU=2,GPU_feeders=1,guard_thresholds_unchanged=True))

def launch(j):
    from peer20_v1_20261105 import start,record
    n='dag-'+j['key'];receipt=BASE/'launch_receipts'/(n+'.json')
    assert not receipt.exists(),('No automatic retry',n)
    unit=start(n,j['script'],j['args']);j['unit']=unit+'.service';j['status']='running';j['launched_at']=stamp()
    record(j['name'],dict(status='running_finite_DAG_step',owner=j['unit'],resource_owner=True))

def run(once=False):
    from peer20_v1_20261105 import record
    lock=(OUT/'owner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    statepath=OUT/'state.json';jobs=load(statepath)['jobs'] if statepath.exists() else load(OUT/'plan.json')['jobs']
    while True:
        # Light remote receipt check only, no scientific imports/waiters.
        if not (OUT/'remote_R7_ready.json').exists():
            path='/path/to/remote_project/'+str((BASE/'R7_environment_v2_remote/terminal.json').relative_to(ROOT))
            r=subprocess.run(['ssh','-o','ConnectTimeout=8','remote-gpu','test -f '+path+' && cat '+path],capture_output=True,text=True,timeout=20)
            if r.returncode==0 and json.loads(r.stdout).get('status')=='complete_light_checked':atomic(OUT/'remote_R7_ready.json',json.loads(r.stdout))
        running=[]
        for j in jobs:
            if scientific_ready(j['output']):
                if active(j['unit']):running.append(j);continue
                j['status']='complete';continue
            if active(j['unit']):j['status']='running';running.append(j);continue
            if j['status']=='running':
                j['status']='failed_retained';j['failure_detected_at']=stamp()
                record(j['name'],dict(status='failed_retained',owner=None,resource_owner=False,
                    failure_kind='execution_or_resource_failure_see_unit_and_guard',unit=j['unit']))
            if Path(j['output']).exists() and load(j['output']).get('status')=='failed_retained':j['status']='failed_retained'
        available=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))*1024
        slots=2-len(running)
        if available>=28*1024**3 and slots>0:
            for j in jobs:
                if slots<=0:break
                if j['status']!='pending' or not all(scientific_ready(p) for p in j['deps']):continue
                launch(j);slots-=1
        atomic(statepath,dict(created_at=stamp(),jobs=jobs,available_bytes=available,
            max_heavy_CPU=2,counts={s:sum(j['status']==s for j in jobs) for s in ('pending','running','complete','failed_retained')},
            no_model_wakeup=True,no_retry=True))
        if all(j['status'] in ('complete','failed_retained') for j in jobs):break
        if once:return
        time.sleep(30)
    atomic(OUT/'terminal.json',dict(created_at=stamp(),status='complete_light_checked',
        failed=[j['key'] for j in jobs if j['status']=='failed_retained'],scientific_acceptance_not_implied=True))
    record('finite_resource_DAG_v1',dict(status='complete_light_checked',owner=None,resource_owner=False,terminal=ref(OUT/'terminal.json')))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','run','once'));a=p.parse_args()
    prepare() if a.mode=='prepare' else run(a.mode=='once')
