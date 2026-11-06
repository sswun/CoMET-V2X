"""Full independent CoAlign geometry/source/AP review on each owning host."""
import argparse,inspect,os,time,traceback
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
from peer13_cycle18_v1_20261102 import ROOT,AUTO,AUDIT,CYCLE,C15,atomic,ev,load,now,update,task
import run_peer13_coalign_sources_ROI_v2_20261102 as source
import review_peer12_native_independent_v1_20261101 as original
BASE=CYCLE/'coalign_full_independent_v1';TID='peer13_coalign_full_independent_{}_v1_20261102'

def record(host,v):
    tid=TID.format(host)
    if str(ROOT).startswith('/home/user/'):
        p=BASE/'remote_queue_delta.json';r=load(p) if p.exists() else {};r[tid]=dict(id=tid,parent_task_id=task('K5'),**v);atomic(p,r)
    else:update(tid,task('K5'),v)

def names(host):
    first=['author_evaluation'] if host=='remote' else ['adapted_evaluation','online_adapted_evaluation']
    pp=[C15/'coalign_full_v1'/n for n in first];kind='reference' if host=='remote' else 'graph'
    pp += [CYCLE/'coalign_confirmation_v1'/('seed'+str(s))/(t+'_'+kind)/'evaluation' for s in (20261103,20261104) for t in ('public','online')]
    return [str(p.relative_to(AUDIT)) for p in pp]

def namespace(host):
    spec=dict(producers=names(host),protocols=['official','interpolated','no_interpolation'],rois=['native','common'],groups=['all','veh','coop','inf','veh_coop'],thresholds=[.5,.7])
    expected=2640 if host=='remote' else 2400
    text=inspect.getsource(original.main)
    changes=[("out=CYCLE/'independent_native'","out=REVIEW_OUT"),("spec=load(CYCLE/'contract.json')['native_review']","spec=REVIEW_SPEC"),
        ("bounds=load(AUDIT/'cycle03_v1/contract.json')['native_source_aux']['native_eval_range']","bounds=REVIEW_BOUNDS"),
        ("producer=AUDIT/name;terminal=load(producer/'terminal.json')","producer=AUDIT/name;wait_ready(producer);terminal=load(producer/'terminal.json')"),
        ("and terminal['weight_unchanged'] and not terminal['forward_GT_input']","and load(producer/'input_identity.json')['GT_not_in_model_input']"),
        ("goldroot=AUDIT/spec['source_root']/name","goldroot=GOLDROOT/name"),
        ("with np.load(producer/f'wait{wait:03d}.npz') as shard,np.load(goldroot/f'wait{wait:03d}_outcomes.npz') as gold:","with np.load(producer/f'wait{wait:03d}.npz') as shard,np.load(goldroot/f'wait{wait:03d}_outcomes.npz') as gold,np.load(ROOT/selected[0]['GT_shard']) as gtshard:"),
        ('gt=shard[f"c{r[\'shard_case\']}_gt"]','gt=gtshard[f"c{r[\'GT_case\']}_gt"]'),
        ("ap=metric(f[0,valid],f[1,valid],ss[valid],st['gt'])","ap=metric(f[0],f[1],ss,st['gt'])"),
        ('all1440_metric_panels','all_registered_metric_panels'),('len(panels)==1440','len(panels)==EXPECTED_PANELS'),
        ("ev(CYCLE/'contract.json')","ev(REVIEW_CONTRACT)")]
    for a,b in changes:assert text.count(a)==1,(a,text.count(a));text=text.replace(a,b)
    ns=dict(original.__dict__,REVIEW_OUT=BASE/host,REVIEW_SPEC=spec,REVIEW_BOUNDS=source.RANGE,GOLDROOT=source.BASE/'native_source_aux',REVIEW_CONTRACT=BASE/'contract.json',EXPECTED_PANELS=expected,wait_ready=wait_ready,update=lambda _,values:record(host,values),CYCLE=CYCLE)
    exec(compile(text,__file__+'::independent_float64_shared_GT_replay','exec'),ns);return ns

def wait_ready(producer):
    start=time.monotonic();gold=source.BASE/'native_source_aux'/producer.relative_to(AUDIT)
    while not (producer/'terminal.json').exists() or not (gold/'terminal.json').exists():
        if time.monotonic()-start>34*3600:raise TimeoutError('finite34h producer/source review wait')
        time.sleep(30)
    assert load(producer/'terminal.json')['status']=='native_AP_complete_light_checked'
    assert load(gold/'terminal.json')['status']=='complete_light_checked'

def prepare():
    refs={}
    for n in ('current_handoff.md','main_chat_directive.md'):(AUTO/n).read_text();refs[n]=ev(AUTO/n)
    BASE.mkdir(exist_ok=False);namespace('local');namespace('remote')
    atomic(BASE/'contract.json',dict(created_at=now(),script=ev(__file__),independent_source=ev(original.__file__),source_contract=ev(source.BASE/'contract.json'),instructions=refs,
        prediction_before_result='Full float64 independent geometry/target-first matching/stable primary source and own VOC integration may expose implementation or threshold discrepancies; preserve signs and canonical replay before numerical adjudication.',
        scope='All first-seed author/PUBLIC/ONLINE and fixed seed2/3 PUBLIC/ONLINE graph/reference outputs, each scientific arm exactly once. Native-named scopes retain shared TraF GT/CoAlign prediction ROI; common ROI is the cross-backbone comparison. Not original DAIR training reproduction.',
        cost=dict(GPU_forwards=0,CPU_threads=1,metric_panels=5040,new_disk_MiB='<30 checks/identities only, no copies of predictions/GT/outcomes',wall_h='6-18 parallel finite hosts'),no_model_wakeup=True))
    for host in ('local','remote'):record(host,dict(status='registered_finite_successor',contract=ev(BASE/'contract.json')))

def run(host):
    c=load(BASE/'contract.json');assert c['script']==ev(__file__) and c['independent_source']==ev(original.__file__)
    namespace(host)['main']()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','local','remote']);a=p.parse_args();prepare() if a.mode=='prepare' else run(a.mode)
