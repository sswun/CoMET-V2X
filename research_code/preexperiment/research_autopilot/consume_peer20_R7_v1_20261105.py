"""Sole remote R7 collector -> saved-box independent AP -> paired2000 CI.

No detector reruns, no checkpoint selection. Same-input old K4 and same-budget
PUBLIC/PUBLIC references per seed. Training input remains ONLINE_K1, NOT H1.
Primary preregistered gate is val21 three-seed/four-wait mean; hold14 is reported
separately without inventing a new all-seed/all-cell sign gate. No automatic
main-method adoption: qualifying evidence is handed to the main conversation.
"""
import argparse,hashlib,json,shlex,subprocess,time,traceback
import numpy as np
from peer20_population_core_v1_20261105 import *
import peer20_saved_native_v1_20261105 as native
from run_peer20_K4_decontamination_v1_20261105 import SEEDS
OUT=BASE/'R7_analysis_actual';TRAIN=BASE/'R7_environment_v2'
VAL=[('k4_eval_online_k1','k4_eval_public'),('seed2_online_k1/evaluation','seed2_public/evaluation'),('cycle15_peer11_v1/seed3_online/evaluation','cycle15_peer11_v1/seed3_public/evaluation')]
HOLD=[('cycle09_v1/hold14_online_k1','cycle09_v1/hold14_public'),('cycle21_peer12_v1/hold2_online21/evaluation','cycle15_peer11_v1/hold14_seed2_public_correct_env'),('cycle21_peer12_v1/hold3_online21/evaluation','cycle21_peer12_v1/hold3_public21/evaluation')]

def prepare():
    for name in ['R7_collector',*[f'R7_analysis_seed{s}'+suffix for s in SEEDS for suffix in ('','_independent')],'R7_preregistered_read']:
        register(name,__file__,__doc__,dict(GPU_forwards=0,updates=0,CPU_threads=1,replicates=2000,disk_MiB='<150 collector raw; <60 per-seed scored outcomes/draws'),
            adapter=ev(native.__file__),original_analysis_contract=ev(BASE/'R7_analysis/contract.json'),
            old_val_sources=VAL,old_hold_sources=HOLD,initial_identity=ev(BASE/'R7_environment_v2_local/contract.json'),
            gate='val21 mean clean-minus-originalK4 co-labelled>=.40, paired95CI lower>0, official>=-.20; hold14 separate. No successor.')
    record('R7_analysis',dict(status='implemented_registered_successors',implementation=ev(__file__),actual_output=str(OUT),owner=None,resource_owner=False))

def remote_json(path):
    cmd=['ssh','remote-gpu',shlex.join(['test','-f',str(path)])]
    if subprocess.run(cmd,capture_output=True).returncode:return None
    text=subprocess.check_output(['ssh','remote-gpu',shlex.join(['cat',str(path)])],text=True)
    return json.loads(text)

def collect():
    name='R7_collector';dest=BASE/name
    try:
        terminal=REMOTE_ROOT/str((BASE/'R7_environment_v2_remote/terminal.json').relative_to(ROOT))
        while True:
            state=remote_json(terminal)
            if state is not None:break
            time.sleep(30)
        assert state['status']=='complete_light_checked',state
        root=REMOTE_ROOT/str((TRAIN/'seed20261102').relative_to(ROOT));names=[]
        for role in ('val21','mechanism_holdout14'):
            records=remote_json(root/role/'records.json');assert records
            names.extend(role+'/'+n for n in ['terminal.json','records.json','panels.json','status.json',*sorted({r['shard'] for r in records})])
        names.extend('training/'+n for n in ('terminal.json','status.json','qualification.json','sampling_identity.json','training.jsonl','final.pth'))
        target=TRAIN/'seed20261102';target.mkdir(parents=True,exist_ok=True)
        for n in names:
            (target/n).parent.mkdir(parents=True,exist_ok=True)
            subprocess.run(['rsync','-a','--ignore-existing','remote-gpu:'+str(root/n),str(target/n)],check=True)
        code='import hashlib,json,pathlib; p=pathlib.Path('+repr(str(root))+'); names='+repr(names)+'; print(json.dumps({n:hashlib.sha256((p/n).read_bytes()).hexdigest() for n in names}))'
        result=subprocess.check_output(['ssh','remote-gpu',shlex.join([str(REMOTE_ROOT/'.conda/opencood-linux/bin/python'),'-c',code])],text=True)
        expected=json.loads(result);assert all(hashlib.sha256((target/n).read_bytes()).hexdigest()==v for n,v in expected.items()),'collector conflict'
        local_terminal=BASE/'R7_environment_v2_remote/terminal.json';assert not local_terminal.exists();atomic(local_terminal,state)
        atomic(dest/'manifest.json',dict(remote_SHA=expected,no_delete=True,active_resume_excluded=True))
        atomic(dest/'terminal.json',dict(created_at=now(),status='complete_light_checked',manifest=ev(dest/'manifest.json')))
        record('R7_environment_v2_remote',dict(status='complete_light_checked',terminal=ev(local_terminal),owner=None,resource_owner=False))
        record(name,dict(status='complete_light_checked',terminal=ev(dest/'terminal.json'),owner=None,resource_owner=False))
    except Exception as e:
        atomic(dest/'terminal.json',dict(created_at=now(),status='failed_retained',error=repr(e),traceback=traceback.format_exc()));record(name,dict(status='failed_retained',terminal=ev(dest/'terminal.json'),owner=None,resource_owner=False));raise

def run(seed,review=False):
    name=f'R7_analysis_seed{seed}'+('_independent' if review else '');dest=BASE/name;src=BASE/f'R7_analysis_seed{seed}'
    try:
        if review:wait(src/'terminal.json')
        if seed==20261102:wait(BASE/'R7_collector/terminal.json')
        record(name,dict(status='running_independent_R7_AP_CI' if review else 'waiting_R7_fixed_final_predictions',resource_owner=True))
        ix=SEEDS.index(seed);done=[]
        for role,refs in [('val21',VAL[ix]),('mechanism_holdout14',HOLD[ix])]:
            clean=TRAIN/f'seed{seed}'/role
            while not (clean/'terminal.json').exists():
                for parent in ('R7_environment_v2_local','R7_environment_v2_remote'):
                    if (BASE/parent/'terminal.json').exists():assert load(BASE/parent/'terminal.json')['status']!='failed_retained'
                time.sleep(30)
            for label,path,arm in [('clean',clean,'ONLINE_K1'),('original_K4',AUDIT/refs[0],'ONLINE_K1'),('PUBLIC_finetuned',AUDIT/refs[1],'PUBLIC')]:
                term=load(path/'terminal.json');assert term['status']=='native_AP_complete_light_checked' and not term['forward_GT_input'] and term['weight_unchanged'],(label,path,term)
                tag=role+'_'+label;pred=src/(tag+'_predictions.npz')
                m,rows,p,b=native.assemble('author',role,native.shard_cases(path,arm),label,review)
                if review:
                    with np.load(pred) as z:assert all(np.array_equal(z[k],v) for k,v in p.items()),(label,path)
                else:
                    np.savez_compressed(pred,**p)
                    atomic(src/(tag+'_identity.json'),dict(source=ev(path/'terminal.json'),records=ev(path/'records.json'),seed=seed,input=arm,model=label,GT_join=True,ROI='original official common'))
                save_stats(dest,m,rows,p,label,tag,review,load(src/(tag+'_statistics.json')) if review else None)
                done.append(dict(role=role,method=label,predictions=ev(pred)))
                atomic(dest/'progress.json',dict(created_at=now(),completed=len(done),last=tag))
        status='complete_independently_reviewed' if review else 'complete_light_checked'
        atomic(dest/'terminal.json',dict(created_at=now(),status=status,completed=done,matching_and_statistic_independent=review,training_shared=True))
        record(name,dict(status=status,terminal=ev(dest/'terminal.json'),owner=None,resource_owner=False))
    except Exception as e:
        atomic(dest/'terminal.json',dict(created_at=now(),status='failed_retained',error=repr(e),traceback=traceback.format_exc()));record(name,dict(status='failed_retained',terminal=ev(dest/'terminal.json'),owner=None,resource_owner=False));raise

def read():
    name='R7_preregistered_read';dest=BASE/name;done=[]
    try:
        for s in SEEDS:wait(BASE/f'R7_analysis_seed{s}_independent/terminal.json')
        for role in ('val21','mechanism_holdout14'):
            for pop in POPS:
                for th in (50,70):
                    distributions={};points={}
                    for method in ('clean','original_K4','PUBLIC_finetuned'):
                        dd=[];pp=[]
                        for s in SEEDS:
                            path=BASE/f'R7_analysis_seed{s}';tag=role+'_'+method
                            with np.load(path/(tag+'_draws.npz')) as z:dd.append(z[pop+'_AP'+str(th)].mean(0))
                            pp.append(next(r['AP'] for r in load(path/(tag+'_statistics.json')) if r['population']==pop and r['threshold']==th and r['wait_ms']=='four_wait_mean'))
                        distributions[method]=np.mean(dd,axis=0);points[method]=float(np.mean(pp))
                    for method in ('original_K4','PUBLIC_finetuned'):
                        draw=distributions['clean']-distributions[method];ci=np.quantile(draw,[.025,.975]).tolist()
                        done.append(dict(role=role,population=pop,threshold=th,comparison='clean_minus_'+method,delta_pp=points['clean']-points[method],CI95=ci,
                            clean_AP=points['clean'],reference_AP=points[method],seeds=list(SEEDS),replicates=2000,four_wait_mean=True,seed_average_paired=True))
        def row(role,pop):return next(r for r in done if r['role']==role and r['population']==pop and r['threshold']==70 and r['comparison']=='clean_minus_original_K4')
        a,b=row('val21','colabelled'),row('val21','all');qualifies=a['delta_pp']>=.40 and a['CI95'][0]>0 and b['delta_pp']>=-.20
        atomic(dest/'panels.json',done);atomic(dest/'decision.json',dict(created_at=now(),qualifies_for_main_conversation_discussion=qualifies,automatic_adoption=False,
            main_method_unchanged='CoMET all-ego H1',primary_role='val21',colabelled=a,official=b,hold14_colabelled=row('mechanism_holdout14','colabelled'),
            disposition='main_conversation_discussion_only' if qualifies else 'supplement_no_successor',no_new_learning_followup=True))
        atomic(dest/'terminal.json',dict(created_at=now(),status='complete_independently_reviewed',decision=ev(dest/'decision.json'),panels=ev(dest/'panels.json')))
        record(name,dict(status='complete_independently_reviewed',terminal=ev(dest/'terminal.json'),owner=None,resource_owner=False))
    except Exception as e:
        atomic(dest/'terminal.json',dict(created_at=now(),status='failed_retained',error=repr(e),traceback=traceback.format_exc()));record(name,dict(status='failed_retained',terminal=ev(dest/'terminal.json'),owner=None,resource_owner=False));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','collect','run','review','read'));p.add_argument('--seed',type=int,choices=SEEDS);a=p.parse_args()
    prepare() if a.mode=='prepare' else collect() if a.mode=='collect' else read() if a.mode=='read' else run(a.seed,a.mode=='review')
