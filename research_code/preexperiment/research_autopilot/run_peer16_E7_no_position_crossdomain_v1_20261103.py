"""Execute the preregistered no-position readout gate and matched confirmations."""
import argparse
import inspect
import time
import traceback
import numpy as np
from peer16_v1_20261103 import BASE,ROOT,atomic,ev,load,now,register,record
import run_peer16_E7_readout_v1_20261103 as author_control
from run_peer16_cached_methods_v1_20261103 import materials,DOMAINS,observation
import run_peer15_author_recipe_v1_20261103 as author
import run_crosscheckpoint_temporal_matched39d_v1_20261029 as matched
OUT=BASE/'E7_no_position_crossdomain'
AUTHOR=BASE/'E7_readout_v3'

def gate():
    groups={}
    for kind in ('full39','no_position'):
        values=[]
        for seed in (2023,2024,2025):
            rows=load(AUTHOR/f'{kind}_seed{seed}/panels.json')
            values.append(np.mean([r['ap70'] for r in rows if r['role']=='val21']))
        groups[kind]=values
    difference=float(np.mean(groups['no_position'])-np.mean(groups['full39']))
    assert load(BASE/'E4_E7_independent_AP/terminal.json')['status']=='complete_independently_reviewed'
    return dict(preregistered_loss_threshold_pp=.3,AP70_difference_pp=difference,
        paired_three_seed_AP70=groups,adopt_no_position=difference>-.3,
        rule='peer16 E7 preregistration, not newly chosen from observed panels',
        original_full39_results_retained=True,
        integration='No-position is the readout recipe preference; existing full39 tables remain versioned. Cross-domain results not assumed before actual heads/AP. No main.tex automatic edits.')

def prepare():
    decision=gate();assert decision['adopt_no_position']
    register('E7_no_position_crossdomain',__file__,
        'Author3seed independent no-position loss is below preregistered.3pp. '
        'Execute preference, then measure actual strict60/ViT2024/ViT2025/Where '
        'heading/current-q inputs using paired full39 versus no-position3seed '
        'heads with identical per-domain schedule. Fit24/cal8 only, val21 '
        'four waits only, no new native hold, no per-cell selection. '
        'Do not substitute old CV heads for untested current heading heads.',
        dict(GPU_forwards=0,updates=48000,CPU_threads=1,wall_h='finite20h',disk_MiB='<180 heads/scores; geometry shared'),
        decision=decision,independent=ev(BASE/'E4_E7_independent_AP/terminal.json'),
        author_source=ev(AUTHOR/'terminal.json'),domains=list(DOMAINS[1:]),
        removed_features=list(author_control.REMOVED),budget='4domains x2feature arms x3seeds x2000updates',
        boundary='Readout reliability control, not a recursive geometry microvariant; full/no-position paired training is needed for feature attribution.')

def geometry(m,q):
    obs=observation(m);n=np.array([len(z) for _,z in obs]);z=np.array([r.mean(0) if len(r) else np.zeros(2) for _,r in obs])
    delta=author.kf(m,n,z,q,.6)[0];return author.heading.build_heading(m,delta)

def run():
    start=time.monotonic();record('E7_no_position_crossdomain',dict(status='running_preregistered_readout_confirmation'))
    atomic(OUT/'recipe_decision.json',gate());done={};rows=[]
    try:
        for domain in DOMAINS[1:]:
            assert time.monotonic()-start<20*3600
            data,oldhead,q=materials(domain);dest=OUT/domain;dest.mkdir(exist_ok=False)
            fitcal={}
            for role in ('fit24','calibration8'):
                m,_=data[role];b,f=geometry(m,q);fitcal[role]=matched.geometry_material(m,b,f)
            with np.load(oldhead/'training_schedule.npz') as z:batches=z['batches'].copy();weights=z['weights'].copy()
            assert len(weights)==len(fitcal['fit24']['features']) and batches.max()<len(weights)
            schedule=dest/'training_schedule.npz';np.savez_compressed(schedule,weights=weights,batches=batches)
            m,_=data['val21'];b,f=geometry(m,q)
            for kind in ('full39','no_position'):
                fit=dict(fitcal['fit24']);cal=dict(fitcal['calibration8']);vf=f.copy()
                if kind=='no_position':
                    fit['features']=fit['features'].copy();cal['features']=cal['features'].copy()
                    for a in (fit['features'],cal['features'],vf):a[:,author_control.REMOVED]=0
                for seed in (2023,2024,2025):
                    head=dest/f'{kind}_seed{seed}'
                    code=inspect.getsource(matched.train_head).replace('torch.manual_seed(readout.MODEL_SEED)','torch.manual_seed(SEED)').replace('"seed": readout.MODEL_SEED','"seed": SEED')
                    ns=dict(matched.__dict__,SEED=seed);exec(compile(code,__file__+'::paired_fixed_seed_head','exec'),ns)
                    ns['train_head'](fit,schedule,head);matched.source_map(cal,matched.score(cal,head),head)
                    score=author.refit.score_with_head(m,vf,head);name=f'{kind}_seed{seed}'
                    pp,pred=author.refit.fixed.base.evaluate(m,{name:b},{name:score})
                    rows.extend(dict(r,domain=domain,head_seed=seed,kind=kind,q=q) for r in pp)
                    np.savez_compressed(head/'val21_predictions.npz',**pred)
                    atomic(OUT/'panels.json',rows);atomic(OUT/'status.json',dict(created_at=now(),domain=domain,kind=kind,seed=seed,wall_s=time.monotonic()-start))
            atomic(dest/'terminal.json',dict(created_at=now(),status='complete_light_checked',updates=12000,GPU_forwards=0,q=q,independent_pending=True))
            done[domain]=ev(dest/'terminal.json')
        atomic(OUT/'terminal.json',dict(created_at=now(),status='complete_light_checked',completed=done,updates=48000,GPU_forwards=0,wall_s=time.monotonic()-start,independent_pending=True))
        record('E7_no_position_crossdomain',dict(status='complete_light_checked',terminal=ev(OUT/'terminal.json')))
    except Exception as exc:
        atomic(OUT/'terminal.json',dict(created_at=now(),status='failed_retained',error=repr(exc),traceback=traceback.format_exc(),completed=done))
        record('E7_no_position_crossdomain',dict(status='failed_retained',terminal=ev(OUT/'terminal.json')));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run']);globals()[p.parse_args().mode]()
