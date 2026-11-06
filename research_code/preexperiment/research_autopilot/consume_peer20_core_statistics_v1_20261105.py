"""Paired population contrasts/A1 from shared whole-sequence bootstrap draws.

Never subtract CI endpoints. All compared methods share 2000 draws by design.
Ratios preserve negative denominators and poles; nonfinite draws are counted,
not silently converted to zero. Matched-readout EGO is not native ego.
"""
import argparse,traceback
import numpy as np
from peer20_population_core_v1_20261105 import *
import run_peer20_population_v1_20261105 as core

def prepare():
    for domain in h1.recipe.DOMAINS:
        register('R6bc_paired_'+domain,__file__,__doc__,dict(GPU_forwards=0,updates=0,CPU_threads=1,replicates=2000,disk_MiB='<12'),
            domain=domain,producer=ev(BASE/('R6b_'+domain)/'contract.json'),independent_prerequisite=ev(BASE/('R6b_'+domain+'_independent')/'contract.json'),
            scope='A1 ratios for both annotation denominators plus all core pairwise requested contrasts. Auxiliary descriptive only.')

def get(root,role,method,seed):
    tag=role+'_'+method+'_seed'+str(seed)
    with np.load(root/(tag+'_draws.npz')) as z:draws={k:z[k] for k in z.files}
    values={(r['population'],r['threshold'],r['wait_ms']):r['AP'] for r in load(root/(tag+'_statistics.json'))}
    return draws,values

def run(domain):
    name='R6bc_paired_'+domain;dest=BASE/name;src=BASE/('R6b_'+domain);contrasts=[];ratios=[]
    try:
        wait(BASE/('R6b_'+domain+'_independent')/'terminal.json')
        record(name,dict(status='running_paired_contrasts_and_A1',resource_owner=True))
        for role in ROLES:
            for seed in h1.recipe.SEEDS:
                loaded={k:get(src,role,k,seed) for k in core.METHODS}
                for a,b in [('H1',m) for m in core.METHODS if m!='H1']+[('OFFICIAL_filled','OFFICIAL'),('MA5_filled','MA5')]:
                    for pop in POPS:
                        for th in (50,70):
                            key=pop+'_AP'+str(th);draw=loaded[a][0][key]-loaded[b][0][key];assert draw.shape==(4,2000)
                            for iw,w in enumerate((*WAITS,'four_wait_mean')):
                                dd=draw[iw] if iw<4 else draw.mean(0);point=loaded[a][1][pop,th,w]-loaded[b][1][pop,th,w]
                                ci=np.quantile(dd,[.025,.975]).tolist()
                                contrasts.append(dict(role=role,domain=domain,seed=seed,population=pop,threshold=th,wait_ms=w,
                                    comparison=a+'_minus_'+b,delta_pp=point,CI95=ci,replicates=2000,paired_draws=True,
                                    independent_AP_bootstrap_upstream=True,saved_draw_subtraction_not_CI_subtraction=True))
                for num in ('PUBLIC','H1'):
                    for den in ('OFFICIAL','MA5','OFFICIAL_filled','MA5_filled'):
                        for pop in POPS:
                            for th in (50,70):
                                key=pop+'_AP'+str(th);ego=loaded['EGO'][0][key];a=loaded[num][0][key]-ego;b=loaded[den][0][key]-ego
                                for iw,w in enumerate((*WAITS,'four_wait_mean')):
                                    aa=a[iw] if iw<4 else a.mean(0);bb=b[iw] if iw<4 else b.mean(0)
                                    valid=bb!=0;dd=aa[valid]/bb[valid]
                                    pp=loaded[num][1][pop,th,w]-loaded['EGO'][1][pop,th,w];qq=loaded[den][1][pop,th,w]-loaded['EGO'][1][pop,th,w]
                                    ratios.append(dict(domain=domain,role=role,seed=seed,population=pop,threshold=th,wait_ms=w,
                                        numerator=num,denominator=den,ratio=float(pp/qq) if qq else None,
                                        numerator_gap_pp=pp,denominator_gap_pp=qq,CI95=np.quantile(dd,[.025,.975]).tolist() if len(dd) else None,
                                        zero_denominator_draws=int((~valid).sum()),negative_denominator_draws=int((bb<0).sum()),replicates=2000,
                                        reference='Annotation-assisted denominator, not physical truth',EGO='Same matched readout diagnostic, not native system'))
            atomic(dest/'contrasts.json',contrasts);atomic(dest/'A1_retention.json',ratios)
        atomic(dest/'terminal.json',dict(created_at=now(),status='complete_independently_reviewed',
            contrasts=ev(dest/'contrasts.json'),A1=ev(dest/'A1_retention.json'),scope='Independent upstream AP/draws, direct paired arithmetic; not independent neural training.'))
        record(name,dict(status='complete_independently_reviewed',terminal=ev(dest/'terminal.json'),owner=None,resource_owner=False))
    except Exception as e:
        atomic(dest/'terminal.json',dict(created_at=now(),status='failed_retained',error=repr(e),traceback=traceback.format_exc()));record(name,dict(status='failed_retained',terminal=ev(dest/'terminal.json'),owner=None,resource_owner=False));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','run'));p.add_argument('--domain',choices=h1.recipe.DOMAINS);a=p.parse_args()
    prepare() if a.mode=='prepare' else run(a.domain)
