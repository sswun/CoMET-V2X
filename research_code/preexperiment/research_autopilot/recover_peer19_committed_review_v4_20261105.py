"""Resume only missing independent AP/CI panels after peer20 memory abort.

Frozen scientific kernels unchanged; committed AP groups and complete contrast
draws are reused, with prefix/array identity checks. Original aborts retained.
"""
import argparse, inspect, time
import numpy as np
from peer20_v1_20261105 import *
import review_peer19_recipe_v3_20261104 as old
import recover_peer19_recipe_interface_v2_20261104 as interface

def destination(domain):
    return P19/('recipe_independent_interface_recovery_v2' if domain=='strict60' else 'recipe_independent')/domain

def prepare():
    for d in old.prod.DOMAINS:
        dest=destination(d)
        register('R1_review_recovery_'+d,__file__,__doc__,dict(GPU_forwards=0,updates=0,CPU_threads=1,
            replicates=2000,disk_MiB='<60 missing draws, no prediction duplication'),
            original_script=ev(old.__file__),domain=d,canonical_output=str(dest),
            committed_prefix={n:ev(dest/n) for n in ('AP.json','bootstrap.json') if (dest/n).exists()},
            policy='No restart of completed panels; finish interrupted reviewer, no new science or deadline')

def run(domain):
    name='R1_review_recovery_'+domain;dest=destination(domain)
    assert not (dest/'terminal.json').exists(), 'Never replace an original terminal'
    interface.setup()
    old.OUT=dest.parent
    committed_AP=load(dest/'AP.json') if (dest/'AP.json').exists() else []
    committed_CI=load(dest/'bootstrap.json') if (dest/'bootstrap.json').exists() else []
    atomic(BASE/name/'prefix.json',dict(created_at=now(),AP=ev(dest/'AP.json'),bootstrap=ev(dest/'bootstrap.json'),
        AP_panels=len(committed_AP),CI_panels=len(committed_CI),kernel=ev(old.__file__)))
    raw_contrast=old.contrast
    def contrast(m,p,method,q,baseline,label,identity,output,done):
        existing=[r for r in done if r['comparison']==label]
        if existing:
            assert len(existing)==10
            assert {(r['threshold'],r['wait_ms']) for r in existing}=={(t,w) for t in (50,70) for w in (0,100,200,400,'four_wait_mean')}
            assert all(r['identity']==identity for r in existing)
            for th in (50,70):
                with np.load(output/(label+'_AP'+str(th)+'.npz')) as z:
                    for k in ('wait0','wait100','wait200','wait400','four_wait_mean'):assert z[k].shape==(2000,)
            return
        raw_contrast(m,p,method,q,baseline,label,identity,output,done)
    source=inspect.getsource(old.run)
    source=source.replace('dest.mkdir(parents=True,exist_ok=False)','dest.mkdir(parents=True,exist_ok=True)')
    source=source.replace('allchecks=[];boots=[]','allchecks=list(committed_AP);boots=list(committed_CI)')
    marker='            boxes=geometry(m,delta)'
    assert source.count(marker)==1
    source=source.replace(marker,"""            existing=[r for r in allchecks if r['method']==label and all(r.get(k)==v for k,v in tags.items())]
            if existing:
                assert len(existing)==4 and {r['wait_ms'] for r in existing}=={0,100,200,400}
                assert all(abs(r['error'+str(t)])<1e-8 and r['TP_mismatch'+str(t)]==0 for r in existing for t in (50,70))
                return
"""+marker)
    ns=dict(old.__dict__,committed_AP=committed_AP,committed_CI=committed_CI,contrast=contrast,
        record=lambda unused,values:record(name,values))
    record(name,dict(status='running_committed_prefix_recovery',resource_owner=True))
    exec(compile(source,__file__+'::unchanged_review_missing_panels','exec'),ns);ns['run'](domain)
    assert load(dest/'AP.json')[:len(committed_AP)]==committed_AP
    assert load(dest/'bootstrap.json')[:len(committed_CI)]==committed_CI
    atomic(BASE/name/'terminal.json',dict(created_at=now(),status='complete_independently_reviewed',
        canonical_terminal=ev(dest/'terminal.json'),committed_prefix_retained=True,
        reused_AP_panels=len(committed_AP),reused_CI_panels=len(committed_CI),scientific_kernel_unchanged=True))
    record(name,dict(status='complete_independently_reviewed',terminal=ev(BASE/name/'terminal.json'),owner=None,resource_owner=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','run'));p.add_argument('--domain',choices=old.prod.DOMAINS)
    a=p.parse_args();prepare() if a.mode=='prepare' else run(a.domain)
