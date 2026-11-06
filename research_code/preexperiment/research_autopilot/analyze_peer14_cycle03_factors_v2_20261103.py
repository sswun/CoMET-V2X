"""Compare actual shared GT content, not different producer filenames."""
import hashlib
from pathlib import Path
import numpy as np
from peer14_cycle03_v1_20261103 import ROOT,CYCLE,atomic,ev,now,record
import analyze_peer14_cycle03_factors_v1_20261103 as original
CACHE={}

def GT_identity(path):
    if path not in CACHE:
        sha=hashlib.sha256();count=0
        with np.load(ROOT/path) as z:
            keys=sorted(k for k in z.files if k.endswith('_gt'))
            assert len(keys)==3316
            for key in keys:
                a=z[key];sha.update(key.encode());sha.update(str((a.shape,a.dtype.str)).encode());sha.update(a.tobytes());count+=len(a)
        CACHE[path]=dict(GT_only_SHA256=sha.hexdigest(),cases=len(keys),targets=count)
    return CACHE[path]['GT_only_SHA256']

if __name__=='__main__':
    atomic(CYCLE/'factor_analysis_v1_failure.json',dict(created_at=now(),status='failed_retained',reason='Path-string equality rejected different native prediction shards containing the same shared GT; no factor published. V2 compares per-array dtype/shape/bytes plus condition IDs.',script=ev(original.__file__),GPU_forwards=0))
    record('factor_analysis',dict(status='failed_retained',terminal=ev(CYCLE/'factor_analysis_v1_failure.json')))
    source=Path(original.__file__).read_text()
    replacements={"r['GT_shard'],r['GT_case']":"GT_identity(r['GT_shard']),r['GT_case']",
        "'factor_analysis'":"'factor_analysis_recovery_v2'",
        'CoAlign_three_seed_fair_graph_plugin_upper_factors.csv':'CoAlign_three_seed_fair_graph_plugin_upper_factors_v2.csv',
        'CoAlign_three_seed_factor_synthesis.json':'CoAlign_three_seed_factor_synthesis_v2.json',
        'SPD_annotation_dose_fixed_predictions.pdf':'SPD_annotation_dose_fixed_predictions_v2.pdf',
        'SPD_annotation_dose_fixed_predictions.png':'SPD_annotation_dose_fixed_predictions_v2.png',
        'figure_identity.json':'figure_identity_v2.json'}
    for a,b in replacements.items():assert a in source,a;source=source.replace(a,b)
    ns=dict(__name__='factor_recovery',__file__=str(Path(__file__).resolve()),GT_identity=GT_identity)
    exec(compile(source,__file__+'::content_identity','exec'),ns);ns['run']()
    atomic(CYCLE/'factor_GT_content_identity_audit.json',dict(created_at=now(),sources=CACHE,source_script=ev(original.__file__),repair_script=ev(__file__),scope='Only GT arrays hashed, not prediction arrays; exact case identifiers, sizes, dtypes and bytes. No model rerun.'))
