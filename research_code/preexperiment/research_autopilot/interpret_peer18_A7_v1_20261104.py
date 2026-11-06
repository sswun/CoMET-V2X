"""Audited local call-path evidence, not universal publication/runtime claims."""
import argparse,ast
from peer18_v1_20261104 import *
OUT=BASE/'A7_callpaths'
TRAF='third_party/third_party/related_works/TraF-Align/'
COALIGN='third_party/external/CoAlign_64d1987affe4fb881c439cc498c30ab01bec67e8/'
FF='third_party/external/FFNet_VIC3D_20261030/'
def prepare():
    register('A7_callpaths',__file__,__doc__,dict(detector_forwards=0,updates=0,CPU_threads=1),
        original_scan=ev(BASE/'A7_v2/code_evidence.json'),
        conclusion='DAIR SPD official transform adds offset in vehicle frame; FFNet preprocessing compiles offsets into calibration; CoAlign DAIR dataset constructs corrected partner pose; TraF conditional loc_calib and released DAIR yaml True. No independent original V2X-ViT DAIR default proof.')
def evidence(path,start,end,required):
    p=ROOT/path;lines=p.read_text().splitlines();text='\n'.join(lines[start-1:end])
    assert all(s in text for s in required),(path,start,end)
    # Two distinct parses: requested text span and enclosing syntax tree.
    tree=ast.parse(p.read_text()) if p.suffix=='.py' else None
    functions=[dict(name=n.name,line=n.lineno,end=n.end_lineno) for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.lineno<=start<=n.end_lineno] if tree else []
    return dict(source=ev(p),line_start=start,line_end=end,lines=text,enclosing_functions=functions)
def run():
    rows=[dict(method='DAIR_V2X_Seq_official',classification='inference_dataset_and_transform',
        evidence=[evidence('vendor/DAIR-V2X/v2x/dataset/dair_v2x_for_detection.py',439,445,['system_error_offset']),
            evidence('vendor/DAIR-V2X/v2x/dataset/dataset_utils/frame.py',277,294,['Coord_transformation_spd','self.delta_x']),
            evidence('vendor/DAIR-V2X/v2x/v2x_utils/transformation_utils.py',433,441,['self.delta_x','self.delta_y','new_translationA2C'])],
        conclusion='Checked SPD loader carries per-pair offsets into frame transform; the transform adds translation after world-to-vehicle composition. Does not show every public method used this path.'),
      dict(method='TraF_Align_DAIR',classification='inference_path_conditional_released_config_enabled',
        evidence=[evidence(TRAF+'datasets/Basedataset/v2xseq_basedataset.py',744,767,['system_error_offset','if not self.loc_calib','transformation_matrix']),
            evidence(TRAF+'hypes_yaml/dair_v2x_seq/dair_v2x_seq_Trafalign.yaml',8,13,['loc_calib: True'])],
        conclusion='Delayed partner correction is applied iff loc_calib; checked released DAIR YAML enables it. Code fallback without wild parameters is False, so not unconditional for every configuration.'),
      dict(method='FFNet_V2X_Seq',classification='preprocessing_not_runtime_metadata_read',
        evidence=[evidence(FF+'data/v2x-seq/preprocess.py',56,73,['system_error_offset','inf_lidar2lidar_t']),
            evidence(FF+'data/v2x-seq/preprocess.py',173,184,['system_error_offset','trans_lidar_i2v','calib_lidar_i2v'])],
        conclusion='Checked conversion bakes offset into cooperative lidar_i2v calibration; runtime may read corrected calibration without keyword. Actual published cache provenance not proved solely by converter source.'),
      dict(method='CoAlign_DAIR',classification='dataset_partner_pose_constructor',
        evidence=[evidence(COALIGN+'opencood/data_utils/datasets/basedataset/dairv2x_basedataset.py',93,117,['system_error_offset','inf_side_rot_and_trans_to_trasnformation_matrix','lidar_pose']),
            evidence(COALIGN+'opencood/utils/transformation_utils.py',469,477,['system_error_offset','translation[0][0]','translation[1][0]'])],
        conclusion='Checked DAIR dataset applies offset to infrastructure world translation before pose conversion. This is not a separate proof of original V2X-ViT DAIR published settings.'),
      dict(method='original_V2X_ViT_DAIR',classification='not_established_by_local_source',evidence=[],
        conclusion='Local OpenCOOD-main keyword absence does not prove absence of corrected preprocessing; do not make method-specific or universal default claim without its actual DAIR adapter/config provenance.')]
    atomic(OUT/'callpath_evidence.json',dict(created_at=now(),rows=rows,
        allowed_prose='The checked DAIR official/TraF/CoAlign implementations or FFNet preprocessing support offline offset corrections, under the stated paths/configurations.',
        disallowed_prose='All public cooperative-perception results unconditionally use corrected pair poses.'))
    atomic(OUT/'terminal.json',dict(created_at=now(),status='complete_light_checked',evidence=ev(OUT/'callpath_evidence.json'),
        scope='Human source-path interpretation plus literal-span/AST verification; not dynamic execution or independent replication. Independent source-span checker remains explicit.',GPU_forwards=0))
    record('A7_callpaths',dict(status='complete_light_checked',terminal=ev(OUT/'terminal.json'),owner=None,resource_owner=False))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run']);globals()[p.parse_args().mode]()
