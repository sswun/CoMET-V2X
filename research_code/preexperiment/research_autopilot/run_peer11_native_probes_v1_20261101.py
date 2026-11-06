"""Frozen native evaluator with preregistered ego/annotation-reference inputs.

ego: mask remote voxel slots only at inference, keep original current GT.
upper: noncausal seqmean and centred MA5 of official offsets, never deployment.
"""
import argparse,hashlib,sys
from pathlib import Path
from peer10_tournament_v1_20261101 import ROOT,atomic,ev,load
from peer11_cycle15_v1_20261101 import CYCLE

def upper_inputs(cases,official,np):
 groups={}
 for r in cases:
  raw=official[str(r['current_vehicle_frame'])]['system_error_offset']
  groups.setdefault(str(r['sequence']),{})[int(r['receiver_timestamp_us'])]=np.asarray([raw['delta_x'],raw['delta_y']],np.float64)
 result={}
 for seq,values in groups.items():
  times=sorted(values);v=np.stack([values[t] for t in times]);mean=v.mean(axis=0)
  for i,t in enumerate(times):
   result['OFFICIAL_SEQMEAN',seq,t]=mean
   result['OFFICIAL_MA5',seq,t]=v[max(0,i-2):min(len(v),i+3)].mean(axis=0)
 return result

def ego_input(batch):
 import torch
 import traf_native4_recorded_arrival_20261010 as adapter
 inputs=__import__('copy').deepcopy(adapter.inference_input(batch));voxel=inputs['processed_lidar'];coords=voxel['voxel_coords'];mask=coords[:,0]<4;n=len(coords)
 assert sorted(set(coords[mask,0].tolist()))==list(range(4))
 for k,v in list(voxel.items()):
  assert v.shape[0]==n,(k,v.shape,n)
  voxel[k]=v[mask].clone()
 inputs['record_len']=torch.full_like(inputs['record_len'],4)
 delays=inputs['time_delays']
 if torch.is_tensor(delays):delays[...,1]=0
 else:
  for row in delays:row[1]=0
 assert int(inputs['record_len'][0])==4 and int(voxel['voxel_coords'][:,0].max())==3
 ident={k:hashlib.sha256(v.numpy().tobytes()).hexdigest() for k,v in voxel.items()}
 # Stable per-current-frame ego-input identities must agree across weights.
 return inputs,ident

def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['ego','upper'],required=True);p.add_argument('--workspace',type=Path,required=True);p.add_argument('--contract',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--weight',type=Path,required=True);p.add_argument('--gpu',type=int,required=True);p.add_argument('--compile-check',action='store_true');a=p.parse_args()
 contract=load(a.contract);ref=contract['native_probe']['source'];assert ev(ROOT/ref['path'])==ref
 source=(ROOT/ref['path']).read_text();changes=[]
 if a.mode=='ego':
  changes=[("elif arm!='PUBLIC':raise ValueError(arm)","elif arm not in ('PUBLIC','EGO_ONLY'):raise ValueError(arm)"),
   ('for wait in (0,100,200,400):','for wait in (0,):'),
   ('inputs=copy.deepcopy(adapter.inference_input(batch));',"inputs,ego_identity=ego_input(batch);atomic(out/f'ego_input_{index}.json',ego_identity) if ordinal in (0,len(selected)-1) else None;")]
  arms='EGO_ONLY'
 else:
  changes=[('maps={arm:{} for arm in arms}','upper=upper_inputs(cases,official,np);maps={arm:{} for arm in arms}'),
   ("elif arm!='PUBLIC':raise ValueError(arm)","elif arm in ('OFFICIAL_SEQMEAN','OFFICIAL_MA5'):matrix[:2,3]=upper[arm,key[0],key[1]]\n                elif arm!='PUBLIC':raise ValueError(arm)")]
  arms='OFFICIAL_SEQMEAN,OFFICIAL_MA5'
 for old,new in changes:
  assert source.count(old)==1,(old,source.count(old));source=source.replace(old,new)
 source=source.replace("if refresh.exists():subprocess.run", "if False and refresh.exists():subprocess.run")
 # The actual evaluator writes its own argv-derived contract/weight identity.
 sys.argv=[str(__file__),'--workspace',str(a.workspace),'--contract',str(a.contract),'--output',str(a.output),'--weight',str(a.weight),'--gpu',str(a.gpu),'--arms',arms]
 code=compile(source,__file__+'::'+a.mode,'exec')
 if a.compile_check:print('passed exact-count source compilation: '+a.mode);return
 ns={'__name__':'peer11_frozen_native_probe','__file__':__file__,'upper_inputs':upper_inputs,'ego_input':ego_input};exec(code,ns);ns['main']()
 atomic(a.output/'diagnostic_identity.json',{'mode':a.mode,'source':ref,'adapter':ev(Path(__file__).resolve()),'literal_changes':changes,'ego_current_GT_unchanged':True,'official_upper_annotation_assisted_noncausal':a.mode=='upper','MA5':'centred +/-2 actual val frames within each sequence, clipped ends; label-derived only','not_deployable_method':True,'no_model_wakeup':True})
if __name__=='__main__':main()
