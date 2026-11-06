"""Fixed seed1 50/50 PUBLIC/ONLINE training, followed by val and hold."""
import argparse,sys
from peer10_tournament_v1_20261101 import ROOT,atomic,ev,load,now
from peer12_cycle00_v1_20261102 import CYCLE
from run_peer12_native_additions_v1_20261102 import replace,run_cell

def mixed_flags(np,seed,count):return np.random.RandomState(seed^0x50454552).random_sample(count)<.5

def install_mixed(ds,np,arm,online,official):
 assert arm=='MIXED_PUBLIC_ONLINE';original=ds.retrieve_base_data;ds.peer12_mix_online=False
 def retrieve(index,cur_ego_pose_flag=False):
  data=original(index,cur_ego_pose_flag=cur_ego_pose_flag);a=ds.native4_audit;key=(str(a['sequence']),int(a['decision_us']),int(a['wait_ms']));assert key in online
  d=online[key] if ds.peer12_mix_online else np.zeros(2)
  if not a['no_packet_fallback']:
   remote=next(k for k,v in data.items() if isinstance(k,int) and not v['ego']);m=np.eye(4);m[:2,3]=d;data[remote]['params']['transformation_matrix']=m@np.asarray(data[remote]['params']['transformation_matrix'])
  a['k4_correction']={'arm':'ONLINE_K1' if ds.peer12_mix_online else 'PUBLIC','delta_xy':d.tolist(),'annotation_assisted':False,'mixed_pose_ablation':True};return data
 ds.retrieve_base_data=retrieve

def training_source():
 c=load(CYCLE/'contract.json');ref=c['training_source'];assert ev(ROOT/ref['path'])==ref
 return replace((ROOT/ref['path']).read_text(),[("choices=['PUBLIC','ONLINE_K1','OFFICIAL_REFERENCE']","choices=['MIXED_PUBLIC_ONLINE']"),('install_correction(ds,np,a.arm,online,official)','install_mixed(ds,np,a.arm,online,official)'),('torch.set_num_threads(1)','torch.set_num_threads(2)'),("ledger=(out/'training.jsonl')", "mix=mixed_flags(np,c['seed'],c['updates']*c['gradient_accumulation']);atomic(out/'mixed_assignment.json',{'seed':c['seed'],'bernoulli_p':.5,'online_microbatches':int(mix.sum()),'public_microbatches':int((~mix).sum()),'assignment_sha256':hashlib.sha256(mix.tobytes()).hexdigest(),'input_choice_independent_of_pointcloud_RNG':True,'ablation_only_no_val_selection':True});ledger=(out/'training.jsonl')"),('ds=datasets[wait];np.random.seed', 'ds=datasets[wait];ds.peer12_mix_online=bool(mix[micro]);np.random.seed')])

def main():
 p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);a=p.parse_args();out=CYCLE/'mixed00';out.mkdir(exist_ok=False);c=load(CYCLE/'contract.json');assert ev(ROOT/c['mixed_contract']['path'])==c['mixed_contract']
 sys.argv=[__file__,'--workspace',str(ROOT),'--contract',str(CYCLE/'mixed_training_contract.json'),'--output',str(out/'training'),'--arm','MIXED_PUBLIC_ONLINE','--gpu',str(a.gpu)]
 ns={'__name__':'peer12_mixed_fixed_training','__file__':__file__,'mixed_flags':mixed_flags,'install_mixed':install_mixed};exec(compile(training_source(),__file__+'::fixed_training','exec'),ns);ns['main']()
 t=load(out/'training/terminal.json');assert t['status']=='training_complete_light_checked' and t['updates']==2000 and t['microbatches']==8000
 weight=out/'training/final.pth';run_cell('mixed',weight,a.gpu,out/'val21');run_cell('mixed',weight,a.gpu,out/'hold14',True)
 atomic(out/'terminal.json',{'status':'complete_light_checked','created_at':now(),'training':ev(out/'training/terminal.json'),'weight':ev(weight),'val21':ev(out/'val21/terminal.json'),'hold14':ev(out/'hold14/terminal.json'),'ablation_only_no_val_recipe_selection':True,'no_model_wakeup':True})
if __name__=='__main__':main()
