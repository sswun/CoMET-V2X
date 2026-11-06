"""Full e60/PUBLIC/ONLINE sources for registered finite diagnostic cells."""
import sys
from peer12_cycle03_v1_20261102 import CYCLE,atomic,ev,load,now
from run_peer12_cycle03_recovery_v2_20261102 import adapted
def main():
 remote='--remote' in sys.argv;s=adapted();old="for name in ('public','online_k1'):";assert s.count(old)==1;s=s.replace(old,"for name in ('e60','public','online_k1'):")
 s=s.replace("out=CYCLE/('source_remote' if remote else 'sources03_v2')","out=CYCLE/('source_remote_all' if remote else 'source_local_all')")
 ns={'__name__':'peer12_sources_all_references','__file__':__file__};exec(compile(s,__file__+'::all3weights','exec'),ns);ns['source_lane'](remote)
if __name__=='__main__':main()
