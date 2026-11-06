"""18h peer13 registrations, references and bounded cross-host receipts."""
from peer13_cycle15_v1_20261102 import ROOT,AUTO,AUDIT,CYCLE as C15,C09,C12,atomic,ev,load,now,update,task
CYCLE=AUDIT/'cycle18_peer13_v1_20261102'
def prepare():
    refs={}
    for name in ('current_handoff.md','main_chat_directive.md'):
        (AUTO/name).read_text();refs[name]=ev(AUTO/name)
    CYCLE.mkdir(exist_ok=False)
    atomic(CYCLE/'entry.json',dict(created_at=now(),instructions=refs,script=ev(__file__),official_protocol_unchanged=True,no_model_wakeup=True))
    atomic(CYCLE/'storage_entry.json',load(ROOT/'artifacts/maintenance/storage_status/latest.json'))
if __name__=='__main__':prepare()
