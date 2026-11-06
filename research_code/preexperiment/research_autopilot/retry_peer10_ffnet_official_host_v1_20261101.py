"""One authorized Google official-host retry; never invokes an event/model."""
import os,time,traceback,subprocess
from urllib.parse import urlparse,parse_qs,urljoin
import requests
from peer10_tournament_v1_20261101 import ROOT,AUDIT,atomic,ev,load,now,sha
from peer10_cycle03_registration_v1_20261101 import CYCLE,update
from retry_ffnet_author_checkpoint_after_freeze_v1_20261031 import DownloadForm,is_textual,FILE_ID,SOURCE_URL,TARGET

def main():
    out=CYCLE/'ffnet_official_host_retry';out.mkdir(exist_ok=False);c=load(CYCLE/'contract.json')['ffnet_host_retry'];start=time.monotonic()
    state={'status':'running','started_at':now(),'contract':ev(CYCLE/'contract.json'),'responses':[],'requests_made':0,'owner_released':False}
    partial=TARGET.with_suffix('.pth.partial.peer10_official_host_v1');response=None
    try:
        if TARGET.exists() or partial.exists():raise RuntimeError('refuse to overwrite checkpoint or partial')
        session=requests.Session();session.trust_env=False;session.proxies={'http':c['proxy'],'https':c['proxy']};url=SOURCE_URL
        for ordinal in range(1,c['request_limit']+1):
            parsed=urlparse(url)
            if parsed.scheme!='https' or parsed.hostname not in c['allowed_hosts'] or parse_qs(parsed.query).get('id')!=[FILE_ID]:raise RuntimeError('unapproved host/file identity')
            response=session.get(url,stream=True,timeout=(15,120),allow_redirects=False);state['requests_made']=ordinal
            state['responses'].append({'status_code':response.status_code,'url':response.url,'location':response.headers.get('Location'),'content_type':response.headers.get('Content-Type'),'content_length':response.headers.get('Content-Length')});atomic(out/'status.json',state)
            if response.is_redirect:
                url=urljoin(url,response.headers['Location']);response.close();response=None;continue
            response.raise_for_status()
            if is_textual(response):
                body=response.content;bodyfile=out/f'response{ordinal}.html';bodyfile.write_bytes(body);state['responses'][-1]['body']=ev(bodyfile)
                form=DownloadForm();form.feed(body.decode(response.encoding or 'utf-8',errors='replace'));response.close();response=None
                if not form.action or form.inputs.get('id')!=FILE_ID:raise RuntimeError('text quota/error; no valid same-file confirmation form')
                url=requests.Request('GET',form.action,params=form.inputs).prepare().url;continue
            length=response.headers.get('Content-Length');TARGET.parent.mkdir(parents=True,exist_ok=True)
            with partial.open('xb') as stream:
                for block in response.iter_content(1<<20):
                    if block:stream.write(block)
                stream.flush();os.fsync(stream.fileno())
            if partial.stat().st_size<100000 or (length and partial.stat().st_size!=int(length)):raise RuntimeError('checkpoint size mismatch')
            partial.rename(TARGET);state['checkpoint']=ev(TARGET)
            verify=out/'format_verification.json';result=subprocess.run([str(ROOT/'.conda/ffnet-vic3d-20261031/bin/python'),str(ROOT/'preexperiment/research_autopilot/verify_ffnet_checkpoint_format_v1_20261031.py'),'--checkpoint',str(TARGET),'--output',str(verify)],cwd=ROOT)
            state['format_verification']=ev(verify)
            if result.returncode:raise RuntimeError('CPU checkpoint format failed')
            state['status']='author_weight_verified_native_formal_pending';break
        else:raise RuntimeError('one-shot request limit reached')
    except Exception as exc:state.update(status='download_failed_retained',error=repr(exc),traceback=traceback.format_exc())
    finally:
        if response is not None:response.close()
        state.update(completed_at=now(),wall_s=time.monotonic()-start,partial_bytes=partial.stat().st_size if partial.exists() else 0,no_model_wakeup=True)
        atomic(out/'terminal.json',state)
        update('ffnet_host_retry',{'status':state['status'],'terminal':ev(out/'terminal.json'),'next_action':'native formal after verified author weight; otherwise no additional download and retain 11-03 stop-loss'})
        print(state['status'])
if __name__=='__main__':main()
