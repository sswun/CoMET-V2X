"""Fresh actual author compiled-IoU call, explicit host CUDA error-slot audit."""
import ctypes,ctypes.util,os,sys,time,traceback
from peer10_tournament_v1_20261101 import ROOT,atomic,ev,now
from pathlib import Path

def main():
    out=Path(sys.argv[1]);out.mkdir(exist_ok=False);start=time.monotonic();calls=[]
    try:
        import torch
        sys.path.insert(0,str(ROOT/'third_party/external/FFNet_VIC3D_20261030'))
        from mmdet3d.ops.iou3d.iou3d_utils import boxes_iou_bev
        torch.cuda.set_device(0);device=torch.device('cuda:0')
        # Resolve the already-loaded runtime, not a different installed CUDA.
        maps=Path('/proc/self/maps').read_text().splitlines();paths=sorted({line.split()[-1] for line in maps if 'libcudart.so' in line and line.split()[-1].startswith('/')})
        assert len(paths)==1,('loaded CUDA runtime identity',paths)
        lib=ctypes.CDLL(paths[0]);lib.cudaGetLastError.restype=ctypes.c_int;lib.cudaPeekAtLastError.restype=ctypes.c_int;lib.cudaGetErrorString.argtypes=[ctypes.c_int];lib.cudaGetErrorString.restype=ctypes.c_char_p
        box=torch.tensor([[0.,0.,4.,2.,0.]],device=device);empty=box[:0].contiguous();torch.cuda.synchronize();lib.cudaGetLastError()
        for name,a,b in [('nonempty',box,box),('empty_gt',box,empty),('empty_pred',empty,box),('both_empty',empty,empty)]:
            assert lib.cudaGetLastError()==0
            result=boxes_iou_bev(a,b);peek=lib.cudaPeekAtLastError();code=lib.cudaGetLastError()
            # Clear the host error before the next torch API: this is the
            # diagnostic distinction from the previous launch-blocking trial.
            torch.cuda.synchronize();values=result.detach().cpu().tolist()
            row={'name':name,'shape':list(result.shape),'peek_error':peek,'error_code':code,'error_string':lib.cudaGetErrorString(code).decode(),'values':values};calls.append(row);atomic(out/'status.json',{'calls':calls,'runtime':paths[0]})
        assert calls[0]['error_code']==0 and abs(calls[0]['values'][0][0]-1.)<1e-5
        assert all(r['error_code']==r['peek_error']==9 for r in calls[1:]),('not zero-grid error9',calls)
        guarded=[]
        for a,b in [(box,box),(box,empty),(empty,box),(empty,empty)]:
            lib.cudaGetLastError();result=boxes_iou_bev(a,b) if len(a) and len(b) else a.new_zeros((len(a),len(b)));code=lib.cudaGetLastError();torch.cuda.synchronize();guarded.append({'shape':list(result.shape),'error_code':code,'values':result.cpu().tolist()})
        assert all(r['error_code']==0 for r in guarded) and guarded[0]['values']==calls[0]['values']
        atomic(out/'terminal.json',{'status':'confirmed_empty_IoU_host_error_slot','calls':calls,'guarded_calls':guarded,'runtime':paths[0],'operator':ev(Path(sys.modules[boxes_iou_bev.__module__].__file__)),'wall_s':time.monotonic()-start,'created_at':now(),'author_model_unchanged':True,'not_formal_AP':True,'no_model_wakeup':True})
    except Exception as e:
        atomic(out/'terminal.json',{'status':'failed_retained','calls':calls,'error':repr(e),'traceback':traceback.format_exc(),'wall_s':time.monotonic()-start,'no_model_wakeup':True});raise
if __name__=='__main__':main()
