"""Independent saved fold geometry/AP plus paired sequence bootstrap.

No new head fit, detector call, or official split change.
"""
import argparse
import time
import traceback
import numpy as np
from peer16_v1_20261103 import ROOT, BASE, atomic, ev, load, now, register, record
import run_peer16_E7_location_disjoint_v1_20261103 as producer
from review_peer16_E4_E7_AP_v1_20261103 import geometry, check
from bootstrap_peer15_paper_v1_20261103 import from_pred, contrast

OUT=BASE/'E7_location_folds_independent'


def prepare():
    register('E7_location_folds_independent',__file__,
        'Independently rebuild heading/translation and IoU/matching/AP for all12 '
        'already trained leave-one-intersection-out heads and their all-site '
        'controls. Verify metadata disjointness, then paired2000 whole-sequence '
        'bootstrap within each held site. Preserve all outcomes; no new fitting '
        'or official split. Few-sequence sites have weak/degenerate intervals '
        'and do not prove detector geography generalization.',
        dict(GPU_forwards=0,updates=0,CPU_threads=1,wall_h='1-5',disk_MiB='<30'),
        producer=ev(producer.OUT/'terminal.json'),
        scope='Independent heading geometry/polygon IoU/matching/AP; shared '
              'metadata, candidate IDs and saved neural scores. Shared bootstrap '
              'kernel is not independent statistical implementation.')


def run():
    start=time.monotonic();record('E7_location_folds_independent',dict(status='running_frozen_fold_geometry_AP_CI'))
    try:
        data,_,_=producer.old.materials('author')
        m,rows=data['val21']
        info=load(ROOT/'artifacts/data/v2x_seq_spd_main_domain_shared_v1/V2X-Seq-SPD/vehicle-side/data_info.json')
        lookup={str(r['frame_id']):r.get('intersection_loc','') for r in info}
        loc=np.asarray([lookup[str(r['current_vehicle_frame'])] for r in rows])
        fit_sites={lookup[str(r['current_vehicle_frame'])] for r in data['fit24'][1]}
        cal_sites={lookup[str(r['current_vehicle_frame'])] for r in data['calibration8'][1]}
        with np.load(BASE/'E3_temporal_baselines/val21_states.npz') as z:d=z['KF'].copy()
        b=geometry(m,d);gold=load(producer.OUT/'panels.json');result=[];stats=[]
        sites=load(producer.OUT/'location_audit.json')['sites']
        for site in sites:
            identity=load(producer.OUT/site/'identity.json')
            assert identity['fit_sites']==sorted(fit_sites-{site,''})
            assert identity['cal_sites']==sorted(cal_sites-{site,''})
            assert site not in identity['fit_sites'] and site not in identity['cal_sites']
            sm,sb,_=producer.subset(m,loc==site,b,np.zeros((len(b),1)))
            dest=OUT/site;dest.mkdir(exist_ok=False)
            for kind in ('full39','no_position'):
                with np.load(producer.OUT/site/kind/'val_predictions.npz') as z:p={k:z[k].copy() for k in z.files}
                name=site+'_'+kind
                for arm in (name,name+'_allsite_head'):
                    result.extend(dict(r,site=site,kind=kind) for r in check(sm,sb,p,arm,gold))
                exact=all(abs(r['error'+str(th)])<1e-8 and r['TP_mismatch'+str(th)]==0
                          for r in result for th in (50,70))
                assert exact,'fold independent AP/TP discrepancy'
                for th in (50,70):
                    stats.extend(contrast(dest,kind+'_held_minus_allsite',
                        from_pred(sm,p,name,th),from_pred(sm,p,name+'_allsite_head',th),th,
                        dict(site=site,kind=kind,producer=ev(producer.OUT/site/kind/'val_predictions.npz'),
                             scope='Held-site readout diagnostic; few-sequence CI may be degenerate.')))
                atomic(OUT/'independent_AP.json',result);atomic(OUT/'bootstrap_panels.json',stats)
                atomic(OUT/'status.json',dict(created_at=now(),site=site,kind=kind,reviewed_AP_panels=len(result)))
        atomic(OUT/'terminal.json',dict(created_at=now(),status='complete_independently_reviewed',
            AP=ev(OUT/'independent_AP.json'),bootstrap=ev(OUT/'bootstrap_panels.json'),
            GPU_forwards=0,updates=0,heads=12,wall_s=time.monotonic()-start,
            independent_statistics_not_claimed=True,official_split_unchanged=True))
        record('E7_location_folds_independent',dict(status='complete_independently_reviewed',terminal=ev(OUT/'terminal.json')))
    except Exception as exc:
        atomic(OUT/'terminal.json',dict(created_at=now(),status='failed_retained',error=repr(exc),traceback=traceback.format_exc()))
        record('E7_location_folds_independent',dict(status='failed_retained',terminal=ev(OUT/'terminal.json')))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run'])
    globals()[p.parse_args().mode]()
