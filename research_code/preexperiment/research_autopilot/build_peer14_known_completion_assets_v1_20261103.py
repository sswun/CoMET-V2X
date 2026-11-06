"""Accept now-complete known native/source consumers; figure-only layout update."""
import inspect,csv
from peer14_cycle12_v1_20261103 import ROOT,CYCLE,C09,C06,C21,atomic,ev,load,now,register,record
import analyze_peer14_cycle12_factors_v1_20261103 as factors
import analyze_peer14_native_physical_tables_v1_20261103 as physical
BASE=CYCLE/'known_completion_assets_v1'

def run():
    BASE.mkdir(exist_ok=False)
    register('known_completion_assets',__file__,'Close snapshot-only gaps after own source18 and native-dose18 terminals, without repeating inference/IoU. Keep old17/18 snapshots; new compact table versions explicitly full. Move plot legend outside data, no numbers changed.',dict(GPU_forwards=0,CPU_threads=1,new_disk_MiB='<10 compact source/native table versions and figure'))
    assert load(C06/'composed_consumers_v1/sources/terminal.json')['status']=='complete_light_checked'
    assert load(C21/'native_dose_response_v1/producer/terminal.json')['status']=='complete_light_checked'
    source,missing=factors.source_rows();assert not missing and len(source)==4320
    with (BASE/'CoAlign_full18_source_all4320.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(source[0]));w.writeheader();w.writerows(source)
    ns=dict(physical.__dict__,CYCLE=BASE,__file__=__file__)
    s=inspect.getsource(physical.run).replace("'native_physical_tables'","'native_physical_final_tables'")
    exec(compile(s,__file__+'::completed_native18_tables','exec'),ns);ns['run']()
    final=load(BASE/'native_physical_synthesis.json');assert final['B_native18_completed']==18 and final['B_backbone_completed']==21 and not final['B_native18_missing']
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    summary=load(CYCLE/'scientific_factor_synthesis.json')['graph_pair_summary']
    contrasts=['K1_GRAPH minus PUBLIC_GRAPH','K1_GRAPH minus K1_INPUT','K2_GRAPH minus K1_GRAPH','ICP_GRAPH_REFERENCE minus K1_GRAPH'];labels=['K1 init vs PUBLIC init','Author graph added to K1','K2 init vs K1 init','Sync. ICP ref vs K1 init']
    fig,axes=plt.subplots(1,2,figsize=(11,4),sharey=True)
    for ax,th in zip(axes,(50,70)):
        for train,offset,color in [('public',-.13,'#3269a8'),('online',.13,'#cd7c29')]:
            rr=[next(r for r in summary if (r['training'],r['contrast'],r['threshold'])==(train,k,th)) for k in contrasts]
            ax.errorbar([r['mean'] for r in rr],[i+offset for i in range(4)],xerr=[r['sample_SD'] for r in rr],fmt='o',capsize=3,color=color,ms=4,label=train.upper()+' weights')
        ax.axvline(0,color='black',lw=.7);ax.grid(axis='x',alpha=.2);ax.set_yticks(range(4),labels);ax.set_xlabel(f'Four-wait AP{th} difference (pp), mean ± seed SD')
    axes[0].invert_yaxis();fig.suptitle('Fixed-weight CoAlign: complete uncertainty → author graph → fusion (3 paired seeds)',fontsize=11,y=.98)
    handles,legends=axes[0].get_legend_handles_labels();fig.legend(handles,legends,ncol=2,loc='upper center',bbox_to_anchor=(.6,.915),frameon=False,fontsize=9)
    fig.tight_layout(rect=[0,0,1,.84]);fig.savefig(BASE/'CoAlign_complete_graph_factors_v2.pdf');fig.savefig(BASE/'CoAlign_complete_graph_factors_v2.png',dpi=160);plt.close(fig)
    atomic(CYCLE/'paper_assets_final_addendum.json',dict(created_at=now(),status='complete_light_checked',prior_assets=ev(CYCLE/'paper_assets_manifest.json'),source18_full=ev(BASE/'CoAlign_full18_source_all4320.csv'),native18_backbone21=ev(BASE/'native_physical_synthesis.json'),factor_figure_v2=ev(BASE/'CoAlign_complete_graph_factors_v2.pdf'),factor_data_unchanged=ev(CYCLE/'CoAlign_completed_composed_graph_paired_factors.csv'),scope='All18 composed source producers light checked, NOT all4320 full independent accepted. All39 native A and native18+Where/ViT21 B completed light checked. Quality subset vs label-dose/official populations separate. Old17 snapshots retained.',original_release_closed=False,GPU_forwards=0))
    record('known_completion_assets',dict(status='complete_light_checked',terminal=ev(CYCLE/'paper_assets_final_addendum.json')))

if __name__=='__main__':run()
