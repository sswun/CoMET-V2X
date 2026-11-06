"""Small evidence/status updates only; cannot invoke or schedule a model."""
import fcntl,json,subprocess,sys
from pathlib import Path
from peer10_tournament_v1_20261101 import ROOT,AUDIT,AUTO,TASKS,task,now,atomic,ev,load
def read(p):return load(p) if p.exists() else None
def main():
    status={s:read(AUDIT/s/'status.json') for s in ('k4_public','k4_online','k4_official','k2_k3_formal_resume_v3','cpu_matched','auxiliary_resume_v3','crossbackbone_se2_public_pivot_v2')}
    for name in status:
        terminal=read(AUDIT/name/'terminal_v2.json') or read(AUDIT/name/'terminal.json')
        if terminal and terminal.get('status') not in ('failed_retained',):
            status[name]={**(status[name] or {}),'status':terminal['status'],'terminal_created_at':terminal.get('created_at'),'terminal_overrides_older_progress':True}
    status['K5_labels']=read(AUDIT/'k5_labels_recovery_v2/terminal_v2.json');status['K5_original_identity']=read(AUDIT/'k5_original_identity_audit/terminal.json')
    cycle09=AUDIT/'cycle09_v1';dimension_error=read(cycle09/'auxiliary_dimension_error_audit.json')
    if dimension_error:
        status['auxiliary_resume_v3']={**(status['auxiliary_resume_v3'] or {}),'status':'auxiliary_AP_invalid_LWH_HWL_adapter','invalidation':ev(cycle09/'auxiliary_dimension_error_audit.json'),'official_AP_unaffected':True}
    freeze=read(AUDIT/'k2_k3_cached_recovery_v2/fit_freeze.json');panels=(read(AUDIT/'k2_k3_formal_resume_v3/panels.json') or [])+(read(AUDIT/'k2_k3_formal_resume_v3/matched_panels.json') or [])
    new_results={}
    native_results={}
    native_dirs=['k2_native','k4_eval_public','k4_eval_online_k1','k4_eval_official_reference','seed2_public/evaluation','seed2_online_k1/evaluation','seed2_official_reference/evaluation',
                 'cycle15_peer11_v1/e60_val','cycle15_peer11_v1/seed3_public/evaluation','cycle15_peer11_v1/seed3_online/evaluation']
    for name in native_dirs:
        terminal=read(AUDIT/name/'terminal.json')
        if terminal and terminal.get('status')=='native_AP_complete_light_checked':
            native_results[name]={'terminal':ev(AUDIT/name/'terminal.json'),'panels':read(AUDIT/name/'panels.json'),'protocol':'strict native4 val21 author native ROI; common/source completion reported separately','review_level':'light finite/replay/frozen weight; independent review pending if top3'}
    for method in ('K1_exact','K2_infra','K2_own_matched_U','K3_K1_expanded','K3_K2_expanded'):
        new_results[method]={}
        for role in ('val21','mechanism_holdout14'):
            rows=sorted([r for r in panels if r['method']==method and r['role']==role and r['source_group']=='all'],key=lambda r:r['wait_ms'])
            if rows:new_results[method][role]={'waits':[r['wait_ms'] for r in rows],'ap70':[r['ap70'] for r in rows],'ap50':[r['ap50'] for r in rows],'review_level':'exploratory producer; lifecycle review final pending'}
    pending=read(AUTO/'event_dispatch/state.json') or {};q=read(AUTO/'deadline_20261008/decision_queue.json') or {}
    cross=read(AUDIT/'crossbackbone_se2_public_pivot_v2/panels.json') or []
    cycle=AUDIT/'cycle03_v1';v2=read(cycle/'v2_transfer_float_recovery_v2/terminal.json');v2panels=read(cycle/'v2_transfer_float_recovery_v2/panels.json') or []
    native_source_panels={}
    for name in native_dirs:
        path=cycle/'native_source_aux'/name
        if read(path/'terminal.json') and read(path/'terminal.json').get('status')=='complete_light_checked':
            native_source_panels[name]={'terminal':ev(path/'terminal.json'),'panels':read(path/'panels.json'),'boundary':'fixed original official population auxiliary sensitivity; not auxiliary-centre ROI remasking; not whole K5 release'}
    sources=[]
    for method in ('K2_infra','K2_own_matched_U','K3_K1_expanded','K3_K2_expanded'):
        for role in ('val21','mechanism_holdout14'):
            for group in ('all','veh','coop','inf','veh_coop'):
                left={r['wait_ms']:r for r in panels if r['method']==method and r['role']==role and r['source_group']==group}
                right={r['wait_ms']:r for r in panels if r['method']=='K1_exact' and r['role']==role and r['source_group']==group}
                if len(left)==len(right)==4:sources.append({'method':method,'role':role,'source_group':group,'mean_AP70_delta_pp':sum(left[w]['ap70']-right[w]['ap70'] for w in left)/4})
    crosssummary=[]
    for domain in sorted({r['domain'] for r in cross}):
        for method in ('K2_infra','K3_K1_expanded','K3_K2_expanded'):
            table={(r['role'],r['wait_ms'],r['method']):r for r in cross if r['domain']==domain and r['source_group']=='all'}
            diffs=[r['ap70']-table[(role,w,'K1_exact')]['ap70'] for (role,w,m),r in table.items() if m==method and (role,w,'K1_exact') in table]
            if diffs:crosssummary.append({'domain':domain,'method':method,'cells':len(diffs),'positive':sum(x>0 for x in diffs),'negative':sum(x<0 for x in diffs),'mean_AP70_delta_pp':sum(diffs)/len(diffs),'CPU_cached_not_native':True})
    value={'created_at':now(),'status':status,'fit_selected':{k:freeze.get(k) for k in ('theta_prior_degrees','K3','selection_scores','K3_selection_scores')} if freeze else None,'new_formal_all_GT':new_results,'new_native_results':native_results,'official_source_contrasts':sources,'crossbackbone_current_results':crosssummary,'V2_current_recovery_terminal':v2,'V2_current_panels':v2panels,'FFNet_author_download_terminal':read(cycle/'ffnet_official_host_retry/terminal.json'),'active_owners':q.get('autonomy_active_owners'),'event_pending_count':len(pending.get('pending',[])),'model_wakeup':'3h timer only; this script cannot invoke model','remaining_release_open':['full independent top3 review','K2/K3 Where/ViT actual-input and V2V4Real','all-external and native auxiliary/source panels','true zero-theta-prior control'],'scoreboard':str((ROOT/'docs/innovation_candidate_scoreboard.md').relative_to(ROOT))};atomic(AUDIT/'scoreboard_latest.json',value)
    doc=ROOT/'docs/innovation_candidate_scoreboard.md'
    if not doc.exists():return
    block='<!-- PEER10_GENERATED_LIVE_BEGIN -->\n## 自动小状态（'+value['created_at']+'）\n\n'
    if freeze:block+='K2 fit 选择 θ 先验 '+str(freeze['theta_prior_degrees'])+'°；K3 选择 '+freeze['K3']+'。只按 fit 选择，正式与辅助 AP 不逐格选法。\n\n'
    block+='| 本机阶段 | 实际状态 / 更新 |\n|---|---|\n'
    for name in ('k4_public','k4_online','k4_official','k2_k3_formal_resume_v3','auxiliary_resume_v3','crossbackbone_se2_public_pivot_v2'):
        st=status[name];block+='| '+name+' | '+((' / '.join(str(st.get(k,'')) for k in ('status','updates','completed_role'))).strip(' /') if st else '未在本机生成状态（4090进度以远端为准）')+' |\n'
    for method,roles in new_results.items():
        for role,v in roles.items():block+='\n'+method+' '+role+'：AP70 '+ '/'.join(f'{x:.5f}' for x in v['ap70'])+'；AP50 '+'/'.join(f'{x:.5f}' for x in v['ap50'])+'（探索结果，正式终态/完整复核层级见原件）。\n'
    for name,native in native_results.items():
        ps=native['panels'] or []
        for arm in sorted({arm for row in ps for arm in row['metrics']}):
            vals=sorted(ps,key=lambda r:r['wait_ms']);ap70=[100*r['metrics'][arm]['global_score_sort']['ap70'] for r in vals];ap50=[100*r['metrics'][arm]['global_score_sort']['ap50'] for r in vals]
            block+='\n'+name+' / '+arm+'：native ROI val21 AP70 '+ '/'.join(f'{x:.5f}' for x in ap70)+'；AP50 '+'/'.join(f'{x:.5f}' for x in ap50)+'（严格native4轻检；common/来源完成情况分列；完整独立复核未完成，不混入晚融合或common0旧表）。\n'
    audit=status['K5_original_identity']
    if audit and audit.get('status')=='complete_label_only_audit':block+='\nK5 原标签身份：'+str(audit['objects'])+' 个 inf 专属目标，原始路侧框 + 公开标定 + 官方平移与协同标签中心 3D 最大误差 '+str(audit['centre_3D_error_m']['max'])+' m；该值不是检测 AP。\n'
    for name,result in native_source_panels.items():
        block+='\n'+name+' 来源/辅助全等待已完成（固定官方GT人口和预测，不能与辅助中心重筛ROI混表；veh/coop来源AP会受inf忽略区移动影响）。\n\n| ROI / 标签 | all AP70四等待 | veh∪coop AP70四等待 | inf AP70四等待 |\n|---|---|---|---|\n'
        for roi in ('native','common'):
            for protocol in ('official','interpolated','no_interpolation'):
                cells=[]
                for group in ('all','veh_coop','inf'):
                    rows=sorted([r for r in result['panels'] if r['roi']==roi and r['protocol']==protocol and r['source_group']==group and r['threshold']==.7],key=lambda r:r['wait_ms'])
                    # Several arms share one producer; label each in the cell.
                    cells.append('; '.join(arm+': '+ '/'.join(f"{r['ap']:.5f}" for r in rows if r['arm']==arm) for arm in sorted({r['arm'] for r in rows})))
                block+='| '+roi+' / '+protocol+' | '+' | '.join(cells)+' |\n'
    cpu06=AUDIT/'cycle06_v1/registration_recovery_v2'
    block+='\n06轮新增CPU正式后继（独立运行，无终态模型唤醒）：\n\n'
    for name in ('zero_theta','fixed_population_aux'):
        st=read(cpu06/name/'terminal.json') or read(cpu06/name/'status.json')
        block+='- '+name+'：'+(str(st.get('status')) if st else '已启动，初始化/输入身份核验')+'；输出 `'+str((cpu06/name).relative_to(ROOT))+'`。\n'
    if dimension_error:
        corrected=read(cycle09/'fixed_population_hwl_v3/terminal.json') or read(cycle09/'fixed_population_hwl_v3/status.json') or {}
        block+='\n09轮已确认CPU旧辅助AP的LWH/HWL适配错误，旧两套辅助AP无效，官方AP/GT和显式换序的native来源辅助不受影响；正确全family重评状态 '+str(corrected.get('status','initializing'))+'，详见09轮段落与纠错合同。\n'
    if sources:
        block+='\n来源均值差（相对精确 K1；pp，四等待完整后才汇总）：\n\n| 候选 / 角色 | all | veh | coop | inf | veh∪coop |\n|---|---:|---:|---:|---:|---:|\n'
        for method,role in sorted({(r['method'],r['role']) for r in sources}):
            table={r['source_group']:r['mean_AP70_delta_pp'] for r in sources if r['method']==method and r['role']==role};block+='| '+method+' / '+role+' | '+' | '.join(f'{table[g]:+.6f}' for g in ('all','veh','coop','inf','veh_coop'))+' |\n'
    if crosssummary:
        block+='\nCPU 缓存跨骨干：只汇总已完成角色，不能当 GPU 原生跨骨干。\n\n| 实际骨干 / 候选 | 已出格 | 正 / 负 | AP70均差pp |\n|---|---:|---:|---:|\n'
        for r in crosssummary:block+=f"| {r['domain']} / {r['method']} | {r['cells']} | {r['positive']} / {r['negative']} | {r['mean_AP70_delta_pp']:+.6f} |\n"
    if v2:
        block+='\n当前 K2/K3 V2V4Real：'+v2['status']+'；0/400ms历史 V2 协议、真实移动远端支点、自身q005-U；不代替 K4 原生跨域。\n'
        for role in ('mechanism_holdout14','val21'):
            for method in ('K1_exact','K2_infra','K3_K1_expanded','K3_K2_expanded'):
                rows=sorted([r for r in v2panels if r['role']==role and r['method']==method],key=lambda r:r['wait_ms'])
                if rows:block+=method+' '+role+'：AP70 '+' / '.join(f"{r['ap70']:.6f}" for r in rows)+'。\n\n'
    value['native_source_aux_results']=native_source_panels
    corrected_terminal=read(cycle09/'fixed_population_hwl_v3/terminal.json')
    value['corrected_cached_HWL_terminal']=corrected_terminal
    value['corrected_cached_HWL_independent_review']=read(AUDIT/'cycle12_v1/independent_cached_hwl/terminal.json') or read(AUDIT/'cycle12_v1/independent_cached_hwl/status.json')
    value['remaining_release_open']=['full independent top3 review','K4 hold14/current actual cross-domain/native backbone','all-external and remaining-native auxiliary/source panels']
    if not corrected_terminal or corrected_terminal.get('status')!='complete_light_checked':value['remaining_release_open'].append('corrected cached HWL auxiliary panels')
    value['zero_theta_complete_terminal']=read(cpu06/'zero_theta/terminal.json')
    value['cached_auxiliary_dimension_invalidation']=dimension_error
    v2native=AUDIT/'cycle12_v1/v2_exact_accepted_roots_v3/v2_native_qualification'
    value['K4_actual_V2_native_qualification']=read(v2native/'terminal.json') or read(v2native/'status.json')
    if value['K4_actual_V2_native_qualification']:
        vs=value['K4_actual_V2_native_qualification']
        block+='\nK4当前V2原生资格：'+str(vs['status'])+'，实际CPU边界batch '+str(vs.get('actual_boundary_batches',vs.get('completed_batches',0)))+'；V2专用权重strict加载、冻结全状态/原global index、原clean-fit/calibration根与历史/两几何核验。仅CPU构造设备放置恢复，不是GPU前向、训练或AP；同预算两臂训练/正式评价仍开放。\n'
    atomic(AUDIT/'scoreboard_latest.json',value)
    ff=value['FFNet_author_download_terminal']
    if ff and ff.get('checkpoint'):
        ft=read(cycle/'ffnet_formal/terminal.json');fs=ft or read(cycle/'ffnet_formal/status.json') or {}
        block+='\nFFNet 作者权重已取得并经CPU实际模型载入兼容核验：'+str(ff['checkpoint']['bytes'])+' bytes，SHA '+ff['checkpoint']['sha256']+'；原正式链状态 '+str(fs.get('status','未启动'))+'。'
        diagnosis=read(cycle09/'ffnet_resume/terminal.json')
        if diagnosis:
            value['FFNet_cycle09_diagnostic_terminal']=diagnosis
            block+='09轮实际blocking诊断保留为 '+str(diagnosis.get('status'))+'：trace仍在下一条件voxelize，不满足旧定位门，因此未运行safe-probe/resume，不能称修好。'
        cycle12=AUDIT/'cycle12_v1';newdiag=read(cycle12/'ffnet_error_slot/terminal.json') or read(cycle12/'ffnet_error_slot/operator_probe/terminal.json') or read(cycle12/'gpu1_chain/status.json')
        if newdiag:
            value['FFNet_cycle12_explicit_error_slot']=newdiag
            block+='12轮独立新合同已排当前GPU1 hold14之后：新鲜进程显式cudaGetLastError定位；只有确认空IoU零grid并通过8条件安全重放，才引用1024前缀执行剩余42020，旧失败/合同不改。当前 '+str(newdiag.get('status'))+'。'
        block+='旧失败不覆盖、作者算法不改。'
        if ft and ft.get('status')=='complete_independently_reviewed':
            block+='严格全量和独立复算均已完成，指标原件 `'+str((cycle/'ffnet_formal/producer/producer_metrics_v1.json').relative_to(ROOT))+'`，不将模块或随机权重smoke当作者结果。\n'
            value['FFNet_formal_terminal']=ft;value['FFNet_formal_metrics']=read(cycle/'ffnet_formal/producer/producer_metrics_v1.json');atomic(AUDIT/'scoreboard_latest.json',value)
        else:block+='尚未取得可接受的完整正式AP终态。\n'
        atomic(AUDIT/'scoreboard_latest.json',value)
    peer11=AUDIT/'cycle15_peer11_v1';paired=read(peer11/'scientific_synthesis.json')
    if paired:
        value['peer11_fair_baseline']='same-budget PUBLIC/PUBLIC, not e60; paired seeds; absolute AP not cross-paradigm ranked'
        value['peer11_two_seed_paired']=paired['seed_paired']
        value['peer11_actual_jobs']={name:read(peer11/name/'terminal.json') or read(peer11/name/'status.json') for name in ('seed3_public','seed3_online','e60_val','v2_public','v2_online','hold14_seed2_public_correct_env','ego_probe','upper_split','source_successors')}
        value['peer11_ffnet_v4']=read(peer11/'ffnet_error_slot/terminal.json') or read(peer11/'ffnet_error_slot/gpu_terminal.json')
        value['peer11_release_coverage']=read(peer11/'release_coverage_audit.json')
        block+='\npeer11最新执行口径：K2只保留旋转消融/K5证据，不开原生跨骨干；K3并入K1。K4相对同预算PUBLIC微调，第三seed仅PUBLIC/ONLINE；V2实际训练/完整评价runner已实现并进入有限GPU链，尚未生成的AP不视为完成。e60精确val、ego-only和非因果seqmean/MA5已排；原12轮FFNet/hold环境失败保留，新恢复身份独立。具体实时队列见本文顶部15轮段落，终态判断留下一calendar。\n'
        atomic(AUDIT/'scoreboard_latest.json',value)
    block+='\n<!-- PEER10_GENERATED_LIVE_END -->\n'
    with (AUDIT/'scoreboard.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);text=doc.read_text();marker='<!-- PEER10_GENERATED_LIVE_BEGIN -->';start=text.find(marker)
        if start>=0:
            end=text.index('<!-- PEER10_GENERATED_LIVE_END -->',start)+len('<!-- PEER10_GENERATED_LIVE_END -->');text=text[:start]+block+text[end:].lstrip('\n')
        else:text+='\n'+block
        temp=doc.with_suffix('.tmp');temp.write_text(text);temp.replace(doc)
if __name__=='__main__':main()
