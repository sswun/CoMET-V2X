#!/usr/bin/env python3
"""Independent, zero-forward audit of shared detection-offset transfer.

Inference consumes only boxes, scores, flags, and ordered timestamps. GT is
accessed by the separate fit-only constant and evaluation/oracle functions.
This reimplementation is not a reproduction of the unavailable peer script.
"""
from pathlib import Path
from collections import defaultdict
import argparse
import json
import resource
import time

import numpy as np
from scipy.optimize import linear_sum_assignment

from e1_receiver_value_core_v1_20261020 import atomic, greedy, iou32, sha256, unique_ego_scores
from object_history_selector_core_v1_20261021 import PackedDataset
from review_traf_v2xseq_adapted_fixed60_raw_20261006 import voc_ap

METHODS = ('ego', 'stale', 'cv', 'fit_constant', 'frame_median',
           'frame_shrink', 'causal_kf', 'stale_plus_same_kf', 'oracle_translation')


def views(pool):
    kind = pool['hypothesis_type']
    ei, si, pi = [np.flatnonzero(kind == k) for k in (0, 1, 2)]
    cv = pool['boxes'][si].copy()
    sources = pool['source_detection_indices']
    lookup = {int(sources[j]): j for j in pi}
    for i, j in enumerate(si):
        if int(sources[j]) in lookup:
            cv[i] = pool['boxes'][lookup[int(sources[j])]]
    return ei, si, cv


def pairs(remote, reference, gate, mutual):
    if not len(remote) or not len(reference):
        return np.empty(0, int), np.empty(0, int)
    dist = np.linalg.norm(remote[:, None, :2] - reference[None, :, :2], axis=2)
    a, b = linear_sum_assignment(dist)
    keep = dist[a, b] <= gate
    if mutual:
        keep &= (dist.argmin(axis=1)[a] == b) & (dist.argmin(axis=0)[b] == a)
    return a[keep], b[keep]


def infer(cv, protected_boxes, state):
    """Additive correction d = ego - remote, independent of GT/offset files."""
    a, b = pairs(cv, protected_boxes, 2.5, True)
    residual = protected_boxes[b, :2].astype(np.float64) - cv[a, :2]
    median = np.median(residual, axis=0) if len(a) >= 2 else np.zeros(2)
    shrink = residual.sum(axis=0) / (len(a) + 1)
    mean, variance = state
    prior = variance + .15**2
    if len(a):
        gain = prior / (prior + .6**2 / len(a))
        mean = mean + gain * (residual.mean(axis=0) - mean)
        variance = (1 - gain) * prior
    else:
        variance = prior
    return median, shrink, (mean, variance), a, b, residual


def oracle(cv, gt):
    a, b = pairs(cv, gt, 3.0, False)
    value = (gt[b, :2].astype(np.float64) - cv[a, :2]).mean(axis=0) if len(a) else np.zeros(2)
    return value, len(a)


def rank(ego_scores, remote_scores):
    # Use the accepted E1 tie semantics, encoding after the +2 interval shift.
    es = unique_ego_scores(2.0 + ego_scores)
    rs = unique_ego_scores(remote_scores, stable=True)
    scores = np.r_[es, rs]
    order = np.argsort(-scores)
    return order, scores[order]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    summary = json.loads((args.dataset / 'summary.json').read_text())
    for name in ('pools', 'conditions'):
        path = args.dataset / ('pools.npz' if name == 'pools' else 'conditions.jsonl')
        assert sha256(path) == summary[name + '_sha256']
    contract = dict(status='frozen_before_outcomes', dataset=str(args.dataset),
                    input_summary=summary, methods=METHODS,
                    evaluation_roles=['mechanism_holdout14'], waits=[0, 100, 200, 400],
                    geometry='One CV replacement per accepted stale source identity; fallback stale if no accepted propagated slot. Same source set for every variant. No new post-correction NMS/ROI filtering.',
                    correction_sign='add ego-minus-remote in receiver lidar XY',
                    matching='Hungarian Euclidean XY, 2.5m gate and mutual nearest-neighbor',
                    kf='Per(sequence,wait); initial zero, variance1m2; Q=.15^2 per chronological step; R=.6^2/m; no GT; independent cold starts for sequences.',
                    frame_median_minimum_anchors=2,
                    fit_constant='Mean per-condition GT-matched additive CV residual on fit24 only; supervised simple control, not a label-free estimator.',
                    oracle='Per-condition Hungarian CV/GT within3m, evaluation-only; heuristic oracle not a mathematical AP upper bound.',
                    scope='Independent decision audit; no val21 read, detector forward, training, or formal-main-table replacement.',
                    uncertainty_readout_deferred=True)
    atomic(args.output / 'contract.json', contract)
    ds = PackedDataset(args.dataset)
    # Fit the supervised constant strictly on fit24, never on cal/hold.
    fit_offsets, fit_ids = [], []
    for i, meta in enumerate(ds.meta):
        if meta['role'] != 'fit24':
            continue
        pool = ds.pool(i)
        _, _, cv = views(pool)
        offset, n = oracle(cv, pool['gt_boxes'])
        if n:
            fit_offsets.append(offset)
            fit_ids.append(i)
    constant = np.mean(fit_offsets, axis=0)
    np.savez_compressed(args.output / 'fit_only_constant.npz', offsets=fit_offsets,
                        condition_index=fit_ids, constant=constant)
    atomic(args.output / 'progress.json', dict(stage='constant_fit', conditions=len(fit_ids), constant=constant.tolist()))

    selected = [i for i, m in enumerate(ds.meta) if m['role'] == 'mechanism_holdout14']
    selected.sort(key=lambda i: (ds.meta[i]['sequence'], ds.meta[i]['wait_ms'], ds.meta[i]['receiver_timestamp_us']))
    states, last_time = {}, {}
    results = defaultdict(lambda: defaultdict(lambda: dict(scores=[], tp50=[], tp70=[], matched=[], gt=0, reachable=0,
                         novel_reachable=0, nonanchor_novel=0, ego_no_box_overlap_novel=0, rescued=0, destroyed=0)))
    correction_rows = []
    saved = {k: defaultdict(list) for k in METHODS}
    invariants = dict(timestamp_monotonic=True, state_isolated_by_sequence_wait=True,
                      infer_has_no_gt_argument=True, stored_pool_not_mutated=True)
    original_boxes = ds.data['boxes'].copy()
    with (args.output / 'condition_ledger.jsonl').open('w') as stream:
        for count, i in enumerate(selected):
            meta = ds.meta[i]; pool = ds.pool(i)
            key = (meta['sequence'], meta['wait_ms'])
            timestamp = meta['receiver_timestamp_us']
            assert timestamp > last_time.get(key, -1)
            last_time[key] = timestamp
            ei, si, cv = views(pool)
            ego, stale, gt = pool['boxes'][ei], pool['boxes'][si], pool['gt_boxes']
            ps = np.flatnonzero(pool['is_protected'][ei])
            median, shrink, state, anchors, eb, residual = infer(cv, ego[ps], states.get(key, (np.zeros(2), 1.0)))
            states[key] = (state[0].copy(), state[1])
            # Replay has no access to any later row: append-only state trace is saved.
            ideal, oracle_n = oracle(cv, gt)
            offsets = dict(stale=np.zeros(2), cv=np.zeros(2), fit_constant=constant,
                           frame_median=median, frame_shrink=shrink, causal_kf=state[0],
                           stale_plus_same_kf=state[0], oracle_translation=ideal)
            correction_rows.append([i, len(anchors), state[1], *state[0], *median, *shrink, *ideal, oracle_n])
            ego_matrix = pool['ious'][ei]
            ego_order, ego_rank = rank(pool['raw_scores'][ei], np.empty(0))
            ego_match = greedy(ego_matrix, ego_order, .7)
            ego_covered = set(ego_match[ego_match >= 0].tolist())
            porder = ps[np.argsort(-unique_ego_scores(2.0 + pool['raw_scores'][ei][ps]))]
            pmatch = greedy(ego_matrix, porder, .7)
            protected_covered = set(pmatch[pmatch >= 0].tolist())
            uncovered = np.asarray([j for j in range(len(gt)) if j not in protected_covered], int)
            no_ego_box = ego_matrix.max(axis=0, initial=0.) < .1
            nonanchor = np.ones(len(stale), bool); nonanchor[anchors] = False
            for method in METHODS:
                if method == 'ego':
                    boxes, matrix = ego, ego_matrix
                    order, scores = ego_order, ego_rank
                    remote_matrix = np.zeros((0, len(gt)), np.float32)
                else:
                    rb = (stale if method in ('stale', 'stale_plus_same_kf') else cv).copy()
                    rb[:, :2] += offsets[method].astype(np.float32)
                    remote_matrix = iou32(rb, gt)
                    boxes = np.concatenate((ego, rb))
                    matrix = np.concatenate((ego_matrix, remote_matrix))
                    order, scores = rank(pool['raw_scores'][ei], pool['raw_scores'][si])
                matches = greedy(matrix, order, .7)
                matches50 = greedy(matrix, order, .5)
                covered = set(matches[matches >= 0].tolist())
                reach = remote_matrix.max(axis=0, initial=0.) >= .7
                nonanchor_reach = remote_matrix[nonanchor].max(axis=0, initial=0.) >= .7 if method != 'ego' else reach
                row = results[meta['wait_ms']][method]
                for field, value in [('scores', scores), ('tp50', (matches50 >= 0).astype(np.int8)),
                                     ('tp70', (matches >= 0).astype(np.int8)), ('matched', matches)]:
                    row[field].append(value)
                    saved[method][field].append(value)
                saved[method]['boxes'].append(boxes[order])
                saved[method]['length'].append(len(order))
                row['gt'] += len(gt)
                row['reachable'] += int(reach.sum())
                row['novel_reachable'] += int(reach[uncovered].sum())
                row['nonanchor_novel'] += int(nonanchor_reach[uncovered].sum())
                row['ego_no_box_overlap_novel'] += int((reach & no_ego_box)[uncovered].sum())
                rescued, destroyed = len(covered - ego_covered), len(ego_covered - covered)
                row['rescued'] += rescued; row['destroyed'] += destroyed
                stream.write(json.dumps(dict(index=i, sequence=meta['sequence'], wait_ms=meta['wait_ms'], method=method,
                         outputs=len(order), gt=len(gt), tp70=int((matches>=0).sum()), rescued=rescued, destroyed=destroyed,
                         novel_reachable=int(reach[uncovered].sum()))) + '\n')
            if (count + 1) % 500 == 0:
                atomic(args.output / 'progress.json', dict(stage='hold_evaluation', completed=count+1, total=len(selected), wall_s=time.monotonic()-start))
    assert np.array_equal(ds.data['boxes'], original_boxes)
    panels, archive = [], {}
    for wait, methods in results.items():
        for method, row in methods.items():
            scores = np.concatenate(row['scores'])
            panel = dict(wait_ms=wait, method=method)
            for threshold in (50, 70):
                tp = np.concatenate(row['tp' + str(threshold)])
                metric = voc_ap(tp.tolist(), (1-tp).tolist(), scores.tolist(), row['gt'], True)
                panel['ap' + str(threshold)] = metric['ap'] * 100
                if threshold == 70: panel.update(tp=metric['tp'], fp=metric['fp'], gt=metric['gt'])
            panel.update({k: v for k,v in row.items() if not isinstance(v, list)})
            panels.append(panel)
    for method, rows in saved.items():
        for field, values in rows.items():
            archive[method+'__'+field] = np.asarray(values) if field=='length' else np.concatenate(values)
    archive['condition_index'] = np.asarray(selected)
    archive['correction_columns'] = np.asarray(['index','anchors','variance','kf_dx','kf_dy','median_dx','median_dy','shrink_dx','shrink_dy','oracle_dx','oracle_dy','oracle_matches'])
    archive['corrections'] = np.asarray(correction_rows)
    np.savez_compressed(args.output / 'predictions.npz', **archive)
    result = dict(status='complete_independent_decision_diagnostic', scope=contract['scope'],
                  constant_additive_xy=constant.tolist(), conditions=len(selected), fit_conditions=len(fit_ids),
                  invariants=invariants, panels=panels, predictions_sha256=sha256(args.output/'predictions.npz'),
                  script_sha256=sha256(Path(__file__)), contract_sha256=sha256(args.output/'contract.json'),
                  wall_seconds=time.monotonic()-start, peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,
                  detector_forwards=0, head_updates=0, gpu_seconds=0,
                  limitation='No peer source available; uses accepted packed candidate set and documented KF initialization. Labels only in fit-only supervised constant and evaluation/oracle. Val21 untouched. Full-ego prefix zero destruction is structural.')
    atomic(args.output / 'summary.json', result)
    atomic(args.output / 'progress.json', dict(stage='complete', wall_seconds=result['wall_seconds']))
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
