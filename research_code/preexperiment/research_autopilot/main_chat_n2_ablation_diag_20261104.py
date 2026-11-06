"""Main-session diagnostic: which part of the accepted N2 soft association gives the gain?

Zero-update ablations only. Kernel values are N2's initial state (sigma 1 m, dustbin 0,
8 Sinkhorn iterations, weight/.36), fixed before any result was seen here. Nothing is
selected on val21/hold14; formal tables, heads and training weights are not modified.

  N0        original hard association (protected ego, Hungarian+mutual NN, 2.5 m)
  N2t       saved trained N2 state (seed 2023), reproduction check
  U0        untrained N2 kernel: all ego boxes, prediction-gated
  U1        untrained kernel, no prediction gating (innovation measured from zero)
  U2        untrained kernel, prediction-gated, protected ego only
  U3        hard association, prediction-gated (remote shifted by x before matching)
  N0_altq   N0 with the other q (author .05, others .15)
  U0_altq   U0 with the other q

Stage 2 (--stage2): minimal explanation of the gain.
  H1        hard association against all ego boxes, no prediction gating
  H2        hard association against all ego boxes, prediction-gated
  H3        H2 with a mutual-NN gate of 1.5 m instead of 2.5 m
  U5        U0 with sigma .5 m
  U6        U0 with sigma 2 m
Also stores robust state errors against MA5 (median, fraction < .3 / .5 m) for N0, U0, H2.
"""
import argparse, json, os, sys, time
os.environ.setdefault('OMP_NUM_THREADS', '1')
ROOT = os.path.expanduser('~/Documents/Aprojects/ApplicationsForMA/embodieddatasets')
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, 'preexperiment/research_autopilot'))
import numpy as np
import evaluate_peer18_comet_net_v1_20261104 as E

engine = E.engine
engine.imports()
final = engine.final
OUT = os.path.join(ROOT, 'artifacts/research_autopilot/status_audit_20261104_peer18/main_chat_n2_diag')
FORMAL = os.path.join(ROOT, 'artifacts/research_autopilot/status_audit_20261104_peer18/CoMET_Net/formal/seed2023')
SEED = 2023


def sinkhorn_weight(residual, centre, sigma=1.):
    nr, ne = residual.shape[:2]
    innovation = residual - centre
    A = -np.sum(innovation ** 2, -1) / (2 * sigma ** 2)
    full = np.zeros((nr + 1, ne + 1))
    full[:nr, :ne] = A
    total = nr + ne
    mu = np.ones(nr + 1) / total
    nu = np.ones(ne + 1) / total
    mu[-1] = ne / total
    nu[-1] = nr / total
    logp = full
    for _ in range(8):
        logp = logp + np.log(mu)[:, None] - np.logaddexp.reduce(logp, axis=1, keepdims=True)
        logp = logp + np.log(nu)[None, :] - np.logaddexp.reduce(logp, axis=0, keepdims=True)
    return np.exp(logp[:nr, :ne]) * total


def run_filter(m, rows, q, mode):
    states = {}
    delta = np.zeros((len(rows), 2))
    mass = np.zeros(len(rows))
    cv = m['cv_boxes']
    for i in range(len(rows)):
        a, b = map(int, m['condition_offsets'][i:i + 2])
        ri = a + np.flatnonzero(m['source'][a:b] == 1)
        ei = a + np.flatnonzero(m['source'][a:b] == 0)
        fixed = a + np.flatnonzero((m['source'][a:b] == 0) & (m['protected'][a:b] == 1))
        key = str(m['sequence'][i]), int(m['wait_ms'][i])
        x, p = states.get(key, (np.zeros(2), np.ones(2)))
        prior = p + q ** 2
        z, R = None, None
        if mode in ('N0', 'U3', 'H1', 'H2', 'H3'):
            target = fixed if mode in ('N0', 'U3') else ei
            shifted = cv[ri].astype(float).copy()
            if mode in ('U3', 'H2', 'H3') and len(shifted):
                shifted[:, :2] += x
            xx, yy = engine.match(shifted, cv[target], 1.5 if mode == 'H3' else 2.5)
            if len(xx):
                z = (cv[target[yy], :2].astype(float) - cv[ri[xx], :2]).mean(0)
                R = .36 / len(xx)
                mass[i] = len(xx) / .36
        else:
            e = ei if mode in ('U0', 'U1', 'U5', 'U6') else fixed
            if len(ri) and len(e):
                residual = cv[e, :2].astype(float)[None, :, :] - cv[ri, :2][:, None, :]
                sigma = dict(U5=.5, U6=2.).get(mode, 1.)
                w = sinkhorn_weight(residual, x if mode in ('U0', 'U2', 'U5', 'U6') else np.zeros(2), sigma) / .36
                total = w.sum()
                if total > 1e-12:
                    z = (w[:, :, None] * residual).sum((0, 1)) / total
                    R = 1 / total
                    mass[i] = total
        if z is not None:
            K = prior / (prior + R)
            x = x + K * (z - x)
            p = (1 - K) * prior
        else:
            p = prior
        states[key] = (x, p)
        delta[i] = x
    return delta, mass


def score_and_ap(m, domain, delta, name):
    head = final.head(domain, 'no_position', SEED)
    boxes, f = final.old.author.heading.build_heading(m, delta)
    f[:, engine.REMOVED] = 0
    direct = final.old.matched.readout.score_direct_u(dict(m, features=f), head)
    maps = json.load(open(head / 'source_maps.json'))
    score = final.old.matched.readout.apply_source_map(direct, m['source'], maps['maps']['direct_to_utility'])
    tau = json.load(open(final.OUT / domain / 'calibration.json'))['selected_tau']
    return final.old.author.refit.fixed.base.evaluate(final.protected(m, tau), {name: boxes}, {name: score})


def ap(score, tp, n_gt):
    if n_gt == 0 or len(score) == 0:
        return 0.0
    order = np.argsort(-score, kind='stable')
    t = tp[order].astype(float)
    c = np.cumsum(t)
    prec = c / np.arange(1, len(t) + 1)
    rec = c / n_gt
    prec = np.maximum.accumulate(prec[::-1])[::-1]
    return float(np.sum(np.diff(np.concatenate(([0.], rec))) * prec) * 100)


def per_sequence(m, pred, name, thr=70):
    off = pred[name + '__offset']
    gt = m['gt_offsets']
    seq = np.array([str(s) for s in m['sequence']])
    waits = np.array([int(w) for w in m['wait_ms']])
    out = {}
    for s in np.unique(seq):
        vals = []
        for w in np.unique(waits):
            idx = np.flatnonzero((seq == s) & (waits == w))
            sc = np.concatenate([pred[name + '__score'][off[i]:off[i + 1]] for i in idx])
            tp = np.concatenate([pred[name + '__tp%d' % thr][off[i]:off[i + 1]] for i in idx])
            vals.append(ap(sc, tp, int(sum(gt[i + 1] - gt[i] for i in idx))))
        out[s] = float(np.mean(vals))
    return out


def ma5_rmse(rows, delta):
    from run_peer18_analyses_v1_20261104 import references
    ma = references(rows)[0]
    return float(min(np.sqrt(np.mean(np.sum((delta - ma) ** 2, 1))), np.sqrt(np.mean(np.sum((delta + ma) ** 2, 1)))))


def robust(rows, delta):
    from run_peer18_analyses_v1_20261104 import references
    ma = references(rows)[0]
    e = min((np.linalg.norm(delta - ma, axis=1), np.linalg.norm(delta + ma, axis=1)), key=lambda v: np.median(v))
    return dict(median=float(np.median(e)), lt03=float((e < .3).mean()), lt05=float((e < .5).mean()),
                gt1=float((e > 1.).mean()), rmse=float(np.sqrt(np.mean(e ** 2))))


def stage2(domain):
    data, _, q = engine.materials(domain)
    result = dict(domain=domain, q=q, roles={})
    path = os.path.join(OUT, domain + '_stage2.json')
    for role in ('val21', 'mechanism_holdout14', 'calibration8'):
        m, rows = data[role]
        res = {}
        seqs = {}
        for name in ('N0', 'U0', 'H1', 'H2', 'H3', 'U5', 'U6'):
            delta, mass = run_filter(m, rows, q, name)
            panels, pred = score_and_ap(m, domain, delta, name)
            res[name] = dict(panels=[{k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in r.items()
                                      if k in ('wait_ms', 'ap50', 'ap70', 'method')} for r in panels],
                             mean_mass=float(mass.mean()), informative_fraction=float((mass > 0).mean()))
            try:
                res[name]['robust'] = robust(rows, delta)
            except Exception as e:
                res[name]['robust_error'] = repr(e)
            seqs[name] = per_sequence(m, pred, name)
            result['roles'][role] = dict(methods=res, per_sequence_ap70=seqs)
            json.dump(result, open(path + '.tmp', 'w'))
            os.replace(path + '.tmp', path)
    result['status'] = 'complete'
    json.dump(result, open(path + '.tmp', 'w'))
    os.replace(path + '.tmp', path)


def main(domain):
    os.makedirs(OUT, exist_ok=True)
    data, _, q = engine.materials(domain)
    altq = .05 if q > .1 else .15
    result = dict(domain=domain, q=q, altq=altq, roles={})
    path = os.path.join(OUT, domain + '.json')
    for role in ('val21', 'mechanism_holdout14', 'calibration8'):
        m, rows = data[role]
        waits = np.array([int(w) for w in m['wait_ms']])
        methods = [('N0', q, 'N0'), ('U0', q, 'U0'), ('U1', q, 'U1'), ('U2', q, 'U2'), ('U3', q, 'U3'),
                   ('N0_altq', altq, 'N0'), ('U0_altq', altq, 'U0')]
        res = {}
        seqs = {}
        for name, qq, mode in methods:
            t = time.time()
            delta, mass = run_filter(m, rows, qq, mode)
            panels, pred = score_and_ap(m, domain, delta, name)
            res[name] = dict(panels=[{k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in r.items()
                                      if k in ('wait_ms', 'ap50', 'ap70', 'AP50', 'AP70', 'method', 'threshold', 'ap')} for r in panels],
                             mean_mass=float(mass.mean()), informative_fraction=float((mass > 0).mean()), wall=time.time() - t)
            try:
                res[name]['MA5_RMSE'] = ma5_rmse(rows, delta)
            except Exception as e:
                res[name]['MA5_RMSE_error'] = repr(e)
            seqs[name] = per_sequence(m, pred, name)
            result['roles'][role] = dict(methods=res, per_sequence_ap70=seqs)
            json.dump(result, open(path + '.tmp', 'w'))
            os.replace(path + '.tmp', path)
        trained = os.path.join(FORMAL, domain + '__to__' + domain + '__N2', role + '_state.npz')
        if domain == 'author' or os.path.exists(trained):
            if os.path.exists(trained):
                with np.load(trained) as z:
                    delta = z['delta']
                panels, pred = score_and_ap(m, domain, delta, 'N2t')
                res['N2t'] = dict(panels=[{k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in r.items()
                                           if k in ('wait_ms', 'ap50', 'ap70', 'AP50', 'AP70', 'method', 'threshold', 'ap')} for r in panels])
                try:
                    res['N2t']['MA5_RMSE'] = ma5_rmse(rows, delta)
                except Exception as e:
                    res['N2t']['MA5_RMSE_error'] = repr(e)
                seqs['N2t'] = per_sequence(m, pred, 'N2t')
        json.dump(result, open(path + '.tmp', 'w'))
        os.replace(path + '.tmp', path)
    result['status'] = 'complete'
    json.dump(result, open(path + '.tmp', 'w'))
    os.replace(path + '.tmp', path)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('domain')
    p.add_argument('--stage2', action='store_true')
    a = p.parse_args()
    (stage2 if a.stage2 else main)(a.domain)
