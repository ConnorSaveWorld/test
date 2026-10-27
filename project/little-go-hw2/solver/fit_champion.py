"""Step 2: fit candidate models of the Championship player by likelihood.

Unique positions (canonical, moves merged into symmetry orbits) with the
champion's observed choice counts; each model is sampled R times per
position. Score = mean log-likelihood per decision (smoothed) and the share
of decisions where the observed move equals the model's most frequent move.
"""
import ctypes, sys, os, json, math, random, time
from collections import defaultdict
from multiprocessing import Pool
from lg import *

HERE = os.path.dirname(os.path.abspath(__file__))
_cm = None
def cm():
    global _cm
    if _cm is None:
        _cm = ctypes.CDLL(os.path.join(HERE, os.environ.get('CM_LIB', 'libcm.so')))
        u32 = ctypes.c_uint32
        _cm.mcts_choose.argtypes = [u32, u32, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.POINTER(ctypes.c_int)]
        _cm.cm_seed.argtypes = [ctypes.c_uint64]
        _cm.cm_config.argtypes = [ctypes.c_int, ctypes.c_int]
    return _cm

def stabilizer(s):
    return [t for t in range(8) if transform(s, t).key() == s.key()]

def orbit(m, stab):
    if m < 0: return m
    return min(tmove(m, t) for t in stab)

def dataset(subs=None):
    tab = defaultdict(lambda: defaultdict(int))
    for s, m, g in champion_decisions():
        if subs and g['submission'] not in subs: continue
        k, t = canon(s)
        cs = State(*k)
        tab[k][orbit(tmove(m, t), stabilizer(cs))] += 1
    return {k: dict(v) for k, v in tab.items()}

def run_model(args):
    key, model, R, seed = args
    kind = model['kind']
    s = State(*key)
    stab = stabilizer(s)
    cnt = defaultdict(int)
    if kind == 'mcts':
        lib = cm(); lib.cm_config(model.get('policy', 0), model.get('select', 0))
        if hasattr(lib, 'cm_config2'):
            lib.cm_config2(model.get('reward', 0), model.get('nopass', 0))
        if hasattr(lib, 'cm_config3'):
            lib.cm_config3.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_int]
            lib.cm_config3(model.get('noise', 2.0), model.get('connect', 0.5), model.get('cutoff', 0))
        out = (ctypes.c_int * 26)()
        for r in range(R):
            lib.cm_seed(seed * 1000003 + r * 7919 + 1)
            m = lib.mcts_choose(s.B, s.W, s.side, s.ko, s.count, s.lp, model['n'], model['c'], out)
            cnt[orbit(m, stab)] += 1
    return key, dict(cnt), len(s.legal())

def score(model, data, R=10, pool=None, seed=1):
    jobs = [(k, model, R, seed) for k in data]
    res = pool.map(run_model, jobs, chunksize=4)
    ll = 0.0; n = 0; top = 0
    per = {}
    for k, cnt, L in res:
        mode = max(cnt, key=cnt.get) if cnt else None
        for m, c in data[k].items():
            p = (cnt.get(m, 0) + 0.5) / (R + 0.5 * L)
            ll += c * math.log(p); n += c
            top += c * (m == mode)
        per[k] = cnt
    return ll / n, top / n, per

if __name__ == '__main__':
    data = dataset()
    print('unique positions', len(data), 'decisions', sum(sum(v.values()) for v in data.values()), flush=True)
    # baseline: uniform over legal moves
    ll = sum(c * math.log(1.0 / len(State(*k).legal())) for k, v in data.items() for c in v.values())
    print('uniform baseline ll/decision %.3f' % (ll / sum(sum(v.values()) for v in data.values())), flush=True)
