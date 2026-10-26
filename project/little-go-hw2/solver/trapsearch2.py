"""Expectimax search for white trap lines against the championship model.
Model: at a black node with count c and nwin winning moves, the champion plays a
losing move with probability err(c, nwin); otherwise a winning move (uniform).
"""
import sys, os, json, pickle, time
from multiprocessing import Pool
from lg import *

PROXY = None
def _proxy_mod():
    global PROXY
    if PROXY is None:
        import importlib.util
        spec = importlib.util.spec_from_file_location('cur', str(ROOT / 'stage2_candidate' / 'my_player3.py'))
        PROXY = importlib.util.module_from_spec(spec); spec.loader.exec_module(PROXY)
    return PROXY

PROBES = (('d', 2), ('d', 4), ('d', 6), ('n', 100000))
def _proxy_work(args):
    """fraction of heuristic-search probes that choose a losing move (canonical frame)"""
    key, prevkey, wins = args
    mod = _proxy_mod()
    s = State(*key)
    board = tuple(s.board())
    prev = tuple(State(*prevkey).board()) if prevkey else board
    bad = 0
    for kind, v in PROBES:
        for name in ('analyze', '_bit_groups', '_bit_static', '_bit_candidates'):
            getattr(mod, name).cache_clear()
        if kind == 'd':
            m, _ = mod.search(prev, board, s.side, s.count, bool(s.lp), cpu_seconds=120, wall_seconds=120, max_depth=v, use_symmetry=False)
        else:
            m, _ = mod.search(prev, board, s.side, s.count, bool(s.lp), cpu_seconds=120, wall_seconds=120, max_nodes=v, use_symmetry=False)
        bad += m not in wins
    return key, bad / len(PROBES)

def err2(c, nwin, f):
    if c >= 16:
        return 0.0
    scale = 0.5 if c >= 14 else 1.0
    if nwin == 1:
        return scale * (0.05 + 0.40 * f)
    if nwin == 2:
        return scale * (0.01 + 0.25 * f)
    return scale * (0.005 + 0.10 * f)

def err(c, nwin):
    if nwin == 1:
        return 0.25 if c <= 12 else (0.05 if c <= 14 else 0.0)
    if nwin == 2:
        return 0.03 if c <= 12 else 0.0
    return 0.005 if c <= 12 else 0.0

CACHE_FILE = 'moves_cache.pkl'
PCACHE_FILE = 'proxy_cache.pkl'
pcache = pickle.load(open(PCACHE_FILE, 'rb')) if os.path.exists(PCACHE_FILE) else {}
cache = {}
if os.path.exists(CACHE_FILE):
    cache = pickle.load(open(CACHE_FILE, 'rb'))

def _work(key):
    s = State(*key)
    return key, moves(s)

pool = None
def batch_moves(states):
    """canonical move-value dicts for many states, using cache + pool"""
    keys = []
    for s in states:
        k, t = canon(s)
        keys.append((k, t))
    todo = sorted({k for k, t in keys if k not in cache}, key=lambda k: k)
    if todo:
        for k, mv in pool.imap_unordered(_work, todo):
            cache[k] = mv
    res = []
    for k, t in keys:
        inv = INV[t]
        res.append({tmove(m, inv): v for m, v in cache[k].items()})
    return res

INV = [[i for i in range(8) if all(SYM[i][SYM[t][p]] == p for p in range(25))][0] for t in range(8)]

def proxy_f(items):
    """items: list of (black-to-move state, parent state, winning moves) -> list of f"""
    jobs = {}
    keys = []
    for s, parent, wins in items:
        k, t = canon(s)
        pk = transform(parent, t).key()
        keys.append(k)
        if k not in pcache and k not in jobs:
            jobs[k] = (k, pk, [tmove(b, t) for b in wins])
    if jobs:
        for k, f in pool.imap_unordered(_proxy_work, list(jobs.values())):
            pcache[k] = f
    return [pcache[k] for k in keys]

def save():
    tmp2 = PCACHE_FILE + '.tmp'
    pickle.dump(pcache, open(tmp2, 'wb')); os.replace(tmp2, PCACHE_FILE)
    tmp = CACHE_FILE + '.tmp'
    pickle.dump(cache, open(tmp, 'wb')); os.replace(tmp, CACHE_FILE)

K = int(os.environ.get("TRAP_K", "4"))
NWMAX = int(os.environ.get("TRAP_NWMAX", "2"))
CMAX = 13
memo = {}
def eval_white(s, trace=False):
    """value (white win prob) of white-to-move state s, assumed lost for white unless shown otherwise"""
    if s.count > CMAX:
        return 0.0, None
    ck, ct = canon(s)
    if ck in memo:
        v, cm = memo[ck]
        return v, (None if cm is None else tmove(cm, INV[ct]))
    legal = [m for m in range(25) if s.play(m) is not None]
    children = [s.play(m) for m in legal]
    mvs = batch_moves(children)
    scored = []
    for m, ch, mv in zip(legal, children, mvs):
        wins = [b for b, v in mv.items() if v == 1]
        if not wins:
            memo[ck] = (1.0, tmove(m, ct)); return 1.0, m
        scored.append((len(wins), m, ch, wins))
    scored.sort(key=lambda x: x[0])
    probe = [x for x in scored if x[0] <= NWMAX and x[2].count <= 14]
    fs = dict(zip([x[1] for x in probe], proxy_f([(x[2], s, x[3]) for x in probe])))
    best = (0.0, scored[0][1]); best_detail = None
    for i, (nw, m, ch, wins) in enumerate(scored):
        e = err2(ch.count, nw, fs.get(m, 0.0))
        if i < K and nw <= NWMAX and ch.count + 1 <= CMAX:
            rec = sum(eval_white(ch.play(b))[0] for b in wins) / len(wins)
        else:
            rec = 0.0
        v = e + (1 - e) * rec
        if v > best[0]:
            best = (v, m)
    memo[ck] = (best[0], tmove(best[1], ct))
    if s.count <= 7:
        print('  ' * (s.count - 3) + 'white node count %d value %.3f best %d  (nwin,f %s)' % (s.count, best[0], best[1], [(x[1], x[0], round(fs.get(x[1], -1), 2)) for x in scored[:8]]), flush=True)
        save()
    return best

if __name__ == '__main__':
    init(10)
    pool = Pool(4, initializer=init, initargs=(25,))
    seq = [int(x) for x in sys.argv[1].split(',')]
    if len(sys.argv) > 2: CMAX = int(sys.argv[2])
    s = State()
    for m in seq: s = s.play(m)
    t = time.time()
    v = eval_white(s)
    save()
    old = pickle.load(open('memo2.pkl', 'rb')) if os.path.exists('memo2.pkl') else {}
    old.update(memo); pickle.dump(old, open('memo2.pkl', 'wb'))
    print('ROOT', seq, 'value %.4f best %s  %.0fs' % (v[0], v[1], time.time() - t))
