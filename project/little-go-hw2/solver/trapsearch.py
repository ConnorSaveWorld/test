"""Expectimax search for white trap lines against the championship model.
Model: at a black node with count c and nwin winning moves, the champion plays a
losing move with probability err(c, nwin); otherwise a winning move (uniform).
"""
import sys, os, json, pickle, time
from multiprocessing import Pool
from lg import *

def err(c, nwin):
    if nwin == 1:
        return 0.25 if c <= 12 else (0.05 if c <= 14 else 0.0)
    if nwin == 2:
        return 0.03 if c <= 12 else 0.0
    return 0.005 if c <= 12 else 0.0

CACHE_FILE = 'moves_cache.pkl'
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

def save():
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
    best = (0.0, scored[0][1]); best_detail = None
    for i, (nw, m, ch, wins) in enumerate(scored):
        e = err(ch.count, nw)
        if i < K and nw <= NWMAX and ch.count + 1 <= CMAX:
            rec = sum(eval_white(ch.play(b))[0] for b in wins) / len(wins)
        else:
            rec = 0.0
        v = e + (1 - e) * rec
        if v > best[0]:
            best = (v, m)
    memo[ck] = (best[0], tmove(best[1], ct))
    if s.count <= 7:
        print('  ' * (s.count - 3) + 'white node count %d value %.3f best %d  (nwins %s)' % (s.count, best[0], best[1], [(x[1], x[0]) for x in scored[:8]]), flush=True)
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
    old = pickle.load(open('memo.pkl', 'rb')) if os.path.exists('memo.pkl') else {}
    old.update(memo); pickle.dump(old, open('memo.pkl', 'wb'))
    print('ROOT', seq, 'value %.4f best %s  %.0fs' % (v[0], v[1], time.time() - t))
