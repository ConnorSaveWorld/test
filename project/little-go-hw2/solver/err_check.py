"""Does a champion model predict where the champion errs?
For every Black (champion) node at count 4..14 in our White games where
Black has a winning move and at least one losing move, compare the model's
error probability with the observed errors."""
import sys, json, math, pickle
from collections import defaultdict
from multiprocessing import Pool
import fit_champion as F
from lg import *

def nodes_with_errors():
    init(24)
    cache = pickle.load(open('moves_cache.pkl', 'rb'))
    tab = defaultdict(lambda: defaultdict(int))
    for s, m, g in champion_decisions():
        if s.side != 1 or not (4 <= s.count <= 14): continue
        k, t = canon(s)
        tab[k][tmove(m, t)] += 1
    out = {}
    for k, obs in tab.items():
        s = State(*k)
        mv = cache.get(k)
        if mv is None:
            mv = moves(s)
        wins = {m for m, v in mv.items() if v == 1}
        if not wins or len(wins) == len(mv): continue
        out[k] = dict(obs=dict(obs), wins=sorted(wins), n=sum(obs.values()), err=sum(c for m, c in obs.items() if m not in wins))
    return out

if __name__ == '__main__':
    model = json.loads(sys.argv[1]); R = int(sys.argv[2])
    nodes = nodes_with_errors()
    with Pool(4) as pool:
        res = pool.map(F.run_model, [(k, model, R, 7) for k in nodes])
    rows = []
    for k, cnt, L in res:
        d = nodes[k]
        s = State(*k); stab = F.stabilizer(s)
        winorb = {F.orbit(m, stab) for m in d['wins']}
        perr = sum(c for m, c in cnt.items() if m not in winorb) / R
        rows.append((k[4], len(d['wins']), d['n'], d['err'], perr))
    rows.sort()
    for c, nw, n, e, p in rows:
        flag = ' <== champion erred' if e else ''
        if e or p > 0.05 or nw == 1:
            print('count %2d nwin %2d visits %2d errors %d   model P(err) %.2f%s' % (c, nw, n, e, p, flag))
    # summary
    E = sum(r[3] for r in rows); N = sum(r[2] for r in rows)
    pe = sum(r[2] * r[4] for r in rows)
    print('total visits', N, 'observed errors', E, 'model expected errors %.1f' % pe)
