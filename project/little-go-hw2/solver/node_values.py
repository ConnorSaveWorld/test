"""Value of every White move at a node under the trapsearch3 model."""
import sys, os
import trapsearch3 as T
from multiprocessing import Pool
from lg import *
if __name__ == '__main__':
    init(10)
    T.pool = Pool(4, initializer=init, initargs=(25,))
    seq = [int(x) for x in sys.argv[1].split(',')]
    s = State()
    for m in seq: s = s.play(m)
    legal = [m for m in range(25) if s.play(m) is not None]
    mvs = T.batch_moves([s.play(m) for m in legal])
    rows = []
    for m, mv in zip(legal, mvs):
        ch = s.play(m)
        wins = [b for b, v in mv.items() if v == 1]
        dist, known = T.black_dist(ch, mv)
        v = sum(p for b, p in dist.items() if b not in wins)
        if len(wins) <= 3 or known:
            for b in wins:
                if dist[b] > 0.02:
                    v += dist[b] * T.eval_white(ch.play(b))[0]
        rows.append((v, m, len(wins), known))
    T.save()
    import pickle
    old = pickle.load(open('memo3.pkl', 'rb')) if os.path.exists('memo3.pkl') else {}
    old.update(T.memo); pickle.dump(old, open('memo3.pkl', 'wb'))
    for v, m, n, k in sorted(rows, reverse=True)[:10]:
        print('W%d value %.3f nwin %d %s' % (m, v, n, 'known' if k else ''))
