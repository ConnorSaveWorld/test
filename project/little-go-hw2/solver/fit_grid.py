import sys, json, itertools, time
from multiprocessing import Pool
import fit_champion as F
if __name__ == '__main__':
    data = F.dataset()
    Ns = [int(x) for x in sys.argv[1].split(',')]
    Cs = [float(x) for x in sys.argv[2].split(',')]
    Ps = [int(x) for x in sys.argv[3].split(',')]
    Ss = [int(x) for x in sys.argv[4].split(',')]
    R = int(sys.argv[5]) if len(sys.argv) > 5 else 10
    import os
    extra = json.loads(os.environ.get('EXTRA', '{}'))
    rows = []
    with Pool(4) as pool:
        for n, c, p, sel in itertools.product(Ns, Cs, Ps, Ss):
            t = time.time()
            model = dict(kind='mcts', n=n, c=c, policy=p, select=sel, **extra)
            ll, top, _ = F.score(model, data, R=R, pool=pool)
            rows.append(dict(model, ll=ll, top=top))
            print('N=%6d c=%.2f policy=%d select=%d  ll=%.3f  top=%.3f  %.0fs' % (n, c, p, sel, ll, top, time.time() - t), flush=True)
    json.dump(rows, open(sys.argv[6] if len(sys.argv) > 6 else 'fit_grid.json', 'w'), indent=1)
