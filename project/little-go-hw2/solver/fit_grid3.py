import sys, json, itertools, time, os
from multiprocessing import Pool
os.environ['CM_LIB'] = 'libcm2.so'
import fit_champion as F
if __name__ == '__main__':
    data = F.dataset()
    grid = json.loads(sys.argv[1]); R = int(sys.argv[2]); out = sys.argv[3]
    keys = sorted(grid)
    rows = []
    with Pool(4) as pool:
        for vals in itertools.product(*[grid[k] for k in keys]):
            model = dict(kind='mcts', **dict(zip(keys, vals)))
            t = time.time()
            ll, top, _ = F.score(model, data, R=R, pool=pool)
            rows.append(dict(model, ll=ll, top=top))
            print(json.dumps(model), 'll=%.3f top=%.3f %.0fs' % (ll, top, time.time() - t), flush=True)
            json.dump(rows, open(out, 'w'), indent=1)
