import sys, importlib.util, time
from multiprocessing import Pool
from lg import *
def load():
    spec = importlib.util.spec_from_file_location('cur', '../stage2_candidate/my_player3.py')
    cur = importlib.util.module_from_spec(spec); spec.loader.exec_module(cur); return cur
CUR = None
def job(args):
    global CUR
    if CUR is None: CUR = load()
    prev, board, color, count, lp, mv, depth = args
    for name in ('analyze', '_bit_groups', '_bit_static', '_bit_candidates'):
        getattr(CUR, name).cache_clear()
    m, st = CUR.search(tuple(prev), tuple(board), color, count, lp, cpu_seconds=120, wall_seconds=120, max_depth=depth, use_symmetry=False)
    return count, m == mv
if __name__ == '__main__':
    depths = [int(x) for x in sys.argv[1].split(',')]
    recs = []
    for g in official_games():
        if 'champ' not in g['opponent']: continue
        opp = 3 - g['agent_color']
        for m in g['moves']:
            if m['color'] == opp and m['move_count'] >= 2:
                recs.append((m['previous'], m['board'], m['color'], m['move_count'], m['last_pass'], m['point']))
    with Pool(int(sys.argv[2]) if len(sys.argv) > 2 else 3) as pool:
        for d in depths:
            t = time.time()
            res = pool.map(job, [r + (d,) for r in recs], chunksize=8)
            by = {}
            for c, ok in res:
                b = by.setdefault(c // 4 * 4, [0, 0]); b[0] += ok; b[1] += 1
            print('depth', d, 'match %d/%d' % (sum(ok for _, ok in res), len(res)), {k: '%d/%d' % tuple(v) for k, v in sorted(by.items())}, '%.0fs' % (time.time() - t), flush=True)
