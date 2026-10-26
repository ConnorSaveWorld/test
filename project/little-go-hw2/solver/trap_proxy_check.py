import sys, importlib.util, pickle
from lg import *
import show
spec = importlib.util.spec_from_file_location('cur', '../stage2_candidate/my_player3.py')
cur = importlib.util.module_from_spec(spec); spec.loader.exec_module(cur)
def check(seq, cpu=3.0):
    st = show.replay(seq); s = st[-1]; prev = tuple(st[-2].board()); board = tuple(s.board())
    mv = moves(s); wins = sorted(m for m, v in mv.items() if v == 1)
    res = []
    for d in (2, 4, 6):
        m, _ = cur.search(prev, board, s.side, s.count, False, cpu_seconds=60, wall_seconds=60, max_depth=d, use_symmetry=False)
        res.append(m)
    m, info = cur.search(prev, board, s.side, s.count, False, cpu_seconds=cpu, wall_seconds=cpu + 1, use_symmetry=False)
    flag = ''.join('+' if x in wins else '-' for x in res + [m])
    print('count', s.count, 'wins', wins, 'ours d2/4/6/timed', res, m, flag, 'depth', info['completed_depth'], ' ', ','.join(map(str, seq)))
for q in sys.argv[1:]:
    check([int(x) for x in q.split(',')])
