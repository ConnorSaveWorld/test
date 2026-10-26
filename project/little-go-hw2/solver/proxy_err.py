"""Does our own search (as a stand-in heuristic player) miss the unique winning
move at the nodes where the champion erred / did not err?"""
import json, sys, importlib.util, time
from lg import *
spec = importlib.util.spec_from_file_location('cur', '../stage2_candidate/my_player3.py')
cur = importlib.util.module_from_spec(spec); spec.loader.exec_module(cur)
d = json.load(open('champ_stats_6.json'))
cpu = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0
for k, v in d.items():
    if len(v['wins']) != 1 or v['count'] > 14: continue
    key = eval(k); s = State(*key)
    # reconstruct previous board (ko ignored: use board itself unless ko) -- previous only matters for ko
    board = tuple(s.board()); prev = board if s.lp else tuple(0 for _ in range(25))
    prev = board  # no-ko approximation is fine when ko==0
    if s.ko: print('ko position skipped'); continue
    res = []
    for depth in (2, 4, 6):
        m, st = cur.search(tuple(0 if x else 0 for x in board) if False else board, board, s.side, s.count, False, cpu_seconds=60, wall_seconds=60, max_depth=depth)
        res.append(m)
    m, st = cur.search(board, board, s.side, s.count, False, cpu_seconds=cpu, wall_seconds=cpu + 1)
    print('count', v['count'], 'win', v['wins'], 'champ', v['chosen'], 'ours d2/4/6', res, 'timed', m, 'depth', st['completed_depth'])
