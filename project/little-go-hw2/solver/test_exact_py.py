import time, random, sys
from lg import *
import exact_py as X
init(25)
random.seed(1)
cases = []
for g in official_games():
    for m in g['moves']:
        if m['move_count'] in (11, 13, 15, 17, 19, 21, 23, 12, 14):
            cases.append(from_record(m['previous'], m['board'], m['color'], m['move_count']))
# random walk extra positions
for i in range(200):
    s = State()
    n = random.choice([13, 15, 17, 19, 21])
    while s.count < n:
        lm = s.legal()
        s = s.play(random.choice(lm))
    cases.append(s)
random.shuffle(cases)
cases = cases[:int(sys.argv[1])]
bad = 0; stats = {}
for s in cases:
    r, b = solve(s)
    t = time.process_time()
    st, mv = X.exact_solve(s.B, s.W, s.side, s.ko, s.count, s.lp, 60)
    dt = time.process_time() - t
    ok = (st == 'win') == bool(r)
    if ok and st == 'win':
        # verify the python move is winning per C
        ch = s.play(mv)
        if ch is None: ok = False
        elif ch.count >= 24 or (mv == -1 and s.lp):
            pass
        else:
            ok = not solve(ch)[0]
    if not ok:
        bad += 1; print('MISMATCH', s.key(), r, b, st, mv)
    d = stats.setdefault((s.count, st), [0, 0.0, 0.0]); d[0] += 1; d[1] += dt; d[2] = max(d[2], dt)
print('cases', len(cases), 'bad', bad)
for k in sorted(stats): print(k, 'n=%d avg %.3fs max %.3fs' % (stats[k][0], stats[k][1] / stats[k][0], stats[k][2]))
