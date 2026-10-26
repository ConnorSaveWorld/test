import sys, time, json
from multiprocessing import Pool
from lg import *
CAP = 3
PREF = (12, 7, 11, 13, 17, 6, 8, 16, 18, 2, 10, 14, 22, 1, 3, 5, 9, 15, 19, 21, 23, 0, 4, 20, 24)

def scan(args):
    seq, wm = args
    init(26)
    s = State()
    for m in seq: s = s.play(m)
    s1 = s.play(wm)
    wins = []; losses = []
    t = time.time()
    for b in PREF:
        s2 = s1.play(b)
        if s2 is None: continue
        r, _ = solve(s2)
        if r: losses.append(b)
        else:
            wins.append(b)
            if len(wins) >= CAP: break
    return wm, wins, losses, time.time() - t

if __name__ == '__main__':
    seq = [int(x) for x in sys.argv[1].split(',')]
    s = State()
    for m in seq: s = s.play(m)
    # one representative per symmetry orbit of white moves
    seen = set(); jobs = []
    for m in range(25):
        c = s.play(m)
        if c is None: continue
        k = canon(c)[0]
        if k in seen: continue
        seen.add(k); jobs.append((seq, m))
    print('orbits', len(jobs), flush=True)
    with Pool(int(sys.argv[2]) if len(sys.argv) > 2 else 3) as pool:
        for wm, wins, losses, dt in pool.imap_unordered(scan, jobs):
            print('W%d: black wins(first %d) %s ; losing tried %s  %.0fs' % (wm, CAP, wins, losses, dt), flush=True)
