"""Build the White opening/trap book from trapsearch results.

Book maps canonical White-to-move states to a canonical move:
  * main trap lines (White lost against perfect play): trapsearch's choice;
  * after any Black error (White now wins): an exact winning move, for every
    Black reply, until count RUNTIME_MIN where the player's own exact search
    takes over.
"""
import sys, os, json, pickle, time, base64
from multiprocessing import Pool
from lg import *

RUNTIME_MIN = 13
INV = [[i for i in range(8) if all(SYM[i][SYM[t][p]] == p for p in range(25))][0] for t in range(8)]


def ckey(s):
    k, t = canon(s)
    return k, t


def win_subtree(key):
    """entries for a White-to-move winning state (canonical key)."""
    s0 = State(*key)
    out = {}
    stack = [s0]
    while stack:
        s = stack.pop()
        if s.count >= RUNTIME_MIN or s.count >= 24:
            continue
        k, t = ckey(s)
        if k in out:
            continue
        r, m = solve(s)
        if not r:
            out[k] = ('LOST?', None)
            continue
        out[k] = ('win', tmove(m, t))
        s1 = s.play(m)
        if s1.count >= 24 or (m == -1 and s.lp):
            continue
        for b in list(range(25)) + [-1]:
            s2 = s1.play(b)
            if s2 is None or s2.count >= 24:
                continue
            if b == -1 and s1.lp:
                continue
            stack.append(s2)
    return out


def main():
    memo = pickle.load(open('memo.pkl', 'rb'))
    cache = pickle.load(open('moves_cache.pkl', 'rb'))
    roots = [[int(x) for x in a.split(',')] for a in sys.argv[1:]]
    book = {}       # canonical key -> (kind, canonical move)
    win_jobs = set()
    stats = dict(main=0, missing=0)

    def add(s, m, kind):
        k, t = ckey(s)
        book[k] = (kind, tmove(m, t))

    def black_moves(s1):
        k, t = ckey(s1)
        if k not in cache:
            return None
        return {tmove(m, INV[t]): v for m, v in cache[k].items()}

    def main_white(s):
        k, t = ckey(s)
        if k in book:
            return
        if k not in memo:
            stats['missing'] += 1
            return
        v, cm = memo[k]
        if cm is None:
            return
        m = tmove(cm, INV[t])
        if v >= 1.0:
            win_jobs.add(k)
            return
        add(s, m, 'trap')
        stats['main'] += 1
        s1 = s.play(m)
        mv = black_moves(s1)
        if mv is None:
            return
        wins = [b for b, val in mv.items() if val == 1]
        for b, val in mv.items():
            s2 = s1.play(b)
            if s2 is None or s2.count >= 24:
                continue
            if val == 1:
                if len(wins) <= 2:
                    main_white(s2)
            else:
                win_jobs.add(ckey(s2)[0])

    for seq in roots:
        s = State()
        for i, m in enumerate(seq):
            if s.side == 2:
                add(s, m, 'prefix')
            s = s.play(m)
        main_white(s)
    print('main entries', stats, 'error branches', len(win_jobs), flush=True)
    t0 = time.time()
    with Pool(4, initializer=init, initargs=(24,)) as pool:
        for i, sub in enumerate(pool.imap_unordered(win_subtree, sorted(win_jobs), chunksize=1)):
            for k, val in sub.items():
                if k not in book:
                    book[k] = val
            if i % 20 == 0:
                print('  win subtrees %d/%d  book %d  %.0fs' % (i + 1, len(win_jobs), len(book), time.time() - t0), flush=True)
    bad = [k for k, v in book.items() if v[0] == 'LOST?']
    print('book entries', len(book), 'unexpected lost', len(bad))
    pickle.dump(book, open('book.pkl', 'wb'))


if __name__ == '__main__':
    main()
