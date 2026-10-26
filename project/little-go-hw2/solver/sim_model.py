"""Simulate the candidate's White against a Championship *model*.

The model is not the real opponent: Black replays the recorded Championship
choice distribution where we have observations, otherwise plays a uniformly
random winning move, and with probability err(count, nwin) a random losing
move instead.  Used to check book mechanics (symmetry lookup, coverage after
errors, exact endgame) and to compare candidates under one stated model.
"""
import sys, random, time, importlib.util, os, json
from collections import defaultdict
from lg import *
from trapsearch import err, cache as MCACHE
from trapsearch2 import err2, pcache as PCACHE
MODEL = os.environ.get('MODEL', 'v2')

os.environ['LITTLE_GO_CPU_SECONDS'] = os.environ.get('SIM_CPU', '1.0')


def load_player(path):
    spec = importlib.util.spec_from_file_location('cand', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def empirical():
    tab = defaultdict(lambda: defaultdict(int))
    for s, m, g in champion_decisions():
        if s.side != 1:
            continue
        k, t = canon(s)
        tab[k][tmove(m, t)] += 1
    return tab

INV = [[i for i in range(8) if all(SYM[i][SYM[t][p]] == p for p in range(25))][0] for t in range(8)]


def black_move(s, emp, rng, use_emp=True):
    k, t = canon(s)
    if MODEL == 'v3':
        import trapsearch3 as T3
        if s.count < 6 and k in T3.EMP:
            d = T3.EMP[k]; r = rng.random() * sum(d.values())
            for cm, c in sorted(d.items()):
                r -= c
                if r < 0:
                    return tmove(cm, INV[t]), 'emp'
        if k in MCACHE:
            mv = {tmove(m, INV[t]): v for m, v in MCACHE[k].items()}
        else:
            mv = moves(s)
            MCACHE[k] = {tmove(m, t): v for m, v in mv.items()}
        dist, known = T3.black_dist(s, mv)
        r = rng.random()
        for b in sorted(dist):
            r -= dist[b]
            if r < 0:
                return b, ('error' if mv[b] == 0 else ('emp' if known else 'win'))
        return max(dist, key=dist.get), 'win'
    if use_emp and k in emp:
        d = emp[k]
        tot = sum(d.values())
        r = rng.random() * tot
        for cm, c in sorted(d.items()):
            r -= c
            if r < 0:
                return tmove(cm, INV[t]), 'emp'
    if k in MCACHE:
        mv = {tmove(m, INV[t]): v for m, v in MCACHE[k].items()}
    else:
        mv = moves(s)
        MCACHE[k] = {tmove(m, t): v for m, v in mv.items()}
    wins = [m for m, v in mv.items() if v == 1]
    losing = [m for m, v in mv.items() if v == 0]
    if not wins:
        return rng.choice(sorted(mv)), 'lost'
    e = err(s.count, len(wins)) if MODEL == 'v1' else err2(s.count, len(wins), PCACHE.get(k, 0.0))
    if losing and rng.random() < e:
        return rng.choice(sorted(losing)), 'error'
    return rng.choice(sorted(wins)), 'win'


def to_tuple(s):
    return tuple(s.board())


def play_game(mod, emp, rng):
    s = State()
    prev = (0,) * 25
    log = []
    while s.count < 24:
        if s.side == 1:
            m, how = black_move(s, emp, rng)
        else:
            board = to_tuple(s)
            last_pass = bool(s.lp)
            t = time.process_time()
            m, st = mod.white_policy(prev, board, s.count, last_pass, 1.0, t)
            how = st.get('policy', 'search')
            if m is None:
                m, _ = mod.search(prev, board, 2, s.count, last_pass, cpu_seconds=float(os.environ['LITTLE_GO_CPU_SECONDS']), wall_seconds=10)
        ns = s.play(m)
        assert ns is not None, ('illegal', s.key(), m)
        log.append((s.count, s.side, m, how))
        prev = to_tuple(s)
        end = (m == -1 and s.lp)
        s = ns
        if end:
            break
    b = s.board()
    winner = 1 if b.count(1) - b.count(2) >= 3 else 2
    return winner, log


if __name__ == '__main__':
    init(24)
    mod = load_player(sys.argv[1])
    n = int(sys.argv[2])
    rng = random.Random(int(sys.argv[3]) if len(sys.argv) > 3 else 1)
    emp = empirical()
    wins = 0
    errs = 0
    err_wins = 0
    for i in range(n):
        w, log = play_game(mod, emp, rng)
        had_err = any(h == 'error' for _, _, _, h in log)
        errs += had_err
        wins += (w == 2)
        err_wins += (w == 2 and had_err)
        seq = ','.join('%d%s' % (m, {'book': 'b', 'exact-win': 'x', 'error': '!', 'emp': 'e'}.get(h, '')) for _, _, m, h in log)
        print('game %d winner %d %s' % (i, w, seq), flush=True)
    print('white wins %d/%d; games with model error %d, won after error %d' % (wins, n, errs, err_wins))
