import sys, json, time
from collections import defaultdict
from multiprocessing import Pool
from lg import *
MINC = int(sys.argv[1]) if len(sys.argv) > 1 else 7

def work(key):
    init(25)
    s = State(*key)
    t = time.time()
    mv = moves(s)
    return key, mv, time.time() - t

if __name__ == '__main__':
    dec = champion_decisions()
    pos = defaultdict(lambda: defaultdict(int))
    games = defaultdict(list)
    for s, m, g in dec:
        if s.count < MINC: continue
        k, t = canon(s)
        pos[k][tmove(m, t)] += 1
        games[k].append((g['submission'], g['round'], g['agent_color']))
    keys = sorted(pos, key=lambda k: -k[4])
    print('unique positions', len(keys), flush=True)
    out = {}
    with Pool(4) as pool:
        for key, mv, dt in pool.imap_unordered(work, keys):
            wins = sorted(m for m, v in mv.items() if v == 1)
            chosen = dict(pos[key])
            err = {m: c for m, c in chosen.items() if mv.get(m) != 1}
            out[str(key)] = dict(count=key[4], side=key[2], wins=wins, legal=len(mv), chosen=chosen, games=games[key])
            if wins and err:
                print('ERROR', key[4], 'side', key[2], 'wins', wins, 'chosen', chosen, games[key], '%.1fs' % dt, flush=True)
    json.dump(out, open('champ_stats_%d.json' % MINC, 'w'))
    # summary
    summ = defaultdict(lambda: [0, 0, 0, 0])
    for v in out.values():
        n = sum(v['chosen'].values())
        e = sum(c for m, c in v['chosen'].items() if m not in v['wins'])
        w = len(v['wins'])
        row = summ[(v['count'], min(w, 3))]
        row[0] += 1; row[1] += n; row[2] += e if w else 0
    for k in sorted(summ): print('count', k[0], 'nwin', k[1], 'positions', summ[k][0], 'visits', summ[k][1], 'errors', summ[k][2])
