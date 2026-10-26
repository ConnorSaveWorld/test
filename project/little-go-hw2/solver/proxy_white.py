"""White regression vs local proxy Black engines (not the Championship).
Compares current stage2_candidate (search only) with stage2_trap (book +
exact endgame + same search). Black's first move is forced to 12 so the
book line is exercised; proxies then play by node-limited search."""
import sys, os, json, time, random
sys.path.insert(0, 'tools')
import strong_league
from benchmark import run_game

WHITE_NODES = 200000

def white_play(module, use_policy):
    def play(state, rng):
        for name in ('analyze', '_bit_groups', '_bit_static', '_bit_candidates'):
            getattr(module, name).cache_clear()
        if use_policy:
            m, st = module.white_policy(state['previous'], state['board'], state['move_count'],
                                       state['last_pass'], 7.0, time.process_time())
            if m is not None:
                return m, st
        return module.search(state['previous'], state['board'], 2, state['move_count'],
                             state['last_pass'], cpu_seconds=30, wall_seconds=30, max_nodes=WHITE_NODES)
    return play

def black_play(module, nodes, forced):
    def play(state, rng):
        if state['move_count'] < len(forced):
            return forced[state['move_count']], {}
        for name in ('analyze', '_bit_groups', '_bit_static', '_bit_candidates'):
            getattr(module, name).cache_clear()
        return module.search(state['previous'], state['board'], 1, state['move_count'],
                             state['last_pass'], cpu_seconds=30, wall_seconds=30, max_nodes=nodes)
    return play

if __name__ == '__main__':
    cands = {'current': ('stage2_candidate/my_player3.py', False), 'trap': ('stage2_trap/my_player3.py', True)}
    only = os.environ.get('ONLY')
    if only: cands = {only: cands[only]}
    out = {}
    for cname, (src, pol) in cands.items():
        mod = strong_league.load_player('baseline', src)
        rows = []
        for prof in ('baseline', 'pressure', 'survival', 'space'):
            opp = strong_league.load_player(prof)
            for nodes in (24000, 48000, 96000):
                for forced in ([12], [12, 6, 11]):
                    g = run_game(black_play(opp, nodes, forced), white_play(mod, pol), 1, 'official-file')
                    b = g['board']; margin = b.count(2) + 2.5 - b.count(1)
                    rows.append(dict(profile=prof, nodes=nodes, forced=forced, winner=g['winner'], reason=g['reason'], margin=margin,
                                     seq=[m['point'] for m in g['moves']]))
                    print(cname, prof, nodes, forced, 'winner', g['winner'], g['reason'], 'margin', margin, flush=True)
        out[cname] = rows
        print(cname, 'white wins', sum(r['winner'] == 2 for r in rows), '/', len(rows), flush=True)
    json.dump(out, open(os.environ.get('OUT', 'solver/proxy_white.json'), 'w'), indent=1)
