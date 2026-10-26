from collections import defaultdict
from lg import *
# canonical opening tree from our-white games vs champion
tree = defaultdict(lambda: defaultdict(int))
for g in official_games():
    if 'champ' not in g['opponent'] or g['agent_color'] != 2: continue
    s = State()
    path = []
    # canonicalize whole sequence by the transform that minimizes progressively? use per-node canonical
    for i, m in enumerate(g['moves'][:12]):
        k, t = canon(s)
        tree[k][tmove(m['point'], t)] += 1
        s = s.play(m['point'])
# print in DFS from root
def dfs(s, depth, maxd):
    k, t = canon(s)
    if k not in tree or depth > maxd: return
    tot = sum(tree[k].values())
    for cm, c in sorted(tree[k].items(), key=lambda x: -x[1]):
        # map canonical move back to actual frame: inverse transform
        inv = [i for i in range(8) if all(SYM[i][SYM[t][p]] == p for p in range(25))][0]
        m = tmove(cm, inv)
        print('  ' * depth + '%s%d %d/%d' % ('B' if s.side == 1 else 'W', m, c, tot))
        dfs(s.play(m), depth + 1, maxd)
dfs(State(), 0, 9)
