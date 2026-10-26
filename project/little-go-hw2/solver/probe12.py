import time, sys
from lg import *
init(26)
seen=set()
for g in official_games():
    if g['submission']!=12 or g['agent_color']!=2 or 'champ' not in g['opponent']: continue
    seq=tuple(m['point'] for m in g['moves'])
    if seq in seen: continue
    seen.add(seq)
    for c in [int(x) for x in sys.argv[1:]]:
        m=g['moves'][c]; s=from_record(m['previous'],m['board'],m['color'],c)
        clear(); t=time.time(); n0=nodes(); r,b=solve(s)
        print(g['round'],c,'side',s.side,'win' if r else 'loss','best',b,'played',m['point'],'%.2fs'%(time.time()-t), nodes()-n0, flush=True)
