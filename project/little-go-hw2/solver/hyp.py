import sys, time
from lg import *
dec = champion_decisions()
print('decisions', len(dec))
for ev in [int(x) for x in sys.argv[2].split(',')]:
  for d in [int(x) for x in sys.argv[1].split(',')]:
    t=time.time(); ok=0; tot=0; amb=0; byc={}
    for s, mv, g in dec:
        vals = minimax(s, d, ev)
        bv = max(vals.values())
        bestset = [m for m,v in vals.items() if v==bv]
        tot+=1
        hit = mv in bestset
        ok += hit
        amb += len(bestset)
        c=byc.setdefault(s.count,[0,0]); c[0]+=hit; c[1]+=1
    print('ev',ev,'depth',d,'match %d/%d'%(ok,tot),'avg set %.1f'%(amb/tot),'%.1fs'%(time.time()-t), {k:'%d/%d'%tuple(v) for k,v in sorted(byc.items())}, flush=True)
