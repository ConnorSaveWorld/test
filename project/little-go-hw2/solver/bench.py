import time, sys, ctypes
import lg
from lg import *
init(26)
mode = int(sys.argv[1]); lg._lib.lg_set_order(mode)
seqs = ['12,6,11,13,8,18,7,16,17,22,15,21,9,14,10,24,5,1,2,3,0,6,1,23','12,6,7,17,11,16,18,10,22,3,21,2,15,19,5,1,14,8,24,0,10,13,23,16','12,6,7,17,16,18,11,8,13,14,3,9,21,22,2,24,1,5,10,15,0,6,20,19','12,6,11,13,8,18,9,17,16,21,7,5,14,19,15,2,20,3,23,10,22,1,24,18']
import show
T=0; N=0
for c in [9,7]:
  for q in seqs:
    s = show.replay([int(x) for x in q.split(',')][:c])[-1]
    clear(); t=time.time(); n0=nodes(); r,b=solve(s); dt=time.time()-t; T+=dt; N+=nodes()-n0
    print(c, r, b, '%.2f'%dt, nodes()-n0, flush=True)
print('total %.1fs %d nodes'%(T,N))
