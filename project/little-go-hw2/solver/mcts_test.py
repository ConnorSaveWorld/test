import ctypes, sys
from collections import Counter
import show
from lg import State, canon, tmove
cm = ctypes.CDLL('./libcm.so')
u32 = ctypes.c_uint32
cm.mcts_choose.argtypes = [u32, u32, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.POINTER(ctypes.c_int)]
cm.cm_seed.argtypes = [ctypes.c_uint64]
def dist(seq, iters, runs, c=1.4):
    s = show.replay(seq)[-1]
    out = (ctypes.c_int * 26)()
    cnt = Counter()
    for r in range(runs):
        cm.cm_seed(1234567 + r * 7919)
        m = cm.mcts_choose(s.B, s.W, s.side, s.ko, s.count, s.lp, iters, c, out)
        cnt[m] += 1
    return dict(cnt)
cases = {'W reply to 12 (obs adj 34: diag 6)': [12], 'count4 after 12,6,11,7 (obs 8:30)': [12,6,11,7],
         'count6 (obs 3:8, 18:2)': [12,6,11,7,8,13], 'count8 (obs 17:6, 2:2; wins 16,17)': [12,6,11,7,8,13,3,18]}
for iters in [int(x) for x in sys.argv[1].split(',')]:
    for name, seq in cases.items():
        print(iters, name, dist(seq, iters, int(sys.argv[2])), flush=True)
