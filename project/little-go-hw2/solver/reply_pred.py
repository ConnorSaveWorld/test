import os, sys, json
from collections import Counter
import fit_champion as F
from lg import *
import show
models = [dict(kind='mcts', n=40000, c=0.7, policy=1, select=0), dict(kind='mcts', n=10000, c=0.7, policy=0, select=0),
          dict(kind='mcts', n=3000, c=0.7, policy=2, select=0, noise=4, connect=1, cutoff=0)]
libs = ['libcm.so', 'libcm.so', 'libcm2.so']
for seq in [[int(x) for x in a.split(',')] for a in sys.argv[1:]]:
    s = show.replay(seq)[-1]
    k, t = canon(s)
    inv = [i for i in range(8) if all(SYM[i][SYM[t][p]] == p for p in range(25))][0]
    for mdl, lib in zip(models, libs):
        F._cm = None; os.environ['CM_LIB'] = lib
        _, cnt, _ = F.run_model((k, mdl, 30, 3))
        # back to actual frame
        print(seq, mdl['n'], mdl['policy'], {tmove(m, inv): c for m, c in sorted(cnt.items(), key=lambda x: -x[1])})
