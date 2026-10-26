import pickle, os
from lg import *
book = pickle.load(open('book.pkl', 'rb'))
cache = pickle.load(open('moves_cache.pkl', 'rb'))
INV = [[i for i in range(8) if all(SYM[i][SYM[t][p]] == p for p in range(25))][0] for t in range(8)]
from trapsearch import err
paths = []
def walk(s, path, prob):
    k, t = canon(s)
    if k not in book or book[k][0] != 'trap' and s.count > 3:
        paths.append((path, prob)); return
    kind, cm = book[k]
    m = tmove(cm, INV[t])
    s1 = s.play(m)
    k1, t1 = canon(s1)
    if k1 not in cache:
        paths.append((path + ['W%d' % m], prob)); return
    mv = {tmove(x, INV[t1]): v for x, v in cache[k1].items()}
    wins = [b for b, v in mv.items() if v == 1]
    e = err(s1.count, len(wins))
    for b in wins:
        walk(s1.play(b), path + ['W%d' % m, 'B%d[%d,e=%.2f]' % (b, len(wins), e)], prob * (1 - e) / len(wins))
s = State().play(12)
s = s.play(6).play(11).play(7).play(8)
walk(s, ['B12', 'W6', 'B11', 'W7', 'B8'], 1.0)
for p, pr in paths:
    print('%.3f' % pr, ' '.join(p))
