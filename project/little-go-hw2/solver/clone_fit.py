"""Behaviour-cloning fit: P(move) proportional to exp(w . f(position after move)).
Features are from the mover's perspective. Optionally the features are taken
after a minimax-optimal opponent reply (depth 2) using the same eval."""
import sys, math, json
import numpy as np
from lg import *
import fit_champion as F

FULLB = (1 << 25) - 1
COL0 = sum(1 << p for p in range(0, 25, 5)); COL4 = COL0 << 4
EDGE = sum(1 << p for p in range(25) if p // 5 in (0, 4) or p % 5 in (0, 4))
CORNER = (1 << 0) | (1 << 4) | (1 << 20) | (1 << 24)
def dil(g):
    return (g | (g >> 5) | ((g << 5) & FULLB) | ((g & ~COL0) >> 1) | ((g & ~COL4) << 1)) & FULLB
def groups(stones):
    out = []
    rem = stones
    while rem:
        g = rem & -rem
        while True:
            n = dil(g) & stones
            if n == g: break
            g = n
        out.append(g); rem &= ~g
    return out
pc = lambda x: bin(x).count('1')

NAMES = ['stones', 'libs', 'atari_own', 'atari_opp', 'two_own', 'two_opp', 'groups', 'edge', 'corner', 'captured', 'self_atari', 'contact', 'influence', 'pass']
def feats(s, m, child):
    """features of child (after mover plays m), from mover perspective"""
    mover = s.side
    own = child.B if mover == 1 else child.W
    opp = child.W if mover == 1 else child.B
    empty = ~(own | opp) & FULLB
    go, gp = groups(own), groups(opp)
    lo = [pc(dil(g) & empty) for g in go]; lp = [pc(dil(g) & empty) for g in gp]
    f = np.zeros(len(NAMES))
    f[0] = pc(own) - pc(opp)
    f[1] = sum(min(x, 6) for x in lo) - sum(min(x, 6) for x in lp)
    f[2] = sum(pc(g) for g, x in zip(go, lo) if x == 1)
    f[3] = sum(pc(g) for g, x in zip(gp, lp) if x == 1)
    f[4] = sum(pc(g) for g, x in zip(go, lo) if x == 2)
    f[5] = sum(pc(g) for g, x in zip(gp, lp) if x == 2)
    f[6] = len(go) - len(gp)
    if m >= 0:
        b = 1 << m
        f[7] = 1 if b & EDGE else 0
        f[8] = 1 if b & CORNER else 0
        prev_opp = s.W if mover == 1 else s.B
        f[9] = pc(prev_opp) - pc(opp)
        g = [x for x in go if x & b][0]
        f[10] = 1 if pc(dil(g) & empty) == 1 else 0
        f[11] = 1 if dil(b) & prev_opp else 0
    else:
        f[13] = 1
    ao, ap = dil(own) & empty, dil(opp) & empty
    f[12] = pc(ao & ~ap) - pc(ap & ~ao)
    return f

def build(data):
    X = []; Y = []; C = []
    for k, obs in data.items():
        s = State(*k)
        stab = F.stabilizer(s)
        moves_ = []
        seen = set()
        for m in list(range(25)) + [-1]:
            ch = s.play(m)
            if ch is None: continue
            o = F.orbit(m, stab)
            if o in seen: continue
            seen.add(o); moves_.append((o, feats(s, m, ch)))
        idx = {o: i for i, (o, _) in enumerate(moves_)}
        M = np.array([f for _, f in moves_])
        for o, c in obs.items():
            if o in idx:
                X.append(M); Y.append(idx[o]); C.append(c)
    return X, Y, np.array(C, float)

def nll(w, X, Y, C, l2=0.01):
    tot = 0.0; g = np.zeros_like(w)
    for M, y, c in zip(X, Y, C):
        z = M @ w; z -= z.max(); p = np.exp(z); p /= p.sum()
        tot -= c * math.log(p[y] + 1e-12)
        g -= c * (M[y] - p @ M)
    tot += l2 * (w @ w); g += 2 * l2 * w
    return tot / C.sum(), g / C.sum()

def fit(X, Y, C, iters=400, lr=0.5):
    w = np.zeros(X[0].shape[1])
    for i in range(iters):
        v, g = nll(w, X, Y, C)
        w -= lr * g
    return w, nll(w, X, Y, C, 0)[0]

def top_acc(w, X, Y, C):
    return sum(c * (int(np.argmax(M @ w)) == y) for M, y, c in zip(X, Y, C)) / C.sum()

if __name__ == '__main__':
    train = F.dataset(subs=set(range(2, 13)))
    test = F.dataset(subs={13, 14})
    Xtr, Ytr, Ctr = build(train); Xte, Yte, Cte = build(test)
    w, ll = fit(Xtr, Ytr, Ctr)
    print('train ll/decision %.3f top %.3f' % (-ll, top_acc(w, Xtr, Ytr, Ctr)))
    print('test  ll/decision %.3f top %.3f' % (-nll(w, Xte, Yte, Cte, 0)[0], top_acc(w, Xte, Yte, Cte)))
    print(dict(zip(NAMES, np.round(w, 2))))
