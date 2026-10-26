"""ctypes wrapper around lgsolve.c (offline analysis only)."""
import ctypes, json, os
from pathlib import Path
HERE = Path(__file__).resolve().parent
_lib = ctypes.CDLL(str(HERE / 'liblg.so'))
u32 = ctypes.c_uint32
_lib.lg_init.argtypes = [ctypes.c_int]
_lib.lg_solve.argtypes = [u32, u32, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
_lib.lg_moves.argtypes = [u32, u32, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
_lib.lg_play.argtypes = [u32, u32, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.POINTER(u32), ctypes.POINTER(u32), ctypes.POINTER(ctypes.c_int)]
_lib.lg_nodes.restype = ctypes.c_uint64
_inited = False

def init(log2=26):
    global _inited
    _lib.lg_init(log2); _inited = True

def clear():
    _lib.lg_clear()

class State:
    __slots__ = ('B', 'W', 'side', 'ko', 'count', 'lp')
    def __init__(self, B=0, W=0, side=1, ko=0, count=0, lp=0):
        self.B, self.W, self.side, self.ko, self.count, self.lp = B, W, side, ko, count, lp
    def key(self):
        return (self.B, self.W, self.side, self.ko, self.count, self.lp)
    def play(self, m):
        oB, oW, oko = u32(), u32(), ctypes.c_int()
        if not _lib.lg_play(self.B, self.W, self.side, self.ko, m, ctypes.byref(oB), ctypes.byref(oW), ctypes.byref(oko)):
            return None
        return State(oB.value, oW.value, 3 - self.side, oko.value, self.count + 1, 1 if m == -1 else 0)
    def terminal(self):
        return self.count >= 24
    def legal(self):
        return [m for m in list(range(25)) + [-1] if self.play(m) is not None]
    def board(self):
        return [1 if self.B >> p & 1 else 2 if self.W >> p & 1 else 0 for p in range(25)]

def from_record(prev, board, color, count):
    """Build a state from previous/current 25-lists as the host gives them."""
    B = sum(1 << p for p in range(25) if board[p] == 1)
    W = sum(1 << p for p in range(25) if board[p] == 2)
    lp = 1 if (prev == board and count > 0) else 0
    ko = 0
    opp = 3 - color
    placed = [p for p in range(25) if board[p] == opp and prev[p] != opp]
    died = [p for p in range(25) if prev[p] == color and board[p] != color]
    if len(placed) == 1 and len(died) == 1:
        ko = 1 + placed[0] * 25 + died[0]
    return State(B, W, color, ko, count, lp)

def solve(s):
    if not _inited: init()
    best = ctypes.c_int()
    r = _lib.lg_solve(s.B, s.W, s.side, s.ko, s.count, s.lp, ctypes.byref(best))
    return r, best.value

def moves(s):
    if not _inited: init()
    out = (ctypes.c_int * 26)()
    _lib.lg_moves(s.B, s.W, s.side, s.ko, s.count, s.lp, out)
    return {(m if m < 25 else -1): out[m] for m in range(26) if out[m] >= 0}

def nodes():
    return _lib.lg_nodes()

ROOT = HERE.parent
def official_games():
    out = []
    base = ROOT / 'reports/official-all-submissions-20261025/submit'
    for n in range(2, 13):
        f = base / ('cnt%d' % n) / 'parsed-games.json'
        if not f.exists(): continue
        for r in json.loads(f.read_text())['results']:
            r = dict(r); r['submission'] = n
            out.append(r)
    return out

_lib.lg_minimax.argtypes = [u32, u32, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
def minimax(s, depth, evtype=0):
    out = (ctypes.c_int * 26)()
    _lib.lg_minimax(s.B, s.W, s.side, s.ko, s.count, s.lp, depth, evtype, out)
    return {(m if m < 25 else -1): out[m] for m in range(26) if out[m] > -9999999}

def champion_decisions():
    """(state, move, game) for every championship-player decision on record."""
    res = []
    for g in official_games():
        if 'champ' not in g['opponent']: continue
        opp = 3 - g['agent_color']
        for m in g['moves']:
            if m['color'] != opp: continue
            res.append((from_record(m['previous'], m['board'], m['color'], m['move_count']), m['point'], g))
    return res

# ---- D4 symmetry ----
def _mk_maps():
    maps = []
    for t in range(8):
        mp = []
        for p in range(25):
            r, c = divmod(p, 5)
            for _ in range(t & 3):
                r, c = c, 4 - r
            if t & 4:
                c = 4 - c
            mp.append(r * 5 + c)
        maps.append(mp)
    return maps
SYM = _mk_maps()
def _tb(x, mp):
    y = 0
    while x:
        lsb = x & -x; p = lsb.bit_length() - 1; x ^= lsb
        y |= 1 << mp[p]
    return y
def transform(s, t):
    mp = SYM[t]
    ko = s.ko
    if ko:
        P, Q = divmod(ko - 1, 25); ko = 1 + mp[P] * 25 + mp[Q]
    return State(_tb(s.B, mp), _tb(s.W, mp), s.side, ko, s.count, s.lp)
def tmove(m, t):
    return m if m < 0 else SYM[t][m]
def canon(s):
    """returns (canonical key, transform t) such that transform(s,t) is canonical"""
    best = None
    for t in range(8):
        k = transform(s, t).key()
        if best is None or k < best[0]:
            best = (k, t)
    return best
