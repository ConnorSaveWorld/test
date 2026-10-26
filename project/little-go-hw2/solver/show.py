import sys
from lg import *
def show(s, marks={}):
    b = s.board(); rows=[]
    for r in range(5):
        rows.append(' '.join(marks.get(r*5+c, '.XO'[b[r*5+c]]) for c in range(5)))
    return '\n'.join(rows)
def replay(seq):
    s = State(); out=[s]
    for m in seq:
        s = s.play(m); out.append(s)
    return out
if __name__ == '__main__':
    seq = [int(x) for x in sys.argv[1].split(',')]
    st = replay(seq)
    s = st[-1]
    mv = moves(s)
    print('count', s.count, 'side', s.side)
    print(show(s, {m: ('W' if v else 'l') for m, v in mv.items() if m >= 0}))
    print('pass', mv.get(-1))
