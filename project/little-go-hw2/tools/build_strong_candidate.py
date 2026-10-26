"""Build frozen single-file candidates from our original PVS source."""
import argparse
from benchmark import ROOT
from strong_league import PROFILES

EYE_CODE='''
TOP_EDGE = 31
BOTTOM_EDGE = TOP_EDGE << 20
DIAGONALS = tuple(sum(1 << (5*r+c) for r,c in
    ((p//5-1,p%5-1),(p//5-1,p%5+1),(p//5+1,p%5-1),(p//5+1,p%5+1))
    if 0<=r<5 and 0<=c<5) for p in range(25))
INTERIOR = sum(1 << (5*r+c) for r in range(1,4) for c in range(1,4))


def _eye_count(own,enemy):
    # Eye potential influences heuristics only; never score or legality.
    empty=FULL & ~(own|enemy)
    possible=(empty & ((own<<5)|TOP_EDGE) & ((own>>5)|BOTTOM_EDGE)
              & ((own<<1)|LEFT_EDGE) & ((own>>1)|RIGHT_EDGE))
    eyes=0
    while possible:
        bit=possible & -possible
        possible^=bit
        p=bit.bit_length()-1
        if _count(DIAGONALS[p] & enemy)<=(1 if bit & INTERIOR else 0):
            eyes+=1
    return eyes


@lru_cache(maxsize=32768)
def _bit_static(black,white):
    liberties=atari=weak=0
    for color,stones,libs in _bit_groups(black,white):
        sign=1 if color==1 else -1
        n=_count(libs)
        liberties+=sign*min(n,6)
        if n<=1:atari+=sign*_count(stones)
        elif n==2:weak+=sign*_count(stones)
    empty=FULL & ~(black|white)
    ba,wa=_adj(black),_adj(white)
    influence=_count(ba & empty & ~wa)-_count(wa & empty & ~ba)
    eyes=_eye_count(black,white)-_eye_count(white,black) if WEIGHTS[5] else 0
    return (WEIGHTS[0]*(_count(black)-_count(white)-2.5)+WEIGHTS[1]*liberties
            -WEIGHTS[2]*atari-WEIGHTS[3]*weak+WEIGHTS[4]*influence+WEIGHTS[5]*eyes)/20.0


def evaluate(board,color):
    return _bit_evaluate(*_pack(board),color)
'''

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--profile',choices=tuple(PROFILES),default='space')
    parser.add_argument('--budget',type=int,choices=(3,5),default=3)
    parser.add_argument('--output',default='candidate_space3.py')
    args=parser.parse_args()
    base=(ROOT/'experiments'/'platform_pvs3.py').read_text().split("if __name__ == '__main__':")[0]
    if args.budget==5:
        base=base.replace('budget = min(3.0, remaining / turns_left)',
                          'budget = min(5.0, remaining / turns_left)')
        base=base.replace('wall_seconds=min(5.0, budget + 1.0)',
                          'wall_seconds=min(7.0, budget + 1.0)')
    weights=tuple(round(w*20) for w in PROFILES[args.profile])
    code=base+'\nWEIGHTS = '+repr(weights)+'\n'+EYE_CODE+"\n\nif __name__ == '__main__':\n    main()\n"
    (ROOT/'experiments'/args.output).write_text(code)
