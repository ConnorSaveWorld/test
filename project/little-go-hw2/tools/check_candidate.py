import argparse
import importlib.util
import json
import random
from benchmark import ROOT, GO, apply


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',required=True)
    parser.add_argument('--states',type=int,default=1000)
    args=parser.parse_args()
    path=ROOT/args.source
    spec=importlib.util.spec_from_file_location('checked_candidate',str(path))
    candidate=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(candidate)
    rng=random.Random(5612026)
    previous=board=(0,)*25
    count=0;passed=False;comparisons=0
    for index in range(args.states):
        color=1+count%2
        packed=candidate._pack
        actions={p:candidate._unpack(b,w) for p,b,w in candidate._bit_moves(*packed(previous),*packed(board),color)}
        for p in range(25):
            reference=apply(previous,board,color,p)
            go=GO(5)
            go.set_board(color,[list(previous[r:r+5]) for r in range(0,25,5)],
                         [list(board[r:r+5]) for r in range(0,25,5)])
            official=None
            if go.place_chess(p//5,p%5,color):
                go.remove_died_pieces(3-color)
                official=tuple(c for row in go.board for c in row)
            assert actions.get(p)==reference==official, (index,previous,board,color,p,actions.get(p),reference,official)
            comparisons+=1
        p,nxt=rng.choice(list(actions.items()))
        if count==23 or (passed and p==-1):
            previous=board=(0,)*25;count=0;passed=False
        else:
            previous,board=board,nxt;count+=1;passed=p==-1
    report={'source':args.source,'states':args.states,'comparisons':comparisons,
            'mismatches':0,'seed':5612026,'note':'Internal optimized bitboard path versus independent matrix oracle and original official GO'}
    (ROOT/'reports'/('rules-'+path.stem+'.json')).write_text(json.dumps(report,indent=2))
    print(json.dumps(report))
