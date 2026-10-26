import argparse
import json
import random
from benchmark import ROOT, player, apply, GO


def check(states=1000,seed=561):
    rng=random.Random(seed)
    previous=board=player.EMPTY
    count=0
    passed=False
    comparisons=0
    for index in range(states):
        color=1+count%2
        actions=[]
        for p in range(25):
            go=GO(5)
            go.set_board(color,[list(previous[r:r+5]) for r in range(0,25,5)],
                         [list(board[r:r+5]) for r in range(0,25,5)])
            expected=None
            if go.place_chess(p//5,p%5,color):
                go.remove_died_pieces(3-color)
                expected=tuple(c for row in go.board for c in row)
            actual=player.play_move(previous,board,color,p)
            independent=apply(previous,board,color,p)
            if actual!=expected or actual!=independent:
                raise AssertionError({'index':index,'seed':seed,'previous':previous,
                    'board':board,'color':color,'point':p,'official':expected,
                    'agent':actual,'independent':independent})
            comparisons+=1
            if actual is not None:
                actions.append((p,actual))
        actions.append((-1,board))
        p,nxt=rng.choice(actions)
        if count==23 or (passed and p==-1):
            previous=board=player.EMPTY
            count=0
            passed=False
        else:
            previous,board=board,nxt
            count+=1
            passed=p==-1
    return {'states':states,'placements_compared':comparisons,'seed':seed,
            'mismatches':0,'sources':['submission','independent matrix oracle','official GO']}


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--states',type=int,default=1000)
    args=parser.parse_args()
    result=check(args.states)
    (ROOT/'reports').mkdir(exist_ok=True)
    (ROOT/'reports'/'rules.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result))
