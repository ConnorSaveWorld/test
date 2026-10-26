"""Original compact TD value experiment. Training is local, never deployment.

Values always predict Black's outcome; therefore next-turn TD targets do not
flip signs. The deployment callback flips once when evaluating White's turn.
"""
import argparse
import json
import math
import random
import time
from benchmark import ROOT, player, make_policy, run_game


def transform_action(point, variant):
    if point == -1:
        return -1
    r,c=divmod(point,5)
    if variant>=4:
        c=4-c
    for _ in range(variant%4):
        r,c=c,4-r
    return 5*r+c


def inverse_action(point, variant):
    if point == -1:
        return -1
    return next(p for p in range(25) if transform_action(p,variant)==point)


def transform(board, variant):
    out=[0]*25
    for p,c in enumerate(board):
        out[transform_action(p,variant)]=c
    return tuple(out)


def normal_features(board):
    return tuple(v/scale for v,scale in zip(player.features(board),(12.5,30,25,25,25)))


class LinearValue:
    def __init__(self, weights=None, alpha=.03, gamma=.98):
        self.weights=list(weights if weights is not None else (1,.18,-1.5,-.3,.1125))
        self.alpha=alpha
        self.gamma=gamma

    def predict(self, board):
        return math.tanh(sum(w*x for w,x in zip(self.weights,normal_features(board))))

    def evaluate(self, board, color):
        value=100*self.predict(board)
        return value if color==1 else -value

    def update(self, board, next_board, reward=None):
        current=self.predict(board)
        target=reward if reward is not None else self.gamma*self.predict(next_board)
        delta=self.alpha*(target-current)*(1-current*current)
        for i,x in enumerate(normal_features(board)):
            self.weights[i]+=delta*x


def train(games=1000,seed=700000):
    learner=LinearValue()
    pool=[]
    began=time.monotonic()
    results={1:0,2:0}
    for index in range(games):
        # Freeze policies for each game; mix past versions and simple opponents.
        frozen=LinearValue(learner.weights)
        archived=LinearValue(pool[index%len(pool)] if pool else learner.weights)
        guided=make_policy('hybrid',.1,max_depth=1,evaluator=frozen.evaluate)
        older=make_policy('hybrid',.1,max_depth=1,evaluator=archived.evaluate)
        random_policy=make_policy('random')
        def exploratory(state,rng):
            return random_policy(state,rng) if rng.random()<.2 else guided(state,rng)
        opponent=older if index%3 else make_policy('greedy-proxy')
        match=run_game(exploratory,opponent,seed+index) if index%2==0 else run_game(opponent,exploratory,seed+index)
        results[match['winner']]+=1
        for i,move in enumerate(match['moves']):
            if i+1<len(match['moves']):
                nxt=tuple(match['moves'][i+1]['board'])
                learner.update(tuple(move['board']),nxt)
            else:
                learner.update(tuple(move['board']),tuple(match['board']),
                               reward=1 if match['winner']==1 else -1)
        if index%100==0:
            pool.append(list(learner.weights))
            pool=pool[-8:]
        if (index+1)%100==0:
            print('training {}/{}'.format(index+1,games),flush=True)
    return {'algorithm':'original linear TD(0), Black-oriented zero-sum value',
            'games':games,'seed_start':seed,'alpha':learner.alpha,'gamma':learner.gamma,
            'weights':learner.weights,'wall_seconds':time.monotonic()-began,
            'training_winners':results,'deployment_approved':False,
            'note':'Training outcomes are not test win rates; model is experimental.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--games',type=int,default=1000)
    parser.add_argument('--seed',type=int,default=700000)
    args=parser.parse_args()
    model=train(args.games,args.seed)
    (ROOT/'reports').mkdir(exist_ok=True)
    (ROOT/'reports'/'experimental-value.json').write_text(json.dumps(model,indent=2))
    print(json.dumps(model,indent=2))
