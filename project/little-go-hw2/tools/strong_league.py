"""Original full-legal-search league. No official Championship emulation claim."""
import argparse
import hashlib
import importlib.util
import json
import pathlib
import random
import sys
from functools import lru_cache
from benchmark import ROOT, run_game, apply

LEAGUE_SHA256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()

PROFILES={
    'baseline':(8,.6,6,1.2,.45,0),
    'pressure':(6,1,10,3,1,0),
    'survival':(8,.8,8,2,.4,3),
    'space':(6,.6,8,2,1,3),
    'space-noeyes':(6,.6,8,2,1,0),
    'material':(8,0,0,0,0,0),
}


def eye_count(module,own,enemy):
    empty=module.FULL & ~(own|enemy)
    eyes=0
    while empty:
        bit=empty & -empty
        empty^=bit
        p=bit.bit_length()-1
        if module.ADJ_BITS[p] & own != module.ADJ_BITS[p]:
            continue
        row,col=divmod(p,5)
        diagonals=[5*r+c for r,c in ((row-1,col-1),(row-1,col+1),
                                    (row+1,col-1),(row+1,col+1)) if 0<=r<5 and 0<=c<5]
        bad=sum(bool(enemy & (1<<q)) for q in diagonals)
        if bad<=(1 if len(diagonals)==4 else 0):
            eyes+=1
    return eyes


def load_player(profile='baseline',source='experiments/platform_pvs3.py'):
    path=ROOT/source
    spec=importlib.util.spec_from_file_location('league_'+profile,str(path))
    module=importlib.util.module_from_spec(spec)
    raw=path.read_bytes()
    exec(compile(raw,str(path),'exec'),module.__dict__)
    module.__source_sha256__=hashlib.sha256(raw).hexdigest()
    if profile=='baseline':
        return module
    weights=tuple(round(w*20) for w in PROFILES[profile])
    @lru_cache(maxsize=32768)
    def evaluation(black,white):
        liberties=atari=weak=0
        for color,stones,libs in module._bit_groups(black,white):
            sign=1 if color==1 else -1
            n=module._count(libs)
            liberties+=sign*min(n,6)
            if n<=1:atari+=sign*module._count(stones)
            elif n==2:weak+=sign*module._count(stones)
        empty=module.FULL & ~(black|white)
        ba,wa=module._adj(black),module._adj(white)
        influence=module._count(ba & empty & ~wa)-module._count(wa & empty & ~ba)
        eyes=eye_count(module,black,white)-eye_count(module,white,black) if weights[5] else 0
        # Exact twentieths preserve the PVS evaluation lattice.
        return (weights[0]*(module._count(black)-module._count(white)-2.5)
                +weights[1]*liberties-weights[2]*atari-weights[3]*weak
                +weights[4]*influence+weights[5]*eyes)/20.0
    module._bit_static=evaluation
    return module


def make_opening(seed,count=2):
    rng=random.Random(seed)
    previous=board=(0,)*25
    actions=[]
    for index in range(count):
        color=1+index%2
        legal=[p for p in range(25) if apply(previous,board,color,p) is not None]
        point=rng.choice(legal)
        actions.append(point)
        previous,board=board,apply(previous,board,color,point)
    return actions


def policy(module,nodes,opening,cpu=None):
    def play(state,rng):
        count=state['move_count']
        if count<len(opening):
            return opening[count],{'forced_opening':True}
        for name in ('analyze','_bit_groups','_bit_static','_bit_candidates'):
            getattr(module,name).cache_clear()
        return module.search(state['previous'],state['board'],state['color'],
            count,state['last_pass'],cpu_seconds=8 if cpu is None else cpu,
            wall_seconds=9 if cpu is None else cpu+1,
            max_nodes=nodes if cpu is None else None)
    return play


def league(profile,nodes=12000,games=8,seed=310000,opening_count=2,
           opponent_nodes=None,source='experiments/platform_pvs3.py',tag='',cpu=None,opponent_cpu=None):
    candidate=load_player(profile,source)
    opponents={name:load_player(name) for name in ('baseline','pressure','survival','space')}
    results=[]
    by_opponent={}
    path=ROOT/'reports'/('strong-'+(tag+'-' if tag else '')+profile+'.json')
    for name,opponent in opponents.items():
        subset=[]
        for index in range(games):
            color=1+index%2
            opening=make_opening(seed+index//2,opening_count)
            own=policy(candidate,nodes,opening,cpu)
            other=policy(opponent,opponent_nodes or nodes,opening,cpu if opponent_cpu is None else opponent_cpu)
            game=run_game(own if color==1 else other,other if color==1 else own,
                          seed+index//2,'official-file')
            game.update(agent_color=color,seed=seed+index//2,opponent=name,opening=opening)
            subset.append(game)
            results.append(game)
        wins=sum(g['winner']==g['agent_color'] for g in subset)
        by_opponent[name]={'games':games,'wins':wins,
            'by_color':{str(c):{'games':games//2,'wins':sum(g['winner']==c for g in subset if g['agent_color']==c)} for c in (1,2)}}
        report={'agent':profile,'games':len(results),'wins':sum(g['winner']==g['agent_color'] for g in results),
            'by_opponent':by_opponent,'results':results,'python':sys.version,
            'source_sha256':candidate.__source_sha256__,
            'opponent_source_sha256':{name:module.__source_sha256__ for name,module in opponents.items()},
            'league_sha256':LEAGUE_SHA256,
            'configuration':{'source':source,'profiles':PROFILES,'nodes':nodes,
                'opponent_nodes':opponent_nodes or nodes,'cpu':cpu,'opening_count':opening_count,
                'opponent_cpu':cpu if opponent_cpu is None else opponent_cpu,
                'seed':seed,'cache':'cold module LRU caches per searched turn'},
            'failures':{k:sum(g['reason']==k for g in results) for k in ('invalid','timeout','crash')},
            'limitations':'Original local full-legal-search pool; not official Championship. First opening_count moves forced in paired fixtures, not whole-game official rates.'}
        temp=path.with_suffix('.tmp')
        temp.write_text(json.dumps(report,indent=2))
        temp.replace(path)
        print('{} vs {}: {}/{}'.format(profile,name,wins,games),flush=True)
    print(json.dumps({k:v for k,v in report.items() if k!='results'},indent=2),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--profile',choices=tuple(PROFILES),default='baseline')
    parser.add_argument('--nodes',type=int,default=12000)
    parser.add_argument('--opponent-nodes',type=int)
    parser.add_argument('--games',type=int,default=8)
    parser.add_argument('--seed',type=int,default=310000)
    parser.add_argument('--opening-count',type=int,default=2)
    parser.add_argument('--source',default='experiments/platform_pvs3.py')
    parser.add_argument('--tag',default='')
    parser.add_argument('--cpu',type=float)
    parser.add_argument('--opponent-cpu',type=float)
    args=parser.parse_args()
    if args.games<=0 or args.games%2:parser.error('games must be positive and even')
    league(args.profile,args.nodes,args.games,args.seed,args.opening_count,
           args.opponent_nodes,args.source,args.tag,args.cpu,args.opponent_cpu)
