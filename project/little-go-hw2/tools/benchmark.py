"""Local tournaments; proxy opponents must not be equated with graders."""
import argparse
import hashlib
import importlib.util
import json
import math
import pathlib
import platform
import random
import statistics
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tests'))
sys.path.insert(0, str(ROOT / 'reference'))
from reference_rules import apply
from host import GO
from random_player import RandomPlayer

spec = importlib.util.spec_from_file_location('agent_submission',
                   str(ROOT / 'submission' / 'my_player3.py'))
player = importlib.util.module_from_spec(spec)
spec.loader.exec_module(player)


class InvalidMove(ValueError):
    pass


def run_game(black, white, seed=0, mode='pdf'):
    rng = random.Random(seed)
    previous = board = (0,) * 25
    passed = False
    moves = []
    cpu_by_color = [0.0, 0.0, 0.0]
    for count in range(24):
        color = 1 + count % 2
        state = {'previous': previous, 'board': board, 'color': color,
                 'move_count': count, 'last_pass': passed}
        began_cpu, began_wall = time.process_time(), time.monotonic()
        try:
            action, stats = (black if color == 1 else white)(state, rng)
        except Exception as error:
            reason=('timeout' if isinstance(error,subprocess.TimeoutExpired) else
                    'invalid' if isinstance(error,InvalidMove) else 'crash')
            failed_cpu=time.process_time()-began_cpu+getattr(error,'child_cpu',0.0)
            cpu_by_color[color]+=failed_cpu
            moves.append(dict(state,point=None,cpu=failed_cpu,
                wall=time.monotonic()-began_wall,stats={},error=repr(error)))
            return {'winner': 3-color, 'steps': count+1, 'reason': reason,
                    'error': repr(error), 'failed_color': color, 'board': list(board),
                    'moves': moves, 'cpu_by_color': cpu_by_color}
        cpu, wall = time.process_time()-began_cpu, time.monotonic()-began_wall
        cpu += stats.get('child_cpu',0.0)
        cpu_by_color[color] += cpu
        record = dict(state, point=action, cpu=cpu, wall=wall, stats=stats)
        moves.append(record)
        if cpu > 10:
            reason = 'timeout'
            updated = None
        else:
            updated = apply(previous, board, color, action)
            reason = 'invalid'
        if updated is None:
            return {'winner': 3-color, 'steps': count+1, 'reason': reason,
                    'failed_color': color, 'board': list(board), 'moves': moves,
                    'cpu_by_color': cpu_by_color}
        ends = action == -1 and (passed or
                    (mode == 'official-file' and previous == board))
        previous, board = board, updated
        passed = action == -1
        if ends or count == 23:
            winner = 1 if board.count(1) > board.count(2)+2.5 else 2
            return {'winner': winner, 'steps': count+1,
                    'reason': 'passes' if ends else 'limit', 'board': list(board),
                    'moves': moves, 'cpu_by_color': cpu_by_color}


def make_policy(name, cpu_seconds=0.025, max_depth=None, evaluator=None,
                use_table=True, use_ordering=True):
    def policy(state, rng):
        prev, board, color = state['previous'], state['board'], state['color']
        if name in ('search', 'hybrid'):
            return player.search(prev, board, color, state['move_count'],
                state['last_pass'], cpu_seconds=cpu_seconds,
                wall_seconds=max(1, cpu_seconds+0.5), max_depth=max_depth,
                evaluator=evaluator, use_table=use_table, use_ordering=use_ordering)
        actions = player.legal_moves(prev, board, color)
        placements = [(p,b) for p,b in actions if p != -1]
        if not placements:
            return -1, {}
        if name == 'official-random':
            go = GO(5)
            go.set_board(color, [list(prev[r:r+5]) for r in range(0,25,5)],
                         [list(board[r:r+5]) for r in range(0,25,5)])
            # Course player uses global random; preserve its code and seed per call.
            saved = random.getstate()
            random.seed(rng.getrandbits(64))
            try:
                move = RandomPlayer().get_input(go, color)
            finally:
                random.setstate(saved)
            return (-1 if move == 'PASS' else 5*move[0]+move[1]), {}
        if name == 'random':
            return rng.choice(placements)[0], {}
        if name == 'learned-policy':
            scored = [(evaluator(b,color),p) for p,b in actions]
        elif name == 'greedy-proxy':
            scored = [(board.count(3-color)-b.count(3-color),p) for p,b in placements]
        elif name == 'aggressive-proxy':
            scored = []
            for p,b in placements:
                captured = board.count(3-color)-b.count(3-color)
                replies = player.legal_moves(board,b,3-color)
                lost = max(b.count(color)-reply.count(color) for _,reply in replies)
                scored.append((captured-lost,p))
        elif name == 'alphabeta-proxy':
            # Explicit local approximation: depth 2, 10 placements plus PASS.
            def truncated(a,b,c):
                moves = player.ordered_moves(a,b,c)
                return [m for m in moves if m[0] != -1][:10] + [(-1,b)]
            scored = []
            for p,b in truncated(prev,board,color):
                if ((p == -1 and state['last_pass']) or state['move_count'] == 23):
                    val = player.terminal_value(b,color)
                else:
                    vals = []
                    for q,c in truncated(board,b,3-color):
                        if (q == -1 and p == -1) or state['move_count'] >= 22:
                            vals.append(player.terminal_value(c,color))
                        else:
                            vals.append(player.evaluate(c,color))
                    val = min(vals)
                scored.append((val,p))
        else:
            raise ValueError('Unknown policy ' + name)
        best = max(v for v,_ in scored)
        return rng.choice([p for v,p in scored if v == best]), {}
    return policy


def wilson(wins, n):
    if not n:
        return [0.0,1.0]
    z = 1.959963984540054
    p = wins/n
    den = 1+z*z/n
    center = (p+z*z/(2*n))/den
    half = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [center-half,center+half]


def tournament(opponent, games=200, seed=10000, cpu_seconds=0.025,
               max_depth=None, agent_name='search', evaluator=None,
               use_table=True, use_ordering=True, policy_override=None,
               on_game=None):
    results = []
    agent = policy_override if policy_override is not None else make_policy(agent_name,cpu_seconds,max_depth,evaluator,use_table,use_ordering)
    other = make_policy(opponent,cpu_seconds,max_depth)
    for i in range(games):
        color = 1 + i % 2
        game = run_game(agent if color == 1 else other,
                        other if color == 1 else agent, seed+i, 'official-file')
        game.update(seed=seed+i, agent_color=color)
        results.append(game)
        if on_game is not None:
            on_game(results)
        if (i+1) % 20 == 0:
            print('{} {}/{} wins={}'.format(opponent,i+1,games,
                  sum(g['winner']==g['agent_color'] for g in results)),flush=True)
    wins = sum(g['winner']==g['agent_color'] for g in results)
    agent_moves = [m for g in results for m in g['moves'] if m['color']==g['agent_color']]
    times = sorted(m['cpu'] for m in agent_moves)
    failures = {reason:sum(g['reason']==reason and g.get('failed_color')==g['agent_color']
                          for g in results) for reason in ('invalid','timeout','crash')}
    report = {'opponent':opponent,'agent':agent_name,'games':games,'wins':wins,
              'win_rate':wins/games,'wilson95':wilson(wins,games),
              'configuration':{'cpu_seconds':cpu_seconds,'max_depth':max_depth,
                  'use_table':use_table,'use_ordering':use_ordering},
              'seed_start':seed,'python':sys.version,'platform':platform.platform(),
              'source_sha256':hashlib.sha256((ROOT/'submission'/'my_player3.py').read_bytes()).hexdigest(),
              'mean_cpu':statistics.mean(times) if times else 0,
              'p95_cpu':times[min(len(times)-1,int(.95*len(times)))] if times else 0,
              'max_cpu':max(times) if times else 0,
              'max_wall':max((m['wall'] for m in agent_moves),default=0),
              'max_game_cpu':max((g['cpu_by_color'][g['agent_color']] for g in results),default=0),
              'failures':failures,'by_color':{}, 'results':results,
              'limitations':'In-process local timing; proxy opponents are not hidden grading agents. Wilson assumes independent games.'}
    for color in (1,2):
        subset=[g for g in results if g['agent_color']==color]
        report['by_color'][str(color)]={'games':len(subset),
                        'wins':sum(g['winner']==color for g in subset)}
    return report


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--opponent',default='official-random')
    parser.add_argument('--games',type=int,default=200)
    parser.add_argument('--seed',type=int,default=10000)
    parser.add_argument('--cpu',type=float,default=.025)
    parser.add_argument('--depth',type=int)
    parser.add_argument('--output',default='reports/random.json')
    parser.add_argument('--no-table',action='store_true')
    parser.add_argument('--no-ordering',action='store_true')
    args=parser.parse_args()
    if args.games <= 0 or args.games % 2:
        parser.error('games must be a positive even number')
    report=tournament(args.opponent,args.games,args.seed,args.cpu,args.depth,
                      use_table=not args.no_table,use_ordering=not args.no_ordering)
    path=ROOT/args.output
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='results'},indent=2))
