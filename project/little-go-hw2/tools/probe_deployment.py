"""Full fresh-process matches through the unmodified official file host."""
import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import time
from benchmark import ROOT, GO, player


def validate_action(text,input_path):
    color,previous,board=player.read_input(input_path)
    if text=='PASS':
        return -1
    fields=text.split(',')
    if len(fields)!=2 or any(len(c)!=1 or c not in '01234' for c in fields):
        raise ValueError('Malformed or out-of-range output: '+text)
    row,col=map(int,fields)
    go=GO(5)
    go.set_board(color,[list(previous[r:r+5]) for r in range(0,25,5)],
                 [list(board[r:r+5]) for r in range(0,25,5)])
    if not go.valid_place_check(row,col,color,test_check=True):
        raise ValueError('Official GO rejects output: '+text)
    return 5*row+col


def probe(games=2,cpu=None,output='reports/deployment-interface.json',
          source='submission/my_player3.py',color=None):
    if color not in (None, 1, 2):
        raise ValueError('color must be 1, 2, or None')
    results=[]
    source_path=ROOT/source
    raw=source_path.read_bytes()
    source_hash=hashlib.sha256(raw).hexdigest()
    for game_index in range(games):
        own_color=color if color is not None else 1+game_index%2
        with tempfile.TemporaryDirectory() as folder:
            work=pathlib.Path(folder)
            frozen=work/'_player_snapshot.py'
            frozen.write_bytes(raw)
            (work/'input.txt').write_bytes((ROOT/'reference'/'init'/'input.txt').read_bytes())
            moves=[]
            env=dict(os.environ)
            if cpu is not None:
                env['LITTLE_GO_CPU_SECONDS']=str(cpu)
            else:
                env.pop('LITTLE_GO_CPU_SECONDS',None)
            for count in range(1,25):
                color=1+(count-1)%2
                path=frozen if color==own_color else (ROOT/'reference'/'random_player.py')
                began=time.monotonic()
                completed=subprocess.run([sys.executable,str(path)],cwd=folder,env=env,
                                         timeout=10,capture_output=True)
                if completed.returncode:
                    raise RuntimeError(completed.stderr.decode(errors='replace'))
                elapsed=time.monotonic()-began
                action=(work/'output.txt').read_text().strip()
                validate_action(action,str(work/'input.txt'))
                data={'count':count,'color':color,'action':action,'wall':elapsed}
                if color==own_color:
                    data['history']=json.loads((work/'state.json').read_text())
                moves.append(data)
                host=subprocess.run([sys.executable,str(ROOT/'reference'/'host.py'),'-m',str(count)],
                                     cwd=folder,timeout=10,capture_output=True)
                if host.returncode not in (0,1,2):
                    raise RuntimeError(host.stdout.decode(errors='replace'))
                if b'invalid' in host.stdout.lower():
                    raise RuntimeError('Invalid move: '+host.stdout.decode(errors='replace'))
                if host.returncode:
                    own=[m for m in moves if m['color']==own_color]
                    results.append({'agent_color':own_color,'winner':host.returncode,
                        'steps':count,'max_agent_wall':max(m['wall'] for m in own),
                        'agent_reported_cpu':own[-1]['history']['cpu_used'], 'moves':moves})
                    break
            else:
                raise RuntimeError('File host failed to terminate at move 24')
        print('deployment game {} complete'.format(game_index+1),flush=True)
    report={'python':sys.version,'budget_override':cpu,'games':games,'requested_color':color,'results':results,
            'source_path':str(source_path),'source_sha256':source_hash,
            'source_execution':'private byte snapshot',
            'legality_validation':'fresh official GO for every output',
            'note':'Actual fresh-process interface check against official Random; small unseeded sample, not a win-rate estimate.'}
    (ROOT/'reports').mkdir(exist_ok=True)
    (ROOT/output).write_text(json.dumps(report,indent=2))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--games',type=int,default=2)
    parser.add_argument('--cpu',type=float)
    parser.add_argument('--output',default='reports/deployment-interface.json')
    parser.add_argument('--source',default='submission/my_player3.py')
    parser.add_argument('--color',type=int,choices=(1,2))
    args=parser.parse_args()
    result=probe(args.games,args.cpu,args.output,args.source,args.color)
    print(json.dumps({k:v for k,v in result.items() if k!='results'},indent=2))
    print([(g['agent_color'],g['winner'],g['max_agent_wall'],g['agent_reported_cpu']) for g in result['results']])
