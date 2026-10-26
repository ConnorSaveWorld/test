"""Stage 1 local rehearsal: run the actual submission in a fresh process.

Requires Unix child CPU accounting. Opponent proxies are not official graders.
"""
import argparse
import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile
import time
try:
    import resource
except ImportError:
    resource=None
from benchmark import ROOT, tournament, InvalidMove
from probe_deployment import validate_action


def child_cpu():
    if resource is None:
        raise RuntimeError('Unix resource accounting is required')
    usage=resource.getrusage(resource.RUSAGE_CHILDREN)
    return usage.ru_utime+usage.ru_stime


def submission_policy(folder,cpu_override=None,source_path=None):
    folder=pathlib.Path(folder)
    source_path=ROOT/'submission'/'my_player3.py' if source_path is None else pathlib.Path(source_path)
    env=dict(os.environ)
    env.pop('LITTLE_GO_CPU_SECONDS',None)
    if cpu_override is not None:
        env['LITTLE_GO_CPU_SECONDS']=str(cpu_override)
    def policy(state,rng):
        text=str(state['color'])+'\n'
        for board in (state['previous'],state['board']):
            text+='\n'.join(''.join(map(str,board[r:r+5])) for r in range(0,25,5))+'\n'
        input_path=folder/'input.txt'
        input_path.write_text(text,encoding='ascii')
        output_path=folder/'output.txt'
        if output_path.exists():
            output_path.unlink()
        before=child_cpu()
        try:
            completed=subprocess.run([sys.executable,str(source_path)],
                cwd=str(folder),env=env,timeout=10,capture_output=True)
        except subprocess.TimeoutExpired as error:
            error.child_cpu=child_cpu()-before
            raise
        used=child_cpu()-before
        if completed.returncode:
            error=RuntimeError(completed.stderr.decode(errors='replace'))
            error.child_cpu=used
            raise error
        if not output_path.exists():
            error=InvalidMove('No output.txt from submission')
            error.child_cpu=used
            raise error
        text=output_path.read_text(encoding='ascii').strip()
        try:
            point=validate_action(text,str(input_path))
        except ValueError as original:
            error=InvalidMove(str(original))
            error.child_cpu=used
            raise error
        stats={'child_cpu':used}
        history_path=folder/'state.json'
        if history_path.exists():
            record=json.loads(history_path.read_text())
            stats.update(record.get('last_stats',{}))
            stats['child_cpu']=used
        return point,stats
    return policy


def save_progress(path,results,opponent,games,seed,source_path,wall_seconds,source_hash=None):
    """Atomically retain complete traces, including losses and failures."""
    wins=sum(g['winner']==g['agent_color'] for g in results)
    data={'opponent':opponent,'completed':len(results),'planned':games,
          'wins':wins,'wall_seconds':wall_seconds,'seed_start':seed,
          'source_path':str(source_path),
          'source_sha256':source_hash or hashlib.sha256(source_path.read_bytes()).hexdigest(),
          'results':results,
          'by_color':{str(c):{'games':sum(g['agent_color']==c for g in results),
                            'wins':sum(g['agent_color']==c and g['winner']==c for g in results)} for c in (1,2)}}
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(data,indent=2))
    os.replace(str(temp),str(path))
    return data


def source_limits(path):
    """Read our frozen source's deployment caps; never label a5s run as3s."""
    text=path.read_text()
    cpu=re.search(r'budget = min\(([0-9.]+), remaining / turns_left\)',text)
    wall=re.search(r'wall_seconds=min\(([0-9.]+), budget \+ 1\.0\)',text)
    if cpu is None or wall is None:
        raise ValueError('Cannot identify deployment limits in source')
    cap=float(cpu.group(1))
    return cap,min(float(wall.group(1)),cap+1.0)


def snapshot_source(path,folder):
    raw=path.read_bytes()
    frozen=folder/'_player_snapshot.py'
    frozen.write_bytes(raw)
    return frozen,hashlib.sha256(raw).hexdigest()


def run(opponent,games=20,seed=561100,source=None,tag=''):
    source_path=ROOT/'submission'/'my_player3.py' if source is None else ROOT/source
    prefix='stage1-'+(tag+'-' if tag else '')
    path=ROOT/'reports'/(prefix+opponent+'.json')
    progress=ROOT/'reports'/(prefix+opponent+'-progress.json')
    began=time.monotonic()
    def checkpoint(results):
        data=save_progress(progress,results,opponent,games,seed,source_path,time.monotonic()-began,source_hash)
        print('{} {}/{} wins={}'.format(opponent,len(results),games,data['wins']),flush=True)
    with tempfile.TemporaryDirectory(prefix='littlego-stage1-') as folder:
        frozen,source_hash=snapshot_source(source_path,pathlib.Path(folder))
        cpu_cap,wall_cap=source_limits(frozen)
        report=tournament(opponent,games,seed,cpu_cap,agent_name='submission-process',
                          policy_override=submission_policy(folder,source_path=frozen),on_game=checkpoint)
    report['configuration'].update({'execution':'actual my_player3.py subprocess per move',
        'cpu_accounting':'parent policy CPU + Unix RUSAGE_CHILDREN user/system CPU',
        'cpu_seconds':cpu_cap,'wall_seconds':wall_cap,'persistent_helper':'validated state.json'})
    report['total_agent_cpu']=sum(g['cpu_by_color'][g['agent_color']] for g in report['results'])
    report['source_sha256']=source_hash
    report['source_execution']='private byte snapshot for all fresh-process decisions'
    report['source_path']=str(source_path)
    report['elapsed_wall']=time.monotonic()-began
    report['limitations']='Local Stage1 rehearsal. Only Random is course-original; other opponents are proxies. Vocareum not tested.'
    path.write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='results'},indent=2))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--opponent',required=True,choices=('official-random','greedy-proxy','aggressive-proxy','alphabeta-proxy'))
    parser.add_argument('--games',type=int,default=20)
    parser.add_argument('--seed',type=int,default=561100)
    parser.add_argument('--source')
    parser.add_argument('--tag',default='')
    args=parser.parse_args()
    if args.games<=0 or args.games%2:
        parser.error('games must be positive and even')
    run(args.opponent,args.games,args.seed,args.source,args.tag)
