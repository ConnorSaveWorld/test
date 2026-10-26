"""Exact win/loss search on actual positions; diagnostic only."""
import json
import time
from pathlib import Path
from strong_league import load_player

ROOT = Path(__file__).resolve().parents[1]
SOURCE = 'reports/official-submission12-20261023/cnt12/my_player3.py'


def prove(module, state, seconds=8, max_nodes=1000000):
    start = time.process_time()
    deadline = start + seconds
    nodes = 0
    cache = {}
    def solve(pb, pw, b, w, side, count, passed):
        nonlocal nodes
        nodes += 1
        if nodes % 128 == 0 and (nodes >= max_nodes or time.process_time() >= deadline):
            raise TimeoutError()
        key = (pb, pw, b, w, side, count, passed)
        if key in cache:
            return cache[key]
        actions = module._bit_order(pb, pw, b, w, side, passed=passed)
        for point, nb, nw in actions:
            if count + 1 >= 24 or (point == module.PASS and passed):
                win = module._bit_terminal(nb, nw, side) > 0
            else:
                win = not solve(b, w, nb, nw, 3-side, count+1, point == module.PASS)[0]
            if win:
                result = (True, point)
                cache[key] = result
                return result
        result = (False, actions[0][0])
        cache[key] = result
        return result
    pb,pw=module._pack(state['previous']);b,w=module._pack(state['board'])
    try:
        win, point = solve(pb,pw,b,w,state['color'],state['move_count'],state['last_pass'])
        result = {'complete': True, 'can_force_win':win, 'point':point}
    except TimeoutError:
        result = {'complete':False}
    result.update(nodes=nodes, cache_size=len(cache), cpu=time.process_time()-start)
    return result


if __name__ == '__main__':
    games=json.loads((ROOT/'reports/official-submission12-20261023/cnt12/parsed-games.json').read_text())['results']
    report={'scope':'Exact full-horizon WLD proof on recorded positions, not opponent emulation.', 'results':[], 'complete':False}
    out=ROOT/'reports/submission12-terminal-proof-probes.json'
    seen=set()
    for game in games:
        sequence=tuple(m['point'] for m in game['moves'])
        if game['opponent']!='championship_player' or game['agent_color']!=2 or sequence in seen:continue
        seen.add(sequence)
        for count in (19,15,13,11,9):
            module=load_player('baseline',SOURCE)
            result=prove(module,game['moves'][count])
            row=dict(round=game['round'],count=count,recorded=game['moves'][count]['point'],proof=result)
            report['results'].append(row);out.write_text(json.dumps(report,indent=2))
            print(json.dumps(row),flush=True)
    report['complete']=True;out.write_text(json.dumps(report,indent=2))
