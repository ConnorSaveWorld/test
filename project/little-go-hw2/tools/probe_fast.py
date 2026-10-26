import importlib.util
import json
from benchmark import ROOT


def load(name):
    spec=importlib.util.spec_from_file_location(name,str(ROOT/'experiments'/(name+'.py')))
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__=='__main__':
    baseline=load('baseline_player3')
    fast=load('candidate_fast')
    games=json.loads((ROOT/'reports'/'stage1-alphabeta-proxy.json').read_text())['results']
    results=[]
    for seed in (561101,561109):
        game=next(g for g in games if g['seed']==seed)
        for count in (5,9,13):
            state=next(m for m in game['moves'] if m['move_count']==count)
            for name,module in (('baseline',baseline),('fast',fast)):
                module.analyze.cache_clear()
                if name=='fast':
                    module._bit_groups.cache_clear()
                    module._bit_static.cache_clear()
                p,stats=module.search(tuple(state['previous']),tuple(state['board']),2,
                    count,state['last_pass'],cpu_seconds=3,wall_seconds=4)
                row={'seed':seed,'count':count,'engine':name,'point':p,'stats':stats}
                results.append(row)
                print(json.dumps(row),flush=True)
    (ROOT/'reports'/'fast-position-probes.json').write_text(json.dumps(results,indent=2))
