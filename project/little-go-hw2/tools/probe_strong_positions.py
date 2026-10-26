"""Cold probes of local league losses; not official Championship traces."""
import json
from benchmark import ROOT
from strong_league import load_player

if __name__=='__main__':
    games=json.loads((ROOT/'reports'/'strong-screen-baseline.json').read_text())['results']
    rows=[]
    for seed,color,counts in ((310000,2,(11,13)),(310001,1,(10,12)),(310003,1,(10,12))):
        game=next(g for g in games if g['seed']==seed and g['agent_color']==color and g['opponent']=='space')
        for count in counts:
            state=next(m for m in game['moves'] if m['move_count']==count)
            for profile,cpu in (('baseline',3),('baseline',5),('space',3),('space-noeyes',3)):
                module=load_player(profile)
                point,stats=module.search(tuple(state['previous']),tuple(state['board']),
                    color,count,state['last_pass'],cpu_seconds=cpu,wall_seconds=cpu+1)
                row={'seed':seed,'color':color,'count':count,'profile':profile,
                     'cpu_budget':cpu,'point':point,'stats':stats}
                rows.append(row)
                print(json.dumps(row),flush=True)
    (ROOT/'reports'/'strong-position-probes.json').write_text(json.dumps(rows,indent=2))
