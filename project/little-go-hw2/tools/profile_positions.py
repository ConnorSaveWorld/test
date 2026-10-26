import argparse
import json
import statistics
from benchmark import ROOT, player


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--cpu',type=float,default=.1)
    args=parser.parse_args()
    games=json.loads((ROOT/'reports'/'alphabeta-020.json').read_text())['results']
    selected=[]
    for game in games:
        if game['winner']!=game['agent_color']:
            own=[m for m in game['moves'] if m['color']==game['agent_color']]
            selected.extend(own[i] for i in (0,min(4,len(own)-1),len(own)-1))
            if len(selected)>=12:
                break
    results=[]
    for index,state in enumerate(selected):
        for table,ordering in ((False,False),(False,True),(True,True)):
            action,stats=player.search(tuple(state['previous']),tuple(state['board']),state['color'],
                state['move_count'],state['last_pass'],cpu_seconds=args.cpu,
                use_table=table,use_ordering=ordering)
            results.append({'position':index,'move_count':state['move_count'],
                            'table':table,'ordering':ordering,'point':action,'stats':stats})
    report={'budget':args.cpu,'positions':len(selected),'results':results,
            'note':'Recorded losing positions, one timing sample/config; diagnostic, not a tournament improvement claim.'}
    (ROOT/'reports'/'profile.json').write_text(json.dumps(report,indent=2))
    for table,ordering in ((False,False),(False,True),(True,True)):
        rows=[r for r in results if r['table']==table and r['ordering']==ordering]
        print(table,ordering,'mean_depth=',statistics.mean(r['stats']['completed_depth'] for r in rows),
              'mean_nodes=',statistics.mean(r['stats']['nodes'] for r in rows))
