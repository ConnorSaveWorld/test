import argparse
import hashlib
import json
from benchmark import ROOT, tournament
from train_value import LinearValue


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--games',type=int,default=40)
    parser.add_argument('--cpu',type=float,default=.025)
    parser.add_argument('--opponent',default='alphabeta-proxy')
    args=parser.parse_args()
    model=json.loads((ROOT/'reports'/'experimental-value.json').read_text())
    learner=LinearValue(model['weights'])
    results={}
    for name in ('search','learned-policy','hybrid'):
        report=tournament(args.opponent,args.games,800000,args.cpu,
                          agent_name=name,evaluator=learner.evaluate if name!='search' else None)
        report['experimental_model']=model if name!='search' else None
        report['model_sha256']=hashlib.sha256(json.dumps(model,sort_keys=True).encode()).hexdigest() if name!='search' else None
        (ROOT/'reports'/('learning-'+name+'.json')).write_text(json.dumps(report,indent=2))
        results[name]={k:report[k] for k in ('wins','games','win_rate','by_color','failures','mean_cpu')}
    # No hidden advanced opponent is locally accessible. Keep the search baseline.
    results['deployment']='search-only; local experiment does not establish advanced-grader improvement'
    (ROOT/'reports'/'learning-comparison.json').write_text(json.dumps(results,indent=2))
    print(json.dumps(results,indent=2))
