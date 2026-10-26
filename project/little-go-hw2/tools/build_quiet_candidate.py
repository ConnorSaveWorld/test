"""Experimental bounded tactical leaf extension of our frozen original search."""
import argparse
from benchmark import ROOT

TACTICAL='''
def _tactical_actions(pb,pw,b,w,side):
    forced=any(color==side and _count(libs)<=1 for color,stones,libs in _bit_groups(b,w))
    moves=_bit_order(pb,pw,b,w,side)
    if forced:return True,moves
    enemy=w if side==1 else b
    return False,[(p,nb,nw) for p,nb,nw in moves
                  if p!=PASS and (nw if side==1 else nb)!=enemy]

'''

QUIESCENCE='''
    def quiet(old_b,old_w,b,w,side,count,passed,remaining,alpha,beta):
        stats['nodes']+=1
        stats['q_nodes']+=1
        check()
        if count is not None and count>=24:
            return _bit_terminal(b,w,side)
        static=_bit_evaluate(b,w,side)
        if remaining<=0:return static
        forced,actions=_tactical_actions(old_b,old_w,b,w,side)
        if count is not None and 24-count<=remaining:
            forced=True
            actions=_bit_order(old_b,old_w,b,w,side)
        best=-float('inf') if forced else static
        if not forced and passed:
            best=max(best,_bit_terminal(b,w,side))
        if best>=beta:return best
        alpha=max(alpha,best)
        next_count=None if count is None else count+1
        for point,nb,nw in actions:
            if ((point==PASS and passed) or
                    (next_count is not None and next_count>=24)):
                candidate=_bit_terminal(nb,nw,side)
            else:
                candidate=-quiet(b,w,nb,nw,3-side,next_count,point==PASS,
                                 remaining-1,-beta,-alpha)
            best=max(best,candidate)
            alpha=max(alpha,best)
            if alpha>=beta:break
        return best

'''

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--budget',type=int,choices=(3,5),default=3)
    parser.add_argument('--depth',type=int,choices=(1,2),default=2)
    parser.add_argument('--output',default='candidate_quiet3.py')
    args=parser.parse_args()
    code=(ROOT/'experiments'/'platform_pvs3.py').read_text()
    code=code.replace('\ndef search(', '\n'+TACTICAL+'\ndef search(',1)
    code=code.replace('max_nodes=None, evaluator=None, use_pvs=True):',
                      'max_nodes=None, evaluator=None, use_pvs=True, q_depth='+str(args.depth)+'):')
    code=code.replace("'pvs': use_pvs}","'pvs': use_pvs, 'q_nodes': 0, 'q_depth': q_depth}")
    code=code.replace('    def value(old_b, old_w, b, w, side, count, passed, depth, alpha, beta):',
                      QUIESCENCE+'    def value(old_b, old_w, b, w, side, count, passed, depth, alpha, beta):')
    code=code.replace('return _bit_evaluate(b, w, side), PASS',
                      'return quiet(old_b,old_w,b,w,side,count,passed,q_depth,alpha,beta), PASS')
    if args.budget==5:
        code=code.replace('budget = min(3.0, remaining / turns_left)',
                          'budget = min(5.0, remaining / turns_left)')
        code=code.replace('wall_seconds=min(5.0, budget + 1.0)',
                          'wall_seconds=min(7.0, budget + 1.0)')
    (ROOT/'experiments'/args.output).write_text(code)
