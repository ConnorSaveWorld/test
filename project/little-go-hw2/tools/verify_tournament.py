"""Independently replay completed tournaments with the original course GO."""
import argparse
import json
from benchmark import ROOT, GO


def verify(report):
    assert report['games']==len(report['results'])
    actions=0
    for game in report['results']:
        previous=board=(0,)*25
        passed=False
        for count,move in enumerate(game['moves']):
            color=1+count%2
            assert tuple(move['previous'])==previous
            assert tuple(move['board'])==board
            assert move['color']==color and move['move_count']==count
            assert move['last_pass']==passed
            point=move['point']
            assert point is not None, 'Cannot certify a failed-action game'
            go=GO(5)
            go.set_board(color,[list(previous[r:r+5]) for r in range(0,25,5)],
                               [list(board[r:r+5]) for r in range(0,25,5)])
            if point!=-1:
                assert 0<=point<25 and go.place_chess(point//5,point%5,color)
                go.remove_died_pieces(3-color)
            updated=tuple(cell for row in go.board for cell in row)
            ends=point==-1 and (passed or previous==board)
            previous,board=board,updated
            passed=point==-1
            actions+=1
            assert not ends or count==len(game['moves'])-1
        assert len(game['moves'])==game['steps']
        assert tuple(game['board'])==board
        assert game['reason']==('passes' if ends else 'limit')
        assert ends or game['steps']==24
        assert go.judge_winner()==game['winner']
    wins=sum(g['winner']==g['agent_color'] for g in report['results'])
    assert wins==report['wins']
    assert all(value==0 for value in report['failures'].values())
    return {'games':report['games'],'actions':actions,'wins':wins,
            'source_sha256':report['source_sha256'],
            'validation':'Original GO legality, captures, terminal score, and full trace continuity'}


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('reports',nargs='+')
    parser.add_argument('--output',default='reports/optimized-replay-validation.json')
    args=parser.parse_args()
    results=[]
    for name in args.reports:
        report=json.loads((ROOT/name).read_text())
        checked=verify(report)
        checked['report']=name
        results.append(checked)
    (ROOT/args.output).write_text(json.dumps(results,indent=2))
    print(json.dumps(results,indent=2))
