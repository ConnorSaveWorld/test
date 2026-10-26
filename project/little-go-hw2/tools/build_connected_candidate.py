import argparse
from benchmark import ROOT

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--penalty',type=int,choices=(2,4),default=2)
    parser.add_argument('--output',default='candidate_connect2.py')
    args=parser.parse_args()
    code=(ROOT/'experiments'/'platform_pvs3.py').read_text()
    before='    # The same weights in exact integer twentieths avoid false floating ties.'
    code=code.replace(before,"    group_balance=sum(1 if side==1 else -1 for side,_,_ in _bit_groups(black,white))\n"+before)
    code=code.replace('- 120 * atari - 24 * weak + 9 * influence) / 20.0',
        '- 120 * atari - 24 * weak + 9 * influence - '+str(20*args.penalty)+' * group_balance) / 20.0')
    code=code.replace("if __name__ == '__main__':",'def evaluate(board,color):\n    return _bit_evaluate(*_pack(board),color)\n\n\n'+"if __name__ == '__main__':")
    (ROOT/'experiments'/args.output).write_text(code)
