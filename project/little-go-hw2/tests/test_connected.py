import importlib.util
import sys
import unittest
from test_rules import ROOT
sys.path.insert(0,str(ROOT/'tools'))
from strong_league import load_player


class Connected(unittest.TestCase):
    def test_bridging_groups_changes_only_heuristic_component(self):
        original=load_player('baseline')
        board=[0]*25
        board[10]=board[12]=1
        board=tuple(board)
        connected=original.play_move(original.EMPTY,board,1,11)
        for penalty in (2,4):
            path=ROOT/'experiments'/('candidate_connect'+str(penalty)+'.py')
            spec=importlib.util.spec_from_file_location('connected',str(path))
            a=importlib.util.module_from_spec(spec)
            spec.loader.exec_module(a)
            self.assertAlmostEqual(a.evaluate(board,1),original.evaluate(board,1)-2*penalty)
            self.assertAlmostEqual(a.evaluate(connected,1),original.evaluate(connected,1)-penalty)
            self.assertEqual(a.evaluate(board,1),-a.evaluate(board,2))
            self.assertEqual(a.terminal_value(connected,1),original.terminal_value(connected,1))
            action,stats=a.search(original.EMPTY,board,1,23,False,max_depth=1,cpu_seconds=1)
            self.assertIsNotNone(a.play_move(original.EMPTY,board,1,action))
            self.assertGreater(abs(stats['value']),a.WIN_VALUE)


if __name__=='__main__':unittest.main()
