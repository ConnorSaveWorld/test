import importlib.util
import random
import sys
import unittest
from test_rules import ROOT
sys.path.insert(0,str(ROOT/'tools'))
from strong_league import load_player,eye_count


class Space(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path=ROOT/'experiments'/'candidate_space3.py'
        spec=importlib.util.spec_from_file_location('space_candidate',str(path))
        cls.agent=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.agent)

    def test_fast_eye_features_match_independent_loop(self):
        rng=random.Random(314159)
        for _ in range(500):
            board=tuple(rng.randrange(3) for _ in range(25))
            b,w=self.agent._pack(board)
            self.assertEqual(self.agent._eye_count(b,w),eye_count(self.agent,b,w))
            self.assertEqual(self.agent._eye_count(w,b),eye_count(self.agent,w,b))

    def test_same_evaluation_as_screened_profile(self):
        slow=load_player('space')
        rng=random.Random(271828)
        for _ in range(200):
            board=tuple(rng.randrange(3) for _ in range(25))
            b,w=self.agent._pack(board)
            value=self.agent.evaluate(board,1)
            self.assertEqual(value,slow._bit_static(b,w))
            self.assertEqual(value,-self.agent.evaluate(board,2))
            self.assertLess(abs(value),self.agent.WIN_VALUE)

    def test_pvs_matches_exhaustive_with_changed_evaluation(self):
        a=self.agent
        board=(1,2,0,0,0)+(0,)*20
        expected=max(min(a.terminal_value(nxt2,1) if point==a.PASS and reply==a.PASS
                         else a.evaluate(nxt2,1) for reply,nxt2 in a.legal_moves(board,nxt,2))
                     for point,nxt in a.legal_moves(a.EMPTY,board,1))
        for pvs in (True,False):
            _,stats=a.search(a.EMPTY,board,1,2,False,max_depth=2,cpu_seconds=2,use_pvs=pvs)
            self.assertEqual(stats['completed_depth'],2)
            self.assertAlmostEqual(stats['value'],expected)


if __name__=='__main__':unittest.main()
