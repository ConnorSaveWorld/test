import importlib.util
import unittest
from test_rules import ROOT


class Quiet(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path=ROOT/'experiments'/'candidate_quiet3.py'
        spec=importlib.util.spec_from_file_location('quiet',str(path))
        cls.a=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.a)

    def test_leaf_extension_solves_remaining_last_reply(self):
        a=self.a
        board=[0]*25
        board[0]=2
        for p in (1,6,8,9):board[p]=1
        board=tuple(board)
        action,stats=a.search(a.EMPTY,board,1,22,False,max_depth=1,cpu_seconds=2)
        brute=max(min(a.terminal_value(reply,1) for _,reply in a.legal_moves(board,nxt,2))
                  for _,nxt in a.legal_moves(a.EMPTY,board,1))
        self.assertEqual(stats['completed_depth'],1)
        self.assertEqual(stats['value'],brute)
        self.assertGreater(stats['q_nodes'],0)
        self.assertEqual(action,5)

    def test_forced_atari_keeps_quiet_legal_alternatives(self):
        a=self.a
        board=[0]*25
        board[0]=1;board[1]=2
        b,w=a._pack(tuple(board))
        forced,actions=a._tactical_actions(0,0,b,w,1)
        self.assertTrue(forced)
        self.assertEqual({p for p,_,_ in actions},{p for p,_,_ in a._bit_moves(0,0,b,w,1)})

    def test_quiescence_keeps_ko_illegal(self):
        a=self.a
        board=[0]*25
        for p in (6,10,12,16):board[p]=2
        for p in (7,13,17):board[p]=1
        prev=tuple(board)
        now=a.play_move(a.EMPTY,prev,1,11)
        _,actions=a._tactical_actions(*a._pack(prev),*a._pack(now),2)
        self.assertNotIn(12,[p for p,_,_ in actions])


if __name__=='__main__':unittest.main()
