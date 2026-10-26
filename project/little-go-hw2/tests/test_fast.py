import importlib.util
import pathlib
import random
import sys
import unittest
from test_rules import ROOT, player as baseline

PATH=ROOT/'experiments'/'candidate_fast.py'
if PATH.exists():
    spec=importlib.util.spec_from_file_location('fast_candidate',str(PATH))
    fast=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fast)
else:
    fast=None


class Fast(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(fast,'Bitboard candidate not implemented')

    def test_edges_never_wrap(self):
        self.assertEqual(fast._adj(1<<4),(1<<3)|(1<<9))
        self.assertEqual(fast._adj(1<<20),(1<<15)|(1<<21))

    def test_pack_roundtrip(self):
        board=(1,2,0,2,1)+(0,)*20
        self.assertEqual(fast._unpack(*fast._pack(board)),board)

    def test_moves_and_evaluation_match_reference(self):
        rng=random.Random(9201)
        previous=board=(0,)*25
        for count in range(300):
            color=1+count%2
            pb,pw=fast._pack(previous)
            b,w=fast._pack(board)
            actual={p:fast._unpack(x,y) for p,x,y in fast._bit_moves(pb,pw,b,w,color)}
            expected=dict(baseline.legal_moves(previous,board,color))
            self.assertEqual(actual,expected)
            self.assertAlmostEqual(fast._bit_evaluate(b,w,color),baseline.evaluate(board,color))
            choices=[item for item in expected.items() if item[0]!=-1]
            if count%24==23 or not choices:
                previous=board=(0,)*25
            else:
                _,nxt=rng.choice(choices)
                previous,board=board,nxt

    def test_fixed_depth_matches_original_search(self):
        prev=(1,1,0,0,1,1,1,1,1,2,2,1,2,0,0,2,0,2,0,2,0,0,0,1,2)
        board=(1,1,2,0,1,1,1,1,1,2,2,1,2,0,0,2,0,2,0,2,0,0,0,1,2)
        _,old=baseline.search(prev,board,1,18,False,cpu_seconds=3,max_depth=6)
        for cache in (True,False):
            action,new=fast.search(prev,board,1,18,False,cpu_seconds=3,max_depth=6,use_table=cache)
            self.assertEqual(new['completed_depth'],6)
            self.assertEqual(new['value'],10002.5)
            self.assertEqual(new['value'],old['value'])
            self.assertIsNotNone(baseline.play_move(prev,board,1,action))

    def test_ko_is_rejected_by_fast_path(self):
        board=[0]*25
        for p in (6,10,12,16):board[p]=2
        for p in (7,13,17):board[p]=1
        previous=tuple(board)
        current=baseline.play_move(baseline.EMPTY,previous,1,11)
        self.assertNotIn(12,[p for p,_,_ in fast._bit_moves(*fast._pack(previous),*fast._pack(current),2)])


if __name__=='__main__':unittest.main()
