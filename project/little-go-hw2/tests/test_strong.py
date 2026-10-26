import sys
import unittest
from test_rules import ROOT
sys.path.insert(0,str(ROOT/'tools'))
import strong_league


class Strong(unittest.TestCase):
    def test_profiles_do_not_mutate_frozen_baseline(self):
        a=strong_league.load_player('baseline')
        b=strong_league.load_player('survival')
        board=tuple(map(int,'1100010100000000000000000'))
        black,white=a._pack(board)
        original=a._bit_static(black,white)
        self.assertNotEqual(original,b._bit_static(black,white))
        self.assertEqual(original,a._bit_static(black,white))
        self.assertEqual(a._bit_static(black,white),-a._bit_evaluate(black,white,2))

    def test_paired_openings_are_identical_and_legal(self):
        opening=strong_league.make_opening(561,4)
        self.assertEqual(opening,strong_league.make_opening(561,4))
        a=strong_league.load_player('baseline')
        previous=board=a.EMPTY
        for count,point in enumerate(opening):
            updated=a.play_move(previous,board,1+count%2,point)
            self.assertIsNotNone(updated)
            previous,board=board,updated

    def test_two_real_eyes_help_survival_but_not_terminal_score(self):
        a=strong_league.load_player('survival')
        # Black fills rows0/1 with isolated empty corners at0,4.
        board=tuple(map(int,'0111011111000000000000000'))
        b,w=a._pack(board)
        self.assertEqual(strong_league.eye_count(a,b,w),2)
        self.assertEqual(a._bit_score(b,w),board.count(1)-2.5)


if __name__=='__main__':unittest.main()
