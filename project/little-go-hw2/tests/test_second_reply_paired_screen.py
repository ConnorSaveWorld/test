import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))

from screen_second_reply_wall_paired import paired_delta, white_margin


class SecondReplyPairedScreenTests(unittest.TestCase):
    def test_white_margin_uses_stones_and_komi(self):
        board = [0] * 25
        board[0] = 1
        board[1] = 2
        board[2] = 2
        self.assertEqual(white_margin({'board': board}), 3.5)

    def test_paired_delta_counts_win_and_margin_changes(self):
        reference = [
            {'winner': 1, 'white_margin': -0.5},
            {'winner': 2, 'white_margin': 1.5},
            {'winner': 2, 'white_margin': 2.5},
        ]
        candidate = [
            {'winner': 2, 'white_margin': 0.5},
            {'winner': 2, 'white_margin': 1.5},
            {'winner': 1, 'white_margin': -0.5},
        ]
        self.assertEqual(paired_delta(candidate, reference), {
            'white_win_delta': 0,
            'white_loss_delta': 0,
            'white_margin_delta_sum': -2.0,
            'paired_margin_improvements': 1,
            'paired_margin_regressions': 1,
            'paired_margin_ties': 1,
        })


if __name__ == '__main__':
    unittest.main()
