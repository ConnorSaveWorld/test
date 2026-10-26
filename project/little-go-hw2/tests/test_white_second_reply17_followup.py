import unittest

from reference_rules import apply
from test_root_symmetry import load


class WhiteSecondReply17FollowupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.player = load(
            'experiments/candidate_white_second_reply17_black11followup13.py',
            'second_reply17_followup13_candidate')

    def state_after(self, moves):
        previous = board = (0,) * 25
        for index, point in enumerate(moves):
            updated = apply(previous, board, 1 + index % 2, point)
            self.assertIsNotNone(updated, 'test history must be legal')
            previous, board = board, updated
        return previous, board

    def test_exact_followup_line_uses_legal_counterwall13(self):
        previous, board = self.state_after((12, 6, 7, 17, 18, 16, 11))
        action, stats = self.player.search(previous, board, 2, 7, False,
                                           cpu_seconds=0, wall_seconds=0)
        self.assertTrue(stats.get('opening_policy'))
        self.assertEqual(action, 13)
        self.assertIsNotNone(apply(previous, board, 2, action))

    def test_nearby_but_different_history_stays_in_search(self):
        previous, board = self.state_after((12, 6, 7, 17, 18, 16, 10))
        action, stats = self.player.search(previous, board, 2, 7, False,
                                           cpu_seconds=0.01,
                                           wall_seconds=0.1,
                                           max_nodes=64)
        self.assertFalse(stats.get('opening_policy', False))
        self.assertIsNotNone(apply(previous, board, 2, action))


if __name__ == '__main__':
    unittest.main()
