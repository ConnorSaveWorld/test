import unittest

from test_root_symmetry import load


class WhiteCounterwallTargetedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.player = load('experiments/candidate_white_counterwall_targeted.py',
                          'counterwall_targeted_candidate')

    def position(self, black_approach):
        previous = tuple(1 if p == 12 else 2 if p == 6 else 0
                         for p in range(25))
        board = tuple(1 if p in (12, black_approach) else
                      2 if p == 6 else 0 for p in range(25))
        return previous, board

    def test_documented_black_11_line_uses_counterwall_13(self):
        previous, board = self.position(11)
        action, stats = self.player.search(previous, board, 2, 3, False,
                                           cpu_seconds=0, wall_seconds=0)
        self.assertTrue(stats.get('opening_policy'))
        self.assertEqual(action, 13)
        self.assertIsNotNone(self.player.play_move(previous, board, 2, action))

    def test_black_7_line_keeps_the_proven_reply_18(self):
        previous, board = self.position(7)
        action, stats = self.player.search(previous, board, 2, 3, False,
                                           cpu_seconds=0, wall_seconds=0)
        self.assertTrue(stats.get('opening_policy'))
        self.assertEqual(action, 18)
        self.assertIsNotNone(self.player.play_move(previous, board, 2, action))


if __name__ == '__main__':
    unittest.main()
