import unittest

from test_root_symmetry import load


class WhiteSecondReplySearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.player = load('experiments/candidate_white_second_reply_search.py',
                          'second_reply_search_candidate')

    def position(self, black_approach):
        previous = tuple(1 if p == 12 else 2 if p == 6 else 0
                         for p in range(25))
        board = tuple(1 if p in (12, black_approach) else
                      2 if p == 6 else 0 for p in range(25))
        return previous, board

    def test_black7_reply_is_chosen_by_search_and_is_legal(self):
        previous, board = self.position(7)
        action, stats = self.player.search(
            previous, board, 2, 3, False,
            cpu_seconds=1, wall_seconds=2, max_nodes=128)
        self.assertFalse(stats.get('opening_policy', False))
        self.assertGreater(stats.get('nodes', 0), 0)
        self.assertIsNotNone(self.player.play_move(previous, board, 2, action))

    def test_black11_counterwall_remains_exact_and_legal(self):
        previous, board = self.position(11)
        action, stats = self.player.search(previous, board, 2, 3, False,
                                           cpu_seconds=0, wall_seconds=0)
        self.assertTrue(stats.get('opening_policy'))
        self.assertEqual(action, 13)
        self.assertIsNotNone(self.player.play_move(previous, board, 2, action))


if __name__ == '__main__':
    unittest.main()
