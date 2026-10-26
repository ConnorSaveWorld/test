import unittest

from test_root_symmetry import load


class WhiteSecondReplyVariantTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.players = {
            17: load('experiments/candidate_white_second_reply17.py',
                     'second_reply17_candidate'),
            13: load('experiments/candidate_white_second_reply13.py',
                     'second_reply13_candidate'),
        }

    def position(self, black_approach):
        previous = tuple(1 if p == 12 else 2 if p == 6 else 0
                         for p in range(25))
        board = tuple(1 if p in (12, black_approach) else
                      2 if p == 6 else 0 for p in range(25))
        return previous, board

    def assert_opening_action(self, player, approach, expected):
        previous, board = self.position(approach)
        action, stats = player.search(previous, board, 2, 3, False,
                                      cpu_seconds=0, wall_seconds=0)
        self.assertTrue(stats.get('opening_policy'))
        self.assertEqual(action, expected)
        self.assertIsNotNone(player.play_move(previous, board, 2, action))

    def test_black_7_curriculum_reply_is_exact_and_legal(self):
        for reply, player in self.players.items():
            with self.subTest(reply=reply):
                self.assert_opening_action(player, 7, reply)

    def test_black_11_improved_reply_is_preserved(self):
        self.assert_opening_action(self.players[17], 11, 13)
        self.assert_opening_action(self.players[13], 11, 13)


if __name__ == '__main__':
    unittest.main()
