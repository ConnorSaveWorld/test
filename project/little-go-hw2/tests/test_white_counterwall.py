import unittest

from test_root_symmetry import load


class WhiteCounterwallTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.player = load('experiments/candidate_white_counterwall.py', 'counterwall_candidate')

    def test_white_blocks_the_approach_axis_in_all_dihedral_orientations(self):
        a = self.player
        for mapping in a._SYMMETRY_MAPS:
            previous = tuple(1 if p == mapping[12] else
                             2 if p == mapping[6] else 0 for p in range(25))
            board = tuple(1 if p in (mapping[12], mapping[7]) else
                          2 if p == mapping[6] else 0 for p in range(25))
            action, stats = a.search(previous, board, 2, 3, False,
                                     cpu_seconds=0, wall_seconds=0)
            self.assertTrue(stats.get('opening_policy'))
            self.assertEqual(action, mapping[17])
            self.assertIsNotNone(a.play_move(previous, board, 2, action))


if __name__ == '__main__':
    unittest.main()
