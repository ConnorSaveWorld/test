import unittest
from test_root_symmetry import load


class SymmetryScopeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = load('backups/stage2-submission6-604ea3ee.py', 'scope_original')
        cls.candidate = load('experiments/candidate_symmetry_tree.py', 'scope_candidate')

    def test_default_black_search_preserves_previous_work_and_action(self):
        empty = (0,) * 25
        for depth in (2, 3):
            old_action, old = self.original.search(empty, empty, 1, 0, False,
                max_depth=depth, cpu_seconds=10, wall_seconds=15)
            new_action, new = self.candidate.search(empty, empty, 1, 0, False,
                max_depth=depth, cpu_seconds=10, wall_seconds=15)
            self.assertEqual((new_action, new['value'], new['nodes']),
                             (old_action, old['value'], old['nodes']))

    def test_default_white_search_keeps_symmetry_optimization(self):
        empty = (0,) * 25
        board = tuple(1 if p == 12 else 0 for p in range(25))
        _, old = self.original.search(empty, board, 2, 1, False,
            max_depth=3, cpu_seconds=10, wall_seconds=15)
        _, new = self.candidate.search(empty, board, 2, 1, False,
            max_depth=3, cpu_seconds=10, wall_seconds=15)
        self.assertEqual(new['value'], old['value'])
        self.assertLess(new['nodes'], old['nodes'])


if __name__ == '__main__':
    unittest.main()
