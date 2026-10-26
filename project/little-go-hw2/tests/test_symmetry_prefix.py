import unittest
from test_root_symmetry import load


class SymmetryPrefixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = load('backups/stage2-submission6-604ea3ee.py', 'prefix_original')
        cls.root_only = load('experiments/candidate_symmetry.py', 'prefix_root_only')
        cls.candidate = load('experiments/candidate_symmetry_tree.py', 'prefix_candidate')

    def run_search(self, module, previous, board, color, count, depth, passed=False):
        for name in ('analyze', '_bit_groups', '_bit_static', '_bit_candidates'):
            getattr(module, name).cache_clear()
        options = {'use_symmetry': True} if module is self.candidate else {}
        return module.search(previous, board, color, count, passed,
                             max_depth=depth, cpu_seconds=10, wall_seconds=15,
                             use_table=False, use_pvs=False, **options)

    def test_symmetric_prefix_reduces_work_beyond_root_only_pruning(self):
        empty = (0,) * 25
        _, root = self.run_search(self.root_only, empty, empty, 1, 0, 3)
        _, tree = self.run_search(self.candidate, empty, empty, 1, 0, 3)
        self.assertEqual(tree['value'], root['value'])
        self.assertLess(tree['nodes'], root['nodes'])

    def test_white_values_match_full_search_with_known_and_unknown_counts(self):
        for point in (0, 1, 2, 6, 7, 12):
            board = tuple(1 if p == point else 0 for p in range(25))
            for count in (1, None):
                with self.subTest(point=point, count=count):
                    _, old = self.run_search(self.original, (0,) * 25, board, 2, count, 3)
                    action, new = self.run_search(self.candidate, (0,) * 25, board, 2, count, 3)
                    self.assertEqual(new['value'], old['value'])
                    self.assertIsNotNone(self.candidate.play_move((0,) * 25, board, 2, action))

    def test_broken_history_symmetry_does_not_prune_descendants(self):
        previous = tuple(1 if p == 1 else 0 for p in range(25))
        board = tuple(1 if p == 12 else 0 for p in range(25))
        _, old = self.run_search(self.original, previous, board, 2, None, 3)
        _, new = self.run_search(self.candidate, previous, board, 2, None, 3)
        self.assertEqual(new['nodes'], old['nodes'])
        self.assertEqual(new['value'], old['value'])


if __name__ == '__main__':
    unittest.main()
