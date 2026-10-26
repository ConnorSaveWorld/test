import importlib.util
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def load(relative, name):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RootSymmetryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = load('backups/stage2-submission6-604ea3ee.py', 'symmetry_original')
        cls.candidate = load('experiments/candidate_symmetry.py', 'symmetry_candidate')

    def search(self, module, previous, board, color, count, passed, depth):
        for name in ('analyze', '_bit_groups', '_bit_static', '_bit_candidates'):
            getattr(module, name).cache_clear()
        return module.search(previous, board, color, count, passed,
                             max_depth=depth, cpu_seconds=10, wall_seconds=15,
                             use_pvs=False, use_table=False)

    def test_empty_board_searches_six_move_orbits_and_pass(self):
        action, stats = self.search(self.candidate, (0,) * 25, (0,) * 25,
                                    1, 0, False, 1)
        # Six D4 classes of placements, one PASS, plus the root node.
        self.assertEqual(stats['nodes'], 8)
        self.assertAlmostEqual(stats['value'], -9.8)
        self.assertEqual(action, 6)

    def test_white_opening_values_match_unreduced_search(self):
        for point in (0, 1, 2, 6, 7, 12):
            with self.subTest(black_opening=point):
                board = tuple(1 if p == point else 0 for p in range(25))
                _, old = self.search(self.original, (0,) * 25, board,
                                     2, 1, False, 3)
                action, new = self.search(self.candidate, (0,) * 25, board,
                                          2, 1, False, 3)
                self.assertEqual(new['completed_depth'], 3)
                self.assertEqual(new['value'], old['value'])
                self.assertIsNotNone(self.candidate.play_move((0,) * 25, board, 2, action))

    def test_previous_board_can_break_current_board_symmetry(self):
        previous = tuple(1 if p == 1 else 0 for p in range(25))
        board = tuple(1 if p == 12 else 0 for p in range(25))
        _, old = self.search(self.original, previous, board, 2, 5, False, 1)
        _, new = self.search(self.candidate, previous, board, 2, 5, False, 1)
        self.assertEqual(new['nodes'], old['nodes'])
        self.assertEqual(new['value'], old['value'])

    def test_pass_and_final_white_scoring_are_preserved(self):
        for count, passed in ((1, True), (23, False)):
            with self.subTest(count=count, passed=passed):
                _, old = self.search(self.original, (0,) * 25, (0,) * 25,
                                     2, count, passed, 1)
                _, new = self.search(self.candidate, (0,) * 25, (0,) * 25,
                                     2, count, passed, 1)
                self.assertEqual(new['value'], old['value'])

    def test_asymmetric_custom_evaluator_keeps_full_move_set(self):
        def corner_evaluator(board, color):
            value = 100 if board[24] == 1 else 0
            return value if color == 1 else -value
        action, _ = self.candidate.search((0,) * 25, (0,) * 25, 1, 0, False,
                                          max_depth=1, cpu_seconds=5,
                                          evaluator=corner_evaluator)
        self.assertEqual(action, 24)


if __name__ == '__main__':
    unittest.main()
