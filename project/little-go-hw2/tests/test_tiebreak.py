import importlib.util
import sys
import unittest

from test_rules import ROOT


def load_candidate():
    path = ROOT / 'experiments' / 'candidate_white_tiebreak.py'
    spec = importlib.util.spec_from_file_location('tiebreak_candidate', str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TieBreakTests(unittest.TestCase):
    def test_black_keeps_the_current_equal_priority_order(self):
        player = load_candidate()
        baseline_path = ROOT / 'stage2_candidate' / 'my_player3.py'
        spec = importlib.util.spec_from_file_location('tiebreak_baseline', str(baseline_path))
        baseline = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(baseline)

        self.assertEqual(player._bit_candidates(0, 0, 1),
                         baseline._bit_candidates(0, 0, 1))

    def test_white_opening_ties_are_deterministic_and_not_low_index_only(self):
        player = load_candidate()
        black_opening = 1 << 12
        actions = player._bit_candidates(black_opening, 0, 2)
        baseline_path = ROOT / 'stage2_candidate' / 'my_player3.py'
        spec = importlib.util.spec_from_file_location('tiebreak_baseline_white', str(baseline_path))
        baseline = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(baseline)
        baseline_actions = baseline._bit_candidates(black_opening, 0, 2)

        self.assertEqual(actions, player._bit_candidates(black_opening, 0, 2))
        self.assertNotEqual(actions, baseline_actions)


if __name__ == '__main__':
    unittest.main()
