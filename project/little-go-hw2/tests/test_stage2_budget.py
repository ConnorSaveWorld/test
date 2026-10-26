import importlib.util
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
PATH = ROOT / 'stage2_candidate' / 'my_player3.py'
SPEC = importlib.util.spec_from_file_location('stage2_player', str(PATH))
player = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(player)


class Stage2Budget(unittest.TestCase):
    def test_black_and_unknown_history_keep_safe_standard_cap(self):
        self.assertEqual(player.turn_budget(0.0, 0, 1), 7.0)
        self.assertEqual(player.turn_budget(0.0, None, 2), 7.0)

    def test_white_opening_turns_receive_more_search_time(self):
        cpu_used = 0.0
        early_budgets = []
        for count in (1, 3, 5):
            budget = player.turn_budget(cpu_used, count, 2)
            early_budgets.append(budget)
            cpu_used += budget
        self.assertGreater(early_budgets[0], player.turn_budget(0.0, 0, 1))
        self.assertLessEqual(max(early_budgets), 8.5)
        self.assertAlmostEqual(early_budgets[0], early_budgets[1])
        self.assertAlmostEqual(early_budgets[1], early_budgets[2])

    def test_weighted_white_opening_still_respects_game_budget(self):
        cpu_used = 0.0
        for count in range(1, 24, 2):
            cpu_used += player.turn_budget(cpu_used, count, 2)
        self.assertAlmostEqual(cpu_used, 86.0)
        self.assertEqual(player.turn_budget(90.0, 23, 2), 0.0)


if __name__ == '__main__':
    unittest.main()
