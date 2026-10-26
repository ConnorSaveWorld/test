import importlib
import sys
import unittest
from test_rules import player, ROOT

sys.path.insert(0, str(ROOT / 'tools'))

try:
    import train_value
except ImportError:
    train_value = None


class Learning(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(train_value, 'Learning experiment not implemented')

    def test_all_symmetries_preserve_move_coordinates(self):
        board = (1,) + (0,) * 24
        for variant in range(8):
            transformed = train_value.transform(board, variant)
            action = train_value.transform_action(0, variant)
            self.assertEqual(transformed[action], 1)
            self.assertEqual(transformed.count(1), 1)
            self.assertEqual(train_value.inverse_action(action, variant), 0)
            self.assertEqual(train_value.transform_action(-1, variant), -1)
        self.assertEqual(train_value.transform_action(0, 1), 4)

    def test_learner_changes_towards_terminal_reward(self):
        learner = train_value.LinearValue()
        board = (1, 1, 1) + (0,) * 22
        before = learner.predict(board)
        learner.update(board, board, reward=1)
        self.assertGreater(learner.predict(board), before)
        self.assertAlmostEqual(learner.evaluate(board, 1), -learner.evaluate(board, 2))

    def test_feature_evaluation_is_symmetric(self):
        board = (1, 2, 0, 1, 0, 2) + (0,) * 19
        learner = train_value.LinearValue()
        for variant in range(8):
            self.assertAlmostEqual(learner.evaluate(board, 1),
                                   learner.evaluate(train_value.transform(board, variant), 1))


if __name__ == '__main__':
    unittest.main()
