import importlib.util
import pathlib
import sys
import unittest
from test_rules import player, ROOT

sys.path.insert(0, str(ROOT / 'tools'))
try:
    import benchmark
except ImportError:
    benchmark = None


class Benchmark(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(benchmark, 'Tournament harness not implemented')

    def test_two_passes_and_file_initial_pass_difference(self):
        passing = lambda state, rng: (-1, {})
        game = benchmark.run_game(passing, passing, seed=1)
        self.assertEqual((game['winner'], game['steps'], game['reason']), (2, 2, 'passes'))
        game = benchmark.run_game(passing, passing, seed=1, mode='official-file')
        self.assertEqual((game['winner'], game['steps']), (2, 1))

    def test_invalid_move_forfeits(self):
        bad = lambda state, rng: (25, {})
        passing = lambda state, rng: (-1, {})
        game = benchmark.run_game(bad, passing, seed=1)
        self.assertEqual((game['winner'], game['reason']), (2, 'invalid'))

    def test_twenty_four_step_limit(self):
        def first_free(state, rng):
            actions = player.legal_moves(state['previous'], state['board'], state['color'])
            return actions[0][0], {}
        game = benchmark.run_game(first_free, first_free, seed=1)
        self.assertEqual((game['steps'], game['reason']), (24, 'limit'))

    def test_replay_and_exact_stone_score(self):
        a = benchmark.make_policy('random', 0.01)
        first = benchmark.run_game(a, a, seed=44)
        second = benchmark.run_game(a, a, seed=44)
        self.assertEqual(first['board'], second['board'])
        self.assertEqual([m['point'] for m in first['moves']],
                         [m['point'] for m in second['moves']])
        board = first['board']
        expected = 1 if board.count(1) > board.count(2) + 2.5 else 2
        self.assertEqual(first['winner'], expected)

    def test_exception_is_not_counted_as_normal_win(self):
        def broken(state, rng):
            raise RuntimeError('deliberate test')
        passing = lambda state, rng: (-1, {})
        game = benchmark.run_game(broken, passing, seed=1)
        self.assertEqual((game['winner'], game['reason']), (2, 'crash'))


if __name__ == '__main__':
    unittest.main()
