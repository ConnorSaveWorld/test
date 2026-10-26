import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest
from test_rules import player, PATH


class Protocol(unittest.TestCase):
    def setUp(self):
        for name in ('read_input', 'write_output', 'load_history', 'main'):
            self.assertTrue(callable(getattr(player, name, None)), name + ' not implemented')

    def test_read_exact_boards(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, 'input.txt')
            with open(path, 'w') as f:
                f.write('2\n' + '00000\n' * 5 + '10000\n' + '00000\n' * 4)
            color, prev, board = player.read_input(path)
            self.assertEqual((color, prev, board), (2, (0,) * 25, (1,) + (0,) * 24))

    def test_reject_invalid_input(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, 'input.txt')
            for text in ('1\n00000', '3\n' + '00000\n' * 10,
                         '1\n99999\n' + '00000\n' * 9):
                with open(path, 'w') as f:
                    f.write(text)
                with self.assertRaises(ValueError):
                    player.read_input(path)

    def test_output_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, 'output.txt')
            for action, expected in ((13, b'2,3'), (-1, b'PASS')):
                player.write_output(action, path)
                with open(path, 'rb') as f:
                    self.assertIn(f.read(), (expected, expected + b'\n'))

    def test_initial_turns_and_stale_reset(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, 'state.json')
            with open(path, 'w') as f:
                json.dump({'version': 1, 'color': 1, 'move_count': 22,
                           'after': [1] * 25, 'cpu_used': 80}, f)
            h = player.load_history(1, player.EMPTY, player.EMPTY, path)
            self.assertEqual((h['move_count'], h['cpu_used']), (0, 0))
            h = player.load_history(2, player.EMPTY, (1,) + (0,) * 24, path)
            self.assertEqual((h['move_count'], h['cpu_used']), (1, 0))

    def test_history_continuity_and_color_check(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, 'state.json')
            before = (1,) + (0,) * 24
            current = (1, 2) + (0,) * 23
            with open(path, 'w') as f:
                json.dump({'version': 1, 'color': 1, 'move_count': 0,
                           'after': list(before), 'cpu_used': 1.5}, f)
            h = player.load_history(1, before, current, path)
            self.assertEqual((h['move_count'], h['cpu_used']), (2, 1.5))
            self.assertIsNone(player.load_history(2, before, current, path)['move_count'])
            self.assertIsNone(player.load_history(1, current, before, path)['move_count'])

    def test_bad_helper_does_not_invent_horizon(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, 'state.json')
            board = (1, 2) + (0,) * 23
            for text in ('bad', '[]', '{}', '{"cpu_used": "NaN"}'):
                with open(path, 'w') as f:
                    f.write(text)
                self.assertIsNone(player.load_history(1, board, board, path)['move_count'])

    def test_huge_helper_cpu_cannot_crash(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, 'state.json')
            before = (1,) + (0,) * 24
            with open(path, 'w') as f:
                json.dump({'version': 1, 'color': 1, 'move_count': 0,
                           'after': list(before), 'cpu_used': 10 ** 400}, f)
            self.assertIsNone(player.load_history(1, before, before, path)['move_count'])

    def test_white_empty_opening_is_after_black_pass(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, 'state.json')
            self.assertEqual(player.load_history(2, player.EMPTY, player.EMPTY, path)['move_count'], 1)
            (pathlib.Path(folder) / 'input.txt').write_text('2\n' + '00000\n' * 10)
            env = dict(os.environ, LITTLE_GO_CPU_SECONDS='0')
            result = subprocess.run([sys.executable, str(PATH)], cwd=folder, env=env,
                                    timeout=5, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr.decode())
            self.assertEqual((pathlib.Path(folder) / 'output.txt').read_text().strip(), 'PASS')

    def test_fresh_process_and_helper_turns(self):
        with tempfile.TemporaryDirectory() as folder:
            inp = pathlib.Path(folder) / 'input.txt'
            inp.write_text('1\n' + '00000\n' * 10)
            env = dict(os.environ, LITTLE_GO_CPU_SECONDS='0.01')
            first = subprocess.run([sys.executable, str(PATH)], cwd=folder, env=env,
                                   timeout=5, capture_output=True)
            self.assertEqual(first.returncode, 0, first.stderr.decode())
            text = (pathlib.Path(folder) / 'output.txt').read_text().strip()
            row, col = map(int, text.split(','))
            after = player.play_move(player.EMPTY, player.EMPTY, 1, 5 * row + col)
            self.assertIsNotNone(after)
            record = json.loads((pathlib.Path(folder) / 'state.json').read_text())
            self.assertEqual(record['after'], list(after))
            rows = '\n'.join(''.join(map(str, after[i:i+5])) for i in range(0, 25, 5))
            inp.write_text('1\n' + rows + '\n' + rows)
            second = subprocess.run([sys.executable, str(PATH)], cwd=folder, env=env,
                                    timeout=5, capture_output=True)
            self.assertEqual(second.returncode, 0, second.stderr.decode())
            record = json.loads((pathlib.Path(folder) / 'state.json').read_text())
            self.assertEqual(record['move_count'], 2)


class Search(unittest.TestCase):
    def setUp(self):
        for name in ('search', 'evaluate'):
            self.assertTrue(callable(getattr(player, name, None)), name + ' not implemented')

    def test_zero_time_legal_fallback(self):
        b = (1,) + (0,) * 24
        action, stats = player.search(player.EMPTY, b, 2, 1, False, cpu_seconds=0)
        self.assertIsNotNone(player.play_move(player.EMPTY, b, 2, action))

    def test_last_move_capture(self):
        b = [0] * 25
        b[0] = b[1] = 2
        b[2] = b[6] = 1
        action, stats = player.search(player.EMPTY, tuple(b), 1, 23, False,
                                      cpu_seconds=0.1, max_depth=1)
        self.assertEqual(action, 5)
        self.assertGreater(stats['value'], 10000)

    def test_last_move_white_komi(self):
        b = (1, 1, 1) + (0,) * 22
        action, stats = player.search(player.EMPTY, b, 2, 23, False,
                                      cpu_seconds=0.1, max_depth=1)
        self.assertNotEqual(action, -1)
        self.assertGreater(stats['value'], 10000)

    def test_winning_second_pass_at_shallow_depth(self):
        b = (1,) + (0,) * 24
        action, stats = player.search(b, b, 2, 2, True, cpu_seconds=0.1, max_depth=1)
        self.assertEqual(action, -1)

    def test_alpha_beta_matches_exhaustive_minimax(self):
        b = (1, 2, 0, 0, 0) + (0,) * 20
        def brute(previous, board, color, count, passed, depth):
            if count >= 24:
                return player.terminal_value(board, color)
            if depth == 0:
                return player.evaluate(board, color)
            values = []
            for action, nxt in player.legal_moves(previous, board, color):
                if action == -1 and passed:
                    value = player.terminal_value(nxt, color)
                else:
                    value = -brute(board, nxt, 3-color, count+1, action == -1, depth-1)
                values.append(value)
            return max(values)
        expected = brute(player.EMPTY, b, 1, 22, False, 2)
        for use_table in (False, True):
            action, stats = player.search(player.EMPTY, b, 1, 22, False,
                cpu_seconds=1, max_depth=2, use_table=use_table)
            self.assertAlmostEqual(stats['value'], expected)
            self.assertEqual(stats['completed_depth'], 2)

    def test_heuristic_is_zero_sum_and_bounded(self):
        b = (1, 2, 1, 0, 2) + (0,) * 20
        self.assertAlmostEqual(player.evaluate(b, 1), -player.evaluate(b, 2))
        self.assertLess(abs(player.evaluate(b, 1)), 10000)

    def test_search_does_not_play_ko(self):
        b = [0] * 25
        for p in (6, 10, 12, 16):
            b[p] = 2
        for p in (7, 13, 17):
            b[p] = 1
        prev = tuple(b)
        current = player.play_move(player.EMPTY, prev, 1, 11)
        action, _ = player.search(prev, current, 2, 10, False, cpu_seconds=0.03)
        self.assertNotEqual(action, 12)

    def test_transposition_reuse_preserves_exact_endgame(self):
        previous = (1,1,0,0,1,1,1,1,1,2,2,1,2,0,0,2,0,2,0,2,0,0,0,1,2)
        board = (1,1,2,0,1,1,1,1,1,2,2,1,2,0,0,2,0,2,0,2,0,0,0,1,2)
        a, cached = player.search(previous, board, 1, 18, False,
                                 cpu_seconds=2, max_depth=6)
        b, uncached = player.search(previous, board, 1, 18, False,
                                   cpu_seconds=2, max_depth=6, use_table=False)
        self.assertGreater(cached['cache_hits'], 0)
        self.assertEqual(cached['completed_depth'], 6)
        self.assertEqual(uncached['completed_depth'], 6)
        self.assertEqual(cached['value'], 10002.5)
        self.assertEqual(cached['value'], uncached['value'])
        self.assertEqual(a, b)


if __name__ == '__main__':
    unittest.main()
