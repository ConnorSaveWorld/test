import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest
from test_root_symmetry import ROOT, load
from reference_rules import apply


class WhiteOpeningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous = load('experiments/candidate_symmetry_tree.py', 'opening_previous')
        cls.candidate = load('experiments/candidate_white_opening.py', 'opening_candidate')

    def test_trained_opening_is_legal_and_does_not_spend_search_budget(self):
        empty = (0,) * 25
        board = tuple(1 if p == 12 else 0 for p in range(25))
        action, stats = self.candidate.search(empty, board, 2, 1, False,
            cpu_seconds=0, wall_seconds=0)
        self.assertTrue(stats.get('opening_policy'))
        self.assertIsNotNone(self.candidate.play_move(empty, board, 2, action))
        self.assertEqual(stats['nodes'], 0)
        self.assertIsNone(stats['value'])

    def test_opening_does_not_override_other_contexts_or_fixed_depth_search(self):
        empty = (0,) * 25
        center = tuple(1 if p == 12 else 0 for p in range(25))
        other = tuple(1 if p == 6 else 0 for p in range(25))
        cases = [(empty, empty, 1, 0, None),
                 (empty, other, 2, 1, None),
                 (empty, center, 2, None, None),
                 (empty, center, 2, 3, None),
                 (other, center, 2, 1, None),
                 (empty, center, 2, 1, 2)]
        for previous, board, color, count, depth in cases:
            options = dict(max_depth=depth, cpu_seconds=0 if depth is None else 10,
                           wall_seconds=0 if depth is None else 15)
            old_action, old = self.previous.search(previous, board, color, count, False, **options)
            action, stats = self.candidate.search(previous, board, color, count, False, **options)
            self.assertFalse(stats.get('opening_policy', False))
            self.assertEqual((action, stats['value'], stats['nodes']),
                             (old_action, old['value'], old['nodes']))

    def test_fresh_process_opening_keeps_history_for_next_white_turn(self):
        source = ROOT / 'experiments/candidate_white_opening.py'
        empty = (0,) * 25
        center = tuple(1 if p == 12 else 0 for p in range(25))
        with tempfile.TemporaryDirectory() as directory:
            folder = pathlib.Path(directory)
            def invoke(previous, board):
                rows = ['2'] + [''.join(map(str, b[r:r + 5]))
                    for b in (previous, board) for r in range(0, 25, 5)]
                (folder / 'input.txt').write_text('\n'.join(rows) + '\n')
                result = subprocess.run([sys.executable, str(source)], cwd=str(folder),
                    env=dict(os.environ, LITTLE_GO_CPU_SECONDS='0.01'),
                    timeout=5, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                text = (folder / 'output.txt').read_text().strip()
                point = -1 if text == 'PASS' else sum(a * b for a, b in
                    zip(map(int, text.split(',')), (5, 1)))
                after = apply(previous, board, 2, point)
                self.assertIsNotNone(after)
                saved = json.loads((folder / 'state.json').read_text())
                self.assertEqual(saved['after'], list(after))
                return after, saved
            after, first = invoke(empty, center)
            self.assertEqual(first['move_count'], 1)
            self.assertTrue(first['last_stats']['opening_policy'])
            black_reply = apply(center, after, 1, 6)
            self.assertIsNotNone(black_reply)
            _, second = invoke(after, black_reply)
            self.assertEqual(second['move_count'], 3)
            self.assertGreaterEqual(second['cpu_used'], first['cpu_used'])
            self.assertFalse(second['last_stats'].get('opening_policy', False))


if __name__ == '__main__':
    unittest.main()
