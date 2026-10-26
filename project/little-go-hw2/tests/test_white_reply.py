import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest
from test_root_symmetry import ROOT, load
from reference_rules import apply


class WhiteReplyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old = load('experiments/candidate_symmetry_tree.py', 'reply_old')
        cls.player = load('experiments/candidate_white_reply.py', 'reply_candidate')

    def test_learned_reply_is_legal_in_all_eight_board_orientations(self):
        a = self.player
        for mapping in a._SYMMETRY_MAPS:
            previous = tuple(1 if p == mapping[12] else
                             2 if p == mapping[6] else 0 for p in range(25))
            board = tuple(1 if p in (mapping[12], mapping[7]) else
                          2 if p == mapping[6] else 0 for p in range(25))
            action, stats = a.search(previous, board, 2, 3, False,
                                    cpu_seconds=0, wall_seconds=0)
            self.assertTrue(stats.get('opening_policy'))
            self.assertEqual(action, mapping[18])
            self.assertIsNotNone(a.play_move(previous, board, 2, action))
            self.assertEqual(stats['nodes'], 0)
            self.assertIsNone(stats['value'])

    def test_reply_does_not_override_wrong_history_or_fixed_depth(self):
        previous = tuple(1 if p == 12 else 2 if p == 6 else 0 for p in range(25))
        board = tuple(1 if p in (12, 7) else 2 if p == 6 else 0 for p in range(25))
        for color, count, passed, depth, prev in ((1, 2, False, None, previous),
                (2, None, False, None, previous), (2, 5, False, None, previous),
                (2, 3, True, None, previous), (2, 3, False, None, board),
                (2, 3, False, 2, previous)):
            options = dict(cpu_seconds=10 if depth else 0,
                           wall_seconds=15 if depth else 0, max_depth=depth)
            action, stats = self.player.search(prev, board, color, count, passed, **options)
            old_action, old = self.old.search(prev, board, color, count, passed, **options)
            self.assertFalse(stats.get('opening_policy', False))
            self.assertEqual((action, stats['value'], stats['nodes']),
                             (old_action, old['value'], old['nodes']))

    def test_fresh_process_reply_preserves_helper_for_next_white_turn(self):
        previous = tuple(1 if p == 12 else 2 if p == 6 else 0 for p in range(25))
        board = tuple(1 if p in (12, 7) else 2 if p == 6 else 0 for p in range(25))
        with tempfile.TemporaryDirectory() as directory:
            folder = pathlib.Path(directory)
            (folder / 'state.json').write_text(json.dumps({'version': 1, 'color': 2,
                'move_count': 1, 'cpu_used': 1.0, 'after': list(previous)}))
            def invoke(prev, now):
                rows = ['2'] + [''.join(map(str, b[r:r + 5]))
                    for b in (prev, now) for r in range(0, 25, 5)]
                (folder / 'input.txt').write_text('\n'.join(rows) + '\n')
                result = subprocess.run([sys.executable,
                    str(ROOT / 'experiments/candidate_white_reply.py')],
                    cwd=str(folder), env=dict(os.environ, LITTLE_GO_CPU_SECONDS='0.01'),
                    timeout=5, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                text = (folder / 'output.txt').read_text().strip()
                point = -1 if text == 'PASS' else sum(a * b for a, b in
                    zip(map(int, text.split(',')), (5, 1)))
                after = apply(prev, now, 2, point)
                self.assertIsNotNone(after)
                saved = json.loads((folder / 'state.json').read_text())
                self.assertEqual(saved['after'], list(after))
                return after, saved
            after, first = invoke(previous, board)
            self.assertTrue(first['last_stats']['opening_policy'])
            self.assertEqual(first['move_count'], 3)
            self.assertGreaterEqual(first['cpu_used'], 1.0)
            black_reply = apply(board, after, 1, 17)
            self.assertIsNotNone(black_reply)
            _, second = invoke(after, black_reply)
            self.assertFalse(second['last_stats'].get('opening_policy', False))
            self.assertEqual(second['move_count'], 5)
            self.assertGreaterEqual(second['cpu_used'], first['cpu_used'])


if __name__ == '__main__':
    unittest.main()
