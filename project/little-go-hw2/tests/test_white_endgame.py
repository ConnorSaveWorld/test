import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest

from reference_rules import apply

ROOT = pathlib.Path(__file__).resolve().parents[1]
PREVIOUS = tuple(map(int, '1101122111021000221120210'))
BOARD = tuple(map(int, '1101122111021000221120211'))
SOURCES = ('experiments/candidate_symmetry.py',
           'experiments/candidate_symmetry_tree.py',
           'experiments/candidate_white_opening.py',
           'experiments/candidate_white_reply.py')


class WhiteEndgameTests(unittest.TestCase):
    def test_final_white_move_wins_under_independent_stone_scoring(self):
        for source in SOURCES:
            with self.subTest(source=source):
                spec = importlib.util.spec_from_file_location('white_endgame', str(ROOT / source))
                player = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(player)
                point, stats = player.search(PREVIOUS, BOARD, 2, 23, False,
                                              max_depth=1, cpu_seconds=.2)
                after = apply(PREVIOUS, BOARD, 2, point)
                self.assertIsNotNone(after)
                self.assertEqual(after.count(2) + 2.5 - after.count(1), .5)
                self.assertGreater(stats['value'], player.WIN_VALUE)

    def test_fresh_white_entry_preserves_turn_and_cpu_history(self):
        for source in SOURCES:
            with self.subTest(source=source), tempfile.TemporaryDirectory() as folder:
                work = pathlib.Path(folder)
                rows = ['2'] + [''.join(map(str, board[r:r + 5]))
                                for board in (PREVIOUS, BOARD) for r in range(0, 25, 5)]
                (work / 'input.txt').write_text('\n'.join(rows) + '\n')
                (work / 'state.json').write_text(json.dumps({
                    'version': 1, 'color': 2, 'move_count': 21,
                    'cpu_used': 40.0, 'after': list(PREVIOUS)}))
                env = dict(os.environ, LITTLE_GO_CPU_SECONDS='0.2')
                result = subprocess.run([sys.executable, str(ROOT / source)],
                    cwd=str(work), env=env, timeout=5, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                text = (work / 'output.txt').read_text().strip()
                point = -1 if text == 'PASS' else sum(a * b for a, b in
                                                       zip(map(int, text.split(',')), (5, 1)))
                after = apply(PREVIOUS, BOARD, 2, point)
                self.assertIsNotNone(after)
                self.assertEqual(after.count(2) + 2.5 - after.count(1), .5)
                saved = json.loads((work / 'state.json').read_text())
                self.assertEqual(saved['move_count'], 23)
                self.assertEqual(saved['after'], list(after))
                self.assertGreaterEqual(saved['cpu_used'], 40.0)
                self.assertLess(saved['cpu_used'], 40.5)


if __name__ == '__main__':
    unittest.main()
