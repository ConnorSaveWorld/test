import importlib.util
import json
import pathlib
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    'stage2_history_candidate', str(ROOT / 'experiments' / 'candidate_symmetry.py'))
player = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(player)


class Stage2HistoryTests(unittest.TestCase):
    def test_matching_white_helper_preserves_cpu_when_turn_count_is_unknown(self):
        previous = (1, 2) + (0,) * 23
        board = (1, 2, 1) + (0,) * 22
        with tempfile.TemporaryDirectory() as folder:
            path = pathlib.Path(folder) / 'state.json'
            path.write_text(json.dumps({'version': 1, 'color': 2,
                'after': list(previous), 'move_count': None, 'cpu_used': 40.0}))
            history = player.load_history(2, previous, board, str(path))
        self.assertIsNone(history['move_count'])
        self.assertEqual(history['cpu_used'], 40.0)

    def test_mismatched_helper_still_resets_cpu_and_turn_count(self):
        previous = (1, 2) + (0,) * 23
        with tempfile.TemporaryDirectory() as folder:
            path = pathlib.Path(folder) / 'state.json'
            path.write_text(json.dumps({'version': 1, 'color': 2,
                'after': list((2, 1) + (0,) * 23), 'move_count': None,
                'cpu_used': 40.0}))
            history = player.load_history(2, previous, previous, str(path))
        self.assertEqual(history, {'move_count': None, 'cpu_used': 0.0})


if __name__ == '__main__':
    unittest.main()
