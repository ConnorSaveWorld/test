import importlib.util
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
PATH = ROOT / 'submission' / 'my_player3.py'
if PATH.exists():
    spec = importlib.util.spec_from_file_location('player', str(PATH))
    player = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(player)
else:
    player = None


class Rules(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(player, 'The submission rules have not been implemented')

    def test_corner_liberties(self):
        stones, liberties = player.group((1,) + (0,) * 24, 0)
        self.assertEqual(stones, frozenset([0]))
        self.assertEqual(liberties, frozenset([1, 5]))

    def test_diagonals_do_not_connect(self):
        b = [0] * 25
        b[0] = b[6] = 1
        self.assertEqual(player.group(tuple(b), 0)[0], frozenset([0]))

    def test_shared_liberties_are_unique(self):
        b = [0] * 25
        for p in (0, 1, 5):
            b[p] = 1
        self.assertEqual(player.group(tuple(b), 0)[1], frozenset([2, 6, 10]))

    def test_suicide(self):
        b = [0] * 25
        for p in (7, 11, 13, 17):
            b[p] = 2
        self.assertIsNone(player.play_move((0,) * 25, tuple(b), 1, 12))

    def test_multiple_captures_before_suicide(self):
        b = [0] * 25
        b[1] = b[5] = 2
        for p in (2, 6, 10):
            b[p] = 1
        result = player.play_move((0,) * 25, tuple(b), 1, 0)
        self.assertIsNotNone(result)
        self.assertEqual((result[0], result[1], result[5]), (1, 0, 0))
        self.assertEqual(tuple(b)[1], 2)

    def test_connected_group_capture(self):
        b = [0] * 25
        b[0] = b[1] = 2
        b[2] = b[6] = 1
        result = player.play_move((0,) * 25, tuple(b), 1, 5)
        self.assertEqual((result[0], result[1], result[5]), (0, 0, 1))

    def test_ko_and_intervening_move(self):
        b = [0] * 25
        for p in (6, 10, 12, 16):
            b[p] = 2
        for p in (7, 13, 17):
            b[p] = 1
        old = tuple(b)
        captured = player.play_move((0,) * 25, old, 1, 11)
        self.assertIsNotNone(captured)
        self.assertIsNone(player.play_move(old, captured, 2, 12))
        threat = player.play_move(old, captured, 2, 24)
        reply = player.play_move(captured, threat, 1, 20)
        recapture = player.play_move(threat, reply, 2, 12)
        self.assertIsNotNone(recapture)
        self.assertEqual(recapture[11], 0)

    def test_illegal_coordinates_and_occupied(self):
        b = (1,) + (0,) * 24
        for p in (-2, 25, 0):
            self.assertIsNone(player.play_move((0,) * 25, b, 2, p))

    def test_pass_and_all_opening_actions(self):
        b = (0,) * 25
        self.assertEqual(player.play_move(b, b, 1, -1), b)
        actions = player.legal_moves(b, b, 1)
        self.assertEqual(set(p for p, _ in actions), set(range(25)) | {-1})

    def test_komi_and_stone_only_score(self):
        self.assertEqual(player.score_black((1, 1, 2) + (0,) * 22), -1.5)
        self.assertLess(player.terminal_value((1, 1) + (0,) * 23, 1), 0)
        self.assertGreater(player.terminal_value((1, 1, 1) + (0,) * 22, 1), 0)
        self.assertEqual(player.terminal_value((1,) + (0,) * 24, 1),
                         -player.terminal_value((1,) + (0,) * 24, 2))


if __name__ == '__main__':
    unittest.main()
