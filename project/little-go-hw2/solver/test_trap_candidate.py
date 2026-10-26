"""Checks for stage2_trap/my_player3.py book mechanics (needs liblg.so).

Run: python3 test_trap_candidate.py [path-to-candidate]
"""
import importlib.util
import pickle
import random
import sys
import unittest
from pathlib import Path

from lg import State, SYM, transform, tmove, solve, moves, init

HERE = Path(__file__).resolve().parent
CAND = Path(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].endswith('.py') else HERE.parent / 'stage2_trap' / 'my_player3.py'
spec = importlib.util.spec_from_file_location('cand', str(CAND))
cand = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cand)
BOOK = pickle.load(open(HERE / 'book.pkl', 'rb'))
init(24)


def runtime_move(s):
    return cand.book_move(s.B, s.W, s.ko, s.count, bool(s.lp))


class BookTest(unittest.TestCase):
    def test_decode_matches_pickle(self):
        table = cand._load_book()
        expected = sum(1 for kind, cm in BOOK.values() if cm is not None and kind != 'LOST?')
        self.assertEqual(len(table), expected)
        multi = {k: v[1] for k, v in BOOK.items() if isinstance(v[1], list)}
        self.assertEqual(sum(len(v) > 1 for v in table.values()), len(multi))

    def test_every_entry_in_every_orientation(self):
        rng = random.Random(5)
        keys = list(BOOK)
        rng.shuffle(keys)
        for key in keys[:600]:
            kind, cm = BOOK[key]
            s0 = State(*key)
            for t in range(8):
                s = transform(s0, t)
                m = runtime_move(s)
                self.assertIsNotNone(m, (key, t))
                # the returned move must be the image of the canonical move
                # under some symmetry that maps s back onto the canonical state
                cms = cm if isinstance(cm, list) else [cm]
                ok = any(transform(s, u).key() == key and tmove(m, u) in cms for u in range(8))
                self.assertTrue(ok, (key, t, m, cm))
                self.assertIsNotNone(s.play(m), 'book move illegal')

    def test_win_entries_are_proven_wins(self):
        rng = random.Random(7)
        wins = [k for k, v in BOOK.items() if v[0] == 'win']
        for key in rng.sample(wins, min(300, len(wins))):
            s = State(*key)
            child = s.play(BOOK[key][1])
            self.assertIsNotNone(child)
            if child.count < 24:
                self.assertFalse(solve(child)[0], key)

    def test_trap_entries_are_legal_and_lost(self):
        for key, (kind, cm) in BOOK.items():
            if kind not in ('trap', 'prefix'):
                continue
            s = State(*key)
            for c in (cm if isinstance(cm, list) else [cm]):
                self.assertIsNotNone(s.play(c))

    def test_off_book_returns_none(self):
        s = State().play(0).play(24).play(4)
        self.assertIsNone(runtime_move(s))

    def test_exact_endgame_agrees(self):
        rng = random.Random(11)
        late = [k for k, v in BOOK.items() if v[0] == 'win' and k[4] == 11]
        for key in rng.sample(late, min(40, len(late))):
            s = State(*key)
            status, m = cand.exact_solve(s.B, s.W, 2, s.ko, s.count, bool(s.lp), 30.0)
            self.assertEqual(status, 'win')
            child = s.play(m)
            self.assertFalse(solve(child)[0])


if __name__ == '__main__':
    unittest.main(argv=[sys.argv[0]], verbosity=2)
