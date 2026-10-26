"""Generate the Stage-2 trap-book candidate from the current submission.

Inputs: current my_player3.py (cefadb4f...), exact_py.py, book.pkl.
Output: stage2_trap/my_player3.py. Black's code path is unchanged.
"""
import base64, hashlib, pickle, sys, ast
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SRC = ROOT / 'stage2_candidate' / 'my_player3.py'
OUT = ROOT / 'stage2_trap' / 'my_player3.py'
EXPECTED = 'cefadb4f805e5d95e2f157a5b50619e476ed05dea47bef36e6ac1a0a56661698'


def encode_book(book):
    rows = []
    for k, (kind, cm) in sorted(book.items()):
        if cm is None or kind == 'LOST?':
            continue
        B, W, side, ko, count, lp = k
        assert side == 2
        key = B | (W << 25) | (count << 50) | (lp << 55) | (ko << 56)
        rows.append(key.to_bytes(9, 'big') + bytes([cm + 1]))
    blob = base64.b85encode(b''.join(rows)).decode('ascii')
    lines = [blob[i:i + 100] for i in range(0, len(blob), 100)]
    return len(rows), '\n'.join(lines)


RUNTIME = r'''

# ---------------------------------------------------------------------------
# White trap book and exact endgame (added for Stage 2, White only).
# The book was generated offline by our own exact win/loss solver: on lines
# the Championship opponent has actually played, White chooses moves that
# leave Black a single winning reply, and after any losing Black reply it
# stores a proven winning White move.  Positions are matched up to the eight
# board symmetries, including the ko point.
_BOOK = None


def _load_book():
    global _BOOK
    if _BOOK is None:
        import base64
        raw = base64.b85decode(''.join(_BOOK_DATA.split()))
        table = {}
        for i in range(0, len(raw), 10):
            table[int.from_bytes(raw[i:i + 9], 'big')] = raw[i + 9] - 1
        _BOOK = table
    return _BOOK


def book_move(black, white, ko, count, last_pass):
    """Return the book move for White in this exact state, or None."""
    if count is None:
        return None
    book = _load_book()
    for mapping in _SYMMETRY_MAPS:
        tko = 0
        if ko:
            p, q = divmod(ko - 1, 25)
            tko = 1 + 25 * mapping[p] + mapping[q]
        key = (_transform_bits(black, mapping) | (_transform_bits(white, mapping) << 25)
               | (count << 50) | ((1 if last_pass else 0) << 55) | (tko << 56))
        move = book.get(key)
        if move is not None:
            if move == PASS:
                return PASS
            return mapping.index(move)
    return None


def white_policy(previous, board, count, last_pass, budget, started):
    """Book move, else proven endgame win; None means use the normal search."""
    if count is None or count % 2 != 1:
        return None, {}
    pb, pw = _pack(previous)
    black, white = _pack(board)
    ko = exact_ko(pb, pw, black, white, 2)
    legal = set(p for p, _, _ in _bit_moves(pb, pw, black, white, 2))
    try:
        move = book_move(black, white, ko, count, last_pass)
    except Exception:
        move = None
    if move is not None and move in legal:
        return move, {'policy': 'book'}
    if count >= EXACT_FROM:
        limit = min(EXACT_SECONDS, max(budget - 0.5, 0.5))
        status, move = exact_solve(black, white, 2, ko, count, last_pass, limit)
        if status == 'win' and move in legal:
            return move, {'policy': 'exact-win'}
        return None, {'exact': status}
    return None, {}


EXACT_FROM = 13
EXACT_SECONDS = 6.0
'''

MAIN_OLD = """    last_pass = previous == board and (board != EMPTY or color == 2)
    action, stats = search(previous, board, color, count, last_pass,
                           cpu_seconds=budget, wall_seconds=min(9.5, budget + 1.0))
"""
MAIN_NEW = """    last_pass = previous == board and (board != EMPTY or color == 2)
    action = None
    stats = {}
    if color == 2:
        try:
            action, stats = white_policy(previous, board, count, last_pass,
                                         budget, process_started)
        except Exception:
            action, stats = None, {}
    if action is None:
        spent = time.process_time() - process_started
        remaining = max(0.5, budget - spent)
        action, search_stats = search(previous, board, color, count, last_pass,
                                      cpu_seconds=remaining,
                                      wall_seconds=min(9.5 - spent, remaining + 1.0))
        search_stats.update(stats)
        stats = search_stats
"""


def main():
    src = SRC.read_text()
    assert hashlib.sha256(SRC.read_bytes()).hexdigest() == EXPECTED, 'unexpected base source'
    assert MAIN_OLD in src
    src = src.replace(MAIN_OLD, MAIN_NEW)
    exact = (HERE / 'exact_py.py').read_text()
    exact = exact.split('import time\n', 1)[1]  # module docstring/import already present
    book = pickle.load(open(HERE / (sys.argv[1] if len(sys.argv) > 1 else 'book.pkl'), 'rb'))
    n, data = encode_book(book)
    tail = "\n\nif __name__ == '__main__':\n    main()\n"
    assert src.endswith(tail)
    body = src[:-len(tail)]
    body += '\n\n# Exact win/loss search (original, see header).' + exact + RUNTIME
    body += '\n_BOOK_DATA = """\n' + data + '\n"""\n'
    body += tail
    ast.parse(body, feature_version=(3, 7))
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(body)
    print('book entries', n, 'file bytes', len(body.encode()), 'sha256', hashlib.sha256(body.encode()).hexdigest())


if __name__ == '__main__':
    main()
