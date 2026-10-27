"""Step 1: is any White first reply to B12 a forced win?

For White's first move w, White is lost iff some Black reply b leaves White
(to move at count 3) lost. We try Black replies in a center-first order and
stop at the first proven win for Black. Run one class per process.
"""
import sys, time, json
from lg import *

ORDER = (12, 7, 11, 13, 17, 6, 8, 16, 18, 2, 10, 14, 22, 1, 3, 5, 9, 15, 19, 21, 23, 0, 4, 20, 24)

if __name__ == '__main__':
    w = int(sys.argv[1])
    init(int(sys.argv[2]) if len(sys.argv) > 2 else 27)
    s1 = State().play(12).play(w)
    seen = set()
    result = {'white_first': w, 'tried': []}
    for b in ORDER:
        s2 = s1.play(b)
        if s2 is None: continue
        k = canon(s2)[0]
        if k in seen: continue
        seen.add(k)
        t = time.time(); n0 = nodes()
        white_wins, best = solve(s2)
        row = {'black': b, 'white_wins': bool(white_wins), 'white_best': best,
               'seconds': round(time.time() - t, 1), 'nodes': nodes() - n0}
        result['tried'].append(row)
        print(json.dumps(row), flush=True)
        if not white_wins:
            result['white_first_lost'] = True
            break
    else:
        result['white_first_lost'] = False
    print('RESULT', json.dumps(result), flush=True)
