import sys
from lg import *
games = official_games(); steps = 0; bad = 0
for g in games:
    mv = g['moves']
    s = from_record(mv[0]['previous'], mv[0]['board'], mv[0]['color'], 0)
    for i, m in enumerate(mv):
        r = from_record(m['previous'], m['board'], m['color'], m['move_count'])
        if i > 0 and r.key() != s.key():
            # ko/lp derived from record may differ only if no ko applies; compare boards strictly
            if (r.B, r.W, r.side, r.count) != (s.B, s.W, s.side, s.count) or r.ko != s.ko:
                bad += 1; print('mismatch', g['submission'], g['round'], i, r.key(), s.key()); break
        s2 = s.play(m['point'])
        if s2 is None:
            bad += 1; print('illegal', g['submission'], g['round'], i); break
        s = s2; steps += 1
    final = s.board()
    if final != g['board']:
        bad += 1; print('final mismatch', g['submission'], g['round'])
    bw = 1 if final.count(1) > final.count(2) + 2.5 else 2
    if bw != g['winner']:
        bad += 1; print('winner mismatch', g['submission'], g['round'])
print('games', len(games), 'steps', steps, 'bad', bad)
