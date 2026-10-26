"""Parse downloaded official history and independently validate every board."""
import hashlib
import json
import re
from pathlib import Path
from benchmark import ROOT, GO, apply


def parse(path):
    raw = Path(path).read_bytes()
    text = raw.decode('utf-8').replace('\r\n', '\n')
    sections = re.split(r'^==Playing with ([^=]+)==$', text, flags=re.M)
    games = []
    for index in range(1, len(sections), 2):
        opponent, section = sections[index:index + 2]
        rounds = re.split(r'^=====Round (\d+)=====$', section, flags=re.M)
        for offset in range(1, len(rounds), 2):
            number, chunk = rounds[offset:offset + 2]
            own = 2 if 'Black:TA White:You' in chunk else 1
            assert ('Black:TA White:You' if own == 2 else 'Black:You White:TA') in chunk
            previous = board = (0,) * 25
            passed = False
            moves = []
            for match in re.finditer(r'^(Black|White) (PASS|([0-4]),([0-4]))$', chunk, re.M):
                color = 1 if match[1] == 'Black' else 2
                count = len(moves)
                assert color == 1 + count % 2
                point = -1 if match[2] == 'PASS' else 5 * int(match[3]) + int(match[4])
                updated = apply(previous, board, color, point)
                assert updated is not None, (opponent, number, count, 'illegal')
                start = chunk.index('----------', match.end())
                rows = chunk[start:].splitlines()[1:6]
                assert len(rows) == 5 and all(len(row) >= 9 for row in rows)
                displayed = tuple({' ': 0, 'X': 1, 'O': 2}[row[c * 2]]
                                  for row in rows for c in range(5))
                assert displayed == updated, (opponent, number, count, 'diagram mismatch')
                go = GO(5)
                go.set_board(color, [list(previous[r:r + 5]) for r in range(0, 25, 5)],
                             [list(board[r:r + 5]) for r in range(0, 25, 5)])
                if point != -1:
                    assert go.place_chess(point // 5, point % 5, color)
                    go.remove_died_pieces(3 - color)
                assert tuple(c for row in go.board for c in row) == updated
                moves.append({'previous': previous, 'board': board, 'color': color,
                              'move_count': count, 'last_pass': passed, 'point': point})
                ends = point == -1 and (passed or previous == board)
                previous, board = board, updated
                passed = point == -1
            winner_match = re.search(r'^The winner is ([XO])$', chunk, re.M)
            assert winner_match and moves
            winner = 1 if winner_match[1] == 'X' else 2
            assert go.judge_winner() == winner
            assert ends or len(moves) == 24
            times = {label: float(re.search(r'^' + label + r' time: ([\d.]+)$', chunk, re.M)[1])
                     for label in ('ta', 'student')}
            games.append({'opponent': opponent, 'round': int(number), 'agent_color': own,
                          'winner': winner, 'steps': len(moves), 'moves': moves,
                          'reason': 'passes' if ends else 'limit', 'board': board,
                          'white_margin': board.count(2) + 2.5 - board.count(1),
                          'reported_cumulative_times': times})
    assert len(games) == 20
    return {'source_history_sha256': hashlib.sha256(raw).hexdigest(),
            'games': len(games), 'wins': sum(g['winner'] == g['agent_color'] for g in games),
            'results': games, 'validation': 'All diagrams and moves agree with independent matrix rules and original GO; terminal scores verified.'}


if __name__ == '__main__':
    report = parse(ROOT / 'reports/submission7_snapshot/history.log')
    (ROOT / 'reports/submission7_snapshot/parsed-games.json').write_text(json.dumps(report, indent=2))
    print('Validated', report['games'], 'official games and',
          sum(g['steps'] for g in report['results']), 'actions')
    for game in report['results']:
        print(game['opponent'], game['round'], 'own', game['agent_color'],
              'winner', game['winner'], 'white margin', game['white_margin'],
              'reported times', game['reported_cumulative_times'])
        if game['opponent'] == 'championship_player' and game['agent_color'] == 2:
            print('  points', [m['point'] for m in game['moves']])
