"""Original feature experiments informed by validated official positions.

This module never emulates the official Championship implementation.
"""
import json
from functools import lru_cache
from pathlib import Path
import strong_league

SOURCE = 'reports/submission7_snapshot/my_player3.py'
ROW_STARTS = tuple(sum(1 << (5 * r + c) for r in range(5)
                       for c in range(6 - length)) for length in range(1, 6))


@lru_cache(maxsize=16384)
def longest_line(stones):
    if not stones:
        return 0
    horizontal = vertical = stones
    result = 1
    for offset in range(1, 5):
        horizontal &= stones >> offset
        vertical &= stones >> (5 * offset)
        if (horizontal & ROW_STARTS[offset]) or vertical:
            result = offset + 1
    return result


def install(module, line=0, liberty_cap=6):
    base = module._bit_static
    line_weight = round(20 * line)
    @lru_cache(maxsize=32768)
    def evaluation(black, white):
        value = round(20 * base(black, white))
        if liberty_cap != 6:
            difference = sum((1 if color == 1 else -1) *
                (min(module._count(libs), liberty_cap) - min(module._count(libs), 6))
                for color, _, libs in module._bit_groups(black, white))
            value += 12 * difference
        if line_weight:
            b = max(0, longest_line(black) - 1)
            w = max(0, longest_line(white) - 1)
            value += line_weight * (b - w)
        return value / 20.0
    module._feature_base_static = base
    module._feature_longest_line = longest_line
    module._bit_static = evaluation


def cold(module):
    for value in vars(module).values():
        if callable(getattr(value, 'cache_clear', None)):
            value.cache_clear()


def main():
    # Check row wrapping and compare to a direct scanner on all single rows.
    assert longest_line((1 << 4) | (1 << 5)) == 1
    assert longest_line(sum(1 << p for p in (0, 5, 10, 15, 20))) == 5
    history = json.loads(Path('reports/submission7_snapshot/parsed-games.json').read_text())
    games = [g for g in history['results']
             if g['opponent'] == 'championship_player' and g['agent_color'] == 2]
    profiles = [(0, 6), (.75, 6), (1.5, 6), (3, 6), (6, 6),
                (0, 10), (0, 25), (1.5, 25), (3, 25)]
    results = []
    for line, cap in profiles:
        module = strong_league.load_player('baseline', SOURCE)
        install(module, line, cap)
        for game_index in (0, 4):
            move = games[game_index]['moves'][5]
            cold(module)
            action, stats = module.search(tuple(move['previous']), tuple(move['board']),
                2, 5, False, max_nodes=200000, cpu_seconds=10, wall_seconds=15)
            results.append({'line': line, 'liberty_cap': cap, 'official_round': games[game_index]['round'],
                            'count': 5, 'recorded': move['point'], 'chosen': action, 'stats': stats})
            report = {'source_sha256': module.__source_sha256__, 'complete': len(results) == len(profiles) * 2,
                      'configuration': {'nodes': 200000, 'profiles': profiles},
                      'scope': 'Development probes at real official positions; changed action is not evidence of an official win.',
                      'results': results}
            Path('reports/submission7_snapshot/feature-position-probes.json').write_text(json.dumps(report, indent=2))
            print('line', line, 'cap', cap, 'round', games[game_index]['round'],
                  'action', action, 'depth', stats['completed_depth'], 'value', stats['value'], flush=True)


if __name__ == '__main__':
    main()
