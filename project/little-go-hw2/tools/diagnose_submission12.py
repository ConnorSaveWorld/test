"""Probe actual submitted White losses; this is not an opponent emulator."""
import argparse
import json
from pathlib import Path
import platform
from strong_league import load_player

ROOT = Path(__file__).resolve().parents[1]
SOURCE = 'reports/official-submission12-20261023/cnt12/my_player3.py'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', type=float, default=8.0)
    parser.add_argument('--nodes', type=int, default=2000000)
    args = parser.parse_args()
    history = json.loads((ROOT / 'reports/official-submission12-20261023/cnt12/parsed-games.json').read_text())
    unique = {}
    for game in history['results']:
        if game['opponent'] == 'championship_player' and game['agent_color'] == 2:
            unique.setdefault(tuple(m['point'] for m in game['moves']), game)
    report = {'python': platform.python_version(), 'source': SOURCE,
              'configuration': vars(args), 'complete': False, 'results': [],
              'scope': 'Search diagnostics on recorded positions; no official win claim.'}
    output = ROOT / 'reports/submission12-endgame-diagnosis.json'
    for game in unique.values():
        for count in (19, 17, 15, 13, 11, 9):
            module = load_player('baseline', SOURCE)
            state = game['moves'][count]
            action, stats = module.search(tuple(state['previous']), tuple(state['board']),
                2, count, state['last_pass'], cpu_seconds=args.cpu,
                wall_seconds=args.cpu + 2, max_nodes=args.nodes)
            row = {'round': game['round'], 'count': count, 'recorded': state['point'],
                   'chosen': action, 'stats': stats,
                   'solved_to_limit': stats['completed_depth'] >= 24 - count}
            report['results'].append(row)
            output.write_text(json.dumps(report, indent=2))
            print(json.dumps(row), flush=True)
    report['complete'] = True
    output.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
