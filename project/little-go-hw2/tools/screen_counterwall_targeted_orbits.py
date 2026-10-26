"""Paired local opening screen for the White counterwall hypothesis.

These hand-built opponents are local search variants, not the official grader.
The first three moves recreate the documented Black 12 / White 6 / Black 7
approach in all D4 orientations; after that both sides search freely.
"""
import json
from pathlib import Path

import strong_league


ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    'reply18': 'experiments/candidate_white_reply.py',
    'targeted13': 'experiments/candidate_white_counterwall_targeted.py',
}
PROFILES = ('baseline', 'pressure', 'space', 'wall3')
WHITE_NODES = 200000
BLACK_NODES = 48000
SEED = 771000
ORIENTATIONS = tuple(range(8))
GAMES_PER_PROFILE = len(ORIENTATIONS)


def setup_opponent(name):
    module = strong_league.load_player('baseline' if name == 'wall3' else name)
    if name == 'wall3':
        from official_white_experiments import install
        install(module, line=3, liberty_cap=6)
    return module


def run():
    report = {'purpose': 'Paired offline White opening response screen; not official Championship.',
              'configuration': {'white_nodes': WHITE_NODES,
                  'black_nodes': BLACK_NODES,
                  'games_per_profile': GAMES_PER_PROFILE, 'seed': SEED,
                  'opening': 'D4 transforms of Black12, White6, Black7',
                  'orientations': ORIENTATIONS,
                  'profiles': PROFILES}, 'candidates': {}}
    opponents = {name: setup_opponent(name) for name in PROFILES}
    for label, source in SOURCES.items():
        candidate = strong_league.load_player('baseline', source)
        groups = {}
        for opponent_name, opponent in opponents.items():
            games = []
            for index in ORIENTATIONS:
                mapping = candidate._SYMMETRY_MAPS[index]
                opening = [mapping[p] for p in (12, 6, 7)]
                white = strong_league.policy(candidate, WHITE_NODES, opening)
                black = strong_league.policy(opponent, BLACK_NODES, opening)
                game = strong_league.run_game(black, white, SEED + index,
                                              mode='official-file')
                game.update(agent='White', source=source, opponent=opponent_name,
                            seed=SEED + index, opening=opening,
                            orientation=index)
                games.append(game)
            groups[opponent_name] = {'games': len(games),
                'white_wins': sum(g['winner'] == 2 for g in games),
                'white_losses': sum(g['winner'] == 1 for g in games),
                'ties': sum(g['winner'] == 0 for g in games),
                'invalid_timeout_crash': sum(g['reason'] in ('invalid', 'timeout', 'crash') for g in games),
                'results': games}
            print(label, opponent_name, groups[opponent_name]['white_wins'], '/', len(games), flush=True)
        report['candidates'][label] = {'sha256': candidate.__source_sha256__,
                                       'by_profile': groups}
    out = ROOT / 'reports' / 'counterwall-opening-screen-targeted-d4.json'
    temp = out.with_suffix('.tmp')
    temp.write_text(json.dumps(report, indent=2))
    temp.replace(out)
    print('saved', out, flush=True)


if __name__ == '__main__':
    run()
