"""Matched local D4 screen: delivered B7 reply versus normal minimax search."""
import hashlib
import json
from pathlib import Path
import platform

import strong_league
from screen_second_reply_wall_paired import paired_delta, setup_opponent, white_margin


ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    'delivered_reply18': 'stage2_candidate/my_player3.py',
    'search_after_black7': 'experiments/candidate_white_second_reply_search.py',
}
PROFILES = ('baseline', 'pressure', 'space', 'wall3')
WHITE_NODES = 200000
BLACK_NODES = 48000
SEED = 771000
ORIENTATIONS = tuple(range(8))


def run():
    report = {
        'purpose': ('Compare delivered B7 response with normal White minimax search '
                    'while preserving the Black-11 counterwall. Local proxy screen only.'),
        'configuration': {
            'white_nodes': WHITE_NODES,
            'black_nodes': BLACK_NODES,
            'games_per_profile': len(ORIENTATIONS),
            'seed': SEED,
            'opening': 'D4 transforms of Black12, White6, Black7',
            'orientations': ORIENTATIONS,
            'profiles': PROFILES,
            'paired_seed_rule': 'same seed+orientation for both candidates per profile',
            'white_margin_formula': 'white stones - black stones + 2.5 komi',
            'python': platform.python_version(),
        },
        'candidates': {},
    }
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
                game['white_margin'] = white_margin(game)
                game.update(agent='White', source=source, opponent=opponent_name,
                            seed=SEED + index, opening=opening,
                            orientation=index)
                games.append(game)
            groups[opponent_name] = {
                'games': len(games),
                'white_wins': sum(g['winner'] == 2 for g in games),
                'white_losses': sum(g['winner'] == 1 for g in games),
                'ties': sum(g['winner'] == 0 for g in games),
                'invalid_timeout_crash': sum(
                    g['reason'] in ('invalid', 'timeout', 'crash') for g in games),
                'white_margin_sum': sum(g['white_margin'] for g in games),
                'results': games,
            }
            print('{} vs {}: W{} L{} T{} margin={:+.1f}'.format(
                label, opponent_name,
                groups[opponent_name]['white_wins'],
                groups[opponent_name]['white_losses'],
                groups[opponent_name]['ties'],
                groups[opponent_name]['white_margin_sum']), flush=True)
        report['candidates'][label] = {
            'sha256': candidate.__source_sha256__,
            'by_profile': groups,
        }
    reference = report['candidates']['delivered_reply18']['by_profile']
    candidate = report['candidates']['search_after_black7']
    candidate['paired_vs_delivered_reply18'] = {
        profile: paired_delta(candidate['by_profile'][profile]['results'],
                              reference[profile]['results'])
        for profile in PROFILES
    }
    report['source_hashes'] = {
        label: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
        for label, path in SOURCES.items()
    }
    report['opponent_hashes'] = {
        name: module.__source_sha256__ for name, module in opponents.items()
    }
    out = ROOT / 'reports' / 'second-reply-search-paired-d4.json'
    temp = out.with_suffix('.tmp')
    temp.write_text(json.dumps(report, indent=2))
    temp.replace(out)
    print('saved', out, flush=True)


if __name__ == '__main__':
    run()
