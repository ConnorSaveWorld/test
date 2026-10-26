"""White-only evaluation experiments; local profiles are not Championship."""
import json
from functools import lru_cache
from pathlib import Path
import benchmark
import compare_stage2_candidates as comparison
import strong_league

SOURCE = 'experiments/candidate_symmetry_tree.py'
WEIGHTS = {
    'current': (8, .6, 6, 1.2, .45, 2),
    'connection0': (8, .6, 6, 1.2, .45, 0),
    'connection1': (8, .6, 6, 1.2, .45, 1),
    'connection4': (8, .6, 6, 1.2, .45, 4),
    'material12': (12, .6, 6, 1.2, .45, 2),
    'material4': (4, .6, 6, 1.2, .45, 2),
    'liberties12': (8, 1.2, 6, 1.2, .45, 2),
    'liberties03': (8, .3, 6, 1.2, .45, 2),
    'atari10': (8, .6, 10, 1.2, .45, 2),
    'weak3': (8, .6, 6, 3, .45, 2),
    'pressure': (6, 1, 10, 3, 1, 2),
    'survival': (8, .8, 8, 2, .4, 2),
}


def install(module, weights):
    weights = tuple(round(w * 20) for w in weights)
    @lru_cache(maxsize=32768)
    def evaluation(black, white):
        liberties = atari = weak = groups = 0
        for side, stones, libs in module._bit_groups(black, white):
            sign = 1 if side == 1 else -1
            groups += sign
            n = module._count(libs)
            liberties += sign * min(n, 6)
            if n <= 1:
                atari += sign * module._count(stones)
            elif n == 2:
                weak += sign * module._count(stones)
        empty = module.FULL & ~(black | white)
        ba, wa = module._adj(black), module._adj(white)
        influence = module._count(ba & empty & ~wa) - module._count(wa & empty & ~ba)
        return (weights[0] * (module._count(black) - module._count(white) - 2.5)
                + weights[1] * liberties - weights[2] * atari - weights[3] * weak
                + weights[4] * influence - weights[5] * groups) / 20.0
    module._bit_static = evaluation


def main():
    results = []
    digest = None
    for label, weights in WEIGHTS.items():
        for profile in ('baseline', 'survival'):
            own_module = strong_league.load_player('baseline', SOURCE)
            digest = own_module.__source_sha256__
            install(own_module, weights)
            other_module = strong_league.load_player(profile)
            opening = (12,)
            own = comparison.cold_policy(own_module, 48000, opening, None)
            other = comparison.cold_policy(other_module, 48000, opening, None)
            game = benchmark.run_game(other, own, 910001, 'official-file')
            game.update(agent_color=2, opponent=profile, opening=list(opening),
                        seed=910001, evaluation=label, weights=list(weights))
            results.append(game)
            report = {'complete': len(results) == len(WEIGHTS) * 2,
                      'source_sha256': digest, 'source': SOURCE,
                      'configuration': {'white_nodes': 48000, 'black_nodes': 48000,
                                        'opening': [12], 'weights': WEIGHTS},
                      'scope': 'Development screening only, not independent validation; White evaluation replaced in its private module, opponent separate.',
                      'results': results}
            comparison.write_report(Path('reports/stage2-white-evaluation-screen.json'), report)
            print(label, profile, 'white_win', game['winner'] == 2, flush=True)


if __name__ == '__main__':
    main()
