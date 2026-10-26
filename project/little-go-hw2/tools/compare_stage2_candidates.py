"""Paired, reproducible local comparisons; these opponents are not Championship."""
import argparse
import hashlib
import json
import math
import pathlib
import platform
import sys
import time

import benchmark
import strong_league

ROOT = benchmark.ROOT
OPPONENT_SOURCE = 'experiments/platform_pvs3.py'
OPPONENTS = ('baseline', 'pressure', 'survival', 'space')
FAILURE_REASONS = ('invalid', 'timeout', 'crash')
# Exactly one representative from each D4 orbit of a single black placement.
DEVELOP_OPENINGS = ((0,), (1,), (2,), (6,), (7,), (12,))
# Fixed, distinct multi-move fixtures, never randomly re-sampled or augmented.
HELDOUT_OPENINGS = ((0, 12), (1, 6), (6, 18), (12, 0, 24),
                    (7, 11, 17), (2, 12, 22, 7))
LIMITATIONS = (
    'These four original local full-legal-search profiles are proxies, not the '
    'official Championship agents. Outcomes do not estimate hidden-grader win '
    'rates. Fixtures are deterministic and correlated, and each is run once '
    'per source, opponent, and own color. Opening moves are forced and included '
    'in the 24-move game limit. Node budgets allow reproducible search work but '
    'are not equal CPU budgets. CPU budgets and timings are in-process and '
    'machine/load dependent; hard timeout is checked by benchmark.run_game. '
    'Heldout means a separate literal fixture set, not a claim that its outcomes '
    'have never been inspected. Search APIs are exercised directly; this does '
    'not test file protocol or whole-game budget allocation.'
)


def validate_opening(opening):
    """Validate literal fixtures with the independent matrix rules oracle."""
    previous = board = (0,) * 25
    for index, point in enumerate(opening):
        updated = benchmark.apply(previous, board, 1 + index % 2, point)
        if updated is None:
            raise ValueError('Illegal fixture move {} in {}'.format(point, opening))
        previous, board = board, updated


def make_fixtures(fixture_set='develop', color=None):
    """The pairing key identifies the exact opponent, opening, and own color."""
    if color not in (None, 1, 2):
        raise ValueError('color must be 1, 2, or None for both')
    openings = {'develop': DEVELOP_OPENINGS, 'heldout': HELDOUT_OPENINGS}[fixture_set]
    if len(set(openings)) != len(openings):
        raise ValueError('Duplicate opening fixture')
    fixtures = []
    for index, opening in enumerate(openings):
        validate_opening(opening)
        for opponent in OPPONENTS:
            for agent_color in ((1, 2) if color is None else (color,)):
                fixtures.append({
                    'id': '{}/{}/vs-{}/{}'.format(fixture_set, index, opponent, agent_color),
                    'opening': list(opening), 'opponent': opponent,
                    'agent_color': agent_color, 'seed': 420000 + index,
                })
    return fixtures


def cold_policy(module, nodes, opening, cpu):
    """Also clear new candidate-specific LRU caches, not only known base caches."""
    play = strong_league.policy(module, nodes, opening, cpu)
    caches = {id(value): value for value in vars(module).values()
              if callable(getattr(value, 'cache_clear', None))}

    def invoke(state, rng):
        if state['move_count'] >= len(opening):
            for cache in caches.values():
                cache.cache_clear()
        return play(state, rng)
    return invoke


def player_failed(game):
    return (game['reason'] in FAILURE_REASONS and
            game.get('failed_color') == game['agent_color'])


def summarize_games(games):
    player_moves = [move for game in games for move in game['moves']
                    if move['color'] == game['agent_color']]
    game_cpu = [sum(move['cpu'] for move in game['moves']
                    if move['color'] == game['agent_color']) for game in games]
    wins = sum(game['winner'] == game['agent_color'] for game in games)
    return {
        'games': len(games), 'wins': wins,
        'win_rate': wins / len(games) if games else 0.0,
        'by_color': {str(color): {
            'games': sum(game['agent_color'] == color for game in games),
            'wins': sum(game['agent_color'] == color and game['winner'] == color
                        for game in games),
        } for color in (1, 2)},
        'failures': {reason: sum(game['reason'] == reason and player_failed(game)
                                 for game in games) for reason in FAILURE_REASONS},
        'opponent_failures': {
            reason: sum(game['reason'] == reason and
                        game.get('failed_color') == 3 - game['agent_color']
                        for game in games) for reason in FAILURE_REASONS},
        'total_player_cpu': sum(game_cpu),
        'max_player_move_cpu': max((move['cpu'] for move in player_moves), default=0.0),
        'max_player_game_cpu': max(game_cpu, default=0.0),
        'total_player_nodes': sum(move['stats'].get('nodes', 0) for move in player_moves),
    }


def summarize_pairs(pairs):
    """Compare binary own wins separately from failure creation/resolution."""
    changes = {'improved': 0, 'regressed': 0, 'unchanged': 0,
               'new_player_failures': 0, 'resolved_player_failures': 0,
               'identical_move_sequences': 0}
    seen = set()
    for pair in pairs:
        fixture = pair['fixture']
        if fixture['id'] in seen:
            raise ValueError('Duplicate pairing key: ' + fixture['id'])
        seen.add(fixture['id'])
        baseline, candidate = pair['baseline'], pair['candidate']
        for game in (baseline, candidate):
            for field in ('fixture_id', 'opening', 'opponent', 'agent_color', 'seed'):
                expected = fixture['id'] if field == 'fixture_id' else fixture[field]
                if game[field] != expected:
                    raise ValueError('Mismatched {} in {}'.format(field, fixture['id']))
        before = baseline['winner'] == baseline['agent_color']
        after = candidate['winner'] == candidate['agent_color']
        key = 'improved' if after > before else 'regressed' if after < before else 'unchanged'
        changes[key] += 1
        changes['new_player_failures'] += player_failed(candidate) and not player_failed(baseline)
        changes['resolved_player_failures'] += player_failed(baseline) and not player_failed(candidate)
        changes['identical_move_sequences'] += (
            [move['point'] for move in baseline['moves']] ==
            [move['point'] for move in candidate['moves']])
    return {
        'baseline': summarize_games([pair['baseline'] for pair in pairs]),
        'candidate': summarize_games([pair['candidate'] for pair in pairs]),
        'paired': changes,
    }


def sha256(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def snapshot_players(baseline, candidate):
    """Compile source bytes before the first game and retain their actual hashes."""
    modules = {
        'baseline': strong_league.load_player('baseline', baseline),
        'candidate': strong_league.load_player('baseline', candidate),
    }
    opponents = {name: strong_league.load_player(name, OPPONENT_SOURCE)
                 for name in OPPONENTS}
    hashes = {}
    for source, module in ([(baseline, modules['baseline']), (candidate, modules['candidate'])] +
                           [(OPPONENT_SOURCE, opponents[name]) for name in OPPONENTS]):
        path = str((ROOT / source).resolve())
        digest = module.__source_sha256__
        if path in hashes and hashes[path] != digest:
            raise RuntimeError('Source changed while snapshots were loaded: ' + path)
        hashes[path] = digest
    manifest = {
        'sources': {name: {'path': str((ROOT / source).resolve()),
                           'sha256': modules[name].__source_sha256__}
                    for name, source in (('baseline', baseline), ('candidate', candidate))},
        'opponents': {name: {
            'path': str((ROOT / OPPONENT_SOURCE).resolve()),
            'sha256': opponents[name].__source_sha256__,
            'profile': list(strong_league.PROFILES[name]),
        } for name in OPPONENTS},
        'tools_sha256': {
            str(path.relative_to(ROOT)): sha256(path) for path in (
                pathlib.Path(__file__).resolve(),
                pathlib.Path(strong_league.__file__).resolve(),
                pathlib.Path(benchmark.__file__).resolve(),
                ROOT / 'tests' / 'reference_rules.py',
            )
        },
    }
    return modules, opponents, manifest


def write_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(report, indent=2), encoding='utf-8')
    temp.replace(path)


def resolve_budgets(nodes=12000, cpu=None, opponent_nodes=None, opponent_cpu=None):
    """Use one budget mode for all players; default opponents to the own budget."""
    if cpu is None and (nodes is None or nodes <= 0):
        raise ValueError('nodes must be positive')
    if cpu is not None and (not math.isfinite(cpu) or cpu <= 0):
        raise ValueError('cpu must be finite and positive')
    if cpu is None and opponent_cpu is not None:
        raise ValueError('opponent-cpu requires CPU mode (--cpu)')
    if cpu is not None and opponent_nodes is not None:
        raise ValueError('opponent-nodes requires node mode (--nodes)')
    if opponent_nodes is not None and opponent_nodes <= 0:
        raise ValueError('opponent-nodes must be positive')
    if opponent_cpu is not None and (not math.isfinite(opponent_cpu) or opponent_cpu <= 0):
        raise ValueError('opponent-cpu must be finite and positive')
    if cpu is None:
        return nodes, None, nodes if opponent_nodes is None else opponent_nodes, None
    return None, cpu, None, cpu if opponent_cpu is None else opponent_cpu


def compare(baseline, candidate, output, nodes=12000, cpu=None, fixture_set='develop', color=None,
            opponent_nodes=None, opponent_cpu=None):
    nodes, cpu, opponent_nodes, opponent_cpu = resolve_budgets(
        nodes, cpu, opponent_nodes, opponent_cpu)
    fixtures = make_fixtures(fixture_set, color)
    modules, opponents, manifest = snapshot_players(baseline, candidate)
    path = ROOT / output
    pairs = []
    report = {
        'complete': False, 'expected_pairs': len(fixtures),
        'manifest': manifest, 'python': sys.version, 'platform': platform.platform(),
        'configuration': {
            'fixture_set': fixture_set, 'agent_colors': [1, 2] if color is None else [color],
            'nodes_per_searched_turn': nodes,
            'cpu_seconds_per_searched_turn': cpu,
            'cpu_guard_seconds': 8 if cpu is None else cpu,
            'wall_guard_seconds': 9 if cpu is None else cpu + 1,
            'opponent_nodes_per_searched_turn': opponent_nodes,
            'opponent_cpu_seconds_per_searched_turn': opponent_cpu,
            'opponent_cpu_guard_seconds': 8 if opponent_cpu is None else opponent_cpu,
            'opponent_wall_guard_seconds': 9 if opponent_cpu is None else opponent_cpu + 1,
            'cache': 'all module LRU caches cleared before every searched turn',
            'mode': 'official-file',
            'run_order': 'baseline/candidate order alternates by paired fixture',
            'opponents_receive_same_budget': (nodes, cpu) == (opponent_nodes, opponent_cpu),
        },
        'fixtures': fixtures, 'results': pairs, 'limitations': LIMITATIONS,
    }
    began = time.monotonic()
    for index, fixture in enumerate(fixtures):
        pair = {'fixture': fixture, 'run_order': (
            ['baseline', 'candidate'] if index % 2 == 0 else ['candidate', 'baseline'])}
        for label in pair['run_order']:
            own = cold_policy(modules[label], nodes, fixture['opening'], cpu)
            other = cold_policy(opponents[fixture['opponent']], opponent_nodes,
                                fixture['opening'], opponent_cpu)
            color = fixture['agent_color']
            game = benchmark.run_game(own if color == 1 else other,
                                      other if color == 1 else own,
                                      fixture['seed'], 'official-file')
            game.update(fixture_id=fixture['id'], opening=fixture['opening'],
                        opponent=fixture['opponent'], agent_color=color, seed=fixture['seed'])
            pair[label] = game
        pairs.append(pair)
        report['summary'] = summarize_pairs(pairs)
        report['by_opponent'] = {
            name: summarize_pairs([item for item in pairs if item['fixture']['opponent'] == name])
            for name in OPPONENTS}
        report['completed_pairs'] = len(pairs)
        report['elapsed_wall_seconds'] = time.monotonic() - began
        report['complete'] = len(pairs) == len(fixtures)
        write_report(path, report)
        print('{}/{} {}: baseline={} candidate={}'.format(
            index + 1, len(fixtures), fixture['id'],
            pair['baseline']['winner'] == color, pair['candidate']['winner'] == color), flush=True)
    print(json.dumps(report['summary'], indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--output', required=True)
    budget = parser.add_mutually_exclusive_group()
    budget.add_argument('--nodes', type=int, default=12000)
    budget.add_argument('--cpu', type=float)
    parser.add_argument('--opponent-nodes', type=int,
                        help='opponent node budget in node mode; default is own budget')
    parser.add_argument('--opponent-cpu', type=float,
                        help='opponent CPU seconds in CPU mode; default is own budget')
    parser.add_argument('--fixture-set', choices=('develop', 'heldout'), default='develop')
    parser.add_argument('--color', type=int, choices=(1, 2),
                        help='only test this own color; default tests both')
    args = parser.parse_args()
    try:
        resolve_budgets(args.nodes, args.cpu, args.opponent_nodes, args.opponent_cpu)
    except ValueError as error:
        parser.error(str(error))
    compare(args.baseline, args.candidate, args.output, args.nodes, args.cpu, args.fixture_set,
            args.color, args.opponent_nodes, args.opponent_cpu)


if __name__ == '__main__':
    main()
