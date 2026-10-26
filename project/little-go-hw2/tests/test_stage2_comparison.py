import copy
from functools import lru_cache
import pathlib
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import compare_stage2_candidates as comparison


def game(fixture, win=False, reason='limit', failed_color=None):
    color = fixture['agent_color']
    return dict(fixture_id=fixture['id'], opening=list(fixture['opening']),
                opponent=fixture['opponent'], agent_color=color, seed=fixture['seed'],
                winner=color if win else 3 - color, reason=reason, failed_color=failed_color,
                moves=[{'color': color, 'point': 12, 'cpu': 0.25, 'stats': {'nodes': 5}},
                       {'color': 3 - color, 'point': 0, 'cpu': 0.5, 'stats': {}}])


class Stage2Comparison(unittest.TestCase):
    def test_opponent_budgets_default_to_own_and_allow_stronger_stress(self):
        self.assertEqual(comparison.resolve_budgets(), (12000, None, 12000, None))
        self.assertEqual(comparison.resolve_budgets(nodes=12000, opponent_nodes=48000),
                         (12000, None, 48000, None))
        self.assertEqual(comparison.resolve_budgets(cpu=0.1), (None, 0.1, None, 0.1))
        self.assertEqual(comparison.resolve_budgets(cpu=0.1, opponent_cpu=0.4),
                         (None, 0.1, None, 0.4))

    def test_invalid_or_mixed_opponent_budget_modes_are_rejected(self):
        for options in ({'nodes': 0}, {'cpu': 0}, {'cpu': float('nan')},
                        {'opponent_nodes': 0}, {'opponent_cpu': 0.1},
                        {'cpu': 0.1, 'opponent_nodes': 48000},
                        {'cpu': 0.1, 'opponent_cpu': -1},
                        {'cpu': 0.1, 'opponent_cpu': float('inf')}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                comparison.resolve_budgets(**options)

    def test_opponent_budget_reaches_policy_and_report_without_playing_matches(self):
        fixture = comparison.make_fixtures(color=2)[0]
        for options, own, other in (
                ({'opponent_nodes': 48000}, (12000, None), (48000, None)),
                ({'cpu': 0.1, 'opponent_cpu': 0.4}, (None, 0.1), (None, 0.4)),
                ({}, (12000, None), (12000, None))):
            with self.subTest(options=options):
                with patch.object(comparison, 'make_fixtures', return_value=[fixture]), \
                        patch.object(comparison, 'snapshot_players', return_value=(
                            {'baseline': 'before', 'candidate': 'after'}, {'baseline': 'opponent'}, {})), \
                        patch.object(comparison, 'cold_policy') as policy, \
                        patch.object(comparison.benchmark, 'run_game',
                                     side_effect=[game(fixture), game(fixture)]), \
                        patch.object(comparison, 'write_report'), patch('builtins.print'):
                    report = comparison.compare('before.py', 'after.py', 'unused.json',
                                                color=2, **options)
                calls = policy.call_args_list
                self.assertEqual([(call[0][1], call[0][3]) for call in calls],
                                 [own, other, own, other])
                config = report['configuration']
                self.assertEqual(config['opponent_nodes_per_searched_turn'], other[0])
                self.assertEqual(config['opponent_cpu_seconds_per_searched_turn'], other[1])
                self.assertEqual(config['opponents_receive_same_budget'], own == other)

    def test_develop_fixtures_are_distinct_and_balanced(self):
        fixtures = comparison.make_fixtures()
        self.assertEqual(len(fixtures), 48)
        self.assertEqual(len({item['id'] for item in fixtures}), 48)
        self.assertEqual({tuple(item['opening']) for item in fixtures},
                         {(0,), (1,), (2,), (6,), (7,), (12,)})
        for opponent in comparison.OPPONENTS:
            for opening in comparison.DEVELOP_OPENINGS:
                pair = [item for item in fixtures if item['opponent'] == opponent
                        and item['opening'] == list(opening)]
                self.assertEqual({item['agent_color'] for item in pair}, {1, 2})
                self.assertEqual(len({item['seed'] for item in pair}), 1)

    def test_heldout_literals_are_legal_and_separate(self):
        fixtures = comparison.make_fixtures('heldout')
        self.assertTrue(all(2 <= len(item['opening']) <= 4 for item in fixtures))
        self.assertTrue({tuple(item['opening']) for item in fixtures}.isdisjoint(
            comparison.DEVELOP_OPENINGS))
        with self.assertRaises(ValueError):
            comparison.validate_opening((12, 12))

    def test_color_filter_preserves_exact_fixture_ids_and_counts(self):
        for fixture_set in ('develop', 'heldout'):
            all_fixtures = comparison.make_fixtures(fixture_set)
            for color in (1, 2):
                filtered = comparison.make_fixtures(fixture_set, color)
                self.assertEqual(len(filtered), 24)
                self.assertEqual(filtered, [item for item in all_fixtures
                                            if item['agent_color'] == color])
        with self.assertRaises(ValueError):
            comparison.make_fixtures(color=3)

    def test_pairing_rejects_changed_openings_and_duplicate_games(self):
        fixture = comparison.make_fixtures()[0]
        pair = {'fixture': fixture, 'baseline': game(fixture), 'candidate': game(fixture)}
        changed = copy.deepcopy(pair)
        changed['candidate']['opening'] = [1]
        with self.assertRaisesRegex(ValueError, 'Mismatched opening'):
            comparison.summarize_pairs([changed])
        with self.assertRaisesRegex(ValueError, 'Duplicate pairing key'):
            comparison.summarize_pairs([pair, pair])

    def test_new_candidate_caches_are_cold_for_each_searched_turn(self):
        @lru_cache(maxsize=8)
        def candidate_only_cache(value):
            return value
        module = SimpleNamespace(extra_cache=candidate_only_cache)
        observed_sizes = []

        def play(state, rng):
            observed_sizes.append(candidate_only_cache.cache_info().currsize)
            candidate_only_cache(state['move_count'])
            return 0, {}

        candidate_only_cache(99)
        with patch.object(comparison.strong_league, 'policy', return_value=play):
            policy = comparison.cold_policy(module, 1, [0], None)
            policy({'move_count': 0}, None)  # Forced opening does not search.
            policy({'move_count': 1}, None)
            policy({'move_count': 2}, None)
        self.assertEqual(observed_sizes, [1, 0, 0])

    def test_outcomes_failures_and_cpu_are_counted_by_own_color(self):
        fixtures = comparison.make_fixtures()[:4]
        pairs = [{'fixture': fixture, 'baseline': game(fixture), 'candidate': game(fixture)}
                 for fixture in fixtures]
        pairs[0]['candidate'] = game(fixtures[0], True)
        pairs[1]['baseline'] = game(fixtures[1], True)
        pairs[1]['candidate'] = game(fixtures[1], False, 'crash', fixtures[1]['agent_color'])
        pairs[2]['baseline'] = game(fixtures[2], False, 'invalid', fixtures[2]['agent_color'])
        pairs[3]['candidate'] = game(fixtures[3], True, 'timeout', 3 - fixtures[3]['agent_color'])
        summary = comparison.summarize_pairs(pairs)
        self.assertEqual(summary['paired']['improved'], 2)
        self.assertEqual(summary['paired']['regressed'], 1)
        self.assertEqual(summary['paired']['unchanged'], 1)
        self.assertEqual(summary['paired']['new_player_failures'], 1)
        self.assertEqual(summary['paired']['resolved_player_failures'], 1)
        self.assertEqual(summary['candidate']['failures'],
                         {'invalid': 0, 'timeout': 0, 'crash': 1})
        self.assertEqual(summary['candidate']['opponent_failures']['timeout'], 1)
        self.assertEqual(summary['candidate']['by_color']['1'], {'games': 2, 'wins': 1})
        self.assertEqual(summary['candidate']['by_color']['2'], {'games': 2, 'wins': 1})
        self.assertEqual(summary['candidate']['total_player_cpu'], 1.0)
        self.assertEqual(summary['candidate']['max_player_move_cpu'], 0.25)
        self.assertEqual(summary['candidate']['max_player_game_cpu'], 0.25)
        self.assertEqual(summary['candidate']['total_player_nodes'], 20)


if __name__ == '__main__':
    unittest.main()
