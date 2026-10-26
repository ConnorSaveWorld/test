#!/usr/bin/env python3
"""Original Little-Go agent. Standard library only; Python 3.7 compatible.

Board indexing: 5 * row + column. The scoring rule counts surviving stones.
"""
import json
import math
import os
import time
from functools import lru_cache

EMPTY = (0,) * 25
PASS = -1
NEIGHBORS = tuple(tuple(q for q in (p - 5, p + 5, p - 1, p + 1)
                        if 0 <= q < 25 and
                        abs(p // 5 - q // 5) + abs(p % 5 - q % 5) == 1)
                  for p in range(25))
WIN_VALUE = 10000.0


def group(board, point):
    """Return all connected stones and distinct adjacent empty points."""
    color = board[point]
    if not color:
        return frozenset(), frozenset()
    members = {point}
    liberties = set()
    pending = [point]
    while pending:
        p = pending.pop()
        for q in NEIGHBORS[p]:
            if board[q] == 0:
                liberties.add(q)
            elif board[q] == color and q not in members:
                members.add(q)
                pending.append(q)
    return frozenset(members), frozenset(liberties)


@lru_cache(maxsize=8192)
def analyze(board):
    seen = set()
    groups = []
    for p, color in enumerate(board):
        if color and p not in seen:
            members, liberties = group(board, p)
            seen.update(members)
            groups.append((color, members, liberties))
    return tuple(groups)


def play_move(previous, board, color, point):
    """Apply captures before suicide and compare the final board for KO."""
    if point == PASS:
        return board
    if not 0 <= point < 25 or board[point]:
        return None
    updated = list(board)
    updated[point] = color
    dead = set()
    visited = set()
    for q in NEIGHBORS[point]:
        if updated[q] == 3 - color and q not in visited:
            members, liberties = group(updated, q)
            visited.update(members)
            if not liberties:
                dead.update(members)
    for q in dead:
        updated[q] = 0
    if not group(updated, point)[1]:
        return None
    result = tuple(updated)
    if result == previous:
        return None
    return result


def legal_moves(previous, board, color):
    actions = []
    for p in range(25):
        if board[p] == 0:
            updated = play_move(previous, board, color, p)
            if updated is not None:
                actions.append((p, updated))
    actions.append((PASS, board))
    return actions


def score_black(board):
    return board.count(1) - board.count(2) - 2.5


def terminal_value(board, root_color):
    margin = score_black(board) * (1 if root_color == 1 else -1)
    return (WIN_VALUE if margin > 0 else -WIN_VALUE) + margin


def features(board):
    """Black-oriented features. Threatened stone count is per group."""
    liberty_balance = 0.0
    atari_balance = 0.0
    weak_balance = 0.0
    influence = 0.0
    for color, stones, liberties in analyze(board):
        sign = 1 if color == 1 else -1
        size = len(stones)
        count = len(liberties)
        liberty_balance += sign * min(count, 6)
        if count <= 1:
            atari_balance += sign * size
        elif count == 2:
            weak_balance += sign * size
    # Access to empty space predicts growth/safety; it is not territory score.
    for p, stone in enumerate(board):
        if stone == 0:
            black = sum(board[q] == 1 for q in NEIGHBORS[p])
            white = sum(board[q] == 2 for q in NEIGHBORS[p])
            if black and not white:
                influence += 1
            elif white and not black:
                influence -= 1
    return (score_black(board), liberty_balance, atari_balance,
            weak_balance, influence)


def evaluate(board, root_color):
    f = features(board)
    value = (8.0 * f[0] + 0.6 * f[1] - 6.0 * f[2]
             - 1.2 * f[3] + 0.45 * f[4])
    return value if root_color == 1 else -value


def ordered_moves(previous, board, color, preferred=None, last_pass=False,
                  use_ordering=True):
    actions = legal_moves(previous, board, color)
    if not use_ordering:
        return actions
    own_atari = [stones for side, stones, libs in analyze(board)
                 if side == color and len(libs) == 1]
    enemy_count = board.count(3 - color)
    def priority(item):
        point, updated = item
        if point == PASS:
            rank = 2000 if last_pass and terminal_value(board, color) > 0 else -1000
        else:
            captures = enemy_count - updated.count(3 - color)
            stones, libs = group(updated, point)
            rescued = sum(len(old) for old in own_atari
                          if old & stones and len(libs) > 1)
            # Central access is a modest tie-break hypothesis, not an opening book.
            access = len(NEIGHBORS[point])
            rank = (100 * captures + 40 * rescued + 2 * min(len(libs), 5)
                    + access - (30 * len(stones) if len(libs) == 1 else 0))
        return (point == preferred, rank)
    actions.sort(key=priority, reverse=True)
    return actions


class SearchTimeout(Exception):
    pass


def _tuple_search(previous, board, color, move_count, last_pass, cpu_seconds=3.0,
           wall_seconds=4.0, max_depth=None, use_table=True, use_ordering=True,
           max_nodes=None, evaluator=None):
    """Only complete root iterations replace the validated fallback."""
    started_cpu = time.process_time()
    started_wall = time.monotonic()
    cpu_deadline = started_cpu + max(0.0, cpu_seconds)
    wall_deadline = started_wall + max(0.0, wall_seconds)
    evaluator = evaluate if evaluator is None else evaluator
    stats = {'nodes': 0, 'cache_hits': 0, 'completed_depth': 0,
             'value': None, 'interrupted': False}
    table = {}
    fallback = ordered_moves(previous, board, color, last_pass=last_pass,
                             use_ordering=use_ordering)
    best_action = fallback[0][0]

    def check_time():
        if (time.process_time() >= cpu_deadline or time.monotonic() >= wall_deadline
                or (max_nodes is not None and stats['nodes'] >= max_nodes)):
            raise SearchTimeout()

    def value(prev, current, side, count, passed, depth, alpha, beta):
        stats['nodes'] += 1
        check_time()
        if count is not None and count >= 24:
            return terminal_value(current, side), PASS
        if depth == 0:
            return evaluator(current, side), PASS
        key = (prev, current, side, count, passed)
        entry = table.get(key) if use_table else None
        preferred = entry[3] if entry is not None else None
        original_alpha, original_beta = alpha, beta
        if entry is not None and entry[0] >= depth:
            stats['cache_hits'] += 1
            if entry[2] == 'exact':
                return entry[1], entry[3]
            if entry[2] == 'lower':
                alpha = max(alpha, entry[1])
            else:
                beta = min(beta, entry[1])
            if alpha >= beta:
                return entry[1], entry[3]
        actions = ordered_moves(prev, current, side, preferred, passed, use_ordering)
        best = -float('inf')
        best_point = actions[0][0]
        next_count = None if count is None else count + 1
        for point, updated in actions:
            check_time()
            if ((point == PASS and passed) or
                    (next_count is not None and next_count >= 24)):
                candidate = terminal_value(updated, side)
            else:
                child, _ = value(current, updated, 3 - side, next_count,
                                 point == PASS, depth - 1, -beta, -alpha)
                candidate = -child
            if candidate > best:
                best, best_point = candidate, point
            alpha = max(alpha, best)
            if alpha >= beta:
                break
        if use_table:
            flag = ('upper' if best <= original_alpha else
                    'lower' if best >= original_beta else 'exact')
            if len(table) < 60000 or key in table:
                table[key] = (depth, best, flag, best_point)
        return best, best_point

    limit = 24 - move_count if move_count is not None else 10
    if max_depth is not None:
        limit = min(limit, max_depth)
    for depth in range(1, max(1, limit) + 1):
        try:
            result, action = value(previous, board, color, move_count, last_pass,
                                   depth, -float('inf'), float('inf'))
        except SearchTimeout:
            stats['interrupted'] = True
            break
        best_action = action
        stats['completed_depth'] = depth
        stats['value'] = result
        # A complete horizon can be stopped; apparent shallow wins alone cannot.
        if move_count is not None and depth >= 24 - move_count:
            break
    stats['cpu_seconds'] = time.process_time() - started_cpu
    stats['wall_seconds'] = time.monotonic() - started_wall
    return best_action, stats


def read_input(path='input.txt'):
    with open(path, 'r') as stream:
        lines = stream.read().splitlines()
    if len(lines) != 11 or lines[0] not in ('1', '2'):
        raise ValueError('Expected color and ten five-digit board rows')
    if any(len(row) != 5 or any(c not in '012' for c in row) for row in lines[1:]):
        raise ValueError('Invalid board row')
    previous = tuple(int(c) for row in lines[1:6] for c in row)
    board = tuple(int(c) for row in lines[6:11] for c in row)
    return int(lines[0]), previous, board


def write_output(point, path='output.txt'):
    if point != PASS and not 0 <= point < 25:
        raise ValueError('Invalid action coordinate')
    text = 'PASS' if point == PASS else '{},{}'.format(point // 5, point % 5)
    with open(path, 'w', newline='\n') as stream:
        stream.write(text + '\n')


def load_history(color, previous, board, path='state.json'):
    unknown = {'move_count': None, 'cpu_used': 0.0}
    if previous == EMPTY:
        if color == 1 and board == EMPTY:
            return {'move_count': 0, 'cpu_used': 0.0}
        if color == 2 and board.count(1) <= 1 and board.count(2) == 0:
            return {'move_count': 1, 'cpu_used': 0.0}
    try:
        with open(path, 'r') as stream:
            data = json.load(stream)
        if not isinstance(data, dict) or data.get('version') != 1:
            return unknown
        count = data.get('move_count')
        cpu = data.get('cpu_used')
        if (data.get('color') != color or data.get('after') != list(previous)
                or type(cpu) not in (int, float) or not 0 <= cpu <= 90
                or not math.isfinite(cpu)):
            return unknown
        if count is None:
            return {'move_count': None, 'cpu_used': float(cpu)}
        if (type(count) is not int or not 0 <= count <= 21
                or count % 2 != color - 1):
            return unknown
        return {'move_count': count + 2, 'cpu_used': float(cpu)}
    except (OSError, ValueError, TypeError):
        return unknown


def turn_budget(cpu_used, move_count, color):
    """Prioritize White's opening replies within strict game and move caps."""
    remaining = max(0.0, 86.0 - cpu_used)
    turns_left = max(1, (25 - move_count) // 2) if move_count is not None else 12
    if move_count is not None and color == 2 and move_count <= 5:
        early_turns = sum(1 for offset in range(turns_left)
                          if move_count + 2 * offset <= 5)
        weight = 1.25
        weighted_turns = turns_left + (weight - 1.0) * early_turns
        return min(8.5, remaining * weight / weighted_turns)
    return min(7.0, remaining / turns_left)


def main():
    process_started = time.process_time()
    color, previous, board = read_input()
    history = load_history(color, previous, board)
    count = history['move_count']
    # Keep game CPU targets below the stage's 90-second per-game average.
    budget = turn_budget(history['cpu_used'], count, color)
    # Development override can reduce time, but cannot increase deployment caps.
    try:
        override = float(os.environ.get('LITTLE_GO_CPU_SECONDS', budget))
        if math.isfinite(override):
            budget = min(budget, max(0.0, override))
    except ValueError:
        pass
    last_pass = previous == board and (board != EMPTY or color == 2)
    action = None
    stats = {}
    if color == 2:
        try:
            action, stats = white_policy(previous, board, count, last_pass,
                                         budget, process_started)
        except Exception:
            action, stats = None, {}
    if action is None:
        spent = time.process_time() - process_started
        remaining = max(0.5, budget - spent)
        action, search_stats = search(previous, board, color, count, last_pass,
                                      cpu_seconds=remaining,
                                      wall_seconds=min(9.5 - spent, remaining + 1.0))
        search_stats.update(stats)
        stats = search_stats
    write_output(action)
    after = play_move(previous, board, color, action)
    record = {'version': 1, 'color': color, 'move_count': count,
              'after': list(after), 'cpu_used': history['cpu_used'] +
              time.process_time() - process_started, 'last_stats': stats}
    try:
        with open('state.json.tmp', 'w') as stream:
            json.dump(record, stream, separators=(',', ':'))
        os.replace('state.json.tmp', 'state.json')
    except OSError:
        # History is optional. Legality depends only on the supplied boards.
        pass



# Original bitboard search core; concatenated with our baseline interface.
FULL = (1 << 25) - 1
LEFT_EDGE = sum(1 << p for p in range(0, 25, 5))
RIGHT_EDGE = LEFT_EDGE << 4
ADJ_BITS = tuple(sum(1 << q for q in NEIGHBORS[p]) for p in range(25))


def _symmetry_map(turns, reflected):
    points = []
    for point in range(25):
        row, column = divmod(point, 5)
        if reflected:
            column = 4 - column
        for _ in range(turns):
            row, column = column, 4 - row
        points.append(5 * row + column)
    return tuple(points)


_SYMMETRY_MAPS = tuple(_symmetry_map(turns, reflected)
                       for reflected in (False, True) for turns in range(4))


def _transform_bits(bits, mapping):
    transformed = 0
    while bits:
        bit = bits & -bits
        bits ^= bit
        transformed |= 1 << mapping[bit.bit_length() - 1]
    return transformed


def _root_symmetries(previous_black, previous_white, black, white):
    boards = (black, white, previous_black, previous_white)
    return tuple(mapping for mapping in _SYMMETRY_MAPS
                 if all(_transform_bits(bits, mapping) == bits for bits in boards))


def _root_representatives(actions, symmetries):
    if len(symmetries) == 1:
        return actions
    representatives = []
    covered = 0
    for action in actions:
        point = action[0]
        if point == PASS:
            representatives.append(action)
        elif not covered & (1 << point):
            representatives.append(action)
            for mapping in symmetries:
                covered |= 1 << mapping[point]
    return representatives


def _count(bits):
    return bin(bits).count('1')


def _adj(bits):
    return (((bits << 5) | ((bits & ~RIGHT_EDGE) << 1)) & FULL
            | (bits >> 5) | ((bits & ~LEFT_EDGE) >> 1))


def _pack(board):
    black = white = 0
    for p, color in enumerate(board):
        if color == 1:
            black |= 1 << p
        elif color == 2:
            white |= 1 << p
    return black, white


def _unpack(black, white):
    return tuple(1 if black & (1 << p) else 2 if white & (1 << p) else 0
                 for p in range(25))


@lru_cache(maxsize=16384)
def _bit_groups(black, white):
    empty = FULL & ~(black | white)
    groups = []
    for color, mask in ((1, black), (2, white)):
        remaining = mask
        while remaining:
            stones = remaining & -remaining
            frontier = stones
            while frontier:
                frontier = _adj(frontier) & mask & ~stones
                stones |= frontier
            remaining &= ~stones
            groups.append((color, stones, _adj(stones) & empty))
    return tuple(groups)


def _bit_moves(pb, pw, black, white, color):
    actions = [(p, b, w) for p, b, w, _ in _bit_candidates(black, white, color)
               if b != pb or w != pw]
    actions.sort(key=lambda item: item[0])
    actions.append((PASS, black, white))
    return actions


@lru_cache(maxsize=4096)
def _bit_candidates(black, white, color):
    """Cache history-independent transitions; filter KO in the caller."""
    own, enemy = (black, white) if color == 1 else (white, black)
    contacts = [0] * 25
    captures = [0] * 25
    rescued = [0] * 25
    for side, stones, libs in _bit_groups(black, white):
        if side == color:
            remaining = libs
            while remaining:
                bit = remaining & -remaining
                remaining ^= bit
                p = bit.bit_length() - 1
                contacts[p] |= stones
                if libs == bit:
                    rescued[p] += _count(stones)
        elif libs and not (libs & (libs - 1)):
            captures[libs.bit_length() - 1] |= stones
    actions = []
    remaining = FULL & ~(black | white)
    while remaining:
        bit = remaining & -remaining
        remaining ^= bit
        p = bit.bit_length() - 1
        members = contacts[p] | bit
        dead = captures[p]
        new_own, new_enemy = own | bit, enemy & ~dead
        liberties = _adj(members) & FULL & ~(new_own | new_enemy)
        if not liberties:
            continue
        n = _count(liberties)
        priority = (100 * _count(dead) + 40 * (rescued[p] if n > 1 else 0)
                    + 2 * min(n, 5) + len(NEIGHBORS[p])
                    - (30 * _count(members) if n == 1 else 0))
        b, w = (new_own, new_enemy) if color == 1 else (new_enemy, new_own)
        actions.append((p, b, w, priority))
    actions.sort(key=lambda item: item[3], reverse=True)
    return tuple(actions)


def _bit_score(black, white):
    return _count(black) - _count(white) - 2.5


def _bit_terminal(black, white, color):
    margin = _bit_score(black, white) * (1 if color == 1 else -1)
    return (WIN_VALUE if margin > 0 else -WIN_VALUE) + margin


@lru_cache(maxsize=32768)
def _bit_static(black, white):
    liberties = atari = weak = 0
    for color, stones, libs in _bit_groups(black, white):
        sign = 1 if color == 1 else -1
        n = _count(libs)
        liberties += sign * min(n, 6)
        if n <= 1:
            atari += sign * _count(stones)
        elif n == 2:
            weak += sign * _count(stones)
    empty = FULL & ~(black | white)
    ba, wa = _adj(black), _adj(white)
    influence = _count(ba & empty & ~wa) - _count(wa & empty & ~ba)
    group_balance=sum(1 if side==1 else -1 for side,_,_ in _bit_groups(black,white))
    # The same weights in exact integer twentieths avoid false floating ties.
    return (160 * (_count(black) - _count(white)) - 400 + 12 * liberties
            - 120 * atari - 24 * weak + 9 * influence - 40 * group_balance) / 20.0


def _bit_evaluate(black, white, color):
    value = _bit_static(black, white)
    return value if color == 1 else -value


def _bit_order(pb, pw, black, white, color, preferred=None, passed=False,
               ordering=True):
    if not ordering:
        return _bit_moves(pb, pw, black, white, color)
    actions = [(p, b, w) for p, b, w, _ in _bit_candidates(black, white, color)
               if b != pb or w != pw]
    passing = (PASS, black, white)
    if passed and _bit_terminal(black, white, color) > 0:
        actions.insert(0, passing)
    else:
        actions.append(passing)
    if preferred is not None:
        for index, item in enumerate(actions):
            if item[0] == preferred:
                actions.insert(0, actions.pop(index))
                break
    return actions


def search(previous, board, color, move_count, last_pass, cpu_seconds=3.0,
           wall_seconds=4.0, max_depth=None, use_table=True, use_ordering=True,
           max_nodes=None, evaluator=None, use_pvs=True, use_symmetry=None):
    if evaluator is not None:
        return _tuple_search(previous, board, color, move_count, last_pass,
            cpu_seconds, wall_seconds, max_depth, use_table, use_ordering, max_nodes, evaluator)
    if use_symmetry is None:
        # Keep the proven Black search allocation; optimize White's search tree.
        use_symmetry = color == 2
    started_cpu = time.process_time()
    started_wall = time.monotonic()
    cpu_deadline = started_cpu + max(0.0, cpu_seconds)
    wall_deadline = started_wall + max(0.0, wall_seconds)
    stats = {'nodes': 0, 'cache_hits': 0, 'completed_depth': 0,
             'value': None, 'interrupted': False, 'engine': 'cached-bitboards',
             'pvs': use_pvs}
    pb, pw = _pack(previous)
    black, white = _pack(board)
    # Local targeted hypothesis from the recorded Championship lines:
    # Trial: use reply17 after Black-7 while preserving the proved Black-11
    # counterwall. All other states search.
    # Fixed-depth solver calls retain their requested minimax semantics.
    if color == 2 and move_count == 3 and max_depth is None and not last_pass:
        for mapping in _SYMMETRY_MAPS:
            center, own, approach = (1 << mapping[p] for p in (12, 6, 7))
            if (pb == center and pw == own and black == (center | approach)
                    and white == own):
                opening_action = mapping[17]
                if any(point == opening_action for point, _, _ in
                       _bit_moves(pb, pw, black, white, color)):
                    stats.update(opening_policy=True,
                                 opening_origin='local-second-reply-B7-17-v1',
                                 cpu_seconds=time.process_time() - started_cpu,
                                 wall_seconds=time.monotonic() - started_wall)
                    return opening_action, stats
    # In the pressure-profile loss after White's B7 reply17, Black answered
    # with18, White played16, and Black approached at11. Test a counterwall13
    # only for this exact seven-ply history; other positions still use search.
    if color == 2 and move_count == 7 and max_depth is None and not last_pass:
        for mapping in _SYMMETRY_MAPS:
            expected_previous = EMPTY
            expected_board = EMPTY
            for index, point in enumerate((12, 6, 7, 17, 18, 16, 11)):
                updated = play_move(expected_previous, expected_board,
                                    1 + index % 2, mapping[point])
                if updated is None:
                    break
                expected_previous, expected_board = expected_board, updated
            if (tuple(previous) == expected_previous
                    and tuple(board) == expected_board):
                followup = mapping[13]
                if any(point == followup for point, _, _ in
                       _bit_moves(pb, pw, black, white, color)):
                    stats.update(opening_policy=True,
                                 opening_origin='local-B7-17-B18-W16-B11-counterwall13-v1',
                                 cpu_seconds=time.process_time() - started_cpu,
                                 wall_seconds=time.monotonic() - started_wall)
                    return followup, stats
    table = {}
    root_actions = _bit_order(pb, pw, black, white, color,
                              passed=last_pass, ordering=use_ordering)
    symmetries = (_root_symmetries(pb, pw, black, white) if use_symmetry
                  else (_SYMMETRY_MAPS[0],))
    representatives = _root_representatives(root_actions, symmetries)
    stats['root_actions'] = len(root_actions)
    stats['root_orbits'] = len(representatives)
    stats['root_symmetries'] = len(symmetries)
    stats['symmetry_pruned_actions'] = 0
    best_action = representatives[0][0]

    def check(force=False):
        if max_nodes is not None and stats['nodes'] >= max_nodes:
            raise SearchTimeout()
        if force or stats['nodes'] % 128 == 0:
            if time.process_time() >= cpu_deadline or time.monotonic() >= wall_deadline:
                raise SearchTimeout()

    def value(old_b, old_w, b, w, side, count, passed, depth, alpha, beta,
              active_symmetries=None):
        stats['nodes'] += 1
        check()
        if count is not None and count >= 24:
            return _bit_terminal(b, w, side), PASS
        if depth == 0:
            return _bit_evaluate(b, w, side), PASS
        key = (old_b, old_w, b, w, side, count, passed)
        entry = table.get(key) if use_table else None
        preferred = entry[3] if entry is not None else None
        original_alpha, original_beta = alpha, beta
        if entry is not None and entry[0] >= depth:
            stats['cache_hits'] += 1
            if entry[2] == 'exact':
                return entry[1], entry[3]
            if entry[2] == 'lower':
                alpha = max(alpha, entry[1])
            else:
                beta = min(beta, entry[1])
            if alpha >= beta:
                return entry[1], entry[3]
        actions = _bit_order(old_b, old_w, b, w, side, preferred, passed, use_ordering)
        if active_symmetries is not None:
            action_count = len(actions)
            actions = _root_representatives(actions, active_symmetries)
            stats['symmetry_pruned_actions'] += action_count - len(actions)
        best, point = -float('inf'), actions[0][0]
        next_count = None if count is None else count + 1
        for index, (p, nb, nw) in enumerate(actions):
            if ((p == PASS and passed) or
                    (next_count is not None and next_count >= 24)):
                candidate = _bit_terminal(nb, nw, side)
            else:
                child_symmetries = None
                if active_symmetries is not None:
                    if p == PASS:
                        child_symmetries = active_symmetries
                    else:
                        subgroup = tuple(mapping for mapping in active_symmetries
                                         if mapping[p] == p)
                        if len(subgroup) > 1:
                            child_symmetries = subgroup
                if use_pvs and index > 0 and alpha != -float('inf'):
                    child, _ = value(b, w, nb, nw, 3 - side, next_count,
                                     p == PASS, depth - 1, -alpha - 0.01, -alpha,
                                     child_symmetries)
                    if alpha < -child < beta:
                        child, _ = value(b, w, nb, nw, 3 - side, next_count,
                                         p == PASS, depth - 1, -beta, -alpha,
                                         child_symmetries)
                else:
                    child, _ = value(b, w, nb, nw, 3 - side, next_count,
                                     p == PASS, depth - 1, -beta, -alpha,
                                     child_symmetries)
                candidate = -child
            if candidate > best:
                best, point = candidate, p
            alpha = max(alpha, best)
            if alpha >= beta:
                break
        if use_table and (len(table) < 100000 or key in table):
            flag = ('upper' if best <= original_alpha else
                    'lower' if best >= original_beta else 'exact')
            table[key] = (depth, best, flag, point)
        return best, point

    limit = 24 - move_count if move_count is not None else 10
    if max_depth is not None:
        limit = min(limit, max_depth)
    for depth in range(1, max(1, limit) + 1):
        try:
            check(force=True)
            result, action = value(pb, pw, black, white, color, move_count,
                                   last_pass, depth, -float('inf'), float('inf'),
                                   symmetries if len(symmetries) > 1 else None)
            check(force=True)
        except SearchTimeout:
            stats['interrupted'] = True
            break
        best_action = action
        stats['completed_depth'], stats['value'] = depth, result
        if move_count is not None and depth >= 24 - move_count:
            break
    stats['cpu_seconds'] = time.process_time() - started_cpu
    stats['wall_seconds'] = time.monotonic() - started_wall
    return best_action, stats


def evaluate(board,color):
    return _bit_evaluate(*_pack(board),color)


# Exact win/loss search (original, see header).
_XFULL = (1 << 25) - 1
_XCOL0 = sum(1 << p for p in range(0, 25, 5))
_XCOL4 = _XCOL0 << 4
_XBIT = tuple(1 << p for p in range(25))
_XNEI = tuple((((1 << p) >> 5) | (((1 << p) << 5) & _XFULL) |
               (((1 << p) & ~_XCOL0) >> 1) | (((1 << p) & ~_XCOL4) << 1)) & _XFULL
              for p in range(25))
_XORDER = (12, 7, 11, 13, 17, 6, 8, 16, 18, 2, 10, 14, 22, 1, 3, 5, 9, 15,
           19, 21, 23, 0, 4, 20, 24)


class ExactTimeout(Exception):
    pass


def _xdilate(g):
    return (g | (g >> 5) | ((g << 5) & _XFULL) | ((g & ~_XCOL0) >> 1)
            | ((g & ~_XCOL4) << 1)) & _XFULL


def _xflood(stones, seed):
    g = seed
    while True:
        n = _xdilate(g) & stones
        if n == g:
            return g
        g = n


def _xplay(own, opp, p, ko):
    """Return (own, opp, ko) after own plays p, or None if illegal."""
    bit = _XBIT[p]
    if (own | opp) & bit:
        return None
    own |= bit
    empty = ~(own | opp) & _XFULL
    cap = 0
    nb = _XNEI[p] & opp
    while nb:
        q = nb & -nb
        g = _xflood(opp, q)
        nb &= ~g
        if not (_xdilate(g) & empty):
            cap |= g
    if cap:
        opp &= ~cap
        if not cap & (cap - 1):
            if ko and p == (ko - 1) % 25 and cap == _XBIT[(ko - 1) // 25]:
                return None
            return own, opp, 1 + 25 * p + cap.bit_length() - 1
        return own, opp, 0
    if not (_xdilate(_xflood(own, bit)) & empty):
        return None
    return own, opp, 0


def exact_ko(previous_black, previous_white, black, white, color):
    """Ko code for the side to move from the host's previous/current boards."""
    own_prev = previous_black if color == 1 else previous_white
    own_now = black if color == 1 else white
    opp_prev = previous_white if color == 1 else previous_black
    opp_now = white if color == 1 else black
    placed = opp_now & ~opp_prev
    died = own_prev & ~own_now
    if placed and not placed & (placed - 1) and died and not died & (died - 1):
        return 1 + 25 * (placed.bit_length() - 1) + died.bit_length() - 1
    return 0


def exact_solve(black, white, color, ko, count, last_pass, seconds,
                table=None, node_limit=None):
    """Return ('win', move) / ('loss', None) / ('unknown', None) for color."""
    deadline = time.process_time() + seconds
    if table is None:
        table = {}
    killers = [[-2, -2] for _ in range(27)]
    state = [0]
    popcount = bin

    def black_wins(b, w):
        return popcount(b).count('1') - popcount(w).count('1') >= 3

    def solve(own, opp, ko, count, lp):
        state[0] += 1
        if not state[0] & 1023:
            if time.process_time() > deadline or (
                    node_limit is not None and state[0] > node_limit):
                raise ExactTimeout()
        key = (own, opp, ko, count, lp)
        hit = table.get(key)
        if hit is not None:
            return hit
        side_black = not count & 1
        last = count + 1 >= 24
        kl = killers[count]
        result = False
        tried = 0
        for m in (kl[0], kl[1]) + _XORDER:
            if m < 0 or (tried >> m) & 1:
                continue
            tried |= 1 << m
            nxt = _xplay(own, opp, m, ko)
            if nxt is None:
                continue
            if last:
                b, w = (nxt[0], nxt[1]) if side_black else (nxt[1], nxt[0])
                win = black_wins(b, w) == side_black
            else:
                win = not solve(nxt[1], nxt[0], nxt[2], count + 1, 0)
            if win:
                if kl[0] != m:
                    kl[1] = kl[0]
                    kl[0] = m
                result = True
                break
        if not result:
            if lp or last:
                b, w = (own, opp) if side_black else (opp, own)
                result = black_wins(b, w) == side_black
            else:
                result = not solve(opp, own, 0, count + 1, 1)
        table[key] = result
        return result

    own, opp = (black, white) if color == 1 else (white, black)
    if count is None or count >= 24 or (count & 1) != color - 1:
        return 'unknown', None
    last = count + 1 >= 24
    try:
        for m in _XORDER + (-1,):
            if m == -1:
                if last_pass or last:
                    b, w = (own, opp) if color == 1 else (opp, own)
                    win = black_wins(b, w) == (color == 1)
                else:
                    win = not solve(opp, own, 0, count + 1, 1)
            else:
                nxt = _xplay(own, opp, m, ko)
                if nxt is None:
                    continue
                if last:
                    b, w = (nxt[0], nxt[1]) if color == 1 else (nxt[1], nxt[0])
                    win = black_wins(b, w) == (color == 1)
                else:
                    win = not solve(nxt[1], nxt[0], nxt[2], count + 1, 0)
            if win:
                return 'win', m
    except ExactTimeout:
        return 'unknown', None
    return 'loss', None


# ---------------------------------------------------------------------------
# White trap book and exact endgame (added for Stage 2, White only).
# The book was generated offline by our own exact win/loss solver: on lines
# the Championship opponent has actually played, White chooses moves that
# leave Black a single winning reply, and after any losing Black reply it
# stores a proven winning White move.  Positions are matched up to the eight
# board symmetries, including the ko point.
_BOOK = None


def _load_book():
    global _BOOK
    if _BOOK is None:
        import base64
        raw = base64.b85decode(''.join(_BOOK_DATA.split()))
        table = {}
        for i in range(0, len(raw), 10):
            table[int.from_bytes(raw[i:i + 9], 'big')] = raw[i + 9] - 1
        _BOOK = table
    return _BOOK


def book_move(black, white, ko, count, last_pass):
    """Return the book move for White in this exact state, or None."""
    if count is None:
        return None
    book = _load_book()
    for mapping in _SYMMETRY_MAPS:
        tko = 0
        if ko:
            p, q = divmod(ko - 1, 25)
            tko = 1 + 25 * mapping[p] + mapping[q]
        key = (_transform_bits(black, mapping) | (_transform_bits(white, mapping) << 25)
               | (count << 50) | ((1 if last_pass else 0) << 55) | (tko << 56))
        move = book.get(key)
        if move is not None:
            if move == PASS:
                return PASS
            return mapping.index(move)
    return None


def white_policy(previous, board, count, last_pass, budget, started):
    """Book move, else proven endgame win; None means use the normal search."""
    if count is None or count % 2 != 1:
        return None, {}
    pb, pw = _pack(previous)
    black, white = _pack(board)
    ko = exact_ko(pb, pw, black, white, 2)
    legal = set(p for p, _, _ in _bit_moves(pb, pw, black, white, 2))
    try:
        move = book_move(black, white, ko, count, last_pass)
    except Exception:
        move = None
    if move is not None and move in legal:
        return move, {'policy': 'book'}
    if count >= EXACT_FROM:
        limit = min(EXACT_SECONDS, max(budget - 0.5, 0.5))
        status, move = exact_solve(black, white, 2, ko, count, last_pass, limit)
        if status == 'win' and move in legal:
            return move, {'policy': 'exact-win'}
        return None, {'exact': status}
    return None, {}


EXACT_FROM = 13
EXACT_SECONDS = 6.0

_BOOK_DATA = """
000C40000G00#g73;+Ot01$u-000yK0e}D*01f~EoB%<902u%h005)}L4W`m01*HHtOh}V02u%i000~SL4W`m0TKWJBm_Z#02u)h
005)}L4W`m0TBQIECxY<02u)j00688L4W`m0TTcK8~{Op02u-j001NeL4W`m0ucZJqy#~L02u-i001loL4W`m0uulLtOh}V02u-c
001NeL4W`m0}%iKECxY<02u=l00688L4W`m0}B8E8~{Op02u@k005){L4W`m1QP%NBm_Z#02u@k001loL4W`m1PcHFtOh}V02u@e
0068CL4W`m1QGxMBmzNz02u`n001NeL4W`m1rY!MECxY<02u`f00688L4W`m1q%QGEDAw@02u`m0068CL4W`m1s4DSBmzNz02u}o
001NeL4W`m1`z-NECxY<02u}g00688L4W`m1`_}PEDAw@02u}n0068CL4W`m1`+@OECxY<02v1h001lsL4W`m2ND1P8~{Op02v4q
001NeL4W`m2oV4PECxY<02v4q00688L4W`m2nzrJqzFNP02v4p0068EL4W`m2oV4PBm_Z#02v7p001loL4W`m2@?PStOh}V02v7j
001NiL4W`m2@(JREDS+_02v7p0068EL4W`m2@wDQBm_Z#02vAq001loL4W`m3JU-LtOh}V02vAs001NiL4W`m3K0MRtO`MZ02vAr
001luL4W`m3K0MRECxY<02vDl001lsL4W`m3labTEDS+_02vDr001NcL4W`m3=;qVECxY<02vGm001NiL4W`m3=seTEDAw@02vGt
0068CL4W`m3=#kUEDS+_02vGs001loL4W`m4GRDOEDAw@02vJu001luL4W`m4G#bSECxY<02vMw001lsL4W`m4iW$W8~{Op02vSw
005){L4W`m5EB3ZBm_Z#02vSw001loL4W`m5EB3ZtOh}V02vSq0068CL4W`m5DNeRBmzNz02vVz001NeL4W`m5fK0YECxY<02vVz
00688L4W`m5eonSEDAw@02vVr0068CL4W`m5eonSBmzNz02vY!001NeL4W`m5)l9ZECxY<02vYs00688L4W`m5)%LbEDAw@02vYs
0068CL4W`m5)uFaECxY<02vbt001lsL4W`m6A}ObBmzNz02veu00684Nq_(u6cPXcECxY<02veu001lsL4W`m6c+#htO`MZ02ve)
001lkNq_(u6%qgdECxY<02vhv001lsL4W`m6$=0WECNY@02vk%001loL4W`m783vfEDAw@02vk%001NcL4W`m7!v>hECxY<02vqy
001NiL4W`m7!m*gEDAw@02vqy0068CL4W`m7z+RZEDS+_02vq&0068EL4W`m7!d#fECxY<02vtz001lsL4W`m84CaaEDS+_02vt(
001loL4W`m8WR8jEDAw@02vw*001luL4W`m8W8{hECNY@02v$-001lsL4W`m91;KkEDS+_02v$+000~SL4W`mAQJ!pBm_Z#02v?=
001loL4W`mAQJ!ptOh}V02v?)005*2L4W`mAQ1ontO`MZ02v?>001NeL4W`mArSxoECxY<02v_@00688L4W`mAqxNiBnUx(02v_!
0068AL686$Arb%pEDAw@02v_?001NeL4W`mA`t)pECxY<02v|+00688L4W`mA`1WjBnUx(02v|?001lsL4W`mA_@QitO`MZ02v|@
001loL4W`mBMSfkEC@l602w0+001lsL4W`mBN6}rBmzNz02w3`001loL4W`mBntolBnUx(02w3^001lsL4W`mBoY7stO`MZ02w3_
001loL4W`mB?|xmEC@l602w6;001lsL4W`mB@zGtECxUT02w9|001lsKmY(4CK3PuECxY<02wF?001NiL4W`mC=mbvEDAw@02wF>
0068CL4W`mC=vhwEDS+_02wF|001loL4W`mDGLAqEDAw@02wI~001luL4W`mDG>kwECxY<02wM1001lsL4W`mDiQzyEDAw@02wS2
001NcL4W`mFcSa(ECxY<02wd~001NiL4W`mFcAO%EDAw@02wd~0068CL4W`mFcJU&ECxY<02wh0001lqL686$F%kd(EDAw@02wh0
001loL4W`mG7A6zEDAw@02wk8001lkNq_(uG!g&+EDAw@02wqF001lsL4W`mI1&H=EDS+_02w$D000~SL4W`o01*HHqyj;J02=@j
001NeL4W`o01*HHECxY<02=@k00688L4W`o01E&BBmzNz02=`k001NeL4W`o0TBQIECxY<02=`l00688L4W`o0Sf>CBmzNz02=}m
001NeL4W`o0ucZJECxY<02=}e00688L4W`o0uulLtO`MZ02=}l001loL4W`o0}B8EEDAw@02>1m001NcL4W`o1QP%NtO7}Z02>4n
001loL4W`o1PcHFEDAw@02>4n0068CL4W`o1QGxMECNY@02>7o001loL4W`o1q%QGEDAw@02>7o001lkNq_(w1`+@OECxY<02>Aq
001lsL4W`o1`+@OBmzNz02>Gs001loL4W`o2nzrJBnUx(02>Gr0068CL4W`o2nzrJEDS+_02>Gq0068EL4W`o2oV4PECxY<02>Jl
001lsL4W`o2@3!KEDS+_02>Jr001loL4W`o3KIYTEDAw@02>Mt001luL4W`o3K0MRECNY@02>Sv001lsL4W`o3=#kUEDS+_02>So
001NcL4W`o5EB3ZECxY<02>es00688L4W`o5EB3ZEDAw@02>es0068CL4W`o5ElRdECxY<02>ht001lsL4W`o5f=aeECxY<02>k$
001lsL4W`o5)uFaECNY@02>q%001loL4W`o6bk?VEDAw@02>q+001loL4W`o7!v>hEDAw@02>$!001luL4W`o7!d#fBmzNz02?3^
001loL4W`oAQJ!pBnUx(02?3?001lsL4W`oAQAuotO`MZ02?3@001loL4W`oAqxNiEC@l602?6^001lsL4W`oArb%pECxY<02?9;
001lsL4W`oA`$=qECNY@02?F{001lsL4W`oBoY7sEDAw@02?S0001luL4W`oC=mbvECxY<02?q9001lsL4W`oFc$y-8~{Op0384m
001NeL4W`q01*HHECxY<0384m00688L4W`q022TJqzFNP0384l0068EL4W`q01*HHBm_Z#0387l001loL4W`q0TTcKtOh}V0387n
001NiL4W`q0TKWJEDS+_0387l0068EL4W`q0TBQIBm_Z#038Ao001loL4W`q0uulLBnUx(038An001luL4W`q0ucZJtPDYb038Am
001loL4W`q0}}uMEDS+_038Dn001NcL4W`q1QP%NECxY<038Gm001NiL4W`q1QGxMEDAw@038Gu0068CL4W`q1QGxMEDS+_038Go
0068EL4W`q1Q7rLECxY<038Jn001lsL4W`q1rh)NEDS+_038Jp001loL4W`q1`_}PEDAw@038Mr001luL4W`q1_}TGECxY<038Su
001NiL4W`q2oeAQEDS+_038Ss0068EL4W`q2oV4PECxY<038Vv001luL4W`q2@e1OECxY<038Yw001lsL4W`q3K9SSEDS+_038Yn
001lsL4W`q3=#kUEDS+_038eu001NcL4W`q5EB3ZECxY<038q$001NiL4W`q5E1|YEDAw@038q)0068CL4W`q5E1|YEDS+_038q!
0068EL4W`q5D@?XECxY<038t%001lsL4W`q5fT6ZEDS+_038t#001loL4W`q5)%LbEDAw@038w%001luL4W`q5)l9ZEDAw@038$(
001luL4W`q6cGRbEDAw@038?-001luL4W`q7!d#fECxY<039F`001NiL4W`qAQAuoEDAw@039F-001luL4W`qAQ1ontPDYb039F^
001loL4W`qArk-qEC@l6039I;001luL4W`qArSxoECxY<039L|001lsL4W`qA`$=qEDS+_039L`001lsL4W`qBoY7sEDS+_039R|
001lsL4W`qC>H<#EDS+_039e1001lsL4W`qFcJU&EDS+_039$9001NcL4W`s022TJECxY<03QGo001NiL4W`s01^NItO`MZ03QGn
001luL4W`s01*HHtPDYb03QGm001loL4W`s0TTcKEDAw@03QJo001luL4W`s0TBQIECxY<03QMq001lsL4W`s0ulfKEDS+_03QMo
001lsL4W`s1QGxMEDS+_03QSq001lsL4W`s2oeAQEDS+_03Qeu001loL4W`s5EB3ZEDAw@03Q$+001luL4W`s5D@?XEDAw@03RR{
001luL4W`sAQ1on8~_sl05m`m005){6951-KobA}Bm@%x05m`o0068C6951-KoS4|EDRF>05m`m0068E6951-Ko0-_Bmxrv05m}p
001Ni6951-K@k7|ED93<05m}t0068C6951-K@tD}EDRF>05m}l001Nc6951-LJ|M~ECv$*05n1q006886951-LK6T0ED93<05n1p
0068C6951-LKgr4EDRF>05n1m001lo6951-LlXc1ED93<05n4q001Nc6951-L=yl2BnT4#05n7q001ls6951-L>B-6tO^qV05n7r
001lu6951-L=OM}EC`bT05nAr001ls6951-MG^o2ECv$*05nDq001ls6951-MiKx3Bmxrv05nJw001Ne6951-ND%-4tOgSR05nJw
001ls6951-ND=@5tO^qV05nJv001lu6951-NDlx2ECv$*05nMx001ls6951-NfH16EDRF>05nMv001lo6951-N(KM`ED93<05nPx
001lu6951-N)Z46ECv$*05nVn001ls6951-OcDS9EDRF>05nVy001Nc6951-P!j+EBm@%x05nh&001ls6951-P!a$DtO^qV05nh%
001lu6951-P!RwCtPB$X05nh!001ls6951-Q4#<EEDRF>05nk#001lo6951-QWF3GED93<05nn;001lu6951-QV#$CED93<05nt*
001lu6951-R1W|EECv$*05n(=001ls6951-SP}pLEDRF>05n(+001Nc6951-U=siUBnT4#05o6`001ls6951-U=jcTtO^qV05o6{
001lu6951-U=IKQED93<05o9|001lu6951-VG#fTECv$*05oC~001ls6951-ViEuVEC`bT05oJ0001ls6951-WD)=XECv$*05oV5
001ls6951-Xc7PbEDRF>05oV3001ls6951-a1sCjEDRF>05ot9001Nc6951<KobA}ED93<05(7p0068C6951<KoS4|EDRF>05(7m
001ls6951<K@tD}ECv$*05(Ds001ls6951<LKgr4ED93<05(Jt001lo6951<ND}}6ED93<05(Vx001lu6951<ND%-4ECLe*05(t(
001ls6951<P!a$DEDRF>05(t$001ls6951<U=jcTECxY<09^nR001NiL4W{V01^NIEDAw@09^nV0068CL4W{V01^NIEDS+_09^nP
0068EL4W{V01*HHECxY<09^qS001lsL4W{V0TKWJEDS+_09^qQ001lsL4W{V0ulfKEDS+_09^tK001lsL4W{V1Ofm6EDS+_09^zZ
001lsL4W{V2oeAQEDS+_09^<X001lsL4W{V5E1|YEDS+_09_Cf001lsL4W{VAQAuoEDS+_09_yv001lsL4W{X01^NIEDS+_0ABzR
000~SL4W|6022TJBm_Z#0GR+0001loL4W|6022TJtOh}V0GR*_005*2L4W|601^NItPDYb0GR+0001NeL4W|60TBQIECxY<0GR<3
00688L4W|60Sf>CBnUx(0GR<2001luL4W|60TBQItPDYb0GR;-001NeL4W|60ucZJECxY<0GR>{00688L4W|60uulLBnUx(0GR?3
001luL4W|60ucZJtPDYb0GR?2001loL4W|60}B8EEDS+_0GR_3001NcL4W|61QP%NECxY<0GR{}001NiL4W|61QGxMEDAw@0GR|5
0068CL4W|61Q!4REDS+_0GR{=0068EL4W|61Q7rLECxY<0GR~~001lsL4W|61s4DSEDS+_0GS05001loL4W|61`_}PEDAw@0GS37
001luL4W|61`z-NECxY<0GS92001NiL4W|62oeAQEDS+_0GS980068EL4W|62oV4PECxY<0GSC3001luL4W|62@wDQECxY<0GSFC
001lsL4W|63K9SSEDS+_0GSFA001lsL4W|63=#kUEDS+_0GSLC001NcL4W|65EB3ZECxY<0GSXA001NiL4W|65E1|YEDAw@0GSXA
0068CL4W|65ElRdEDS+_0GSXG0068EL4W|65CQ-IECxY<0GSaB001lsL4W|65f=aeEDS+_0GSa2001loL4W|65)%LbEDAw@0GSdJ
001luL4W|65)l9ZECNY@0GSjL001lsL4W|66c+#hEDS+_0GSjK001lsL4W|67z+RZEDS+_0GSvO001loL4W|6APWEhBnUx(0GS{W
001lsL4W|6AQAuotO`MZ0GS{X001luL4W|6AQ1ontPDYb0GS{W001loL4W|6AqxNiEC@l60GS~Y001lsL4W|6Ar}AuEDS+_0GS~X
001loL4W|6A`1WjEDAw@0GT2Z001luL4W|6A`t)pEDAw@0GT8b001luL4W|6BoP1rEDAw@0GTKf001luL4W|6C=mbvEDAw@0GTis
001luL4W|6FcAO%BmzNz0Gj|3001loL4W|801E&BtOh}V0Gj{{001NiL4W|801^NIEDS+_0Gj|20068EL4W|801*HHECxY<0Gj~|
001luL4W|80TBQIECxY<0Gk36001lsL4W|80ulfKEDS+_0Gk34001lkNq_*G1QGxMECxY<0Gk90001lsL4W|81QGxMEDS+_0Gk96
001loL4W|82nzrJEDAw@0GkLB001luL4W|82oV4PECxY<0GkjC001lsL4W|85ElRdEDS+_0GkjI001loL4W|8AP)clEDAw@0Gl8Z
001luL4W|8AQ1onECxY<0G$96001NiL4W|A01^NIEDS+_0G$940068EL4W|A01*HHECxY<0G$C7001luL4W|A0TBQIECxY<0G$F8
001luL4W|A0ucZJEDAw@0G$L9001luL4W|A1Q7rLEDS+_0G$XC001lsL4W|A5E1|YEDS+_0G$vK001lsL4W|AAQAuoEDS+_0G%Ka
001loL4W|C022TJEDAw@0G|L7001luL4W|C01*HH8~_sl0I)z2005){6953PKobA}Bm@%x0I)z40068C6953PKoS4|EDRF>0I)z2
0068E6953PKo0-_Bmxrv0I)$5001Ni6953PK@k7|ED93<0I)$10068C6953PK^Fi3EDRF>0I)$1001Nc6953PLJ|M~ECv$*0I)(6
006886953PLK6T0ED93<0I)(50068C6953PLJt4{EDRF>0I)(2001lo6953PLlXc1ED93<0I)+3001Nc6953PL=yl2BnT4#0I)<6
001ls6953PL=pf1tO^qV0I)<C001lu6953PL=OM}EC`bT0I)?8001ls6953PMHc`7ECv$*0I)_A001ls6953PMiKx3Bmxrv0I*0C
001Ne6953PND}}6ED93<0I*0B0068C6953PND=@5EDRF>0I*080068E6953PND%-4ED93<0I*3C001lu6953PNf7`5ECv$*0I*6E
001ls6953PN)iA7EDRF>0I*6C001ls6953POcDS9EDRF>0I*CE001Nc6953PP!j+EBm@%x0I*OK001ls6953PP!a$DtO^qV0I*OG
001lu6953PP!RwCtPB$X0I*OG001ls6953PQ5OIJEDRF>0I*RH001lo6953PQWF3GED93<0I*UI001lu6953PQV#$CED93<0I*aS
001lu6953PR1W|EED93<0I*mR001lu6953PSPuXIBmxrv0I*;a001Ni6953PU=jcTED93<0I*;Z0068C6953PU=jcTEDRF>0I*;W
0068E6953PU=aWSED93<0I*>f001lu6953PVG#fTECv$*0I*^c001ls6953PVh;cSEDRF>0I*^a001lqlK=p)WD)=XED93<0I*~d
001lu6953PWDfuUED93<0I+Bh001lu6953PXb}JaED93<0I+Zp001lu6953Pa1Q_gBmxrv0J1<6001ls6953RKoS4|tO^qV0J1<5
001lu6953RKo0-_ED93<0J1?B001lo6953RLK6T0ED93<0J1_7001ls6953RL=pf1ED93<0J2CD001lu6953RNDlx2ECLe*0J2aL
001ls6953RP!a$DEDRF>0J2aI001ls6953RU=jcTEDRF>0J2~Y001Nc6953TKobA}ED93<0JK0C0068C6953TKoS4|EDRF>0JK04
001ls6953TK@tD}ECv$*0JK6A001ls6953TLJ|M~ED93<0JKCB001lo6953TND}}6ED93<0JKOF001lu6953TND%-4ED93<0JKmN
001lu6953TP!9kAED93<0JLBd001lu6953TU=aWSED93<0JcC9001loL4W|+01E&BBnUx(0NDT$001lsL4W|+01^NItO`MZ0NDT%
001luL4W|+00IC2tPDYb0NDTp001loL4W|+0Sf>CEDAw@0NDW&001luL4W|+0R#X5ECxY<0NDZ)001lsL4W|+0ulfKEDS+_0NDZ&
001lsL4W|+1QGxMEDS+_0NDft001lsL4W|+2p0eVEDS+_0NDr;001lsL4W|+5E1|YEDS+_0ND@%001lsL4W|+AQAuoEDS+_0NEe}
001loL4W|;01E&BEDAw@0NVf(001luL4W|;00aO4EDAw@0Nnr*001luL4W|=01*HHBmxrv0PsK(001lo69544KobA}tOgSR0PsK)
001ls69544KoS4|EDRF>0PsK&001lo69544K@$J~ED93<0PsN)001lo69544LK6T0ED93<0PsQv001lo69544L=yl2ED93<0PsW?
001lo69544ND}}6ED93<0Psi#001lu69544ND%-4ECv$*0Ps)~001ls69544P!a$DEDRF>0Ps)|001lo69544U=siUED93<0PtWE
001lu69544U=aWSECv$*0P;W+001ls69546KoS4|ECv$*0Q5i;001ls69548KoS4|90U-60T7T5001NeP=EmtkO=?)qy$ia0T7T0
001lmQh)&vkP`p^tOQbk0T7S_001NeP=EmtkqH0*ECf=30T7WB00686Qh)&vkpln#Bm_`^0T7Z2001lmQh)&vk^=w$tOQbk0T7Y{
001Nm5P$&?k_i9+EDREW0T7Z70068E5`X~^k`@2}ECf=30T7b|001lu5`X~^lNJB~EDaEV0T7c8001NeP=EmtlnDR;ECf=30T7fE
00686Qh)&vloJ2|Bn%LM0T7f9001luP=EmtlnDR;tPD_q0T7f4001lmQh)&vl@kB}EDTV90T7i5001lmQh)&vmJ<K~EDREW0T7lB
001luP=EmtmI(j=Bm_`^0T7r8001lmQh)&vm;(R+tOQbk0T7r2001Nm5P$&?m<|8{EDTV90T7r80068EP=Emtm<9j<ECf=30T7u3
001luP=EmtnFas=ECf=30T7x4001lu5`X~^nhyW~EDTV90T7xA001lmQh)&voD%>5EDTV90T7%C001NeP=Emtun7PFECf=30T8eQ
00686Qh)&vuoD0PECf=30T8hR001lmQh)&vvI77BEDREW0T8kd001lmQh)&vv=aaTEDTV90T8qa001lmQh)&vxDx;XEDTV90T8$e
000~W5P$&@fENG&Bm_`^0TF-`0068EP=EmufC~Tsq=XQF0TF-@0069o5P$&@fEEA%Bm_`^0TF={001luP=EmufeQcttPD_q0TF>0
001O}5P$&@fe-)yEQJt&0TF=}0069o5P$&@fffJ&Bn%LM0TF@_001luP=Emuf(rlutPK!=0TF@@001O}5P$&@f))S(EQJt&0TF@~
006AS5P$&@f)D@zEDTV90TF{2001lv5P$&@gBAb)EQJt&0TF`^001n+5P$&@gAf1!Bn%LM0TF}_001luP=EmugbM%wtPD_q0TF}?
001O}5P$&@gcbk*EQJt&0TF~1006AS5P$&@gcAS&EDTV90TG24001n75P$&@g%$t+EW{9i0TG20001luP=Emuh8F+;EDaEV0TG4{
001n+5P$&@h7$k)Bn%LM0TGA`0068E7=Qs0h!6k(EDTV90TGA`001O}5P$&@h!y|<EQJt&0TGB5006AS5P$&@h!6k(EDRWc0TGE1
001luP=Emui5CC>EQJt&0TGD~001n+5P$&@i4Xt)EDRWc0TGH2001luP=EmuiUt4xEDaEV0TGH8001n+5P$&@iVy$*EDRWc0TGM>
001luP=Emuj28d^EW{9i0TGN7001NeP=EmukP`p^EDTV90TGZ30068EP=EmukOKe!B!m!v0TGZ8001n75P$&@kQD#`tc4JO0TGZE
001luP=Emukrw~}EQJt&0TGcF001luP=Emuk^=w$EDaEV0TGfG001n75P$&@k`@2}EW{9i0TGfA001luP=EmulotR1EQJt&0TGlI
001n+5P$&@loJ2|EDRWc0TGxG001luP=Emum;(R+EQJt&0TGxM001n+5P$&@m=FK}Bm_`^0TG}O00687P=Emupc4Q9EDTV90TG}J
001O}5P$&@pcViCEQJt&0TG}T006AS5P$&@pb!86ECo=20TH1C001luP=Emup$h;2EQJt&0TH1V001n+5P$&@p%4H7ECo=20TH4W
001lv5P$&@q80!EEW{9i0TH4Q001lnP=Emuq!s`GEDTV90THAY001n+5P$&@q!R!DECo=20THMJ001lu7=Qs0r~&{0EDTV90THMR
001n+5P$&@s1N`EECo=20THkk001luP=Emuumb=9EQJt&0THkk001n+5P$&@un+(M90U-60TO@-005)}5`X~`fCB&ktOOE>0TO@?
001NeP=EmvfC&HqECf=30TO@{00686Qh)&xfCB&kBm@$G0TO`&001lm5{Lm3feruwtOOE>0TO{0001NeP=Emvfe8QrECf=30TO`|
00686Qh)&xfdc>lBm@$G0TO}(001lm5{Lm3f))S(tOOE>0TO}^001lmQh)&xf&%~mBn%LM0TO}<001lu5`X~`f(`%xtPB!>0TO}^
001lm5{Lm3gAM=yECf=30TP1)001lu5`X~`g988nEDaEV0TP1_001Ne5`X~`gckq+ECf=30TP50001Nm5P$&^gbn}zEDTV90TP4>
0068EP=Emvgb4rutb`JP0TP4|001lmQh)&xg%bb(EDTV90TP7?001n65`X~`g%AJ$EDREW0TPA|001luP=Emvh6w-wEQAt(0TPA~
001Ne5`X~`hywrsECdpW0TPH7006865{Lm3h!6k(ECf=30TPG<001Nm5P$&^hz<Y%EDTV90TPG_0068EP=Emvhz0-vECdpW0TPK2
001lmQh)&xi30!tEDTV90TPJ@030j?5`X~`iVy%!ECdpO0TPN1001lu5`X~`iVpw)EDTV90TPM{001luP=Emvj0pe$EQAt(0TPT5
001Ne5`X~`kOKe!ECdpW0TPf7006865{Lm3kPZL<Bm_`^0TPf2001lmQh)&xkOKe!tOQbk0TPe{001lm5{Lm3krn^|ECf=30TPh|
001lm5{Lm3k`4d>ECf=30TPk}001lu5`X~`k`4d>ECf=30TPrG001luP=EmvlnDR;EQAt(0TPrD001lm5{Lm3m=FK}ECf=30TP%4
001luP=Emvm<9j<Bm@$G0TQ4C001lm5{Lm3pcViCtOOE>0TQ4N001NeP=Emvpa}o~ECf=30TQ4C00686Qh)&xpc4Q9ECdpW0TQ7O
001lmQh)&xp#uN_ECdpW0TQAP001lmQh)&xq5}W`EDREW0TQAP001lmQh)&xq!R!DEDTV90TQGM001n65`X~`q!0iAECdpW0TQSX
001lmQh)&xs1pDHEDTV90TQSQ001lm5{Lm3unqtKECf=30TQqi001luP=EmwfC~TsB!m!v0TX~0001n75P$&_fE54$ti%w20TX}_
001luP=EmwffoP(EQJt&0TY21001n+5P$&_fe-)yEDaEV0TY52001n+5P$&_f(`%xEDTV90TYB5001n+5P$&_gcAS&EDRWc0TYN2
001luP=EmwhzkG!EW{9i0TYN2001luP=EmwkOKe!EQJt&0TYlG001n+5P$&_kPrX>ECo=20TZAT001n+5P$&_pb!86Bm@$G0ThrE
001NeP=EmxkO=?)ECf=30Thq}00686Qh)&zkP`p^EDREW0Thr90068E5`X~|kPiR=ECf=30Tht~001lu5`X~|kq-a>ECf=30ThxG
001lu5`X~|k`4d>ECf=30Th%M001lu5`X~|lnwv@EDTV90Th%8001lmQh)&zm=gd1EDREW0Th@I001luP=Emxm<a#?ECf=30Ti$k
001lu5`X~|unqtKEDTV90TqA)001O}5P$&{fEEA%EQJt&0TqB1006AS5P$&{fDixxEDTV90TqE4001n75P$&{fffJ&EW{9i0TqD|
001lv5P$&{f))S(EW{9i0TqG}001luP=Emygckq+EW{9i0TqN3001luP=Emyhz0-vEW{9i0TqZ4001luP=EmykOKe!EQJt&0TqxI
001n+5P$&{kPrX>ECo=20TrMY001n+5P$&{pb!86Bm@$G0TzH0001lm5{Lm7fEEA%ECf=30TzG*00686Qh)&#fD-@!EDREW0TzG`
0068E5`X~~fDZrwECdpW0TzK4001lmQh)&#fdc>lEDREW0TzJ|001lm5{Lm7f(`%xECf=30TzN2001lu5`X~~f(`%xECf=30TzT4
001lu5`X~~gbx4!EDTV90TzS_001n65`X~~gbn}zECdpW0Tzf3001lmQh)&#h!X$+EDREW0Tzf4001luP=EmzhzS4yECdpW0Tz%J
001lmQh)&#kP`p^EDREW0Tz%C001lm5{Lm7pbh{4ECf=30T!SW001lu5`X~~pbh{4EW{9i0T+M}001Na!GHl72nqlItN_J;0T~Dr
001NiL4W}n2m=5BtO!Af0T~Dq001luL4W}n2m=5BEC9uT0T~Gs001lqL5Kkv2?_uJEDS+_0T~Go001li#ee}B3IzZFEC@k}0T~Jb
001li#ee}B3<dxIBmlvH0T~bk001li!H@wN5DEYQtN_7~0T~bq001luL4W}n5CZ@JEC9if0T~er001luL4W}n5f1<WEC9if0T~hs
001li#ee}B5(EGMECfN20T~hs05&WD!GHl76bb+UEC9uT0T~no001li#ee}B7!v>hEC@k}0T~z)001luL4W}n7y|$REC9uT0U0O-
001lqL5KkvC<*`oGzdfh0U0V5001li!H@wNFbV(wEC9uT0U0m^001lqL689%FbV(wBmlvH0UH1W001li!H@wP00jU5tN_7~0UH1X
001luL4W}p00RI3EC9if0UH4Y001luL4W}p0S^EGEC9if0UH7Z001lmL689(0t)~DEC9if0UHDb001lqL5Kkx2nzrJEDS+_0UHPl
001li!H@wPAO!#bEC@l60UIC#001li#ee}F2m=5BEC@k}0UZbm001luL4W}r2oC@NEC9if0UZzu001li#ee}F5EB3ZEDS+_0UZzz
001li!H@wT00jU5EDS+_0UrPl000~e5C8!%0S5p8qzn*%0Wbjw001Nm5dZ-&0SN#AtPBx=0Wbjx001lu6954)0S5p8tPB$X0Wbj%
000~e5C8!%0tWy9qzn*%0Wbmx001Nm5dZ-&0to;BtPBx=0Wbmy001lu6954)0tWy9tPB$X0Wbm&001Nm5P$(N0|)>BEDRBV0Wbpz
0068E5r6?O0|@{CEDRF>0Wbp(000~e5C8!%1P1^Bqzn*%0Wbsz001Nm5dZ-&1PK5DtPBx=0Wbs!001lu6954)1P1^BtPB$X0Wbs)
001Nm5P$(N1qc8DEDRBV0Wbv#0068E5r6?O1qlEEEDRF>0Wbv*001Nm5P$(N1_%HEEDRBV0Wbyx0068E5r6?O1_=NFEDRF>0Wby+
0068EkbnU&1_=NFEDRBV0Wb#%001lukbnU&2MGWGBn%LM0Wb*&001lu5r6?O2?+oItPBx=0Wb*(001Nm5&!`(2@e1OEDRF>0Wb*<
001luk^liP2?ziHtPGL>0Wb*=001Nm5P$(N3J3rIEDRBV0Wb;)0068E5r6?O3JU-LBn%P&0Wb;+001lu6954)3J(APtPC0e0Wb;(
001luk^liP3J?GQEDRBV0Wb>*001lu8UO(>3kU!JEDVwW0Wb>?001Nm5P$(N3<v-KEDRBV0Wb^+0068E5r6?O3<&@LBn%P&0Wb^;
001lu6954)3=aSRtPC0e0Wb^}001luk^liP3=jYSEDRBV0Wb{-001lu8UO(>4Hf_ZEDVwW0Wb{^001lu5r6?O4haANEDRa|0Wc00
001lukbnU&4haANBn%LM0Wc8=001Nm5dZ-&5eWbQEDRBV0Wc8>0068E5r6?O5eWbQEDRF>0Wc8{0068E6954)5f1<WBn%LM0WcB>
001Nm5dZ-&5(xkREDRBV0WcB?0068E5r6?O5(xkREDRF>0WcB|0068E6954)5)S|XEDRBV0WcE@001lu6954)6Au6YBn%LM0WcH@
001Nm5dZ-&6bS$TEDRBV0WcH^0068E5r6?O6bS$TEDRF>0WcH~0068E6954)6bAqREDRBV0WcK_001lu6954)6$bzSEDRBV0WcN`
001lu6954)76$+TEDVr<0WcN`001lu5r6?O83_OYEDRF>0WcX4001luk^liP83+IXEDRBV0Wca1001lu6954)8V>*fEDRa|0WcZ}
001luk^liP8V~>gEDRBV0Wcg1001lu6954)90vdZEDRa|0WcgE001luk^liP91s8iBn%LM0Wcv6001lu5r6?OAqfBgtPBx=0Wcv6
001lu5`Y0PAqW5ftPB!>0WcvC001lu6954)ArAlmBn%LM0Wcy7001lu5r6?OA_)KhtPBx=0Wcy7001lu5`Y0PA_xEgtPB!>0Wcy6
001lu6954)A`bunEDRBV0Wc#8001lu5`Y0PBM1NhBn%LM0Wc&9001lu5r6?OBnbcjtPBx=0Wc&9001lu5`Y0PBnSWitPB!>0Wc&8
001lu6954)Bo6=pEDRBV0Wc*A001lu5`Y0PB?tfjEDRBV0Wc;B001lu5`Y0PCI|okEDVr<0Wc;B001Nm5P$(NC<y=nBn%P&0Wc^J
001lu5`Y0PC<p)mtPB!>0Wc^J001lu6954)C=UPtEDVwW0Wc^C0068Ek^liPC=dVuEDRBV0Wc{E001lu5`Y0PDGvYuEDVwW0Wc{L
001lu5r6?ODhmJrEDREW0Wc~E001lu8UO(>DhL1oEDVwW0Wc~M001lu5r6?OEC~PrEDREW0Wd5G001lu8UO(>EEWI&EDVwW0Wd5O
001Nm5P$(NFbMzvBn%M%0WdHL001lu5`Y0PFbDtutPB!>0WdHR001lu6954)Fb4nttPB$X0WdHR001lu5r6?OF$n+wEDREW0WdKS
001lu6954)F%JL$EDRBV0WdNP001lu5`Y0PG6(<wEDRF>0WdNT001lu5r6?OGzkCzEDREW0WdTO001lu6954)GzS0xEDREW0WdfZ
001lu6954)I1c~-EDVwW0Wdfa000~S6954)Kmq^&qyQ5H0Wd%k001Ne6954)KobA}tOOGT0Wd%h001lu6954)Ko0-_tPB$X0Wd%h
001Na69fS;K@$J~ECdq-0Wd)i001Ni6954)K@R``tOyeX0Wd)i0068AlK=rQK@tD}EDRF>0Wd)i001OR5C8!%K?wi=EKCsq0Wd)i
0068^5&!`(K??u?EDRF>0Wd-j001OR5C8!%LJ0r>EKCsq0Wd-j0068^5&!`(LJI%@EC`bT0Wd=n001mZ5&!`(Lkj=^Bmffx0Wd@p
001lm69fS;L<;}_BnT4#0Wd@Y0068A69fS;L=pf1EDRF>0Wd@l001OR5C8!%L<s-@EKCsq0Wd@h0068^5&!`(L=ON0EC>?-0Wd`m
05mKJ6954)MGpV~EC`bT0Wd`p001mZ5&!`(MGpV~EKCvr0Wd}n001Na69fS;ND}}6Bm@%x0We4p001lm69fS;NDBY}tOSz)0We4t
0068A69fS;NCp4^Bn%P&0We4p001lu6954)NDlx2EDVwW0We4q0068Ek^liPNC*G`Buo$h0We4j001mZ5dZ-&NDlx2tV|LB0We4p
001lmlK=rQNfQ77EC>?>0We7m001lqlK=rQNfH16EDRF>0We7q001luk^liPNe}=4EKCvr0We7m001lmlK=rQN)rG8EDRF>0WeAr
001lu8UO(>N(cY|EDVwW0WeAk001mZ5&!`(N(%r0ECiDP0WeGx001lq69fS;ObP%1EDRF>0WeGt001lu8UO(>OcnqDEDVwW0WeG!
001mZ5&!`(ObY-2Bmffx0WeS#001Ne6954)P!j+EECdq-0WeSx0068A69fS;PzwM6Bn%M%0WeSr001lu6954)P!9kAtPB$X0WeSx
001OR5C8!%PzeA4EKCsq0WeSx0068^5&!`(PzwM6EC>?>0WeVy001lqlK=rQQ4#<EEDRF>0WeVy001mZ5&!`(Q40V7EDRF>0WeYz
001mZ5&!`(QVRe8EC>?>0Weew001lu6954)R0{wAEKCvr0Wee#001lmlK=rQSQ7vMEC>?>0Weqw001lu6954)SPuXIEDVwW0Weqy
001mZ5&!`(SPuXIBmffx0We?>00682lLP@UU=jcTECdq-0We?>001Ni6954)U=IKQtO%0;0We?^001lu6954)U=IKQBuo$h0We?*
001mZ5dZ-&U=IKQtV|LB0We?>001lilLP@UVG;lUEC>?>0We_;001lqlK=rQVG;lUEDRF>0We_?001mZ5&!`(VG95NEC`bT0We|`
001mZ5&!`(VhaEOEC7=P0Wf3|001lq69fS;WC{QPEC`bT0Wf3|001mZ5&!`(WDfuUEC7=P0WfG1001lmlK=rQXcGVcEC>?>0WfF}
001lqlK=rQXc7PbEDRF>0WfF}001luk^liPXb1oREKCvr0WfF}001lilLP@Ua1sCjEC>?>0Wfe2001lqlK=rQa1sCjEDRF>0Wfe6
001mZ5&!`(a0>tc90U-60Wg3U005*67ytn<fEEA%tV|dH0Wg3N001Nm5P$(Nfe8QrtPB!>0Wg6B001Nm7ytn<fffJ&EKC>x0Wg6O
0068^7ytn<fe-)yB!m!v0Wg6O001n75P$(NffWD%tc4JO0Wg6U001Nm5P$(Nf)D@zEDREW0Wg9V001Nm7ytn<f)D@ztPGHV0Wg9I
0068F5P$(Nf))S(tPL0d0Wg9V001mZ7ytn<f)xM&B!m!v0Wg9V001n75P$(Nf)xM&ti%w20Wg9P001lu5`Y0Pg988nEDVr<0WgCJ
001lv5P$(NgAV`zEDaa{0WgCW001mZ7ytn<gAf1!EQJt&0WgCW001n+5P$(NgAf1!Bn%LM0WgFK0068E5`Y0Pgbx4!Bn%h;0WgFR
0068F7ytn<gcbk*EKC>x0WgFW001O}5P$(Ngcbk*EQJt&0WgFW006AS5P$(NgcAS&EDREW0WgIR001lv7ytn<g%$t+EKC>x0WgIS
001n75P$(Ng%$t+EW{9i0WgIV001lu5`Y0Ph7SM$EDVr<0WgLM001lv5P$(Nh6w-wEDaa{0WgLZ001n+5P$(Nh7$k)Bn%LM0WgRQ
001Nm5&!`(hzkG!Bn%h;0WgRV0068E7=QsVh!6k(tPC0e0WgRb001luk^liPh!6k(tPL0d0WgRb001mZ7ytn<h!p?;B!m!v0WgRb
001n75P$(Nh!p?;ti%w20WgRV001lu5`Y0Pi30!tEDRWc0WgUP001lu8UO(>i536=EDVwW0WgUW001lv7ytn<i3tDzEKC>x0WgUW
001n75P$(Ni536=EW{9i0WgUW001lu5`Y0PiVpw)EDRWc0WgXX001lu8UO(>iWUF>EDVr<0WgXX001lv5P$(NiWUF>EDaa{0WgXd
001n+5P$(NiVy$*EDREW0WgdU001lu7=QsVj0pe$EDRa|0Wgdf001lv7ytn<j1~X@EW{9i0Wgdc001Nm7ytn<kQM*{EKC>x0Wgpi
0068^7ytn<kO=?)EDREW0WgsR001mZ7ytn<kq`g?EQJt&0Wgsk001lu5`Y0Pk`@2}EDVr<0WgvY001lv5P$(Nk`@2}EDaa{0Wgvl
001mZ7ytn<k`Mp@EQJt&0WgvY001n+5P$(Nk`Mp@EDREW0Wg#g001lv7ytn<lokL0EKC>x0Wg#h001n75P$(NlokL0EW{9i0Wg#k
001lu7=QsVm=FK}EDRa|0Wg>r001luk^liPm=FK}EDaa{0Wg>r001mZ7ytn<m=FK}EQJt&0Wg>r001n+5P$(Nm=FK}Bn%LM0WhEm
001lu5`Y0Ppbr25tPB!>0WhEs001Nm7ytn<pcViCEKC>x0WhEy0068^7ytn<pb!86B!m!v0WhEz001n75P$(NpcMcBti%w20WhEt
001lu5`Y0Pp$`B6EKC>x0WhHu001n75P$(Np%wrDEW{9i0WhHu001lu5`Y0Pq7MK7EDVr<0WhKo001lv5P$(Nq80!EEDaa{0WhK#
001mZ7ytn<q7VQ8EW{9i0WhKv001lu5`Y0Pq!0iAEDaa{0WhQ%001mZ7ytn<q!0iAEW{9i0WhQ!001lu5`Y0Ps1E=DEDRWc0Whc#
001lu8UO(>s1^VKEDVwW0Whc#001lv7ytn<s1^VKEKC>x0Whc#001n+5P$(Ns1N`EEDREW0Wh!+001mZ7ytn<un+(MEQJt&0Wh!@
001n+5P$(Nun+(MBn%h;0WiQ20068F7ytn<z!m@iBuo$h0WiP`0068^5&!`(zz+ZbEKC>x0WiQ7001lv7ytn<!4?1jEKCvr0WiT2
001mZ7ytn<!4LodEDaa{0WiWB001mZ5&!`(!VdrdEDaa{0WicC001mZ5&!`(#18-fEDRa|0WioG001luk^liP$QA$qEDaa{0WioG
001mZ5&!`($PWMjEDaa{0Wi=O001mZ5&!`(&<_9rEKC>x0Wi=I001lv7ytn<;1&P?EKCvr0WjbX001mZ7ytn<;1B=+Bn%LM0Wkpx
001lu5r6?P0T%!OtPBx=0Wkps001Nm7ytn=0TuuNEKC>x0Wkp(0068^7ytn=0T2KHBn%LM0Wks(001lu5r6?P0v7-PBn%h;0Wks)
0068EkbnU(0t)~DtPL0d0Wks=001mZ7ytn=0u=xNEDRBV0Wkvu001lukbnU(0}B8EEDaa{0Wkv>001mZ7ytn=0}ucJBn%LM0Wky!
001lu5r6?P1Q!4RtPBx=0Wky*001Nm7ytn=1P}lKtPL0d0Wky?001mZ7ytn=1Qh@PEDRBV0Wk#^001lv7ytn=1r`7REKC>x0Wk#-
001lu5r6?P1_S^AEDVr<0Wk&!001lv7ytn=1{MGSEDRBV0Wk>|001lv7ytn=2^IhVEKC>x0Wk>>001lu5r6?P3JU-LEDVr<0Wk^&
001lv7ytn=3KjqWEDRBV0Wk~@001lv7ytn=3>E+YEDRBV0WlE|001mZ7ytn=5fA_XEDRBV0WlH}001lukbnU(5(@wTEDaa{0WlH?
001mZ7ytn=5)c3YEDRBV0WlO0001lv7ytn=6czvgEKC>x0WlO1001Nm5P$(OAQu1tEDRBV0WlyK001Nm7ytn=AQk`sEKC>x0WlyI
0068^7ytn=AP@imtc(zV0Wly8001lu5r6?PAr}AuEKC>x0Wl#E001nE5P$(OAqxNiEDRBV0Wl&A001lukbnU(A`1WjEDaa{0Wl&L
001mZ7ytn=A`k!oEQ}C<0Wl&A001lu5r6?PBntolEDaa{0Wl;N001mZ7ytn=BoF`qEQ}C<0Wl;H001lv7ytn=C>8(!EKC>x0Wl~L
001nE5P$(OC<_1pEDRBV0WmNS001mZ7ytn=Fc1I$EQ}C<0WmNO001Nm7ytn=Ko9@`tPL0d0Wm-p001OR5C8!&Knnl>EKCsq0Wm-e
001mZ7ytn=KotN0tV|dH0Wm-j001lv7ytn=K^6c2EKC>x0Wm=p001lv7ytn=LKgr4EKC>x0Wm@Y001lv7ytn=L>2%5EKC>x0Wm}m
001lv7ytn=NEQG9EKC>x0WnAe001lv7ytn=P!<3HEKC>x0WnY&001lv7ytn=U={!XEKC>x0Wn|@001Nm7ytn=fDixxtPL0d0Wp9F
001mZ7ytn=fE54$EDaa{0WpCG001mZ7ytn=fe-)yEQJt&0WpCV001n+5P$(OffE1#EDVr<0WpFQ001lv5P$(Of(8HpEDaa{0WpFH
001n+5P$(Of)fA$EDaa{0WpLJ001n+5P$(OgcAS&EDRWc0WpXI001lv7ytn=hz9@wEW{9i0WpXZ001lv7ytn=kOu$&EKC>x0Wpve
001lv7ytn=pa%c|EKC>x0WqKz001nE5P$(OpbG#1EW{9i0WqKx001lv7ytn=z!m@iEKC>x0WrV>001Nm5P$(P0SEv9EDRBV0Wtvz
0068E5r6?Q0SN#AEDRF>0Wtv(001Nm5P$(P0tf&AEDRBV0Wty!0068E5r6?Q0vG@QEDRF>0Wty)001lu5r6?Q0~i1RBn%LM0Wt&#
001lu5r6?Q1Q-AStPBx=0Wt&$001lu6954+1P=fJEDRBV0Wt*%001lu5r6?Q1_=NFEDVr<0Wt;&001lu5r6?Q2^atXEDVwW0Wt{|
001lu5r6?Q3K#$YEDRa|0Wt~*001luk^liR3KjqWEDRBV0Wu5;001lu8UO(@3>E+YEDVwW0Wu60001lu5r6?Q5f}gfEDRF>0WuK}
001lu5r6?Q5*PpgEDRF>0WuN~001lu5r6?Q6bS$TEDRF>0WuT^001Nm5P$(PAPE2fEDRBV0Wu&70068E5r6?QAQ%7uEDREW0Wu&6
0068E5`Y0RAP4{eEDRF>0Wu&D001lu5r6?QAs7GvEDREW0Wu*7001lu5r6?QA{YPwEDREW0Wu;8001lu5r6?QBnbcjEDREW0Wu^A
001lu5r6?QC>Q_$EDREW0Wv5E001luk^liRC>8(!EDRBV0WvTc001lu5`Y0RFbDtuEDRF>0WvTT001Na69fS=Knnl>tN<GX0Wv@m
001lm69fS=Ko0-_EDRF>0Wv@j001OR5C8!(Knnl>EKCsq0Wv@j0068^7ytn>Ko9@`EC3q>0Wv`n001lq69fS=K??u?EC`bT0Wv`n
001mZ5&!`*K??u?EKC>x0Wv`l001mZ5&!`*LJI%@EKC>x0Wv}f001li8w3F|L=pf1EC>?>0Ww4e001mZ5&!`*L=OM}EKC>x0Ww4h
001li8w3F|ND=@5ECiDP0WwGv001lq69fS=NCp4^EDRF>0WwGr001luk^liRNEQG9EKCvr0WwGr001mZ7ytn>NC^M{EC3q>0Wwe$
001lq69fS=PzwM6EDRF>0Wwez001mZ5&!`*PzwM6EKC>x0Wwe!001li8w3F|U=IKQEC7=P0Wx3<001lqlK=rSU=jcTEKCvr0Wx3^
001mZ7ytn>U=RQRECdpW0WyIW001lu5`Y0Rfdc>lEDaa{0WyIW001mZ7ytn>fe-)yEQAt(0WyIQ001n75P$(PfffJ&EQ}C<0WyIJ
001n+5P$(Pfe-)yECdpW0WyLR001lu5`Y0Rf&%~mEDVr<0WyLR001lv5P$(Pf(ZZsEDaa{0WyLX001n65`Y0Rf)D@zEQ}C<0WyLX
001n+5P$(Pf)D@zEDREW0WyRS001lv7ytn>gcbk*EQAt(0WyRT001nE5P$(Pgcbk*EW{9i0WyRW0ZuFg5QqUXh!6nkEDRU`0Wydd
001lu7=QsXhywrsEDaa{0Wydd001lm5{LmZpb!86EDREW0WzQv001lv7ytn>pcViCEKC>x0WzQv001n65`Y0Rpb!86EQ}C<0WzQ#
001n+5P$(Ppb!86EDaa{0W!cA001mZ5&!`*zz+ZbEKC>x0W!b|001lu5r6?R0T=)PEDaa{0W$#z001mZ7ytn?0T2KHEQ}C<0W$#>
001lu5r6?R0vG@QEDVr<0W$&!001lv7ytn?0u}%OEQ}C<0W$&?001lu5r6?R1Q-ASEDaa{0W$;^001nE5P$(Q1QY-OEDRBV0W%;M
001lv7ytn?AQk`sEKC>x0W%;F001nE5P$(QAPWEhEDaa{0W&}r001mZ7ytn?Ko9@`Bn%LM0W<*#001lu5r6?S0SN#AtPBx=0W<*#
001lu5`Y0T0SEv9tPB!>0W<*v001lu6954;0S^EGBn%LM0W<;$001lu5r6?S0to;BEDREW0W<;#0068E5`Y0T0tf&AEDRF>0W<;+
0068EkbnU+0to;BEDRBV0W<>%001lu5`Y0T0|)>BEDVr<0W<>%001Nm5P$(R1PK5DEDRBV0W<^&0068E5r6?S1Oxy8EDREW0W<^%
0068E5`Y0T1PA~CEDRF>0W<^;001lu5r6?S1s4DSEDREW0W<{&001lu5r6?S1_S^AEDREW0W<~(001lukbnU+1_=NFBn%LM0W=5+
001Nm5&!`-2oC@NEDREW0W=5*0068E5`Y0T2nYZGEDRF>0W=5?001luk^liT2oL}OtPGL>0W=5}001lu5r6?S2?+oIEDREW0W=8+
001luk^liT2^IhVEDRBV0W=B;001lu5`Y0T3J3rIEDVr<0W=B;001luk^liT3J3rIEDRBV0W=H`001lu5`Y0T3<v-KEDVwW0W=I2
001Nm5P$(R5D5SPBn%M%0W=T^001lu5`Y0T5C{MOtPB!>0W=T;001lu6954;5Dx$VtPB$X0W=T~001lu5r6?S5eWbQEDREW0W=W<
001lu6954;5f1<WEDRBV0W=Z`001lu5`Y0T5(oeQEDRF>0W=a1001lukbnU+5(oeQEDRBV0W=f@001lu5`Y0T6bJwSEDRF>0W=f`
001lu5`Y0T7zh9WEDRF>0W=s7001luk^liT7!UveEDRBV0W={A001lu5`Y0TAqW5fEDRBV0W=~B001lu5`Y0TA_xEgEDVr<0W=~B
001lu5r6?SBm@8eEDREW0W>5C001lu5`Y0TC<p)mEDVwW0W>HU001lu5`Y0TFbDtuEDRF>0W>fV0R|)h0|WsyKo0-_tN?=q0W?4o
0R}7t0|WsyKo0<-EDQqx0W?4l0R}7#0{{UuKo0-_EC7Q90W?7p0R}7x0|WsyK@R~2EC_=D0W?7Z0R}8g0ssLtK@R``EKGm^0W?7g
0R}8g0ssLtLJt4{EKGm^0W?Ah001lig9HIIL=yo9EC>Sx0W?Gp0R}8g0ssLtL=OM}EKGm^0W?Gj001lig9HIIND~1DECho90W?Sx
0R}7x0|WsyNDly_EDVAG0W?S!0R}7#f&c+DNC*K2EKC9b0W?St001mZfB*qBNC^M{EC7Q90W?q(0R}7x0|WsyP!9m2EDQqx0W?q#
0R}7#0{{UuP!9o|EDQqx0W?q#0R}8g0ssLtP!9kAEKGm^0W?qv001lig9HIIU=slbEC7Q90W@F}0R}7xg8%_EU=jfaEKC9b0W@F_
001mZfB*qBU<m*KBm@$G0W^RR006875`Y0TfCc~nBn%LM0W^RK001lu5`Y0TfDZrwtPB!>0W^RQ001O}5P$(RfEEA%EQJt&0W^RW
006AS5P$(RfDixxECmvP0W^UV001lu5`Y0Tfe!!xEQJt&0W^UY001n+5P$(Rfe-)yECmvP0W^XJ001lu5`Y0Tf)4-yEDVr<0W^XM
001lv5P$(Rf(8HpEW{9i0W^XT001ln5`Y0TgcAS&EDREW0W^dc001n+5P$(RgaQBnECmvP0W^pc001lu5`Y0Thz|e&EDVwW0W^pf
001n+5P$(Rh!6k(ECmvP0W^>X001lu5`Y0TkPiR=EQJt&0W^>n001n+5P$(RkPrX>ECmvP0W_c%001lu5`Y0Tpbr25EW{9i0W_cx
0R}8g0ssLtzz+ZbEKGm^0W`n~001Nm5P$(S02crNEDRBV0W|;@0069v5P$(S02TlMEDRBV0W|>^001nE5P$(S0TuuNEDVr<0W|^!
001nE5P$(S0u}%OEDRBV0W|~!001nE5P$(S1Qq}QEQ}C<0W}B~001lu5r6?T5Dx$VEQ}C<0W}Z^001nE5P$(SAQk`sEKGm^0X0An
001nE5P$(SfEEA%EW{9i0X2XV001lu5r6?U0TuuNEDREW0X6{$001lu5r6?U0u}%OEDREW0X6~%001lukbnU;0to;BEDRBV0X75{
001lu5`Y0V1PA~CEDRBV0X7I0001lu5`Y0V2nYZGEDVwW0X7I0001lu5r6?U5EcLcEDREW0X7f_001lu6954=5Dx$VEDRBV0X85O
001lu5`Y0VAP4{eEC7Q90X9Gs001mZfB*qDKnVZ<ECmvP0XBdW001lu5`Y0VfDZrwEQAt(0XBdT001nE5P$(TfEEA%EW{9i0XBdT
001lu5r6?V02TlMEQ}C<0XF~^001lig9HIQKobA}G);g20X;wp001NeP=EnYfe8QrECf=30Z@Sx00686Qh)(afdc>lBn%LM0Z@Ss
001luP=EnYfe8QrtPD_q0Z@Sn001lmQh)(af&%~mBn%LM0Z@Vo001lu5`Y0vf))S(tPB!>0Z@Vt001luP=EnYf(ZZstPD_q0Z@Vo
001lmQh)(ag988nEDREW0Z@Yu001luP=EnYg9!itEDaEV0Z@Yu001lmQh)(agcAS&Bn%LM0Z@bv001luP=EnYgb4rutPD_q0Z@bs
001lmQh)(ag%bb(EDTV90Z@et001lu5`Y0vh86$-EDTV90Z@hu001lmQh)(ai30!tEDTV90Z@qv001lu5`Y0viVgq(EDTV90Z@tw
001luP=EnYj0*q&ECf=30Z@?x001luP=EnYkqH0*ECf=30Z@_?001lu5`Y0vk`4d>EDTV90Z@_&001lmQh)(alotR1EDTV90Z^0+
001NeP=EnYpc4Q9ECf=30Z^a=001Nm5P$(tpbh{4EDTV90Z^a`0068EP=EnYpa}o~ECf=30Z^d>001luP=EnYp$Py0ECf=30Z^g?
001lu5`Y0vq7DE6EDTV90Z^g|001luP=EnYq!s`GECf=30Z^zD001luP=EnYs0jc7ECf=30Z_0L001luP=EnYun7PFEDTV90a1Yq
001n75P$(ufffJ&EW{9i0a1Yv001luP=EnZf(iftEDaEV0a1b$001n+5P$(uf)D@zEDTV90a1ht001n+5P$(ugb)A#ECo=20a2h6
001luP=EnZpcViCEW{9i0a2h3001lm5{Lm(fffJ&ECf=30aAej001luP=Enafe8QrEQAt(0aAew001lm5{Lm(f(`%xEDREW0aAhv
001luP=Enaf(ZZsEQAt(0aAhx001luP=EnagbM%wEQAt(0aAnz001lm5{Lm(pbh{4ECf=30aBn7001luP=Enapa}o~EQAt(0aBn4
001Ne5`Y0zfD-@!ECf=30aSnk001Nm5P$(xfDQlvEDREW0aSnv0068E5`Y0zfDQlvEDTV90aSnr0068EP=EncfC&HqECf=30aSql
001lu5`Y0zferuwEDTV90aSqr001lu5`Y0zf(`%xEDTV90aSts001lu5`Y0zgbn}zEDTV90aSzu001lu5`Y0zhz<Y%EDTV90aS<(
001lmQh)(ekP`p^EDREW0aTC<001luP=EnckO=?)ECf=30aTz9001lu5`Y0zpbh{4EDTV90aTzC001luP=EndfC>NrEW{9i0abt#
001lmQh)(gfD-@!EDREW0akzx001luP=EnefC&HqEQAt(0akzz001lqL5Kla0R#X5EDS+_0a*b9001lmL68Ai0tx^CEC@k}0a*eI
001luL4W~S0tx^CEC@k}0a*wO001luL4W~S2nzrJEC@k}0a*|P001lqL5KlaAOrvaEC@l60a+jl001lqL5Klc00aO4EC@k}0bKwD
001luL4W~W00RI3Bn%LM0dN5b001lu5r6@30SN#AtPBx=0dN5c001lu6955l0S5p8Bn%LM0dN8c001lu5r6@30to;BtPBx=0dN8d
001lu6955l0tWy9EDRBV0dNBe001lu5r6@31s4DSEDRBV0dNKv001lu5r6@32?+oIEDRa|0dNTx001lu5r6@33JCxJEDRa|0dNWy
001lu5r6@35eWbQEDRF>0dNrq001lu5r6@35(xkREDRF>0dNur001lu5r6@3AqfBgEDREW0dOG*001lu5r6@3A_)KhEDREW0dOJ+
001lu5r6@3Bo_bxEDREW0dOP;001lu5r6@3C<y=nEDREW0dOb?001lu8UO)sC>8(!EDRBV0dO!0001lu5`Y14FbDtuEDRF>0dOz}
001Na69fTpKnnl>tN<GX0dPPD001lm69fTpKnnl>EDRF>0dPPI001li8w3GxK?VQ-EC>?>0dPSJ001mZ5&!{kK??u?EKC>x0dPSO
001mZ5&!{kLJI%@EKC>x0dPVP001li8w3GxL=pf1EC>?>0dPbM001mZ5&!{kL<;}_EKC>x0dPbJ001li8w3GxND=@5EC>?>0dPnL
001lu8UO)sNEQG9EKCvr0dPnQ001mZ7ytoqNDu%3EC3q>0dP<T001lq69fTpPzwM6EDRF>0dP<Y001mZ5&!{kPzwM6EKC>x0dP<d
001li8w3GxU;zLCEKCvr0dQat001mZ7ytoqU=RQREDREW0dRpA001lv7ytoqfd>EpEKC>x0dRp3001n75P$)2fffJ&EW{9i0dRp3
001lu5`Y14f&%~mEDaEV0dRr_001lv7ytoqf(HNqEW{9i0dRs4001lu5`Y14gcbk*EDaa{0dRyA001lu5`Y14paTE^EDaa{0dSxP
001mZ7ytoqpb!86EW{9i0dSxY001lv7ytoqz!U%gEKCvr0dT+z001mZ7ytoqzzF~VEDRBV0dWBf001lv7ytor0SEv9EKC>x0dWBc
001lu5r6@40t)~DEDaa{0dWEc001lu5r6@4APWEhEDaa{0dXJ)001mZ7ytorAP@imEQ}C<0dXJ@001lv7ytorKo$T1EKC>x0dYVG
001lu5r6@50SN#AEDRBV0dfKf001lu5r6@5APE2fEDREW0dgP+001li8w3GzKoS4|EKC>x0dhbI001lu5r6@70SN#AEDREW0dxTf
001lu5r6@70to;BEDREW0dxWg001lu5r6@71PK5DEDREW0dxci001lu5r6@72nhfHEDREW0dxom001lu5r6@75D5SPEDREW0dx=u
001lu6955p5C;GNECmvP0d#;9001lu5`Y18fCc~nEW{9i0d#;9001NeP=EoDkO=?)ECf=30g#Xb00686Qh))FkOKe!Bn%LM0g#Xm
001luP=EoDkO=?)tPD_q0g#Xh001lmQh))Fkpln#EDTV90g#ai001lmQh))Fk^=w$EDREW0g#do001luP=EoDk_i9+ECf=30g#jv
001luP=EoDlnDR;ECf=30g#vj001luP=EoDm<a#?ECf=30g$j0001lu5`Y1aunqtKEDTV90g$i>001lu7=QthhynlrEDTV90g;FX
001n+5P$)Zh!p?;EDRWc0g;dc001luP=EoEkOKe!EW{9i0g;ds001Ne5`Y1cfENG&ECdpW0g`|g001lmQh))HfCB&kBn%LM0g`|N
0068E5QqVifC&HqEDTV90g`|T0069n5`Y1cfDQlvECdpW0g{0h001lmQh))Hfdc>lEDR8c0g{0U001luP=EoFfe8QrEQAt(0g{0Z
001lm5{Lnkf(`%xEDR8c0g{3V001lu5`Y1cf(`%xEQAt(0g{3i001luP=EoFgb4ruEQAt(0g{9k001lm5{Lnkhz<Y%EDR8c0g{Lg
001luP=EoFhzS4yEQAt(0g{Lo001lm5{LnkkQM*{ECf=30g{jd001lu5QqVikO=?)EDTV90g{jj001n65`Y1ckPZL<ECdpW0g|8&
001lmQh))Hpc4Q9EDR8c0g|8z001lu5`Y1cpbh{4EQAt(0g|8&001lu5QqVjfC~TsEDRWc0h53O001n+5P$)bfE54$ECf=30hEvv
001lu5`Y1ekPZL<EDTV90hEvl001lmQh))LfD-@!EDR8c0hWLX001lu5`Y1gfDQlvEQAt(0hWLk001li!H@x&5DEYQEC9uT0htg4
001lqL5KmF5Ci}KEDS+_0htgF001li!H@x)00jU5EC@k}0h<5>001luL4X0900IC2Bn%LM0k8oG001lu5r6@(0SN#AtPBx=0k8oH
001Nm5&!|P0SEv9EDRF>0k8oN0068E6956Q0S5p8Bn%LM0k8rH001lu5r6@(0to;BtPBx=0k8rI001Nm5&!|P0tf&AEDRF>0k8rO
0068E6956Q0tWy9EDRBV0k8uJ001lu6956Q0|x*ABn%LM0k8xJ001lu5r6@(1PK5DtPBx=0k8xK001Nm5&!|P1P1^BtPB!>0k8xJ
001lu6956Q1P=fJEDRBV0k8!L001lu5`Y1)1qc8DEDRF>0k8!J001lu5r6@(1_=NFEDREW0k8%L001lu6956Q1_uBDEDVr<0k8%M
001Nm5&!|P2nYZGEDRF>0k8-U0068E6956Q2nPTFEDVwW0k8-N0068Ek^lj)2nYZGEDRBV0k8=P001lu6956Q2?qcGEDVwW0k8=O
001lu5r6@(3JCxJEDRF>0k8@O001lu8UO*X3J3rIEDVwW0k8@P001lu5r6@(3<&@LEDREW0k8}R001lu6956Q3<m%JEDRa|0k8}f
001luk^lj)3<v-KBn%M%0k9AW001Nm5&!|P5C{MOEDRF>0k9Ac0068E6956Q5C;GNEDRBV0k9DX001lu6956Q5eEPOEDRBV0k9GY
001lu6956Q5(fYPEDRBV0k9Ma001lu5`Y1)6bJwSEDRF>0k9MY001lu6956Q7zY3VEDVwW0k9Yd001lu5r6@(AqfBgEDREW0k9zm
001lu6956QAqM~eEDRBV0k9$q001lu5`Y1)A_xEgEDRF>0k9$m001lu5r6@(BnbcjEDREW0k9+p001lu6956QBnJQhEDREW0k9|t
001lu6956QC<g!lEDVwW0k9|t001lu5`Y1)FbDtuEDRF>0kAL!001lq69fUUK@R``EC`bT0kA<5001lu6956QK@R``EDVwW0kA;`
001mZ5&!|PK??u?EKHC90kA;{001lu6956QLJt4{EDVwW0kA>{001mZ5&!|PLJI%@EKHC90kA>|001lq69fUUL<#@^EDVwW0kA{}
001mZ5&!|PL=OM}EKHC90kA{~001lmlK=s*ND}}6EC>?>0kB90001lu6956QNDlx2EDVwW0kB92001mZ5&!|PNDlx2EKHC90kB93
001lq69fUUPzwM6EDRF>0kBXH001luk^lj)PzV43EKCvr0kBXD001mZkN^R&PzeA4EC7=P0kB{b001lq69fUUU=IKQEC`bT0kB{a
001lu6956QU=IKQBn%LM0kD7y001Nm5&!|PfDZrwtPB)@0kD7x001Nm7ytpVfEEA%tPGL>0kD7;001mZ7ytpVfE54$tV|dH0kD7-
001O}5P$)&fEEA%ti%w20kD7*001lu5`Y1)fdc>lEDRKY0kDAy001luk^lj)fffJ&EKC>x0kDA;001n+5P$)&ffE1#EDREW0kDD(
001lu6o3J+f)4-yEDVr<0kDDz001luk^lj)f))S(EKC>x0kDD<001n+5P$)&f&%~mEDREW0kDJ*001lu6o3J+gbx4!EDVwW0kDJz
001mZ7ytpVgcSe)EW{9i0kDJ#001lu6o3J+hz|e&EDRWc0kDVy001lu8UO*Xh!y|<EDVwW0kDV`001mZ7ytpVh!p?;EW{9i0kDV(
001lu6o3J+kO=?)EDVwW0kDu3001mZ7ytpVkQD#`EW{9i0kDu0001lu5`Y1)paTE^EDRKY0kEJ6001luk^lj)pa%c|EKC>x0kEJI
001n+5P$)&paK8@EDVwW0kFUW001mZ5&!|Pzz+ZbEKC>x0kFUo001mZkN^R&zzF~VEDRBV0kHuW001mZ7ytpW0TloMEDRBV0kHxD
001lukbnWP0t5g6EKC>x0kHxV001lu5r6@)1Oxy8EKC>x0kH%X001mZ7ytpW2o(STEDRBV0kIGd001mZ7ytpW5ETFbEKC>x0kI$z
001nE5P$)(APWEhEKC>x0kJ?8001mZkN^R(Knnl>G)#a10kL2T001lu6o3J-fDZrwEDRWc0kMD<001mZ7ytpWfCm5oEW{9i0kMD!
001lu5r6@*0SN#AEDRF>0kQ!P001lu5r6@*0to;BEDRF>0kQ%Q001lu5r6@*1PK5DEDREW0kQ-L001lu6956S1P=fJEDRBV0kQ}Q
001lu6956S2oC@NEDVwW0kQ}P001lu5r6@*5D5SPEDRF>0kRMe001lu5r6@*APE2fEDREW0kR+n001lu6956SAP)clEC3q>0kS|3
001lq69fUWKo0-_EDRF>0kS|3001luk^lj+KnMT;EKC>x0kS|9001mZkN^R)Knnl>G)#a10kU8T001lu5QqV?fC~TsEDVwW0kVJ#
001mZ7ytpXfE54$EQAt(0kVJ=001nE5P$))fEEA%EW{9i0kVJs001lu5r6@-0SN#AEDREW0ki=K001lu6956U0S5p8EDRBV0ki@M
001lu5`Y1;0tf&AEDRF>0ki@K001lukbnWS0to;BEDRBV0ki}J001lu5`Y1;1PA~CEDREW0kjAR001lu6956U2nPTFEDVwW0kjAf
001lu5`Y1;5C{MOEDRF>0kjYY001lu5`Y1;fCB&kEDRKY0knV#001luk^lj;fCB&kEW{9i0knVu001lmQh))_fCB&kBn%LM0nmUB
001luP=Eo@fC&HqtPD_q0nmU6001lmQh))_fdc>lEDTV90nmX7001lu5`Y2Ff(`%xEDTV90nma8001luP=Eo@gb4ruEDTV90nmsE
001lmQh))_kP`p^EDTV90nm^M001lmQh))_pc4Q9EDREW0nnfh001luP=Eo@pcViCEDRWc0nvaK001luP=Eo^fC>NrEW{9i0nva0
001lu5QqWNfC&HqEDTV90n&g8001n65`Y2HfEEA%EDREW0n~sF001luP=Eo{fC&HqEDRBV0q_9{001lu5`Y2l0SEv9EDRF>0q_9_
001lu5r6^k0to;BEDREW0q_C{001lu695750tWy9EDRBV0q_I~001lu5`Y2l1PA~CEDRBV0q_V3001lu5`Y2l2nYZGEDRF>0q_V1
001lu8UO+C2o?YUEDRBV0q_tB001lu5`Y2l5C{MOEDRF>0q_t9001lu5`Y2lfCB&kEDRKY0q}qe001mZ7ytqAfE54$EW{9i0q}qV
001-$9Do7vf))S(Bmxrv0uWIS001lkQvd=GQ4;_FtO8R20uWIX001Nk5dZ=ZQ3(J5ED93<0uWIb0068C6957bQ5OIJEDBKo0uWIb
001lkQvd=GQxgCHED93<0uWOV001lsQ2+uEQwab7ECN#i0uWUb001ls6957bRSp0EECN#i0uWgf001ls6957bSq}gJBmxrv0uXQx
001lkQvd=Ga1#IktO8R20uXQy001lsQ2+uEa2EgoECN#i0uXT%001ls6957baTfppECN#i0uXW&001ls6957bau)yqEDBKo0uXWu
001lkQvd=GbQ1soED93<0uXc#001lkQvd=GcoP5sED93<0uXo?001NiL4X1gFbn_yEC^A60ue9~0068AQGfywFcAO%Bnm+Q0ue9=
001lsL4X1gFc$y-tO`MZ0ueA5001lsQ2+uFFc$y-EC^A60ueD0001lsL4X1gF&6*;EC^A60ueF;001lsL4X1gG7A6zEC^A60ueM3
001lsL4X1gGz<U$EQmpX0ueM3001lqQGfywI1vB<EDAw@0ueY1001lsQ2+uFI0pa#ED8}o0ufOc001nC5dZ=aQ4RnAEDA*c0ugW*
001nCK>z{~a0~zdBmxrv0uo>j001lkQvd=IU=siUtO8R20uo>o001lsQ2+uGU>5)YECN#i0uo^p001ls6957dVHW@ZECN#i0uo{q
001ls6957dViy1aEDBKo0uo{g001lkQvd=IWD@`YED93<0up2o001n46957dWD@`YECN#i0upEw001ls6957dXcqtgECN#i0upc&
001Nc6957fP!0e9ECN#i0u)da00684Qvd=KP!j+EBnlA#0u)dQ001ls6957fP!0e9tO^qV0u)dW001lsQ2+uIP!|9IECN#i0u)gb
001ls6957fQ5OIJEQ%2T0u)gW001lkQvd=KQWF3GED93<0u)jX001lsQ2+uIQWpRKEQ%2T0u)jX001lkQvd=KR1*LIED93<0u)pZ
001lkQvd=KSQ7vMED93<0u)#e001lkQvd=Ka1#IkED93<0u*o#001NiL4X1k5DWkSEC^A60u>Mu0068AQGfy!5D@?XBnm+Q0u>Mk
001lsL4X1k5ElRdtO`MZ0u>Mp001lsQ2+uJ5ElRdEC^A60u>Pv001lsL4X1k5extTEC^A60u>Sw001lsL4X1k5)1$UEC^A60u>Yx
001lsL4X1k6bt|WEC^A60u>k$001lsL4X1k7z_XaEDBKo0u>ks001lqQGfy!FcAO%EDAw@0u?X}001nAL4X1kFbn_yED8}o0u@jf
001lsMF0X7Py+w}EQ&z@0u@jU001Nc6957hKn?%^ECN#i0v13M00684Qvd=MKmz~(BnlA#0v13C001ls6957hKn?%^tO^qV0v13I
001lsQ2+uKKo<Z2ECN#i0v167001ls6957hK^Fi3EQ%2T0v16I001lkQvd=MLIVH*ED93<0v19J001lsQ2+uKLKgr4EQ%2T0v19J
001lkQvd=ML=yl2ED93<0v1FL001n46957hL=yl2ECN#i0v1RE001ls6957hNDlx2ECN#i0v1pc001ls6957hP!9kAECN#i0v2Ec
001ls6957hU=9EPEC^A60v7=h001lqQGfy$0ucZJEDAw@0v7@d001lqQGfy$1Q7rLEDAw@0v7}f001lqQGfy$2oV4PEDAw@0v8Aj
001lqQGfy$5D@?XEDAw@0v8Yr001lqQGfy$AQ1onEQmpX0v8|=001ls5kLYLKo$T1BmlvH0vQkl001li!H@zO5DEYQtN_7~0vQkr
001NcL4X1o5DNeRtO7}Z0vQkl001lsL4X1o5ElRdEC9if0vQns001lkNq_<w5fcCaEC@vi0vQnm001lsL4X1o5f=aeEQ&z@0vQns
001li!H@zO5()qSEC9uT0vQq!001lkNq_<w5)%LbEC@vi0vQqn001lsL4X1o5(@wTEQ&z@0vQqt05&WD!GHo86bb+UEC9uT0vQwp
0B9@%L4X1o6aoMMECNY@0vQw&001lsL4X1o6b}FZEQ&z@0vQws001li#ef1C7!v>hECNY@0vQ++001lsL4X1o7z+RZEJ#6s0vQ+)
001nCK>z|77zzLYEC9if0vRw0001li#ef1CFa!VqECNY@0vRv_001lqK^OuVFbV(wEC@l60vRw0001lqMF;{JFa!VqEDAw@0vRwD
001nAL4X1oFcAO%EQ&z@0vRw0001lqK^OuVPznG5EC@vi0vS*Q001lsMF0XBP!|9IEQ&z@0vS*W001li!H@zQ0R;d6ECNY@0viDY
001lqK^OuX0S5p8ED1q?0viDl001lsL4X1q0Sf>CEC9if0viGa001lkNq_<y0uulLEC@ju0viGc001lrL4X1q0t)~DEDAw@0viGg
001li!H@zQ1O)&9ECNB00viMq001lkNq_<y1Q!4RED1q?0viMi001lsL4X1q1Q!4RECNY@0viYu001lrL4X1q2oV4PEDAw@0viYm
001li!H@zQAO!#bECNY@0vjL?001lqK^OuXAQ%7uEC@l60vjL;001lrL4X1qAPWEhEDAw@0vjL~001nAL4X1qAQ1onEC@ju0vkXN
001li!H@zS5DEYQEC9uT0v!+&001lqK^OuZ5DEYQEC@vi0v!+p001lsL4X1s5Dx$VEQ&z@0v!+s001lq69fV@L=gZ0ED93<0yIPy
001lk5exz}ND}}6ECLZo0yIbh001ls5kLYoNC^M{ED93<0yIbu001lk5exz}PzC@1ECLZo0yIzp001ls5kLYoPzeA4ED93<0yIz$
001lk5exz~Kn4H+ED8}o0yRJv001lk5ex!0Kn4H+ECLe*0yaPd001ls5kLYqKnVZ<ED93<0yaPo001n46957>Kmq^&ECN#i0#rd0
001ls6958KK@I=_EQ%2T0#rc>001lkQvd=~LK6T0ED93<0#rf{001nC5dZ>ILJj}`ECN#i0#rm4001ls6958KL=XS~ECN#i0#ry7
001ls6958KNDcr1ECN#i0#r~F001ls6958KP!0e9EC^A60#zUp001lsL4X2PAPfKiEQmpX0#zUp001ls5kLY}Kmz~(Bmxrv0+3J+
001NcQ2+vvP!j+EECN#i0+3J>00684Qvd>xP!j+EEDBKo0+3J_0068CQ2+vvPzeA4ECN#i0+3M?001ls6958`Q4RnAEDBKo0+3M&
001lkQvd>xQWF3GED93<0+3P;001lsQ2+vvQV9S6ECN#i0+3V_001ls6958`R1N?DEDBKo0+3V*001lkQvd>xSQ7vMED93<0+3h^
001lsQ2+vvSP1|CECN#i0+4VI001ls6958`a1H<fEDBKo0+4VC001NkK>z}g5C;GNEDAw@0+A3G0068CL4X305ElRdEDBKo0+A3G
001nCK>z}g5DWkStcpPZ0+A2|001lqQGf!G5fK0YEDAw@0+A6H001nCK>z}g5d;7LEDAw@0+A96001nCK>z}g5(EGMEDAw@0+AF9
001nCK>z}g6bb+UEDAw@0+ARC001lsQ2+vw7zY3VEQ&z@0+ARD001lqQGf!GFarPpEDAvg0+BEb001lsL4X30Fc$y-EQ&z@0+BET
001ls5kLZwPy+w}EDA*c0+CP`001lsQ2+vwP!|9IEP_!00+CP?001nCK>z}gPz(S7Bmxrv0+K)u001NcQ2+vxKobA}ECN#i0+K)z
00684Qvd>zKobA}EDBKo0+K)%0068CQ2+vxKnVZ<ECN#i0+K-!001ls6958|K@I=_EDBKo0+K-q001lkQvd>zLK6T0ED93<0+K=w
001lsQ2+vxLJ0r>ECN#i0+K`%001ls6958|L=FG|EDBKo0+K`t001n46958|L=yl2ECN#i0+L7r001ls6958|NDcr1EDBKo0+L7x
001lkQvd>zP!j+EEDBKo0+LV(001lkQvd>zU=siUED93<0+L`3001lsQ2+vxU<m*KEC^A60+Rs|001lsK?nkq0So{DEQmpX0+Rs|
001nCK>z}i0R#X5EC^A60+Rv}001lsK?nkq0t^5EEDAw@0+Rw4001nAL4X320ucZJEQ&z@0+Rv+001lqQGf!I1Q7rLEDAw@0+R#_
001nAL4X321Q7rLEQ&z@0+R#;001lqQGf!I2oV4PEDAvg0+R>~001lsL4X322p0eVEDBKo0+R>_001nAL4X322oV4PEQ&z@0+R>~
001lqQGf!I5D@?XEDAvg0+SF7001lsL4X325ElRdEQmpX0+SFC001nCK>z}i5Ci}KEC^A60+S#E001lsK?nkqAOrvaEQmpX0+S#S
001ls5kLZyKmz~(EDAvg0+T=t001lsQ2+vyKo<Z2EP_!00+T=!001nCK>z}iKnwr?ECN#i0+dh_001ls6958~P!0e9EDBKo0+dh*
001lqQGf!K5D@?XEDAvg0+kR7001lsL4X345DWkSEQ&z@0+kR1001lkQvd>%Kmz~(ED93<0+v7y001lsQ2+v#KnVZ<EC9if0+|pB
001li#ef2t5Ci}KECNY@0+|p5001lqK^Ov=5DEYQEC@k}0+|p5001lqMF;|!5Ci}KEDAvg0+|pB001lsL4X385ElRdEQ&z@0+|pB
001li8w3KdNCp4^EC3ro0<cH|001ls6959RNDlx2EC3q>0<cg9001lk5ex#bPzC@1ECLZo0<cg5001ls6959RP!9kAECLY>0<k~_
001ls5kLa5Ko<Z2EC3q>0<u5`001li8$bfGKmq^&ECLY>0<u5`001lk69fXXKobA}ED93<0<u5`001n46959TKobA}ECN#i0?<Jd
001ls6959xK@I=_EDBKo0?<JT001lkQvd?cLK6T0ED93<0?<MZ001lsQ2+waLJ0r>ECN#i0?<SP001ls6959xL=FG|ECN#i0?<ek
001ls6959xNDcr1EDBKo0?<ea001lkQvd?cP!j+EEDBKo0?<$i001lqQGf!`0TBQIEDAw@0?`2r001lsQ2+wb0S5p8EQ&z@0?`2k
001lsL4X3$0t)~DEDBKo0?`5o001nCK>z~L0t^5EEDAw@0?`Bv001lsQ2+wb1P1^BEQ&z@0?`Bv001lsL4X3$2n+xKEDBKo0?`N;
001nCK>z~L2n+xKEDAw@0?`l`001lsQ2+wb5C;GNEQ&z@0?`l*001lqQGf!`AOipZEDAvg0?{A}001lsL4X3$APWEhEDBKo0?{A`
001ls5kLabKo<Z2EDA*c0?|Mi001lsQ2+wbKo<Z2EP_!00?|Md001nCK>z~LKnef=ECN#i0@6Se001lsQ2+wcKnVZ<EP@jN0@6Se
001lqMF;~C5d;7LEC^A60}&Am001lqMF;~C5(EGMEC|7X0}&Df001lqK^OxOFbV(wEC@vi0}(I;001lqQGf#xFbe<xEQmpX0}(I-
001lqK^OxOPznG5EC@vi0})UJ0BAIbL;wR3R2BdLEC@ju0}}xU001lqQGf#z0Sf>CEC|7X0}}xY001lqK^OxQ0tWy9EC^A60}}!d
001lq!GHr30t5g6EC@ju0}~(x001lqQGf#zAPWEhEC|7X0}~(&001nAL4X4jAOipZEC@ju0~0_G001lqK^OxS5DEYQEC@vi0~HVi
001lqQGf##5CZ@JEC9if0~rts001li#ef4D5EB3ZEC3q>129kq001li8$bgvPyzq|EC>?>129kz001li8w3L~KobA}EC3ro12RAY
001lq69fY?Ko0-_EC@ju15p7A001lqMF;~?0R#X5EC^A615p7B001lqK^Oy30tx^CEC@vi15pA5001lqK^Oy3APN8gEC^A615qFY
001nAL4X5MAP@imEC@ju15rQ<001lqMF;~?Knnl>EC@ju1CbC4001lqMF<0t5Ci}KECf=31dxaW001NgLI4Dih!g+-ECxb=1dxaf
001loQUC;yhz9@wtOim51dxaa0068OLI4DihzkG!ECf=31dxdX001loLVyI2i3|V$ECx~l1dxdb001l&LI4Dii3<P#ECxb=1dxgg
001loQUC;yiU$AyEEYll1dxgg001loQUC;yj0XS!EEYll1dxmj001NeQUC;ykOu$&ECf=31dxys00686Qh)@IkOKe!BnCnN1dxyn
001loQUC;ykOu$&tOim51dxyi001lmQh)@Ikpln#ECx~l1dx#j001lmQh)@Ik^=w$ECx~l1dx&n001lmQh)@Ilo0>`ECx~l1dx;m
001lmQh)@Im;(R+ECx~l1dx~q001l&LI4Dim<s>^ECfP`1dyl<001loLVyI2r~?21ECx~l1dyl~001l&LI4Dis0#o9EQCUU1dyl<
001lmLWl&AuowUUECf=31dy-;001loQUC;yuonOTEQCUU1dy-{001loLI?zq$QJ+rECx~l1dzxJ001l&LI4Di$QS?sErdb<1dzxS
001loLI?zq&<p?oECx~l1dz}d001nMLI4Di&=>#!Erdb<1dz}a001lmQh)@Kfe`=zECx~l1d@RV001lmQh)@Kf)M}!ECx~l1d@UW
001lmQh)@Kgb@G$ECx~l1d@aY001lmQh)@KhywrsECxb=1d@mh001loQUC;!hz9@wEEYll1d@mh001lmQh)@KkOKe!ECx~l1d@;k
001lmLWl&CpbP*2ECf=31d^Z;001loQUC;!pceoDEQCUU1d^Z(001loLI?zsz!v}jECx~l1d_lD001l$QUC;!zykmPErdb<1d_lM
001loLI?zuhz9@wECxb=1eAyj001loQUC;$hz9@wEEYll1eAyj001lmQh)@MkOKe!ECxad1eA~m001loQUC;$kOu$&ECxY<1eg#D
001NgLI4Dq5C;GNECxb=1eg#A00688LVyIA5EK9aECxdW1eg#900688LjVMr5CZ@JECxY<1eg&E001loLVyIA5flIbECxdW1eg&6
001loLVyIA5(@wTECxdW1eg*B001loNPq;G5(oeQECxb=1eg>E001loLjVMr6bAqRECxb=1eh2U001loLjVMr7zY3VEe1jW1eh2D
001loLVyIAFarPpECxdW1eh=f001loLjVMrPy+w}EEd231ej15001opLI4DqPzV43Ee5~<1ej11001loLI?zyhzkG!EEYll1ek~e
001l&zyJi8h!_9>ECxad1elNq001loLVyIAkQ4v_EEd231elN*001nMLI4DqkQe{}Ee1jW1elN%001opzyJi8kQ4v_Erdb<1elN%
001li!H@)*0SW*BECNB01epO3001loL4X9A0Sf>CEe1e<1epO3001opzyJi90SEv9EC9if1epQ~001lkL68KQ0tx^CECfk_1epR6
001loL4X9A0t)~DECxt`1epQ|001opK!6090t)~DEe5~<1epQ|001li!H@)*1Q7rLECNB01epX2001opK!6091PcHFEe5~<1epW~
001opK!6092oC@NEe5~<1epj3001li!H@)*5DEYQECNB01ep*E001lkNq_{I5E1|YEe1e<1ep*I001opzyJi95C{MOEC9if1eqWT
001l&zyJi9Ko|f3Ee5~<1erh?001loL4X9B0Sf>CECxb=1eyT|001loLVyIC0tf&AECxt`1eyW}001loLVyIC1PA~CECxdW1eyc~
001loLVyIC2nYZGECxdW1eyp3001opLI4Ds2m=5BECxb=1ey>C001loLjVMt5C;GNECxb=1ezcS001loLjVMtAO`>dEEd231e!n?
001n8LI4DsKo0-_Ee1jW1e!n;001opzyJiAKokG~ECxb=1e_2Q001loLjVMv5C;GNEC3q>1h7a3001li8$blGND=@5ECLe<1h7a3
001lm69fdXND%-4EC3q>1h7yB001lk69@#bPzwM6ECv$*1h7yG001lmQh)@|fe`=zECxb=1kiyC001loQUC<dffoP(ECxb=1ki#D
001loQUC<df(iftECxb=1ki*G001loQUC<dga-fsECxb=1ki{K001loQUC<dh!+3=EEYll1ki{J001lmQh)@|kOKe!ECxb=1kjKR
001loQUC<dkQV>|ECxY<1lR!!001loLVyI=0SEv9ECxb=1lR%y001loNPq;`0vG@QECxb=1lR-!001loLjVNW1P1^BECxb=1lR}&
001loLjVNW2nPTFECxyd1lR}%001opLI4EV2nYZGECxb=1lSM=001loLjVNW5C;H=Gy(yT1lb@J001lmQh)^zkpln#ECmvP1rU)J
001lmLWl(runYhIECf)91rV?a001lmQh)^zumb=9ECmvP1rV?n001l$LVyJjumb=9ER;e31rX2}001q7LI4F2&=ddwECf)91reYS
001lnP=EyypcViCEQJt&1reYX001n+5P$^{pb!86ECdpW1rneT001lmP>2N*pa}o~ECf=31rneG001ln5`YB~pcViCECf=31r(42
001ln5`YC1kOKe!EEGb31r(42001lmQh)`JkOKe!EEGb31(1*z001lk5ex>9PzC@1ECLZo29Qt!001lkQ9uTePyzq|ECN#i29Qt>
001loL4XLE0Sf>CEDAw@2$=yA001luL4XLE0RjL3ECxY<2$=#C001lsL4XLE0ulfKEDS+_2$=#A001lsL4XLE1Q!4REDS+_2$=*C
001lsL4XLE2nzrJEDS+_2$={G001lsL4XLE5ElRdEDS+_2$>K9001lmQh*7Nfdc>lEDTV936Oya001lu5`YPif(`%xEDTV936O#b
001luP=E=Lgb4ruEDTV936O{h001lmQh*7NkOKe!EDTV936PKp001lu5r7G>0SN#AEDREW39tbO001lu695UY0S5p8EDRBV39teQ
001lu5`YP?0tf&AEDRF>39teO001mZ5P%7=0s{a5EDRBV39tkS001lu5`YP?1PA~CEDREW39twV001lu695UY2nPTFEDVwW39twV
001lu5`YP?5C{MOEDRF>39t|c001ls695X3K@I=_EDBKo3Xnkw001nC5daF1K@I=_ED93<3Xnn$001lsQ2+{%LJ0r>EQ%2T3Xnn$
001lkQveE(L=yl2ED93<3Xnt&001lsQ2+{%L<s-@EQ%2T3Xntw001ls695X3NDcr1EDBKo3Xn(%001nC5daF1NDcr1ECN#i3Xo6}
001lsQ2+{%PzeA4EQ%2T3Xo6^001NcL4Xh$5EB3ZBm_Z#5E&2=001loL4Xh$5DNeRtOh}V5E&2)001lsL4Xh$5DNeRtO`MZ5E&2)
001loL4Xh$5eonSEDAw@5E&5*001loL4Xh$5)%LbEDAw@5E&8@001lkNq`U;6cPXcECxY<5E&E`001lsL4Xh$6bk?VECxY<5E&Q?
001lsL4Xh$7z+RZEDS+_5E&Q|001loL4Xh$Fbe<xEDAw@5E(ER001loL4Xh&0Sf>CECxY<5E}v$001lsL4Xh&0ulfKECNY@5E}#%
001loL4Xh&1PcHFEDAw@5E}#%001loL4Xh&2onGREDAw@5E}>!001luL4Xh&2oV4PECxY<5E~E^001lsL4Xh&5ElRdECxY<5FHQ`
001lsL4Xh)5E1|YEDS+_5FHQ^001Nc695n}P!a$DBm@%x5HL^^001lo695n}P!j+EtOgSR5HL^^001lu695n}P!RwCtPB$X5HL^=
001lo695n}Q4;_FED93<5HL{>001lu695n}Q4atBECv$*5HL~`001lu695n}QV#$CECv$*5HM5|001ls695n}R2KjMEDRF>5HM5^
001lo695n}SQ7vMEDRF>5HMH|001lo695n}a1#IkEDRF>5HN5L001lo695o0K^Fi3ED93<5Hdj*001lo695o0LKgr4ED93<5Hdm%
001lo695o0L=yl2ED93<5Hds(001lo695o0NEZMAED93<5Hd&-001lu695o0NDlx2ECLe*5He5_001lo695o0P!|9IED93<5He5(
001lu695o0P!9kAECv$*5HwH|001ls695o2P!a$DEDRF>5HwH^001loL4Xih0Sf>CEDAw@5Lp2d001loL4Xih0uulLEDAw@5Lp5e
001luL4Xih0ucZJECxY<5LpBh001lsL4Xih1QGxMECxY<5LpNd001lsL4Xih2p0eVEDS+_5LpNj001loL4Xih5DNeREDAw@5Lpls
001lo695o!K@$J~ED93<5O6^f001lo695o!LK6T0ECv$*5O72j001ls695o!L=pf1ECv$*5O7En001lu695o!ND%-4ECv$*5O7cv
001lu695o!P!RwCECxY<5Sb7Q001lsL4XjM5ElRdEDS+_5Sb7H001lo695pfP!j+EED93<5U@}W001lu695pfP!9kAEDTV95fP9B
001n75P%U8kQM*{ECdpW5fXtG001lmQh*T>fdc>lEDREW5fXs|001lm5{MBJf(`%xEDREW5fXwA001lu5`YmBgbn}zEDTV95fX$6
001n65`YmBgb)A#ECdpW5fX?H001lu5`YmBhz|e&EDTV95fX?7001lm5{MBJkPZL<ECf=35fYFC001lu5`YmBkOKe!EDRBV5ik)6
001lu5`Ymf5f1<WEDRF>5ik)C001lu5r7de5(xkREDREW5ik-D001lu695q~5)S|XEDRBV5ik@9001lu5`Ymf6bJwSEDRF>5ik@7
001lu5`YmfFarPpEDRF>5il?h001lq69f@3PzwM6EDRF>5in2>001luk^m7fPzV43EKCvr5in2>001lm5{MBnkPrX>EDREW5ipPs
001mZ7yuD4kO=?)EDRBV5i$W7001lu5`Ymh0RsR4EDRBV5i$Z8001lu5`Ymh0s{a5EDRBV5i$e`001lu5`Ymh1PA~CECf=35m12x
001lu5`Ym<feruwEDTV95m12%001lu5`Ym<f(`%xEDTV95m15&001lu5`Ym<gbn}zEDTV95m1B+001lu5r7eJ0to;BEDREW5pV(s
001lkQvebYK@$J~ED93<5)wfd001lsQ2-JWK^Fi3ECN#i5)wia001lsQ2-JWLKgr4ECN#i5)woc001ls695ttL=FG|EDBKo5)wog
001n4695ttL=yl2ECN#i5)w!Q001ls695ttNDcr1EDBKo5)w!k001lkQvebYP!j+EEDBKo5)x1s001lqQGgN?0Sf>CEDBKo5)%Oz
001lqQGgN?0ucZJEDAw@5)%R!001lsQ2-JX0v7-PEC3q>5-?B(001ls695u0P!9kAECN#i5>P@D001ls695uWLJj}`EDBKo5>P@H
001lkQvecBL=*r3ED93<5>P}A001lsQ2-K9L>B-6ECN#i5>QAJ001ls695uWNDcr1EDBKo5>QAN001lkQvecBP!j+EEDBKo5>QYV
001lqQGgOr0TBQIEDAw@5>WvQ001lsQ2-KA0S5p8EDAw@5>WyR001lsQ2-KA0tWy9ECf=36cCXE001loQUDYXkp}<(ECf=36cCaF
001loQUDYXk_rF-ECfP`6cDf&001lmQh*c?umb=9ECx~l6cDfn001loQUDYX&=&vzER;e36cErD001lmQh*c^fe`=zECx~l6cT|4
001lmQh*c^f)M}!ECx~l6cU05001loL4Xt*0Sf>CECxb=6dC~w001loL4Xt*0t)~DECxb=6dD2x001loNPrX?0u%rMECxb=6i|W-
001loQUDZCf(iftECxY<6j%Za001loLVy%l0tf&AECxt`6j%Zn001loL4X(;0uulLEDAw@7#RW*001lo695=6LK6T0EDREW84!XG
001luP=FZ_f(ZZsEDaEV84!XG001lsQ2-hcLKgr4EQ%2T8W2Jb
"""


if __name__ == '__main__':
    main()
