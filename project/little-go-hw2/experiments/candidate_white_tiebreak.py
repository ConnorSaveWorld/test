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
                or type(count) is not int or not 0 <= count <= 21
                or count % 2 != color - 1
                or type(cpu) not in (int, float) or not 0 <= cpu <= 90
                or not math.isfinite(cpu)):
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
    # Leave 8 seconds below the 90-second game allowance for host accounting.
    budget = turn_budget(history['cpu_used'], count, color)
    # Development override can reduce time, but cannot increase deployment caps.
    try:
        override = float(os.environ.get('LITTLE_GO_CPU_SECONDS', budget))
        if math.isfinite(override):
            budget = min(budget, max(0.0, override))
    except ValueError:
        pass
    last_pass = previous == board and (board != EMPTY or color == 2)
    action, stats = search(previous, board, color, count, last_pass,
                           cpu_seconds=budget, wall_seconds=min(9.5, budget + 1.0))
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
    tie_seed = 0
    if color == 2:
        tie_seed = ((black * 0x9E3779B1) ^ (white * 0x85EBCA77)
                    ^ (color * 0xC2B2AE3D)) % 25
    actions.sort(key=lambda item: (item[3], -((item[0] + tie_seed) % 25)),
                 reverse=True)
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
           max_nodes=None, evaluator=None, use_pvs=True):
    if evaluator is not None:
        return _tuple_search(previous, board, color, move_count, last_pass,
            cpu_seconds, wall_seconds, max_depth, use_table, use_ordering, max_nodes, evaluator)
    started_cpu = time.process_time()
    started_wall = time.monotonic()
    cpu_deadline = started_cpu + max(0.0, cpu_seconds)
    wall_deadline = started_wall + max(0.0, wall_seconds)
    stats = {'nodes': 0, 'cache_hits': 0, 'completed_depth': 0,
             'value': None, 'interrupted': False, 'engine': 'cached-bitboards',
             'pvs': use_pvs}
    pb, pw = _pack(previous)
    black, white = _pack(board)
    table = {}
    best_action = _bit_order(pb, pw, black, white, color,
                             passed=last_pass, ordering=use_ordering)[0][0]

    def check(force=False):
        if max_nodes is not None and stats['nodes'] >= max_nodes:
            raise SearchTimeout()
        if force or stats['nodes'] % 128 == 0:
            if time.process_time() >= cpu_deadline or time.monotonic() >= wall_deadline:
                raise SearchTimeout()

    def value(old_b, old_w, b, w, side, count, passed, depth, alpha, beta):
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
        best, point = -float('inf'), actions[0][0]
        next_count = None if count is None else count + 1
        for index, (p, nb, nw) in enumerate(actions):
            if ((p == PASS and passed) or
                    (next_count is not None and next_count >= 24)):
                candidate = _bit_terminal(nb, nw, side)
            else:
                if use_pvs and index > 0 and alpha != -float('inf'):
                    child, _ = value(b, w, nb, nw, 3 - side, next_count,
                                     p == PASS, depth - 1, -alpha - 0.01, -alpha)
                    if alpha < -child < beta:
                        child, _ = value(b, w, nb, nw, 3 - side, next_count,
                                         p == PASS, depth - 1, -beta, -alpha)
                else:
                    child, _ = value(b, w, nb, nw, 3 - side, next_count,
                                     p == PASS, depth - 1, -beta, -alpha)
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
                                   last_pass, depth, -float('inf'), float('inf'))
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


if __name__ == '__main__':
    main()
