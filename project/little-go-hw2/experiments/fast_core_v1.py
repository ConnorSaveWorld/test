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
    own, enemy = (black, white) if color == 1 else (white, black)
    groups = _bit_groups(black, white)
    own_groups = [(stones, libs) for side, stones, libs in groups if side == color]
    enemy_groups = [(stones, libs) for side, stones, libs in groups if side != color]
    empty = FULL & ~(own | enemy)
    actions = []
    remaining = empty
    while remaining:
        bit = remaining & -remaining
        remaining ^= bit
        captured = 0
        for stones, libs in enemy_groups:
            if libs == bit:
                captured |= stones
        if not captured and not (_adj(bit) & empty):
            if not any(libs & bit and libs & ~bit for _, libs in own_groups):
                continue
        new_own, new_enemy = own | bit, enemy & ~captured
        b, w = (new_own, new_enemy) if color == 1 else (new_enemy, new_own)
        if b == pb and w == pw:
            continue
        actions.append((bit.bit_length() - 1, b, w))
    actions.append((PASS, black, white))
    return actions


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
    return (8.0 * _bit_score(black, white) + 0.6 * liberties - 6.0 * atari
            - 1.2 * weak + 0.45 * influence)


def _bit_evaluate(black, white, color):
    value = _bit_static(black, white)
    return value if color == 1 else -value


def _bit_order(pb, pw, black, white, color, preferred=None, passed=False,
               ordering=True):
    actions = _bit_moves(pb, pw, black, white, color)
    if not ordering:
        return actions
    groups = _bit_groups(black, white)
    own_groups = [(stones, libs) for side, stones, libs in groups if side == color]
    enemy = white if color == 1 else black
    def rank(item):
        p, b, w = item
        if p == PASS:
            priority = 2000 if passed and _bit_terminal(black, white, color) > 0 else -1000
        else:
            bit = 1 << p
            members = bit
            for stones, libs in own_groups:
                if libs & bit:
                    members |= stones
            liberties = _adj(members) & FULL & ~(b | w)
            liberty_count = _count(liberties)
            rescued = sum(_count(stones) for stones, libs in own_groups
                          if _count(libs) == 1 and stones & members and liberty_count > 1)
            captured = _count(enemy) - _count(w if color == 1 else b)
            priority = (100 * captured + 40 * rescued + 2 * min(liberty_count, 5)
                        + len(NEIGHBORS[p])
                        - (30 * _count(members) if liberty_count == 1 else 0))
        return (p == preferred, priority)
    actions.sort(key=rank, reverse=True)
    return actions


def search(previous, board, color, move_count, last_pass, cpu_seconds=3.0,
           wall_seconds=4.0, max_depth=None, use_table=True, use_ordering=True,
           max_nodes=None, evaluator=None):
    if evaluator is not None:
        return _tuple_search(previous, board, color, move_count, last_pass,
            cpu_seconds, wall_seconds, max_depth, use_table, use_ordering, max_nodes, evaluator)
    started_cpu = time.process_time()
    started_wall = time.monotonic()
    cpu_deadline = started_cpu + max(0.0, cpu_seconds)
    wall_deadline = started_wall + max(0.0, wall_seconds)
    stats = {'nodes': 0, 'cache_hits': 0, 'completed_depth': 0,
             'value': None, 'interrupted': False, 'engine': 'bitboards'}
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
        for p, nb, nw in actions:
            if ((p == PASS and passed) or
                    (next_count is not None and next_count >= 24)):
                candidate = _bit_terminal(nb, nw, side)
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
