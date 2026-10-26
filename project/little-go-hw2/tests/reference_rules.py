"""Independent matrix-style oracle, not using submission helpers."""


def apply(previous, board, color, point):
    if point == -1:
        return tuple(board)
    if type(point) is not int or not 0 <= point < 25 or board[point] != 0:
        return None
    cells = [list(board[r*5:r*5+5]) for r in range(5)]
    cells[point//5][point%5] = color
    def neighbors(r, c):
        return [(a, b) for a, b in ((r-1,c),(r+1,c),(r,c-1),(r,c+1))
                if 0 <= a < 5 and 0 <= b < 5]
    def connected(r, c):
        side = cells[r][c]
        found = set()
        pending = [(r,c)]
        alive = False
        while pending:
            a,b = pending.pop()
            if (a,b) in found:
                continue
            found.add((a,b))
            for x,y in neighbors(a,b):
                if cells[x][y] == 0:
                    alive = True
                elif cells[x][y] == side and (x,y) not in found:
                    pending.append((x,y))
        return found, alive
    visited = set()
    remove = set()
    for r in range(5):
        for c in range(5):
            if cells[r][c] == 3-color and (r,c) not in visited:
                members, alive = connected(r,c)
                visited.update(members)
                if not alive:
                    remove.update(members)
    for r,c in remove:
        cells[r][c] = 0
    if not connected(point//5,point%5)[1]:
        return None
    result = tuple(x for row in cells for x in row)
    return None if result == tuple(previous) else result
