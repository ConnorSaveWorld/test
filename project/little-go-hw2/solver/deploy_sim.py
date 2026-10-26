"""Fresh-process White games against the Championship model through the real
input.txt / output.txt / state.json protocol. Black is the model (C solver).
Checks legality with the course GO class and records per-move time."""
import sys, os, json, time, random, subprocess, tempfile, pathlib, resource
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'reference'))
from host import GO
from lg import *
from sim_model import black_move, empirical
os.environ.pop("LITTLE_GO_CPU_SECONDS", None)

def write_input(path, color, prev, board):
    rows = [''.join(str(x) for x in prev[r*5:r*5+5]) for r in range(5)] + \
           [''.join(str(x) for x in board[r*5:r*5+5]) for r in range(5)]
    path.write_text('%d\n' % color + '\n'.join(rows) + '\n')

def main():
    src = pathlib.Path(sys.argv[1]).resolve()
    n = int(sys.argv[2]); rng = random.Random(int(sys.argv[3]))
    py = sys.argv[4] if len(sys.argv) > 4 else sys.executable
    init(24); emp = empirical()
    tot = 0; wins = 0; worst = 0.0; worst_cpu = 0.0; game_cpu = []
    for gi in range(n):
        with tempfile.TemporaryDirectory() as d:
            work = pathlib.Path(d)
            (work / 'my_player3.py').write_bytes(src.read_bytes())
            s = State(); prev = (0,) * 25; seq = []; cpu_sum = 0.0
            while s.count < 24:
                if s.side == 1:
                    m, how = black_move(s, emp, rng)
                else:
                    board = tuple(s.board())
                    write_input(work / 'input.txt', 2, prev, board)
                    r0 = resource.getrusage(resource.RUSAGE_CHILDREN)
                    t = time.monotonic()
                    cp = subprocess.run([py, 'my_player3.py'], cwd=d, capture_output=True, timeout=10)
                    dt = time.monotonic() - t
                    r1 = resource.getrusage(resource.RUSAGE_CHILDREN)
                    cpu = (r1.ru_utime + r1.ru_stime) - (r0.ru_utime + r0.ru_stime)
                    cpu_sum += cpu
                    worst = max(worst, dt); worst_cpu = max(worst_cpu, cpu)
                    assert cp.returncode == 0, cp.stderr.decode()
                    out = (work / 'output.txt').read_text().strip()
                    if out == 'PASS': m = -1
                    else:
                        r, c = map(int, out.split(',')); m = 5 * r + c
                        go = GO(5)
                        go.set_board(2, [list(prev[i:i+5]) for i in range(0, 25, 5)], [list(board[i:i+5]) for i in range(0, 25, 5)])
                        assert go.valid_place_check(r, c, 2, test_check=True), ('GO rejects', out)
                    st = json.loads((work / 'state.json').read_text()).get('last_stats', {})
                    how = st.get('policy', 'search')
                ns = s.play(m); assert ns is not None
                seq.append('%d%s' % (m, {'book': 'b', 'exact-win': 'x', 'error': '!', 'emp': 'e'}.get(how, '')))
                end = (m == -1 and s.lp)
                prev = tuple(s.board()); s = ns
                if end: break
            b = s.board(); w = 1 if b.count(1) - b.count(2) >= 3 else 2
            wins += (w == 2); game_cpu.append(cpu_sum)
            print('game %d winner %d cpu %.1fs %s' % (gi, w, cpu_sum, ','.join(seq)), flush=True)
    print('white wins %d/%d  worst move wall %.2fs cpu %.2fs  max game cpu %.1fs' % (wins, n, worst, worst_cpu, max(game_cpu)))

main()
