import json, sys
from lg import *
init(25)
d = json.load(open(sys.argv[1]))
for r in d['trap']:
    if r['winner'] == 2: continue
    s = State(); notes = []
    for m in r['seq']:
        if s.side == 2 and 7 <= s.count < 24:
            w, _ = solve(s)
            notes.append('%d%s' % (m, '*' if w else ''))
        else:
            notes.append(str(m))
        s = s.play(m)
    print(r['profile'], r['nodes'], r['forced'], ' '.join(notes), flush=True)
