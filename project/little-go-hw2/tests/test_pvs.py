import importlib.util
import unittest
from test_rules import ROOT, player as baseline

path=ROOT/'experiments'/'candidate_pvs.py'
if path.exists():
    spec=importlib.util.spec_from_file_location('candidate_pvs',str(path))
    candidate=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(candidate)
else:candidate=None


class PVS(unittest.TestCase):
    def setUp(self):self.assertIsNotNone(candidate,'Second search candidate missing')

    def test_narrow_window_matches_full_window(self):
        fixtures=[
            ('0112220102021121120210010','0112220102021121122210010',-10001.5),
            ('1111110002200221101220202','1111110002200221101220222',-10001.5),
            ('0110100010221220201021102','0110100010221220201221102',10001.5),
            ('1020201112000120101210122','1020201112000120121210122',10005.5)]
        for p,b,expected in fixtures:
            prev=tuple(map(int,p));board=tuple(map(int,b))
            for pvs in (False,True):
                _,stats=candidate.search(prev,board,1,18,False,cpu_seconds=3,
                                         max_depth=6,use_pvs=pvs)
                self.assertEqual(stats['completed_depth'],6)
                self.assertAlmostEqual(stats['value'],expected)

    def test_cached_candidates_keep_ko_context_separate(self):
        board=[0]*25
        for p in (6,10,12,16):board[p]=2
        for p in (7,13,17):board[p]=1
        prev=tuple(board)
        now=baseline.play_move(baseline.EMPTY,prev,1,11)
        pack=candidate._pack
        self.assertNotIn(12,[p for p,_,_ in candidate._bit_moves(*pack(prev),*pack(now),2)])
        self.assertIn(12,[p for p,_,_ in candidate._bit_moves(0,0,*pack(now),2)])


if __name__=='__main__':unittest.main()
