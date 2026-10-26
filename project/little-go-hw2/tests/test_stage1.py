import pathlib
import json
import random
import subprocess
import sys
import tempfile
import unittest
from test_rules import ROOT, player

sys.path.insert(0,str(ROOT/'tools'))
import benchmark
try:
    import stage1
except ImportError:
    stage1=None


class Stage1(unittest.TestCase):
    def test_report_uses_actual_candidate_budget(self):
        path=ROOT/'experiments'/'candidate_space5.py'
        self.assertEqual(stage1.source_limits(path),(5.0,6.0))
        self.assertEqual(stage1.source_limits(ROOT/'experiments'/'platform_pvs3.py'),(3.0,4.0))

    def test_progress_keeps_complete_games_and_source_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            path=pathlib.Path(folder)/'progress.json'
            source=pathlib.Path(folder)/'agent.py'
            source.write_text('original agent')
            games=[{'agent_color':1,'winner':1,'reason':'normal','moves':[{'action':12}]},
                   {'agent_color':2,'winner':1,'reason':'timeout','moves':[{'error':'timeout'}]}]
            stage1.save_progress(path,games,'test-opponent',20,762000,source,1.25)
            data=json.loads(path.read_text())
            self.assertEqual(data['results'],games)
            self.assertEqual((data['completed'],data['planned'],data['wins']),(2,20,1))
            self.assertEqual(data['seed_start'],762000)
            self.assertEqual(len(data['source_sha256']),64)
            self.assertEqual(data['by_color']['2'],{'games':1,'wins':0})
            self.assertFalse(path.with_suffix('.tmp').exists())

    def test_failed_subprocess_time_is_not_omitted_from_report(self):
        def failing(state,rng):
            error=subprocess.TimeoutExpired('test-process',10)
            error.child_cpu=9.0
            raise error
        report=benchmark.tournament('random',games=2,policy_override=failing)
        self.assertEqual(report['failures']['timeout'],2)
        self.assertGreaterEqual(report['max_cpu'],9.0)

    def test_child_cpu_is_included_in_timeout(self):
        def over_budget(state,rng):
            return 0, {'child_cpu':11.0}
        other=lambda state,rng:(-1,{})
        result=benchmark.run_game(over_budget,other)
        self.assertEqual((result['winner'],result['reason']),(2,'timeout'))

    @unittest.skipUnless(sys.platform.startswith('linux'),'Unix child CPU accounting')
    def test_actual_submission_process_returns_legal_action_and_cpu(self):
        self.assertIsNotNone(stage1,'Fresh-process Stage 1 harness missing')
        with tempfile.TemporaryDirectory() as folder:
            policy=stage1.submission_policy(pathlib.Path(folder),cpu_override=0)
            state={'previous':player.EMPTY,'board':player.EMPTY,'color':1,
                   'move_count':0,'last_pass':False}
            action,stats=policy(state,random.Random(0))
            self.assertIsNotNone(player.play_move(player.EMPTY,player.EMPTY,1,action))
            self.assertGreater(stats['child_cpu'],0)


if __name__=='__main__':
    unittest.main()
