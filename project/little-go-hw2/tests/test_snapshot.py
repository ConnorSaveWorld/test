import hashlib
import pathlib
import random
import sys
import tempfile
import unittest
from test_rules import ROOT
sys.path.insert(0,str(ROOT/'tools'))
import strong_league
import stage1


class Snapshot(unittest.TestCase):
    def test_import_records_loaded_bytes_after_original_changes(self):
        source=(ROOT/'experiments'/'platform_pvs3.py').read_bytes()
        with tempfile.TemporaryDirectory() as folder:
            path=pathlib.Path(folder)/'source.py'
            path.write_bytes(source)
            module=strong_league.load_player('baseline',str(path))
            path.write_text('raise RuntimeError("edited after import")')
            self.assertEqual(module.__source_sha256__,hashlib.sha256(source).hexdigest())
            self.assertIsNotNone(module.play_move(module.EMPTY,module.EMPTY,1,12))

    def test_subprocess_source_is_frozen_before_original_changes(self):
        source=(ROOT/'experiments'/'platform_pvs3.py').read_bytes()
        with tempfile.TemporaryDirectory() as folder:
            original=pathlib.Path(folder)/'original.py'
            original.write_bytes(source)
            frozen,digest=stage1.snapshot_source(original,pathlib.Path(folder))
            original.write_text('raise RuntimeError("edited before next move")')
            self.assertEqual(frozen.read_bytes(),source)
            self.assertEqual(digest,hashlib.sha256(source).hexdigest())
            self.assertEqual(stage1.source_limits(frozen),(3.0,4.0))
            if sys.platform.startswith('linux'):
                policy=stage1.submission_policy(folder,cpu_override=0,source_path=frozen)
                action,stats=policy({'previous':(0,)*25,'board':(0,)*25,'color':1,
                                     'move_count':0,'last_pass':False},random.Random(0))
                self.assertIn(action,range(25))
                self.assertGreater(stats['child_cpu'],0)


if __name__=='__main__':unittest.main()
