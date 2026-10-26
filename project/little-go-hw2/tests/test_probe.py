import pathlib
import sys
import tempfile
import unittest
from test_rules import ROOT

sys.path.insert(0,str(ROOT/'tools'))
import probe_deployment


class Probe(unittest.TestCase):
    def test_illegal_output_cannot_be_counted_as_normal_loss(self):
        self.assertTrue(callable(getattr(probe_deployment,'validate_action',None)),
                        'Independent output validation is not implemented')
        with tempfile.TemporaryDirectory() as folder:
            path=pathlib.Path(folder)/'input.txt'
            path.write_text('1\n'+'00000\n'*10)
            with self.assertRaises(ValueError):
                probe_deployment.validate_action('5,0',str(path))
            self.assertEqual(probe_deployment.validate_action('0,0',str(path)),0)
            path.write_text('2\n'+'00000\n'*5+'10000\n'+'00000\n'*4)
            with self.assertRaises(ValueError):
                probe_deployment.validate_action('0,0',str(path))


if __name__=='__main__':
    unittest.main()
