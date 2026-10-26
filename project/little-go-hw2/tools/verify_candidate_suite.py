"""Run source-parametric rule/protocol/search tests without replacing delivery."""
import argparse
import hashlib
import importlib.util
import json
import pathlib
import sys
import unittest

ROOT=pathlib.Path(__file__).resolve().parents[1]

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',required=True)
    args=parser.parse_args()
    path=ROOT/args.source
    sys.path.insert(0,str(ROOT/'tests'))
    import test_rules
    spec=importlib.util.spec_from_file_location('source_under_test',str(path))
    agent=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(agent)
    test_rules.PATH=path
    test_rules.player=agent
    import test_protocol
    assert pathlib.Path(test_protocol.player.__file__).resolve()==path.resolve()
    assert test_protocol.PATH==path
    suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromModule(module)
                             for module in (test_rules,test_protocol)])
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    report={'source':args.source,'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'python':sys.version,'tests_run':result.testsRun,'failures':len(result.failures),
            'errors':len(result.errors),'successful':result.wasSuccessful(),
            'scope':'Actual selected source: complete rule, protocol and search suites. Historical candidate-comparison tests remain separate.'}
    (ROOT/'reports'/('suite-'+path.stem+'.json')).write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)
    sys.exit(0 if result.wasSuccessful() else 1)
