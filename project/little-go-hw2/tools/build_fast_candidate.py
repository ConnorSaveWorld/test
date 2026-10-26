from pathlib import Path
import argparse

ROOT=Path(__file__).resolve().parents[1]
base=(ROOT/'experiments'/'baseline_player3.py').read_text()
prefix=base.split("if __name__ == '__main__':")[0].replace('def search(', 'def _tuple_search(', 1)
core=(ROOT/'experiments'/'fast_core.py').read_text()
parser=argparse.ArgumentParser()
parser.add_argument('--output',default='candidate_fast.py')
args=parser.parse_args()
(ROOT/'experiments'/args.output).write_text(prefix+'\n'+core+"\n\nif __name__ == '__main__':\n    main()\n")
