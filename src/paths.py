"""Folder layout of the repository. Every path can be overridden with an environment variable."""
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA = os.environ.get('ASDRCD_DATA', os.path.join(ROOT, 'data'))            # downloaded data (not in the repo)
RESULTS = os.environ.get('ASDRCD_RESULTS', os.path.join(ROOT, 'results'))   # JSON results behind every number
CACHE = os.environ.get('ASDRCD_CACHE', os.path.join(ROOT, 'cache'))         # cached baseline features

XIONG_DIR = os.environ.get('XIONG_DIR', os.path.join(DATA, 'Incipient-fault-waveform-dataset'))
DATA_NPZ = os.path.join(DATA, 'xiong.npz')                                    # 236 records, labels, subtypes
OSC_7Z = os.environ.get('OSCGRID_7Z', os.path.join(DATA, 'oscgrid', 'unlabeled_50_6400.7z'))
OSC_NPZ = os.path.join(DATA, 'oscgrid', 'u6400_recs.npz')

for _d in (RESULTS, CACHE):
    os.makedirs(_d, exist_ok=True)
