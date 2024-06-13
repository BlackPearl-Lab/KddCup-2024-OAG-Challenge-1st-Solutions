import os
from os.path import abspath, dirname, join


PROJ_DIR = join(abspath(dirname(__file__)))
DATA_DIR = join(PROJ_DIR, "data")
OUT_DIR = join(PROJ_DIR, "out")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)
DATA_TRACE_DIR = DATA_DIR
