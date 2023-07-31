from pathlib import Path
from glob import glob
import shutil

files = list(map(str, Path('.').glob('*.bin')))
for f in files:
    shutil.move(f, f.split('_s_')[1])



