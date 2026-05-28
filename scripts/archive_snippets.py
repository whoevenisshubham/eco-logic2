import shutil
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[1]
data_snippets = root / 'data' / 'snippets'
archive = root / 'data' / 'snippets_archive.zip'

if not data_snippets.exists():
    print('No data/snippets directory found, skipping.')
    sys.exit(0)

print(f'Archiving {data_snippets} -> {archive}')
shutil.make_archive(str(archive.with_suffix('')), 'zip', root_dir=str(data_snippets))
print('Archive created:', archive)

# Remove the directory tree
shutil.rmtree(data_snippets)
print('Removed original snippets directory')
