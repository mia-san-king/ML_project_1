"""Package the exact repository sources used for the private lab run."""
from pathlib import Path
import hashlib
import io
import tarfile

run = Path(__file__).resolve().parent
repo = run.parents[1] / 'ML_project_1_time'
files = [repo / 'models/__init__.py', repo / 'models/geoclip_time/__init__.py']
for directory in ('lib','scripts'):
    files.extend(sorted((repo / 'models/geoclip_time' / directory).glob('*.py')))
assert (repo / 'models/geoclip_time/scripts/rerank_locations.py') in files
assert (repo / 'models/geoclip_time/lib/temporal_location_retrieval.py') in files
lines = []
with tarfile.open(run/'sources.tar.gz', mode='w:gz') as archive:
    for path in files:
        name = path.relative_to(repo).as_posix()
        raw = path.read_bytes()
        lines.append(f'{hashlib.sha256(raw).hexdigest()}  {name}')
        info = tarfile.TarInfo(f'source/{name}')
        info.size = len(raw)
        info.mode = 0o644
        archive.addfile(info, io.BytesIO(raw))
(run/'source.sha256').write_text('\n'.join(lines)+'\n')
