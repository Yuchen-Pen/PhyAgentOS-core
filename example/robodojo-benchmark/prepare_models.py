"""Download or check the pinned inference checkpoint; never import a model."""

import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil

from preflight import MODEL_LOCK_SHA256
from prepare_assets import REVISION, REPO, atomic_json, fetch, verified

LOCK = Path(__file__).with_name('pi05-checkpoint.lock.json')
PREFIX = 'ckpt/RoboDojo/Pi_05/RoboDojo-sim-arx_x5-joint-0/'


def locked_files():
    raw = LOCK.read_bytes()
    if hashlib.sha256(raw).hexdigest() != MODEL_LOCK_SHA256:
        raise ValueError('Model file manifest differs from the pinned inference checkpoint')
    lock = json.loads(raw)
    if lock['repo_id'] != REPO or lock['revision'] != REVISION or lock['checkpoint'] != '59999':
        raise ValueError('Unexpected model repository or revision')
    result, seen = [], set()
    for item in lock['files']:
        name = item['path']
        path = PurePosixPath(name)
        if (not name or path.is_absolute() or '..' in path.parts or str(path) != name
                or not name.startswith(('59999/params/', '59999/assets/')) or name in seen):
            raise ValueError('Invalid or duplicate checkpoint path')
        if type(item['size_bytes']) is not int or item['size_bytes'] < 0 or not re.fullmatch('[0-9a-f]{64}', item['sha256']):
            raise ValueError('Invalid checkpoint size or hash')
        seen.add(name)
        result.append({'path': PREFIX + name, 'relative': name, 'size': item['size_bytes'], 'sha256': item['sha256']})
    if not result:
        raise ValueError('Empty inference checkpoint')
    return result


def prepare(cache, *, check_only=False, workers=4):
    if type(workers) is not int or not 1 <= workers <= 16:
        raise ValueError('workers must be between 1 and 16')
    entries = locked_files()
    cache = Path(cache).absolute()
    if cache.resolve() != cache:
        raise ValueError('Model cache must not be redirected')
    ancestor = next(p for p in (cache, *cache.parents) if p.exists())
    if ancestor.stat().st_uid != os.getuid():
        raise ValueError('Model cache must belong to this user')
    cache.mkdir(parents=True, exist_ok=True)
    root = cache / 'models' / MODEL_LOCK_SHA256 / 'payload'
    if root.resolve() != root:
        raise ValueError('Model cache/payload must not be redirected')
    root.mkdir(parents=True, exist_ok=True)
    if (cache / 'pi05-models.lock').is_symlink() or (cache / 'pi05-models.json').is_symlink():
        raise ValueError('Model guard/receipt must not be redirected')
    with (cache / 'pi05-models.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX)
        receipt = cache / 'pi05-models.json'
        atomic_json(receipt, {'schema_version': 1, 'status': 'checking' if check_only else 'preparing'})
        if not check_only:
            remaining = sum(e['size'] for e in entries if not (root / e['relative']).is_file())
            if shutil.disk_usage(cache).free < remaining + 2 * 1024**3:
                raise ValueError('Insufficient free disk for the inference checkpoint')
        def ensure(entry):
            target = root / entry['relative']
            if not target.resolve().is_relative_to(root) or target.is_symlink():
                raise ValueError('Checkpoint path escapes payload')
            if verified(target, entry):
                return
            if check_only:
                raise ValueError('Missing or corrupt checkpoint file: ' + entry['relative'])
            target.parent.mkdir(parents=True, exist_ok=True)
            part = target.with_name(target.name + '.part')
            if part.is_symlink():
                raise ValueError('Checkpoint partial cannot be a symlink')
            if shutil.disk_usage(cache).free < max(0, entry['size'] - (part.stat().st_size if part.exists() else 0)) + 2 * 1024**3:
                raise ValueError('Insufficient free disk for checkpoint transfer')
            fetch(entry, part)
            part.replace(target)
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                list(pool.map(ensure, entries))
        except Exception:
            atomic_json(receipt, {'schema_version': 1, 'status': 'incomplete'})
            raise
        checkpoint = root / '59999'
        value = {'schema_version': 1, 'status': 'ready', 'checkpoint_root': str(checkpoint),
                 'params_dir': str(checkpoint / 'params'), 'assets_dir': str(checkpoint / 'assets'),
                 'checkpoint_lock_sha256': MODEL_LOCK_SHA256}
        atomic_json(receipt, value)
        return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cache', type=Path)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    print(json.dumps(prepare(args.cache, check_only=args.check_only, workers=args.workers), indent=2))
