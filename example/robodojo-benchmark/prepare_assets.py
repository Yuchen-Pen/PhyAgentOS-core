"""Prepare the pinned RoboDojo Assets tree, verifying every LFS SHA256.

Readable existing assets are copied and checked; unavailable or changed files
are downloaded. Only the caller's destination is written, never the reuse tree.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
import shutil
import time
import urllib.parse
import urllib.request
from pathlib import Path, PurePosixPath

REPO = "RoboDojo-Benchmark/RoboDojo"
REVISION = "43dacb13d3051a9ccd421f4e7e08e4e3eb29dfab"


def fetch_catalog(output):
    """Record the official pinned tree before transferring asset contents."""
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("catalog output exists; retain it or choose a new output path")
    base = f"https://huggingface.co/api/datasets/{REPO}/tree/{REVISION}/"
    with urllib.request.urlopen(base + "Assets?recursive=false&limit=1000", timeout=90) as response:
        folders = json.load(response)

    def folder_entries(folder):
        url = base + urllib.parse.quote(folder["path"], safe="/") + "?recursive=true&limit=1000"
        entries = []
        while url:
            if not url.startswith(base):
                raise ValueError("unexpected asset catalog pagination URL")
            with urllib.request.urlopen(url, timeout=90) as response:
                entries.extend(json.load(response))
                links = response.headers.get("Link", "")
            url = next((part.split(">")[0].strip().lstrip("<") for part in links.split(",") if 'rel="next"' in part), None)
        return entries

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        entries = [entry for group in pool.map(folder_entries, folders) for entry in group]
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, {"repo": REPO, "revision": REVISION, "entries": entries})
    files = load_catalog(output)
    print("CATALOG_READY", len(files), sum(entry["size"] for entry in files), flush=True)
    return output


def stamp(path):
    value = path.stat()
    return [value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns]


def atomic_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    temporary.replace(path)


def load_catalog(path):
    catalog = json.loads(Path(path).read_text())
    if catalog.get("repo") != REPO or catalog.get("revision") != REVISION:
        raise ValueError("asset catalog must identify the pinned RoboDojo dataset revision")
    files, seen = [], set()
    for entry in catalog["entries"]:
        if entry["type"] != "file":
            continue
        name = entry["path"]
        parts = PurePosixPath(name).parts
        if not name.startswith("Assets/") or any(p in {".", "..", ""} for p in name.split("/")) or "\\" in name:
            raise ValueError("unsafe asset path")
        if name in seen:
            raise ValueError("duplicate asset path")
        seen.add(name)
        sha = entry.get("lfs", {}).get("oid")
        size = entry["size"]
        if not isinstance(sha, str) or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
            raise ValueError("every asset must have an LFS SHA256")
        if type(size) is not int or size < 0 or entry["lfs"]["size"] != size:
            raise ValueError("inconsistent asset size")
        files.append({"path": name, "size": size, "sha256": sha})
    if not files:
        raise ValueError("empty asset catalog")
    return sorted(files, key=lambda entry: entry["path"])


def verified(path, entry):
    if path.is_symlink() or not path.is_file() or path.stat().st_size != entry["size"]:
        return False
    before = stamp(path)
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024**2), b""):
            digest.update(chunk)
    return before == stamp(path) and digest.hexdigest() == entry["sha256"]


def fetch(entry, part):
    url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/" + urllib.parse.quote(entry["path"], safe="/") + "?download=true"
    for attempt in range(5):
        try:
            if part.exists() and part.stat().st_size >= entry["size"]:
                if verified(part, entry):
                    return
                part.unlink()  # Only a generated, unverified partial in our destination.
            offset = part.stat().st_size if part.exists() else 0
            headers = {"Accept-Encoding": "identity", "User-Agent": "PAOS-RoboDojo-assets/1.0"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=90) as response:
                if response.status == 206:
                    content_range = response.headers.get("Content-Range", "")
                    if not content_range.startswith(f"bytes {offset}-") or not content_range.endswith(f"/{entry['size']}"):
                        raise ValueError("invalid range response")
                    mode = "ab"
                elif response.status == 200:
                    mode, offset = "wb", 0
                else:
                    raise ValueError("unexpected download response")
                count = offset
                with part.open(mode) as target:
                    while chunk := response.read(8 * 1024**2):
                        count += len(chunk)
                        if count > entry["size"]:
                            raise ValueError("download exceeds locked size")
                        target.write(chunk)
            if not verified(part, entry):
                raise ValueError("asset size or SHA256 mismatch")
            return
        except Exception as error:
            if attempt == 4:
                # Do not expose redirect URLs, signed credentials or response bodies.
                raise RuntimeError("asset download failed (" + type(error).__name__ + ")") from None
            time.sleep(min(16, 2**attempt))


def prepare(catalog, destination, reuse=None, workers=8, reserve_gib=20):
    files = load_catalog(catalog)
    identity = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    destination = Path(destination).resolve()
    ancestor = next(p for p in (destination, *destination.parents) if p.exists())
    if ancestor.stat().st_uid != os.getuid():
        raise ValueError("asset destination must be in a directory owned by this user")
    destination.mkdir(parents=True, exist_ok=True)
    reuse = Path(reuse).resolve() if reuse else None
    if reuse is not None and (destination == reuse or reuse.is_relative_to(destination)):
        raise ValueError("reuse source must be outside the destination")
    with (destination / ".prepare.lock").open("a") as guard:
        fcntl.flock(guard, fcntl.LOCK_EX)
        marker = destination / "verified.json"
        try:
            prior = json.loads(marker.read_text())
        except (OSError, ValueError):
            prior = {}
        if prior.get("catalog_sha256") != identity:
            prior = {}
        records = prior.get("files", {})
        atomic_json(destination / "assets-ready.json", {"schema_version": 1, "status": "preparing", "catalog_sha256": identity})
        total = sum(entry["size"] for entry in files)
        remaining = sum(entry["size"] for entry in files if not (destination / entry["path"]).is_file())
        if shutil.disk_usage(destination).free < remaining + reserve_gib * 1024**3:
            raise ValueError("insufficient free space for assets and reserved headroom")
        print(f"ASSETS_START files={len(files)} bytes={total} workers={workers}", flush=True)

        def one(entry, shared_source=None):
            path = destination / entry["path"]
            if not path.resolve().is_relative_to(destination) or path.is_symlink():
                raise ValueError("asset destination escapes its personal tree")
            old = records.get(entry["path"], {})
            if verified(path, entry):
                # Same-size rewrites can share coarse filesystem timestamps.
                # Always rehash an existing asset; stat alone is not proof.
                origin = "verified_cache" if old.get("sha256") == entry["sha256"] else "existing_verified"
                return entry, {"sha256": entry["sha256"], "stamp": stamp(path), "origin": origin}
            if shutil.disk_usage(destination).free < reserve_gib * 1024**3:
                raise RuntimeError("free space fell below reserved headroom")
            path.parent.mkdir(parents=True, exist_ok=True)
            part = path.with_name(path.name + ".part")
            if part.is_symlink():
                raise ValueError("unexpected partial symlink")
            origin = "download"
            candidates = ([shared_source] if shared_source is not None else []) + ([reuse / entry["path"]] if reuse is not None else [])
            for source in candidates:
                if part.exists():
                    break
                try:
                    if source.is_file() and source.stat().st_size == entry["size"]:
                        shutil.copyfile(source, part)
                        if verified(part, entry):
                            origin = "copied_same_hash" if source == shared_source else "copied_and_verified"
                            break
                        else:
                            part.unlink()
                except OSError:
                    if part.exists():
                        part.unlink()
            if origin not in {"copied_and_verified", "copied_same_hash"}:
                fetch(entry, part)
            part.replace(path)
            return entry, {"sha256": entry["sha256"], "stamp": stamp(path), "origin": origin}

        groups = {}
        for entry in files:
            groups.setdefault(entry["sha256"], []).append(entry)

        def group_one(entries):
            def preference(entry):
                path = destination / entry["path"]
                part = path.with_name(path.name + ".part")
                source = reuse / entry["path"] if reuse is not None else None
                try:
                    reusable = source is not None and source.is_file() and source.stat().st_size == entry["size"] and os.access(source, os.R_OK)
                except OSError:
                    reusable = False
                return (path.is_file(), reusable, part.stat().st_size if part.is_file() else 0)
            ordered = sorted(entries, key=preference, reverse=True)
            first, record = one(ordered[0])
            results = [(first, record)]
            for entry in ordered[1:]:
                results.append(one(entry, destination / first["path"]))
            return results

        print(f"ASSETS_UNIQUE_HASHES {len(groups)}", flush=True)
        failures, completed, last_report = [], 0, 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            pending = {pool.submit(group_one, entries): entries for entries in groups.values()}
            while pending:
                done, waiting = concurrent.futures.wait(pending, timeout=10, return_when=concurrent.futures.FIRST_COMPLETED)
                for future in done:
                    try:
                        for entry, record in future.result():
                            records[entry["path"]] = record
                            completed += 1
                    except Exception as error:
                        for entry in pending[future]:
                            failures.append({"path": entry["path"], "error": str(error)})
                            print("ASSET_ERROR", entry["path"], type(error).__name__, flush=True)
                pending = {future: pending[future] for future in waiting}
                if time.monotonic() - last_report > 15 or not pending:
                    atomic_json(marker, {"catalog_sha256": identity, "files": records})
                    print(f"ASSETS_PROGRESS verified={completed}/{len(files)} errors={len(failures)} pending={len(pending)}", flush=True)
                    last_report = time.monotonic()
        atomic_json(destination / "failures.json", failures)
        if failures:
            atomic_json(destination / "assets-ready.json", {"schema_version": 1, "status": "incomplete", "failed_files": len(failures)})
            raise RuntimeError(f"{len(failures)} assets incomplete; keep files and resume the same command")
        receipt = {
            "schema_version": 1, "status": "ready", "repo": REPO, "revision": REVISION,
            "assets_root": str(destination / "Assets"), "files": len(files), "bytes": total,
            "catalog_sha256": identity,
            "origins": {origin: sum(record["origin"] == origin for record in records.values())
                        for origin in sorted({record["origin"] for record in records.values()})},
        }
        atomic_json(destination / "assets-ready.json", receipt)
        print("ASSETS_READY", json.dumps(receipt, ensure_ascii=False), flush=True)
        return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("catalog", nargs="?")
    parser.add_argument("destination", nargs="?")
    parser.add_argument("--write-catalog", help="Only fetch pinned size/hash metadata to a new JSON file")
    parser.add_argument("--reuse", help="Read-only repository root containing Assets/")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.write_catalog:
        fetch_catalog(args.write_catalog)
        return
    if not args.catalog or not args.destination:
        parser.error("catalog and destination are required for asset preparation")
    if not 1 <= args.workers <= 32:
        parser.error("workers must be 1..32")
    prepare(args.catalog, args.destination, args.reuse, args.workers)


if __name__ == "__main__":
    main()
