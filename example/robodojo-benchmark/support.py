"""Shared deployment identity, classified validation and durable file publication.

Keep this module standard-library-only: the bundled launcher uses system Python.
"""

import hashlib
import json
import os
import stat
import tempfile
from enum import Enum
from pathlib import Path


class ErrorCode(str, Enum):
    CONTROL_REJECTED = "ROBODOJO_CONTROL_REJECTED"
    RESULT_INVALID = "ROBODOJO_RESULT_INVALID"
    BUNDLE_MISMATCH = "ROBODOJO_BUNDLE_MISMATCH"
    PATCH_MISMATCH = "ROBODOJO_PATCH_MISMATCH"
    ACCEPTANCE_UNKNOWN = "ROBODOJO_ACCEPTANCE_UNKNOWN"
    QUERY_UNKNOWN = "ROBODOJO_QUERY_UNKNOWN"
    STOP_UNKNOWN = "ROBODOJO_STOP_UNKNOWN"


class BenchmarkError(ValueError):
    """A stable error code, independent of the human-readable explanation."""

    def __init__(self, code, message):
        self.code = ErrorCode(code)
        super().__init__(message)


def require(condition, message, code=ErrorCode.RESULT_INVALID):
    if not condition:
        raise BenchmarkError(code, message)


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def deployment_identity(config):
    """Same canonical JSON identity as Core's interaction.contracts.digest."""
    value = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(value.encode()).hexdigest()


def sha256(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as source:
        for data in iter(lambda: source.read(8 * 1024**2), b""):
            checksum.update(data)
    return checksum.hexdigest()


def sync_directory(directory):
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write(path, content, *, immutable=False):
    """Publish complete bytes, then sync the directory; immutable writes never replace.

    A same-directory hard link atomically claims an immutable evidence name.
    Replays compare an existing regular file without following symlinks.
    """
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        if immutable:
            try:
                os.link(temporary, path)
            except FileExistsError:
                descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
                with os.fdopen(descriptor, "rb") as existing:
                    require(stat.S_ISREG(os.fstat(existing.fileno()).st_mode), "evidence is not a regular file")
                    require(os.fstat(existing.fileno()).st_size == len(content) and existing.read() == content,
                            "evidence changed on replay: " + path.name)
        else:
            os.replace(temporary, path)
        Path(temporary).unlink(missing_ok=True)
        sync_directory(path.parent)
    finally:
        Path(temporary).unlink(missing_ok=True)


def immutable_file(path, content):
    atomic_write(path, content, immutable=True)


def verify_bundle(bundle, config):
    """One identity/hash check for both service admission and node launch."""
    bundle = Path(bundle).resolve()
    deployed = json.loads((bundle / "deployment.json").read_text())
    require(deployment_identity(deployed) == deployment_identity(config),
            "installed native Skill uses another deployment; rebuild before running", ErrorCode.BUNDLE_MISMATCH)
    provenance = json.loads((bundle / "PROVENANCE.json").read_text())
    # Old bundles omit this field; their deployment.json is still hashed below.
    require(provenance.get("deployment_identity", deployment_identity(config)) == deployment_identity(config),
            "bundle deployment identity mismatch", ErrorCode.BUNDLE_MISMATCH)
    files = provenance["bundled_files"]
    require(isinstance(files, dict) and "deployment.json" in files,
            "bundle has no deployment file inventory", ErrorCode.BUNDLE_MISMATCH)
    for relative, expected in files.items():
        path = bundle / relative
        require(not Path(relative).is_absolute() and path.resolve().is_relative_to(bundle)
                and not path.is_symlink() and path.is_file() and sha256(path) == expected,
                "bundled source/config changed: " + relative, ErrorCode.BUNDLE_MISMATCH)
    return provenance
