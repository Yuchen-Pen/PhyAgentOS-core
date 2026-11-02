"""Strict, versioned source transformations for the pinned native evaluator.

The catalogue is data, never executable input. Validate every edit in memory
before publishing files inside prepare's temporary build directory.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from preflight import UPSTREAM_COMMIT
from support import BenchmarkError, ErrorCode, require, sha256

CATALOGUE = Path(__file__).resolve().parent / "patches/native-v1.json"
PATCH_SET = "robodojo-pi05-native/v1"


def checked(condition, message):
    require(condition, message, ErrorCode.PATCH_MISMATCH)


def text_hash(text):
    return hashlib.sha256(text.encode()).hexdigest()


@dataclass(frozen=True)
class TextEdit:
    identity: str
    before: str
    after: str

    def apply(self, text, patch_id):
        count = text.count(self.before)
        checked(count == 1, f"{patch_id}/{self.identity}: expected exactly once, found {count}")
        result = text.replace(self.before, self.after, 1)
        return result, {"id": self.identity, "before_sha256": text_hash(text), "after_sha256": text_hash(result)}


@dataclass(frozen=True)
class SourcePatch:
    identity: str
    path: str
    edits: tuple[TextEdit, ...]

    def apply(self, text):
        before = text_hash(text)
        receipts = []
        for edit in self.edits:
            text, receipt = edit.apply(text, self.identity)
            receipts.append(receipt)
        try:
            compile(text, self.path, "exec")  # Syntax only; no upstream code is executed.
        except SyntaxError as error:
            raise BenchmarkError(ErrorCode.PATCH_MISMATCH, f"{self.identity}: invalid generated syntax") from error
        return text, {"id": self.identity, "path": self.path, "before_sha256": before,
                      "after_sha256": text_hash(text), "edits": receipts}


def load_catalogue(path=CATALOGUE):
    value = json.loads(Path(path).read_text())
    checked(type(value.get("schema_version")) is int and value["schema_version"] == 1 and value.get("patch_set") == PATCH_SET,
            "unsupported native patch catalogue version")
    checked(value.get("upstream_commit") == UPSTREAM_COMMIT, "patch catalogue targets another source commit")
    patches, identities, paths = [], set(), set()
    for record in value["patches"]:
        identity, relative = record["id"], record["path"]
        checked(isinstance(identity, str) and identity and identity not in identities, "duplicate/invalid patch identity")
        checked(isinstance(relative, str) and relative and not Path(relative).is_absolute()
                and ".." not in Path(relative).parts and relative not in paths, "duplicate/unsafe patch path")
        identities.add(identity)
        paths.add(relative)
        edits, edit_ids = [], set()
        for edit in record["edits"]:
            checked(isinstance(edit["id"], str) and edit["id"] and edit["id"] not in edit_ids,
                    "duplicate/invalid edit identity: " + identity)
            checked(all(isinstance(edit[k], str) and edit[k] for k in ("before", "after"))
                    and edit["before"] != edit["after"], "empty/no-op patch: " + identity)
            edits.append(TextEdit(edit["id"], edit["before"], edit["after"]))
            edit_ids.add(edit["id"])
        checked(bool(edits), "empty patch: " + identity)
        patches.append(SourcePatch(identity, relative, tuple(edits)))
    checked(bool(patches), "empty patch catalogue")
    return value, tuple(patches)


def apply_native_patches(root, catalogue=CATALOGUE):
    root = Path(root).resolve()
    definition, patches = load_catalogue(catalogue)
    prepared, receipts = [], []
    for patch in patches:
        path = root / patch.path
        checked(path.resolve().is_relative_to(root) and path.is_file() and not path.is_symlink(),
                "patch source is missing/redirected: " + patch.path)
        text, receipt = patch.apply(path.read_text())
        prepared.append((path, text))
        receipts.append(receipt)
    for path, text in prepared:
        path.write_text(text)
    return {"patch_set": definition["patch_set"], "upstream_commit": definition["upstream_commit"],
            "catalogue_sha256": sha256(catalogue), "files": receipts}
