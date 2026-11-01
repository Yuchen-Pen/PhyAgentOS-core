"""Asset preparation checks with local synthetic files; no remote downloads."""

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SCRIPT = Path(__file__).resolve().parents[1] / "example/robodojo-benchmark/prepare_assets.py"
spec = importlib.util.spec_from_file_location("robodojo_assets", SCRIPT)
assets = importlib.util.module_from_spec(spec)
spec.loader.exec_module(assets)


class AssetPreparationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.catalog = self.root / "catalog.json"
        self.source = self.root / "existing"
        self.target = self.root / "personal"
        self.entry = {"type": "file", "path": "Assets/Robots/x5/robot.urdf", "size": 8,
                      "lfs": {"size": 8, "oid": hashlib.sha256(b"verified").hexdigest()}}
        self.write_catalog(self.entry)

    def write_catalog(self, entry):
        self.catalog.write_text(json.dumps({"repo": assets.REPO, "revision": assets.REVISION, "entries": [entry]}))

    def source_file(self, data):
        source = self.source / self.entry["path"]
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(data)
        return source

    def test_read_only_reuse_is_copied_verified_and_cached(self):
        source = self.source_file(b"verified")
        before = assets.stamp(source)
        with mock.patch.object(assets, "fetch", side_effect=AssertionError("must not download")):
            receipt = assets.prepare(self.catalog, self.target, self.source, workers=1, reserve_gib=0)
            self.assertEqual(receipt["status"], "ready")
            self.assertEqual(receipt["origins"], {"copied_and_verified": 1})
            self.assertEqual(assets.stamp(source), before)
            self.assertFalse((self.target / self.entry["path"]).is_symlink())
            receipt = assets.prepare(self.catalog, self.target, self.source, workers=1, reserve_gib=0)
            self.assertEqual(receipt["origins"], {"verified_cache": 1})

    def test_same_size_wrong_content_cannot_be_accepted_as_reuse(self):
        source = self.source_file(b"bad data")
        def download(entry, part):
            self.assertFalse(part.exists())
            part.write_bytes(b"verified")
        with mock.patch.object(assets, "fetch", side_effect=download) as fetch:
            receipt = assets.prepare(self.catalog, self.target, self.source, workers=1, reserve_gib=0)
        self.assertEqual(fetch.call_count, 1)
        self.assertEqual(receipt["origins"], {"download": 1})
        self.assertEqual(source.read_bytes(), b"bad data")

    def test_changed_destination_is_not_accepted_from_cached_stamp(self):
        self.source_file(b"verified")
        assets.prepare(self.catalog, self.target, self.source, workers=1, reserve_gib=0)
        (self.target / self.entry["path"]).write_bytes(b"bad data")
        receipt = assets.prepare(self.catalog, self.target, self.source, workers=1, reserve_gib=0)
        self.assertEqual(receipt["origins"], {"copied_and_verified": 1})
        self.assertEqual((self.target / self.entry["path"]).read_bytes(), b"verified")

    def test_failed_download_never_publishes_ready_receipt(self):
        with mock.patch.object(assets, "fetch", side_effect=RuntimeError("network unavailable")):
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                assets.prepare(self.catalog, self.target, workers=1, reserve_gib=0)
        self.assertNotEqual(json.loads((self.target / "assets-ready.json").read_text())["status"], "ready")
        self.assertEqual(len(json.loads((self.target / "failures.json").read_text())), 1)

    def test_path_traversal_and_missing_hash_are_rejected(self):
        for entry in (self.entry | {"path": "Assets/../escape"}, self.entry | {"lfs": {}},
                      self.entry | {"size": 9}):
            self.write_catalog(entry)
            with self.assertRaises(ValueError):
                assets.load_catalog(self.catalog)

    def test_target_symlink_cannot_write_to_reuse_source(self):
        source = self.source_file(b"verified")
        target = self.target / self.entry["path"]
        target.parent.mkdir(parents=True)
        target.symlink_to(source)
        with self.assertRaises(RuntimeError):
            assets.prepare(self.catalog, self.target, self.source, workers=1, reserve_gib=0)
        self.assertEqual(source.read_bytes(), b"verified")

    def test_identical_content_at_two_paths_is_downloaded_once(self):
        value = json.loads(self.catalog.read_text())
        value['entries'].append(self.entry | {'path': 'Assets/Room/shared.urdf'})
        self.catalog.write_text(json.dumps(value))
        with mock.patch.object(assets, 'fetch', side_effect=lambda entry, part: part.write_bytes(b'verified')) as fetch:
            receipt = assets.prepare(self.catalog, self.target, workers=2, reserve_gib=0)
        self.assertEqual(fetch.call_count, 1)
        self.assertEqual(receipt['files'], 2)
        self.assertEqual((self.target / 'Assets/Room/shared.urdf').read_bytes(), b'verified')


if __name__ == "__main__":
    unittest.main()
