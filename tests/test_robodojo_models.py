"""Small local checkpoint fixtures; no weights or network downloads."""

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

EXAMPLE = Path(__file__).resolve().parents[1] / 'example/robodojo-benchmark'
sys.path.insert(0, str(EXAMPLE))
import prepare_models


class ModelPreparationTests(unittest.TestCase):
    def test_shipped_lock_has_expected_identity_and_inference_only_paths(self):
        entries = prepare_models.locked_files()
        self.assertEqual(len(entries), 17)
        self.assertTrue(all(e['relative'].startswith(('59999/params/', '59999/assets/')) for e in entries))

    def test_existing_verified_payload_produces_receipt_without_network(self):
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary)
            entry = {'relative': '59999/params/fixture', 'path': 'unused', 'size': 7,
                     'sha256': hashlib.sha256(b'fixture').hexdigest()}
            target = cache / 'models' / prepare_models.MODEL_LOCK_SHA256 / 'payload' / entry['relative']
            target.parent.mkdir(parents=True)
            target.write_bytes(b'fixture')
            with patch.object(prepare_models, 'locked_files', return_value=[entry]), patch.object(prepare_models, 'fetch') as fetch:
                report = prepare_models.prepare(cache, check_only=True)
                self.assertEqual(report['status'], 'ready')
                self.assertEqual(Path(report['checkpoint_root']), target.parents[1])
                fetch.assert_not_called()
            target.write_bytes(b'changed')
            with patch.object(prepare_models, 'locked_files', return_value=[entry]):
                with self.assertRaisesRegex(ValueError, 'Missing or corrupt'):
                    prepare_models.prepare(cache, check_only=True)
            self.assertEqual(json.loads((cache / 'pi05-models.json').read_text())['status'], 'incomplete')

    def test_lock_tampering_is_rejected_before_cache_creation(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'changed-lock.json'
            path.write_text('{}')
            with patch.object(prepare_models, 'LOCK', path):
                with self.assertRaisesRegex(ValueError, 'differs from the pinned'):
                    prepare_models.prepare(Path(temporary) / 'cache')
            self.assertFalse((Path(temporary) / 'cache').exists())
