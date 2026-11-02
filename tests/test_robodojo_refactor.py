"""Lifecycle recovery, versioned patch rejection and durable publication contracts."""

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

EXAMPLE = Path(__file__).resolve().parents[1] / 'example/robodojo-benchmark'
sys.path.insert(0, str(EXAMPLE))
import test_robodojo_profile as profiles
from PhyAgentOS.forge.interaction.contracts import digest
from lifecycle import (EpisodePhase, RootPhase, RootState, claim_episode, request_episode_stop,
                       request_root_stop, settle_roots, transition_episode, validate_journal)
from source_patches import CATALOGUE, TextEdit, apply_native_patches, load_catalogue
from support import BenchmarkError, ErrorCode, atomic_write, deployment_identity, immutable_file, sha256


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.root = dict(task_id='root', run_id='run', binding_digest='binding', stopped=False, stop_requested=False)
        self.scope = {k: k for k in ('subtask_id', 'attempt_id', 'worker_id', 'world_epoch')}
        self.scope.update({k: self.root[k] for k in ('task_id', 'run_id', 'binding_digest')})
        self.episode = dict(scope=self.scope, stop_requested=False, history=[])
        self.journal = dict(roots={'root': self.root}, episodes={}, active=None)
        claim_episode(self.journal, self.episode)
        self.transition(EpisodePhase.RUNNING)

    def transition(self, phase, **kwargs):
        return transition_episode(self.journal, self.episode, state=phase, observation={'stage': phase.value}, **kwargs)

    def test_unknown_is_recoverable_but_cannot_release_worker_or_close_root(self):
        self.transition(EpisodePhase.UNKNOWN)
        request_root_stop(self.root)
        request_episode_stop(self.journal, self.episode)
        self.assertEqual(RootState.from_record(self.root).phase, RootPhase.STOPPING)
        self.assertEqual(self.journal['active'], 'subtask_id')
        self.transition(EpisodePhase.RUNNING)
        self.transition(EpisodePhase.TERMINAL, outcome='success', score=1, stopped=True)
        self.assertEqual(RootState.from_record(self.root).phase, RootPhase.STOPPED)
        self.assertIsNone(self.journal['active'])
        validate_journal(self.journal)

    def test_completed_episode_requires_acknowledgement_before_worker_reuse(self):
        self.transition(EpisodePhase.TERMINAL, outcome='failure', score=0, stopped=True)
        self.assertEqual(self.journal['active'], 'subtask_id')
        self.assertTrue(request_episode_stop(self.journal, self.episode))
        self.assertFalse(request_episode_stop(self.journal, self.episode))
        self.assertIsNone(self.journal['active'])
        validate_journal(self.journal)

    def test_terminal_is_absorbing_and_replay_does_not_change_history(self):
        first = self.transition(EpisodePhase.TERMINAL, outcome='success', score=1, stopped=True)
        self.assertEqual(first, self.transition(EpisodePhase.TERMINAL, outcome='success', score=1, stopped=True))
        history = copy.deepcopy(self.episode['history'])
        for phase in (EpisodePhase.RUNNING, EpisodePhase.UNKNOWN):
            with self.subTest(phase=phase), self.assertRaises(BenchmarkError):
                self.transition(phase)
        self.assertEqual(history, self.episode['history'])

    def test_recovery_rejects_missing_worker_cross_root_and_boolean_coercion(self):
        for mutate in (lambda j: j.update(active=None),
                       lambda j: j['episodes']['subtask_id']['scope'].update(run_id='other'),
                       lambda j: j['roots']['root'].update(stopped=1)):
            journal = copy.deepcopy(self.journal)
            mutate(journal)
            with self.subTest(journal=journal), self.assertRaises(BenchmarkError):
                validate_journal(journal)

    def test_settling_another_episode_cannot_release_active_owner(self):
        other = copy.deepcopy(self.episode)
        other['scope']['subtask_id'] = 'historical'
        other.update(stop_requested=True, snapshot={**other['snapshot'], 'subtask_id': 'historical',
                                                   'state': 'terminal', 'stopped': True})
        self.journal['episodes']['historical'] = other
        settle_roots(self.journal)
        self.assertEqual(self.journal['active'], 'subtask_id')


class PatchTests(unittest.TestCase):
    setUp = profiles.ProfileTests.setUp
    git = profiles.ProfileTests.git
    build = profiles.ProfileTests.build

    def test_receipt_maps_every_original_file_to_the_generated_bytes(self):
        bundle = self.build()
        receipt = json.loads((bundle / 'PROVENANCE.json').read_text())['patches']
        self.assertEqual(receipt['catalogue_sha256'], sha256(CATALOGUE))
        self.assertEqual(len(receipt['files']), 5)
        self.assertEqual(sum(len(p['edits']) for p in receipt['files']), 15)
        for patch in receipt['files']:
            self.assertEqual(patch['before_sha256'], sha256(self.repo / 'forge_robodojo_eval/robodojo' / patch['path']))
            self.assertEqual(patch['after_sha256'], sha256(bundle / 'assets/robodojo' / patch['path']))
        self.assertEqual(sha256(bundle / 'NATIVE_PATCHES.json'), sha256(CATALOGUE))

    def test_late_patch_failure_leaves_all_source_files_unchanged(self):
        root = self.repo / 'forge_robodojo_eval/robodojo'
        policy = root / 'policy_node.py'
        policy.write_text(policy.read_text().replace('                phase = "running"', '                phase = "changed"'))
        before = {p: p.read_bytes() for p in root.rglob('*.py')}
        with self.assertRaisesRegex(BenchmarkError, 'persistent-policy/wait-for-reset'):
            apply_native_patches(root)
        self.assertEqual(before, {p: p.read_bytes() for p in root.rglob('*.py')})

    def test_missing_and_ambiguous_anchors_are_rejected_with_patch_identity(self):
        edit = TextEdit('anchor', 'old', 'new')
        for text in ('missing', 'old old'):
            with self.subTest(text=text), self.assertRaises(BenchmarkError) as failure:
                edit.apply(text, 'budget/v1')
            self.assertEqual(failure.exception.code, ErrorCode.PATCH_MISMATCH)
            self.assertIn('budget/v1/anchor', str(failure.exception))

    def test_wrong_version_commit_and_unsafe_paths_are_rejected(self):
        original = json.loads(CATALOGUE.read_text())
        for field, value in (('patch_set', 'robodojo-pi05-native/v2'), ('upstream_commit', '0' * 40),
                             ('schema_version', 2), ('path', '../escape.py')):
            definition = copy.deepcopy(original)
            if field == 'path':
                definition['patches'][0]['path'] = value
            else:
                definition[field] = value
            catalogue = self.root / 'catalogue.json'
            catalogue.write_text(json.dumps(definition))
            with self.subTest(field=field), self.assertRaises(BenchmarkError):
                load_catalogue(catalogue)

    def test_bundled_launcher_works_without_core_or_site_packages(self):
        bundle = self.build()
        script = ('import sys; sys.path.insert(0,sys.argv[1]); import launch; '
                  'assert launch.launch_plan(sys.argv[1], "tool", {})["role"] == "tool"')
        result = subprocess.run(['/usr/bin/python3', '-I', '-S', '-c', script, str(bundle)],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


class PersistenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.path = self.root / 'evidence.json'

    def test_deployment_identity_matches_core_and_rejects_nonfinite_values(self):
        config = {'path': '/个人目录/with spaces', 'fraction': 0.3, 'gpu': None}
        self.assertEqual(deployment_identity(config), digest(config))
        with self.assertRaises(ValueError):
            deployment_identity({'fraction': float('nan')})

    def test_failed_prepublication_sync_never_leaves_partial_evidence(self):
        with mock.patch('support.os.fsync', side_effect=OSError('injected write interruption')):
            with self.assertRaises(OSError):
                immutable_file(self.path, b'complete evidence')
        self.assertEqual(list(self.root.iterdir()), [])

    def test_failed_mutable_write_preserves_previous_journal(self):
        self.path.write_bytes(b'old journal')
        with mock.patch('support.os.replace', side_effect=OSError('injected publication interruption')):
            with self.assertRaises(OSError):
                atomic_write(self.path, b'new journal')
        self.assertEqual(self.path.read_bytes(), b'old journal')
        self.assertEqual(list(self.root.iterdir()), [self.path])

    def test_immutable_replay_and_concurrent_conflict_preserve_complete_winner(self):
        def publish(content):
            try:
                immutable_file(self.path, content)
                return content
            except BenchmarkError:
                return None
        candidates = [b'a' * 100000, b'b' * 100000]
        with ThreadPoolExecutor(max_workers=2) as pool:
            winners = [value for value in pool.map(publish, candidates) if value is not None]
        self.assertEqual(len(winners), 1)
        self.assertEqual(self.path.read_bytes(), winners[0])
        immutable_file(self.path, winners[0])
        self.assertEqual(list(self.root.iterdir()), [self.path])

    def test_immutable_replay_refuses_a_symlink_without_changing_its_target(self):
        target = self.root / 'target'
        target.write_bytes(b'original')
        self.path.symlink_to(target)
        with self.assertRaises(OSError):
            immutable_file(self.path, b'original')
        self.assertEqual(target.read_bytes(), b'original')
