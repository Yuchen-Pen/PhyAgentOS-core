"""Protect camera validity and terminal footage without extra physics/render calls."""
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'example/robodojo-benchmark'))
from render_state import TerminalObservation, record_final_frame, verify_render_sync


class RenderStateTests(unittest.TestCase):
    def test_reject_kit_physical_gpu_mismatch_after_cuda_renumbering(self):
        simulation = SimpleNamespace(is_fabric_enabled=lambda: False)
        settings = {'/physics/updateToUsd': True, '/renderer/activeGpu': 0}
        with self.assertRaisesRegex(RuntimeError, 'physical NVIDIA GPU'):
            verify_render_sync(simulation, settings, expected_render_gpu='3')
        settings['/renderer/activeGpu'] = 3
        self.assertEqual(verify_render_sync(simulation, settings, '3')['render_gpu_index'], 3)

    def test_reject_frozen_camera_configuration(self):
        simulation = SimpleNamespace(is_fabric_enabled=lambda: False)
        with self.assertRaisesRegex(RuntimeError, 'both disabled'):
            verify_render_sync(simulation, {'/physics/updateToUsd': False})
        self.assertTrue(verify_render_sync(simulation, {'/physics/updateToUsd': True})['physics_update_to_usd'])
        simulation.is_fabric_enabled = lambda: True
        self.assertTrue(verify_render_sync(simulation, {})['fabric_enabled'])

    def test_tap_retains_native_final_observation_without_new_capture(self):
        first = {'env_idx': 0, 'frame': 'initial'}
        last = {'env_idx': 0, 'frame': 'after last action'}
        original = Mock(side_effect=[[first], [last]])
        environment = SimpleNamespace(get_obs_batch=original)
        capture = TerminalObservation(environment)
        self.assertIs(environment.get_obs_batch([0])[0], first)
        self.assertIsNone(capture.observation)
        self.assertIs(environment.get_obs_batch([0], last_frame=True)[0], last)
        self.assertIs(capture.observation, last)
        self.assertEqual(original.call_count, 2)
        original.assert_called_with(env_idx_list=[0], last_frame=True)

    def test_append_terminal_frame_and_cap_frame_without_policy_publish(self):
        for done, step in [(True, 7), (False, 10)]:
            with self.subTest(done=done, step=step):
                recorder = SimpleNamespace(add=Mock())
                frame = {'head': 'final RGB'}
                adapter = SimpleNamespace(to_images=Mock(return_value=frame))
                observation = {'final': True}
                outcome = SimpleNamespace(done=done, success=False, obs=observation)
                record_final_frame(recorder, adapter, outcome, step, 10)
                adapter.to_images.assert_called_once_with(observation)
                recorder.add.assert_called_once_with(frame)

    def test_ongoing_step_does_not_duplicate_recording(self):
        recorder = SimpleNamespace(add=Mock())
        adapter = SimpleNamespace(to_images=Mock())
        record_final_frame(recorder, adapter, SimpleNamespace(done=False, success=False), 1, 10)
        recorder.add.assert_not_called()
        adapter.to_images.assert_not_called()
