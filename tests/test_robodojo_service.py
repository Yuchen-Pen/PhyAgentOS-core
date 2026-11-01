"""Lifecycle isolation checks. No model, GPU or simulator is started."""

import json
import os
from pathlib import Path
import socket
import sys
import subprocess
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'example/robodojo-benchmark'))
from dora_scope import DoraScope, identity, listening_ports
from service import stop_allowed
from service import renew_scope
from types import SimpleNamespace


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.work = Path(self.tmp.name)
        self.scope = DoraScope(self.work)
        self.spec = {'binary': '/pinned/dora', 'control_port': 19032,
                     'daemon_port': 19033, 'local_port': 19034}
        self.patch = mock.patch.object(self.scope, 'load', return_value=self.spec)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def test_core_commands_always_use_private_control_endpoint(self):
        with mock.patch.object(self.scope, 'owned'):
            for args in (['check'], ['list', '--format', 'json'],
                         ['start', '--name', 'paos-robodojo-bench-pi05-pi05', 'dataflow.yaml'],
                         ['stop', '--name', 'paos-robodojo-bench-pi05-pi05']):
                command = self.scope.proxy_command(args)
                self.assertIn('--coordinator-port', command)
                self.assertEqual(command[command.index('--coordinator-port') + 1], '19032')
                self.assertNotIn('6012', command)

    def test_unscoped_or_foreign_mutation_is_rejected(self):
        with mock.patch.object(self.scope, 'owned'):
            for args in (['up'], ['destroy'], ['stop', '--name', 'someone-else'],
                         ['check', '--coordinator-port', '6012']):
                with self.assertRaises(ValueError):
                    self.scope.proxy_command(args)

    def test_missing_ownership_fails_instead_of_falling_back(self):
        with self.assertRaises(FileNotFoundError):
            self.scope.proxy_command(['check'])

    def test_real_socket_ownership_distinguishes_listeners(self):
        with socket.socket() as listener:
            listener.bind(('127.0.0.1', 0))
            listener.listen()
            port = listener.getsockname()[1]
            self.assertIn(port, listening_ports(os.getpid()))
        self.assertNotIn(port, listening_ports(os.getpid()))

    def test_pid_reuse_refuses_termination(self):
        record = identity(os.getpid())
        record['start_ticks'] = 'wrong'
        with mock.patch('dora_scope.pidfd_call') as send:
            with self.assertRaisesRegex(ValueError, 'PID reuse'):
                self.scope.terminate({'daemon': record})
            send.assert_not_called()

    def test_pidfd_stops_only_the_recorded_child(self):
        child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
        try:
            self.scope.terminate({'daemon': identity(child.pid)})
            self.assertEqual(child.wait(timeout=5), -15)
        finally:
            if child.poll() is None:
                child.terminate()
                child.wait(timeout=5)

    def test_busy_port_refuses_start_before_any_child(self):
        with socket.socket() as listener:
            listener.bind(('127.0.0.1', 0))
            self.spec['control_port'] = listener.getsockname()[1]
            with mock.patch('dora_scope.subprocess.Popen') as spawn:
                with self.assertRaises(OSError):
                    self.scope.start()
                spawn.assert_not_called()

    def test_unknown_episode_blocks_service_shutdown(self):
        path = self.work / 'control/state.json'
        path.parent.mkdir()
        path.write_text(json.dumps({'active': 'unknown-native-acceptance'}))
        with self.assertRaisesRegex(ValueError, 'active/unknown'):
            stop_allowed(self.work)
        path.write_text(json.dumps({'active': None}))
        stop_allowed(self.work)

    def test_renew_retains_dead_receipt_without_starting_processes(self):
        self.scope.root.mkdir()
        content = json.dumps({'coordinator': {'pid': 2147483647}, 'daemon': {'pid': 2147483646}})
        self.scope.receipt.write_text(content)
        with mock.patch('dora_scope.subprocess.Popen') as spawn:
            target = renew_scope(self.scope, self.work, SimpleNamespace(status='stopped'))
            spawn.assert_not_called()
        self.assertEqual(target.read_text(), content)
        self.assertFalse(self.scope.receipt.exists())

    def test_renew_refuses_live_recorded_process(self):
        self.scope.root.mkdir()
        self.scope.receipt.write_text(json.dumps({'coordinator': identity(os.getpid())}))
        with self.assertRaisesRegex(ValueError, 'still live'):
            renew_scope(self.scope, self.work, SimpleNamespace(status='stopped'))
        self.assertTrue(self.scope.receipt.exists())


if __name__ == '__main__':
    unittest.main()
