"""Private Dora 0.4.1 endpoints and Linux process ownership, without default-service fallback."""

import contextlib
import ctypes
import fcntl
import hashlib
import json
import os
import platform
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time


def pidfd_call(name, *args):
    # Conda builds can omit both stdlib and glibc wrappers. These are the Linux
    # x86_64 syscall numbers from asm/unistd_64.h on the target server.
    if sys.platform != 'linux' or platform.machine() != 'x86_64':
        raise ValueError('Process signalling requires the verified Linux x86_64 host')
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    number = {'pidfd_open': 434, 'pidfd_send_signal': 424}[name]
    result = libc.syscall(ctypes.c_long(number), *args)
    if result < 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))
    return result


def atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def identity(pid):
    proc = Path('/proc') / str(pid)
    fields = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
    if proc.stat().st_uid != os.getuid() or fields[0] == 'Z':
        raise ValueError('Dora process is not live and owned by this user')
    return {'pid': pid, 'start_ticks': fields[19], 'exe': str((proc / 'exe').resolve()),
            'argv': (proc / 'cmdline').read_bytes().rstrip(b'\0').decode().split('\0')}


def listening_ports(pid):
    sockets = set()
    for fd in (Path('/proc') / str(pid) / 'fd').iterdir():
        try:
            link = os.readlink(fd)
            if link.startswith('socket:['):
                sockets.add(link[8:-1])
        except FileNotFoundError:
            pass
    ports = set()
    for name in ('tcp', 'tcp6'):
        for line in (Path('/proc/net') / name).read_text().splitlines()[1:]:
            fields = line.split()
            if fields[3] == '0A' and fields[9] in sockets:
                ports.add(int(fields[1].split(':')[1], 16))
    return ports


class DoraScope:
    def __init__(self, work):
        self.root = Path(work) / 'private-dora'
        self.definition = self.root / 'definition.json'
        self.receipt = self.root / 'processes.json'

    @contextlib.contextmanager
    def lock(self):
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / 'operation.lock').open('a') as stream:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            yield

    def configure(self, binary, ports):
        binary = Path(binary).resolve()
        version = subprocess.check_output([str(binary), '--version'], text=True, timeout=10)
        if version.splitlines()[0] != 'dora-cli 0.4.1':
            raise ValueError('Expected the pinned Dora CLI 0.4.1')
        if len(set(ports)) != 3 or any(type(p) is not int or not 1024 <= p <= 65535 for p in ports):
            raise ValueError('Choose three distinct unprivileged private ports')
        value = {'binary': str(binary), 'binary_sha256': hashlib.sha256(binary.read_bytes()).hexdigest(),
                 'control_port': ports[0], 'daemon_port': ports[1], 'local_port': ports[2],
                 'proxy_source': str(Path(__file__).resolve()),
                 'proxy_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        if self.definition.exists():
            if self.load() != value:
                raise ValueError('Private Dora definition already exists; retain it for reconciliation')
            return
        self.root.mkdir(parents=True, exist_ok=True)
        atomic_json(self.definition, value)
        bindir = self.root / 'bin'
        bindir.mkdir(exist_ok=True)
        wrapper = bindir / 'dora'
        wrapper.write_text('#!/usr/bin/python3\nimport os,sys\nos.execv(' + repr(sys.executable) +
                           ', [' + repr(sys.executable) + ', ' + repr(str(Path(__file__).resolve())) +
                           ', ' + repr(str(self.root.parent)) + '] + sys.argv[1:])\n')
        wrapper.chmod(0o755)

    def load(self):
        value = json.loads(self.definition.read_text())
        for path, expected in ((value['binary'], value['binary_sha256']),
                               (value['proxy_source'], value['proxy_sha256'])):
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
                raise ValueError('Private Dora executable/source changed')
        return value

    def activate(self):
        self.load()
        os.environ['PATH'] = str(self.root / 'bin') + os.pathsep + os.environ.get('PATH', '')

    def owned(self, *, ports=True):
        spec = self.load()
        records = json.loads(self.receipt.read_text())
        if set(records) != {'coordinator', 'daemon'}:
            raise ValueError('Private Dora process receipt is incomplete')
        for role, record in records.items():
            if identity(record['pid']) != record or record['exe'] != spec['binary']:
                raise ValueError('Private Dora process identity changed')
            if record['argv'] != self.command(role):
                raise ValueError('Private Dora process arguments differ from its definition')
            required = ({spec['control_port'], spec['daemon_port']} if role == 'coordinator'
                        else {spec['local_port']})
            if ports and not required <= listening_ports(record['pid']):
                raise ValueError('Private Dora endpoints are not owned by the recorded process')
        return records

    def command(self, role):
        s = self.load()
        if role == 'coordinator':
            return [s['binary'], role, '--interface', '127.0.0.1', '--port', str(s['daemon_port']),
                    '--control-interface', '127.0.0.1', '--control-port', str(s['control_port'])]
        if role == 'daemon':
            return [s['binary'], role,
                    '--local-listen-port', str(s['local_port']), '--coordinator-addr', '127.0.0.1',
                    '--coordinator-port', str(s['daemon_port'])]
        raise ValueError('Unknown private Dora role')

    def proxy_command(self, args):
        s = self.load()
        if args == ['--version']:
            return [s['binary'], '--version']
        self.owned()
        if not args or args[0] not in {'check', 'start', 'list', 'stop'}:
            raise ValueError('Dora up/destroy and other unscoped operations are forbidden')
        if any(a.startswith('--coordinator-') for a in args):
            raise ValueError('Cannot override private Dora endpoints')
        if args[0] in {'start', 'stop'}:
            if '--name' not in args or args[args.index('--name') + 1] != 'paos-robodojo-bench-pi05-pi05':
                raise ValueError('Cannot operate on another dataflow')
        command = [s['binary'], *args, '--coordinator-addr', '127.0.0.1',
                   '--coordinator-port', str(s['control_port'])]
        if args[0] == 'start':
            command.append('--detach')
        return command

    def start(self):
        spec = self.load()
        if self.receipt.exists():
            self.owned()
            return
        # Reject occupied ports before launching anything; never attach to a foreign service.
        with contextlib.ExitStack() as stack:
            for p in (spec['control_port'], spec['daemon_port'], spec['local_port']):
                sock = stack.enter_context(socket.socket())
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind(('127.0.0.1', p))
        records = {}
        try:
            for role in ('coordinator', 'daemon'):
                with (self.root / (role + '.log')).open('ab') as log:
                    process = subprocess.Popen(self.command(role), stdout=log, stderr=subprocess.STDOUT,
                                               stdin=subprocess.DEVNULL, start_new_session=True)
                # Popen may return before exec has installed the final executable.
                deadline = time.monotonic() + 10
                while True:
                    if process.poll() is not None:
                        raise ValueError(role + ' exited; inspect its private log')
                    record = identity(process.pid)
                    if record['exe'] == spec['binary']:
                        records[role] = record
                        atomic_json(self.receipt, records)
                        break
                    if time.monotonic() > deadline:
                        raise ValueError(role + ' did not exec the pinned binary')
                    time.sleep(.05)
                time.sleep(.3)
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline:
                try:
                    self.owned()
                    result = subprocess.run(self.proxy_command(['check']), capture_output=True, timeout=5)
                    if result.returncode == 0:
                        return
                except (OSError, ValueError):
                    pass
                time.sleep(.25)
            raise ValueError('Private Dora did not become ready; inspect private logs')
        except BaseException:
            # Only known child identities; no port-based/global process termination.
            self.terminate(records)
            raise

    def terminate(self, records=None):
        if records is None:
            records = self.owned(ports=False)
        for role in ('daemon', 'coordinator'):
            record = records.get(role)
            if record is None:
                continue
            try:
                if identity(record['pid']) != record:
                    raise ValueError('Refuse termination after PID reuse')
                fd = pidfd_call('pidfd_open', record['pid'], 0)
                try:
                    if identity(record['pid']) != record:
                        raise ValueError('Refuse termination after PID reuse')
                    pidfd_call('pidfd_send_signal', fd, signal.SIGTERM, None, 0)
                finally:
                    os.close(fd)
            except (FileNotFoundError, ProcessLookupError):
                pass
        # Retain the receipt as evidence; restarting requires explicit reconciliation.


if __name__ == '__main__':
    try:
        scope = DoraScope(sys.argv[1])
        os.execv(scope.load()['binary'], scope.proxy_command(sys.argv[2:]))
    except Exception as error:
        print('Private Dora: ' + str(error), file=sys.stderr)
        raise SystemExit(1)
