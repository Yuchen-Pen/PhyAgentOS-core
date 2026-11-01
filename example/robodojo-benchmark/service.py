"""Configure/start/status/stop this user's native evaluator on private Dora endpoints."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import os
import hashlib

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dora_scope import DoraScope, identity
from preflight import check, load_config
from runtime import NAME, NativeRuntime


def stop_allowed(work):
    journal = Path(work) / 'control/state.json'
    if journal.exists():
        value = json.loads(journal.read_text())
        if value.get('active') is not None:
            raise ValueError('Native episode is active/unknown; reconcile its boundary stop before service shutdown')


def renew_scope(scope, work, state):
    """Retain dead process receipts before an explicitly requested service restart."""
    stop_allowed(work)
    if state is None or state.status != 'stopped':
        raise ValueError('Renew requires a normally stopped managed runtime')
    journal = Path(work) / 'control/state.json'
    if journal.exists():
        value = json.loads(journal.read_text())
        if any(not r['stopped'] for r in value['roots'].values()):
            raise ValueError('Reconcile all old roots before renewing service processes')
    records = json.loads(scope.receipt.read_text())
    for record in records.values():
        try:
            current = identity(record['pid'])
        except FileNotFoundError:
            continue
        except ValueError:
            proc = Path('/proc') / str(record['pid'])
            fields = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
            if proc.stat().st_uid == os.getuid() and fields[0] == 'Z' and fields[19] == record['start_ticks']:
                continue
            raise
        raise ValueError('An old process is still live or its PID was reused; inspect before renewing')
    history = scope.root / 'history'
    history.mkdir(exist_ok=True)
    target = history / ('processes-' + hashlib.sha256(scope.receipt.read_bytes()).hexdigest()[:16] + '.json')
    if target.exists():
        raise ValueError('Historical receipt already exists; inspect rather than overwrite')
    os.rename(scope.receipt, target)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('operation', choices=['configure', 'start', 'status', 'stop', 'renew'])
    parser.add_argument('--dora-binary')
    parser.add_argument('--ports', type=int, nargs=3, default=[19032, 19033, 19034],
                        metavar=('CONTROL', 'DAEMON', 'LOCAL'))
    args = parser.parse_args()
    config = load_config(args.config)
    scope = DoraScope(config['work_root'])
    with scope.lock():
        if args.operation == 'configure':
            if not args.dora_binary:
                parser.error('configure requires --dora-binary')
            scope.configure(args.dora_binary, args.ports)
            print(json.dumps({'private_dora': scope.load(), 'process_started': False}))
            return
        runtime = NativeRuntime(config)
        runtime.verify_bundle()
        if args.operation == 'renew':
            print(json.dumps({'archived_process_receipt': str(renew_scope(scope, config['work_root'], runtime.state.load(NAME))),
                              'process_started': False}))
            return
        if args.operation == 'start':
            resources = check(config)
            if not resources['static_resources_present']:
                raise ValueError('Deployment preflight failed')
            scope.start()
            # Never call RuntimeManager with default-service fallback.
            scope.owned()
            scope.activate()
            print(json.dumps(runtime.manager.start(NAME, 'pi05').to_dict()))
        elif args.operation == 'status':
            state = runtime.state.load(NAME)
            if state is not None and state.status == 'stopped':
                print(json.dumps({'state': state.to_dict(), 'ready': False, 'live_checked': False}))
                return
            scope.owned()
            report = runtime.manager.status(NAME)
            print(json.dumps({'ready': report.ready, **asdict(report)}))
        else:
            scope.owned()
            stop_allowed(config['work_root'])
            state = runtime.manager.stop(NAME)
            if runtime.manager._flow_running(state.flow_name):
                raise ValueError('Native flow is still running; keep private Dora alive')
            scope.terminate()
            print(json.dumps(state.to_dict()))


if __name__ == '__main__':
    main()
