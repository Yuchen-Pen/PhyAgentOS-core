"""Query an existing journal and optionally close its root; never dispatch an episode."""

import argparse
import asyncio
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from controller import RoboDojoAdapter
from native_client import NativeClient
from preflight import load_config
from run import read_plan
from runtime import NativeRuntime
from PhyAgentOS.forge.tool_client import ForgeToolClient


async def reconcile(config, plan, close_root=False):
    runtime = NativeRuntime(config)
    runtime.verify_bundle()
    runtime.scope.owned()
    # Reconciliation still works when a policy node has exited. It requires the
    # owned Gateway, rather than claiming all four nodes are ready.
    gateway = f"http://127.0.0.1:{config['gateway_port']}"
    async with ForgeToolClient(gateway, timeout_s=10) as client:
        native = NativeClient(client, config)
        await native.describe()
        adapter = RoboDojoAdapter(native, plan)
        try:
            if not adapter.resumed:
                raise ValueError('Reconciliation requires an existing journal')
            for episode in adapter.state['episodes'].values():
                await adapter.poll(episode)
                if close_root:
                    # stop() sends no native POST after an authoritative terminal
                    # snapshot; it only releases the worker and records the ack.
                    await adapter.stop(episode)
            if close_root:
                for root in adapter.state['roots'].values():
                    root['stop_requested'] = True
                adapter.settle_roots()
                adapter.persist()
            return {'episodes_dispatched': 0, 'reconciled_existing_journal': True,
                    'active': adapter.state['active'], 'roots': adapter.state['roots'],
                    'episodes': {key: value['snapshot'] for key, value in adapter.state['episodes'].items()}}
        finally:
            adapter.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('plan')
    parser.add_argument('--close-root', action='store_true', help='Request boundary stop of existing work and close its root only after acknowledgement')
    args = parser.parse_args()
    config = load_config(args.config)
    journal = Path(config['work_root']) / 'control/state.json'
    if not journal.is_file():
        raise ValueError('No existing controller journal; nothing to reconcile')
    print(json.dumps(asyncio.run(reconcile(config, read_plan(args.plan, config), args.close_root)),
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
