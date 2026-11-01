"""Check a case plan; --execute connects to an already-managed native Pi0.5 service."""

import argparse
import asyncio
import json
from pathlib import Path
import subprocess
import sys
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from PhyAgentOS.forge.benchmark.contracts import BenchmarkPlan
from PhyAgentOS.forge.tool_client import ForgeToolClient

from control_bundle import CoreExperiment
from controller import RoboDojoAdapter
from interrupts import graceful_signals
from native_client import NativeClient
from preflight import check, load_config, sha256
from results import immutable_file, json_bytes
from runtime import NativeRuntime


def read_plan(path, config):
    value = json.loads(Path(path).read_text())
    required = {"cases", "rounds", "parallelism", "max_decisions", "episode_timeout_s", "model_timeout_s"}
    if set(value) != required:
        raise ValueError("plan must explicitly set cases, rounds, parallelism and all budgets")
    plan = BenchmarkPlan.model_validate(value).model_dump()
    if plan["parallelism"] != 1 or plan["rounds"] != 1 or plan["max_decisions"] != 1:
        raise ValueError("first implementation requires rounds=parallelism=max_decisions=1")
    for case in plan["cases"]:
        environment = case["environment"]
        if set(environment) != {"native_task_id", "task_name", "init_state_id", "layout_seed"}:
            raise ValueError("case must explicitly identify the native task, initial state and layout directory")
        for key in ("native_task_id", "init_state_id", "layout_seed"):
            if type(environment[key]) is not int or environment[key] < 0:
                raise ValueError("invalid case integer: " + key)
        if not isinstance(environment["task_name"], str) or not environment["task_name"]:
            raise ValueError("task_name must be nonempty")
        if environment["layout_seed"] != config["layout_seed"]:
            raise ValueError("case layout seed differs from deployment")
    return plan


async def execute(config, plan, *, new_batch=False):
    resources = check(config)
    if not resources["static_resources_present"]:
        raise ValueError("deployment preflight failed: " + ", ".join(c["name"] for c in resources["checks"] if not c["ok"]))
    runtime = NativeRuntime(config)
    state = await asyncio.to_thread(runtime.require_ready)
    root = Path(config["work_root"]) / "core-runs" / ("run_" + uuid4().hex)
    root.mkdir(parents=True)
    configuration = {"deployment": config, "plan": plan, "explicit_new_batch": new_batch,
                     "managed_runtime": state.to_dict(),
                     "native_provenance": runtime.verify_bundle(), "control_sources": {
                         p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")},
                     "policy_environment": json.loads((Path(config["work_root"]) / "policy-ready.json").read_text()),
                     "models": json.loads(Path(config["models_receipt"]).read_text()),
                     "core_commit": subprocess.check_output(["git", "-C", str(Path(__file__).resolve().parents[2]), "rev-parse", "HEAD"], text=True).strip()}
    immutable_file(root / "configuration.json", json_bytes(configuration))
    async with ForgeToolClient(state.gateway_url, timeout_s=10) as client:
        native = NativeClient(client, config)
        capabilities = await native.describe()
        immutable_file(root / "native-capabilities.json", json_bytes(capabilities))
        adapter = RoboDojoAdapter(native, plan, allow_new_batch=new_batch)
        experiment = CoreExperiment(root, adapter)
        try:
            with graceful_signals() as stop_event:
                await experiment.setup()
                report = await experiment.run(stop_event=stop_event)
            print(json.dumps({"report": str(root / "core-report.json"), "settled": report["run"]["settled"],
                              "outer_model_calls": report["outer_model_calls"]}, ensure_ascii=False))
            return 0 if report["run"]["settled"] else 1
        finally:
            await experiment.close()
            if hasattr(experiment, "task_id"):
                immutable_file(root / "shutdown-report.json", json_bytes(experiment.report()))
            adapter.close()
    # No automatic native deployment stop: unknown physical work keeps its own
    # runtime alive. Native service lifecycle and per-case stop are separate.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("plan")
    parser.add_argument("--execute", action="store_true", help="Dispatch actual episodes on the already-managed personal deployment")
    parser.add_argument('--new-batch', action='store_true', help='Explicitly admit another identical plan only after every previous root/episode is proven stopped; retains history')
    args = parser.parse_args()
    config = load_config(args.config)
    plan = read_plan(args.plan, config)
    if not args.execute:
        print(json.dumps({"plan": plan, "native_capabilities_checked": False,
                          "process_started": False, "episodes_dispatched": 0}, ensure_ascii=False, indent=2))
        return 0
    return asyncio.run(execute(config, plan, new_batch=args.new_batch))


if __name__ == "__main__":
    raise SystemExit(main())
