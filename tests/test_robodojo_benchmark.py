"""Real Core and private control path over an explicitly synthetic native Gateway.

No simulation, model weights, CUDA or model/API requests are used by these tests.
"""

import asyncio
import copy
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import httpx

EXAMPLE = Path(__file__).resolve().parents[1] / "example/robodojo-benchmark"
sys.path.insert(0, str(EXAMPLE))

from controller import RoboDojoAdapter
from control_bundle import CoreExperiment
from native_client import NativeClient
from results import archive_result, validate_result
from step_limits import resolve_step_limit
from runtime import NativeRuntime
from run import read_plan
from PhyAgentOS.forge.benchmark.private import PrivateBenchmarkTransport
from PhyAgentOS.forge.interaction.client import query_outputs
from PhyAgentOS.forge.interaction.contracts import digest
from PhyAgentOS.forge.tool_client import ForgeToolAPIError, ForgeToolClient


def case(key="a"):
    return {"case_key": key, "environment": {"native_task_id": 0,
            "task_name": "put_bottles_into_dustbin", "init_state_id": 0, "layout_seed": 0}}


def plan():
    return {"cases": [case(), case("b")], "rounds": 1, "parallelism": 1,
            "max_decisions": 1, "episode_timeout_s": 3, "model_timeout_s": 1}


class SyntheticGateway:
    def __init__(self, config):
        self.config = config
        self.starts, self.stops, self.records = [], [], {}
        self.outcomes = [True, False]
        self.pending = False
        self.delay = 0
        self.max_active = 0
        self.lose_start = self.query_error = self.missing_file = False
        self.phase_only = self.reset_error = False

    def repro(self):
        return {"env_cfg_type": "arx_x5", "action_type": "joint", "policy_id": "Pi_05",
                "step_limit_mode": "native"}

    def capabilities(self):
        return {"benchmark": "robodojo", "suite": self.config["suite"], "policy_id": "Pi_05",
                "num_tasks": 1, "tasks": [{"task_id": 0, "name": self.config["suite"], "num_init_states": 1}],
                "defaults": {"seed": 0}, "reproducibility": self.repro()}

    def result(self, invocation):
        success = self.outcomes[(int(invocation.split("-")[-1]) - 1) % len(self.outcomes)]
        record = self.records[invocation]
        stopped = record.get("stopped", False)
        episode = {"episode_id": "synthetic-episode", "task_id": 0, "init_state_id": 0, "run_index": 0,
                   "task_description": "synthetic fixture", "success": success, "num_steps": 10,
                   "termination": "success" if success else "timed_out", "return_value": float(success),
                   "error_code": None, "metrics": {"layout_id": 7,
                   "step_limits": resolve_step_limit(self.config["max_steps"], 10, "native")}}
        raw = {"schema_version": "benchmark_execution_result_v1", "benchmark": "robodojo",
               "suite": self.config["suite"], "execution_mode": "target_native", "seed": 0,
               "reproducibility": self.repro(), "status": "stopped" if stopped else "succeeded",
               "max_steps": 10, "episodes": [episode], "total_episodes": 1, "completed_episodes": 1,
               "valid_episodes": 1, "successes": int(success), "success_rate": float(success), "num_steps": 10,
               "primary_metric": {"name": "success_rate", "value": float(success)}, "metrics": {}}
        if self.reset_error:
            raw.update(status='failed', episodes=[], completed_episodes=0, valid_episodes=0,
                       successes=0, success_rate=0.0, num_steps=0,
                       primary_metric={'name': 'success_rate', 'value': 0.0})
        path = Path(self.config["work_root"]) / "results" / f"{self.config['suite']}_{invocation}.json"
        path.parent.mkdir(exist_ok=True)
        if not self.missing_file:
            path.write_text(json.dumps(raw))
        outputs = {"run_id": invocation, "status": "partial" if stopped else "succeeded",
                   "completed_episodes": 1, "total_episodes": 1, "success_rate": float(success),
                   "result_file": str(path), "failure_summary": None}
        if self.reset_error:
            outputs.update(status='failed', completed_episodes=0, success_rate=0.0)
            return {'status': 'failed', 'outputs': outputs,
                    'error': {'code': 'BENCH_RUN_FAILED'}}, raw
        return {"status": "stopped" if stopped else "succeeded", "outputs": outputs}, raw

    async def __call__(self, request):
        path = request.url.path
        args = json.loads(request.content).get("arguments", {}) if request.content else {}
        def response(value, status=200):
            return httpx.Response(status, json={"ok": True, "data": value})
        if path.startswith("/tools/") and request.method == "GET":
            tool = path.split("/")[2]
            return response({"tool_id": tool, "endpoint_id": "benchmark.controller",
                "operation": "describe" if tool.endswith("describe") else "run",
                "semantics": "query" if tool.endswith("describe") else "session"})
        if path == "/tools/benchmark.controller/describe:invoke":
            return response({"response": {"result": {"status": "succeeded", "outputs": self.capabilities()}}})
        if path == "/tools/benchmark.run:invoke":
            self.starts.append(args)
            invocation = "native-" + str(len(self.starts))
            self.records[invocation] = {}
            self.max_active = max(self.max_active, sum(not r.get("terminal") for r in self.records.values()))
            if self.lose_start:
                raise httpx.ReadTimeout("synthetic lost acceptance")
            return response({"invocation_id": invocation, "attempt_id": invocation}, 202)
        if path.startswith("/invocations/"):
            invocation = path.split("/")[2]
            if path.endswith("/stop"):
                self.stops.append(invocation)
                self.records[invocation]["stopped"] = True
                return response({"status": "requested"}, 202)
            if self.query_error:
                raise httpx.ReadTimeout("synthetic temporary disconnect")
            if path.endswith("/result"):
                tool, _ = self.result(invocation)
                return response({"status": "available", "result": tool})
            record = self.records[invocation]
            record["polls"] = record.get("polls", 0) + 1
            phase = "running" if self.pending or record["polls"] <= self.delay else ("stopped" if record.get("stopped") else "completed")
            if self.reset_error:
                phase = 'failed'
            record["terminal"] = phase != "running"
            return response({'invocation_id': invocation, 'phase': phase} if self.phase_only else
                            {"invocation_id": invocation, "status": phase, "phase": phase})
        raise AssertionError(path)


class BenchmarkTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="robodojo core fixture ")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config = {"work_root": str(self.root), "suite": "put_bottles_into_dustbin",
                       "gateway_port": 19031, "layout_seed": 0, "max_steps": 700, "step_limit_mode": "native"}
        self.gateway = SyntheticGateway(self.config)
        self.client = ForgeToolClient("http://127.0.0.1:19031", transport=httpx.MockTransport(self.gateway))
        self.addAsyncCleanup(self.client.close)
        self.native = NativeClient(self.client, self.config)
        await self.native.describe()
        self.adapter = RoboDojoAdapter(self.native, plan())
        self.addCleanup(lambda: self.adapter.close())
        self.private = ForgeToolClient("http://benchmark.private", transport=PrivateBenchmarkTransport(self.adapter))
        self.addAsyncCleanup(self.private.close)
        self.root_args = {"task_id": "task", "run_id": "core-run", "binding_digest": "binding",
                          "plan_digest": digest(self.adapter.plan)}
        await self.private.start_session("benchmark.run", self.root_args)
        self.scope = {**{k: self.root_args[k] for k in ("task_id", "run_id", "binding_digest")},
                      "subtask_id": "child", "attempt_id": "attempt", "worker_id": "robodojo-worker", "world_epoch": "world"}

    def start_args(self, **extra):
        return {**self.scope, "operation": "episode.start", "request_id": "start-1", "case": case(),
                "lessons_digest": digest([]), **extra}

    async def submit(self, arguments):
        return await self.private.invoke_action("benchmark.submit", arguments)

    async def snapshot(self, **extra):
        return query_outputs(await self.private.invoke_query_tool("benchmark.snapshot", {**self.scope, "view": "episode", **extra}))

    async def test_case_maps_to_one_native_episode_and_replay_does_not_resend(self):
        args = self.start_args()
        first = await self.submit(args)
        self.assertEqual(first, await self.submit(args))
        self.assertEqual(self.gateway.starts, [{"task_ids": [0], "init_state_ids": [0], "num_runs": 1, "seed": 0, "max_steps": 700}])
        with self.assertRaises(ForgeToolAPIError):
            await self.submit({**args, "case": case("b")})

    async def test_native_success_and_original_evidence_are_archived(self):
        await self.submit(self.start_args())
        snapshot = await self.snapshot()
        self.assertEqual((snapshot["state"], snapshot["outcome"], snapshot["native_score"], snapshot["stopped"]),
                         ("terminal", "success", 1, True))
        evidence = snapshot["observation"]["evidence"]
        self.assertEqual(evidence["native_invocation"], "native-1")
        directory = Path(self.adapter.state["episodes"]["child"]["directory"])
        self.assertTrue((directory / "native-result.json").is_file())
        self.assertEqual(json.loads((directory / "result-index.json").read_text())["core_scope"], self.scope)

    async def test_tool_success_can_be_robot_failure(self):
        self.gateway.outcomes = [False]
        await self.submit(self.start_args())
        result = await self.snapshot()
        self.assertEqual((result["outcome"], result["native_score"]), ("failure", 0))

    async def test_real_gateway_phase_without_status_is_terminal(self):
        self.gateway.phase_only = True
        await self.submit(self.start_args())
        result = await self.snapshot()
        self.assertEqual((result['state'], result['outcome']), ('terminal', 'success'))

    async def test_reset_error_with_no_episode_is_infrastructure_failure(self):
        self.gateway.phase_only = self.gateway.reset_error = True
        await self.submit(self.start_args())
        result = await self.snapshot()
        self.assertEqual((result['state'], result['outcome'], result['native_score']),
                         ('terminal', 'infra_error', None))

    async def test_missing_result_is_infrastructure_error_after_native_terminal(self):
        self.gateway.missing_file = True
        await self.submit(self.start_args())
        result = await self.snapshot()
        self.assertEqual((result["outcome"], result["stopped"]), ("infra_error", True))

    async def test_stop_acceptance_does_not_release_worker_before_boundary(self):
        self.gateway.pending = True
        await self.submit(self.start_args())
        await self.submit({**self.scope, "operation": "episode.stop", "request_id": "stop-1"})
        snapshot = await self.snapshot()
        self.assertFalse(snapshot["stopped"])
        self.assertEqual(self.adapter.state["active"], "child")
        with self.assertRaises(ForgeToolAPIError):
            await self.submit(self.start_args(subtask_id="child-2", request_id="start-2", case=case("b")))
        self.gateway.pending = False
        self.assertTrue((await self.snapshot())["stopped"])
        self.assertIsNone(self.adapter.state["active"])
        self.assertEqual(self.gateway.stops, ["native-1"])

    async def test_lost_native_start_stays_unknown_and_prevents_another_case(self):
        self.gateway.lose_start = True
        args = self.start_args()
        await self.submit(args)
        await self.submit(args)
        self.assertEqual((await self.snapshot())["state"], "unknown")
        await self.submit({**self.scope, "operation": "episode.stop", "request_id": "stop-1"})
        self.assertFalse((await self.snapshot())["stopped"])
        with self.assertRaises(ForgeToolAPIError):
            await self.submit(self.start_args(subtask_id="child-2", request_id="start-2"))
        self.assertEqual(len(self.gateway.starts), 1)

    async def test_known_invocation_disconnect_can_reconcile_without_restart(self):
        await self.submit(self.start_args())
        self.gateway.query_error = True
        self.assertEqual((await self.snapshot())["state"], "unknown")
        self.gateway.query_error = False
        self.assertEqual((await self.snapshot())["outcome"], "success")
        self.assertEqual(len(self.gateway.starts), 1)

    async def test_restart_keeps_journal_and_never_resends_native_start(self):
        self.gateway.lose_start = True
        args = self.start_args()
        await self.submit(args)
        self.adapter.close()
        self.adapter = RoboDojoAdapter(self.native, plan())
        self.private._client._transport.adapter = self.adapter
        await self.submit(args)
        self.assertEqual((await self.snapshot())["state"], "unknown")
        self.assertEqual(len(self.gateway.starts), 1)

    async def test_explicit_new_batch_rejects_unknown_previous_execution(self):
        self.gateway.lose_start = True
        await self.submit(self.start_args())
        self.adapter.close()
        with self.assertRaisesRegex(ValueError, 'authoritatively stopped'):
            RoboDojoAdapter(self.native, plan(), allow_new_batch=True)
        self.assertEqual(len(self.gateway.starts), 1)

    async def test_explicit_new_batch_keeps_all_history_after_proven_stop(self):
        await self.submit(self.start_args())
        await self.snapshot()
        root = self.adapter.state['roots']['task']
        await self.private.stop_session(root['session_invocation_id'])
        self.assertTrue(root['stopped'])
        self.adapter.close()
        self.adapter = RoboDojoAdapter(self.native, plan(), allow_new_batch=True)
        self.private._client._transport.adapter = self.adapter
        await self.private.start_session('benchmark.run', {'task_id': 'task2', 'run_id': 'core-run2',
            'binding_digest': 'binding2', 'plan_digest': digest(self.adapter.plan)})
        await self.submit(self.start_args(task_id='task2', run_id='core-run2', binding_digest='binding2',
                                         subtask_id='child2', request_id='start2'))
        self.assertTrue(self.adapter.state['roots']['task']['stopped'])
        self.assertEqual(len(self.adapter.state['episodes']), 2)
        self.assertEqual(len(self.gateway.starts), 2)

    async def test_worker_has_one_persistent_owner(self):
        with self.assertRaises(BlockingIOError):
            RoboDojoAdapter(self.native, plan())

    async def test_stopping_historical_root_cannot_stop_the_current_owner(self):
        await self.submit(self.start_args())
        await self.snapshot()
        old_invocation = self.adapter.state['roots']['task']['session_invocation_id']
        await self.private.stop_session(old_invocation)
        self.adapter.close()
        self.adapter = RoboDojoAdapter(self.native, plan(), allow_new_batch=True)
        self.private._client._transport.adapter = self.adapter
        await self.private.start_session('benchmark.run', {'task_id': 'task2', 'run_id': 'core-run2',
            'binding_digest': 'binding2', 'plan_digest': digest(self.adapter.plan)})
        self.gateway.pending = True
        await self.submit(self.start_args(task_id='task2', run_id='core-run2', binding_digest='binding2',
                                         subtask_id='child2', request_id='start2'))
        await self.private.stop_session(old_invocation)
        self.assertEqual(self.gateway.stops, [])
        self.assertEqual(self.adapter.state['active'], 'child2')
        self.assertFalse(self.adapter.state['roots']['task2']['stop_requested'])

    async def test_changed_journal_refuses_dispatch_before_native_post(self):
        self.adapter.path.write_text("{}")
        with self.assertRaisesRegex(RuntimeError, "journal"):
            await self.submit(self.start_args())
        self.assertEqual(self.gateway.starts, [])

    async def test_root_stop_waits_for_the_active_episode(self):
        self.gateway.pending = True
        await self.submit(self.start_args())
        invocation = self.adapter.state["roots"]["task"]["session_invocation_id"]
        await self.private.stop_session(invocation)
        self.assertEqual((await self.private.invocation_status(invocation))["data"]["status"], "running")
        self.gateway.pending = False
        await self.snapshot()
        self.assertEqual((await self.private.invocation_status(invocation))["data"]["status"], "stopped")

    async def test_cross_scope_and_outer_decisions_are_rejected(self):
        await self.submit(self.start_args())
        with self.assertRaises(ForgeToolAPIError):
            await self.snapshot(world_epoch="another-world")
        with self.assertRaises(ForgeToolAPIError):
            await self.submit({**self.scope, "operation": "decision.submit", "request_id": "decision"})

    async def test_history_is_stable_when_polling_an_unchanged_episode(self):
        self.gateway.pending = True
        await self.submit(self.start_args())
        first = await self.snapshot()
        self.assertEqual(first, await self.snapshot())
        history = await self.snapshot(view="history")
        self.assertEqual(digest(history["records"]), first["history_digest"])

    async def test_capability_task_and_layout_validation(self):
        for change in ({"native_task_id": 8}, {"task_name": "another-task"}, {"init_state_id": 1},
                       {"layout_seed": 1}, {"init_state_id": True}):
            value = case();value["environment"].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.native.arguments_for(value)
        self.assertEqual(self.gateway.starts, [])

    async def run_real_core(self, *, lost_native_start=False, lost_control_reply=False):
        # Use a separate clean journal; the fixture above deliberately admitted a root.
        self.adapter.close()
        core_root = (Path(os.environ["ROBODOJO_TEST_EVIDENCE_DIR"]) / "core-integration"
                     if os.environ.get("ROBODOJO_TEST_EVIDENCE_DIR") and not lost_native_start and not lost_control_reply
                     else self.root / "core-integration")
        core_root.mkdir()
        config = {**self.config, "work_root": str(core_root)}
        gateway = SyntheticGateway(config)
        gateway.phase_only = True  # Match the actual Gateway 1.1.1 status envelope.
        gateway.delay = 1
        gateway.lose_start = lost_native_start
        async with ForgeToolClient("http://127.0.0.1:19031", transport=httpx.MockTransport(gateway)) as client:
            native = NativeClient(client, config)
            await native.describe()
            adapter = RoboDojoAdapter(native, plan())
            experiment = CoreExperiment(core_root / "run", adapter)
            try:
                await experiment.setup()
                experiment.supervisor.service.deployment.poll_interval_s = 0.001
                if lost_control_reply:
                    class LostReply:
                        dropped = False
                        async def handle(self, request):
                            result = await adapter.handle(request)
                            args = json.loads(request.content).get("arguments", {}) if request.content else {}
                            if args.get("operation") == "episode.start" and not self.dropped:
                                self.dropped = True
                                raise httpx.ReadTimeout("synthetic lost private acceptance")
                            return result
                    experiment.transport.adapter = LostReply()
                report = await experiment.run()
                if os.environ.get("ROBODOJO_TEST_EVIDENCE_DIR") and not lost_native_start and not lost_control_reply:
                    (core_root / "VALIDATION_SCOPE.json").write_text(json.dumps({
                        "native_gateway": "synthetic test fixture", "real_core_used": True,
                        "simulation_started": False, "model_loaded": False, "gpu_used": False}, indent=2))
                return report, gateway
            finally:
                await experiment.close()
                adapter.close()

    async def test_real_core_two_cases_settle_without_outer_model_calls(self):
        report, gateway = await self.run_real_core()
        self.assertTrue(report["run"]["settled"], report["run"])
        self.assertEqual([c["outcome"] for c in report["children"]], ["success", "failure"])
        self.assertTrue(all(c["settled"] and c["history"] for c in report["children"]))
        self.assertTrue(all(s["state"] != "ready" for c in report["children"] for s in c["history"]))
        self.assertEqual(report["outer_model_calls"], 0)
        self.assertEqual(len(gateway.starts), 2)
        self.assertEqual(gateway.max_active, 1)

    async def test_core_interrupt_waits_for_boundary_and_does_not_start_queued_case(self):
        self.adapter.close()
        root = self.root / 'interrupt-integration'
        root.mkdir()
        config = {**self.config, 'work_root': str(root)}
        gateway = SyntheticGateway(config)
        gateway.pending = True
        gateway.phase_only = True
        async with ForgeToolClient('http://127.0.0.1:19031', transport=httpx.MockTransport(gateway)) as client:
            native = NativeClient(client, config)
            await native.describe()
            adapter = RoboDojoAdapter(native, plan())
            experiment = CoreExperiment(root / 'run', adapter)
            run_task = None
            try:
                await experiment.setup()
                stop_event = asyncio.Event()
                run_task = asyncio.create_task(experiment.run(stop_event=stop_event))
                async with asyncio.timeout(5):
                    while not gateway.starts:
                        await asyncio.sleep(0.01)
                    stop_event.set()
                    while not gateway.stops:
                        await asyncio.sleep(0.01)
                self.assertFalse(run_task.done())
                self.assertIsNotNone(adapter.state['active'])
                self.assertEqual(len(gateway.starts), 1)
                gateway.pending = False
                report = await asyncio.wait_for(run_task, timeout=5)
                self.assertTrue(report['run']['settled'], report['run'])
                self.assertTrue(report['task']['cancellation_requested'])
                self.assertTrue(all(child['settled'] for child in report['children']))
                self.assertEqual(report['children'][1]['outcome'], 'cancelled')
                self.assertEqual(len(gateway.starts), 1)
                self.assertEqual(report['outer_model_calls'], 0)
                self.assertIsNone(adapter.state['active'])
                self.assertTrue((root / 'run/stop-request.json').is_file())
            finally:
                if run_task is not None and not run_task.done():
                    run_task.cancel()
                    await asyncio.gather(run_task, return_exceptions=True)
                await experiment.close()
                adapter.close()

    async def test_real_core_reconciles_lost_control_receipt_without_native_resend(self):
        report, gateway = await self.run_real_core(lost_control_reply=True)
        self.assertTrue(report["run"]["settled"], report["run"])
        self.assertEqual(len(gateway.starts), 2)
        self.assertTrue(any(r["status"] == "unknown" for revision in report["task"]["revisions"]
                            for r in revision["execution_records"]))

    async def test_real_core_retains_worker_when_native_acceptance_is_unknown(self):
        report, gateway = await self.run_real_core(lost_native_start=True)
        self.assertFalse(report["run"]["settled"])
        self.assertEqual(report["run"]["state"], "blocked")
        self.assertEqual(len(gateway.starts), 1)
        self.assertFalse(report["children"][0]["settled"])


class ResultTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config = {"work_root": str(self.root), "suite": "put_bottles_into_dustbin", "max_steps": 700, "step_limit_mode": "native"}
        self.gateway = SyntheticGateway(self.config)
        self.gateway.records["native-1"] = {}
        self.tool, self.raw = self.gateway.result("native-1")
        self.envelope = {"data": {"status": "available", "result": self.tool}}

    def validate(self):
        Path(self.tool["outputs"]["result_file"]).write_text(json.dumps(self.raw))
        return validate_result(self.config, case(), "native-1", self.envelope, "succeeded")[0]

    def test_episode_error_is_not_a_robot_failure(self):
        self.raw["episodes"][0]["error_code"] = "policy_disconnected"
        self.raw["valid_episodes"] = 0
        self.assertEqual(self.validate()["outcome"], "infra_error")

    def test_shared_structure_validator_rejects_malformed_metrics_and_boolean_seed(self):
        from results import validate_native_result
        from support import BenchmarkError, ErrorCode
        for mutate in (lambda raw: raw.update(seed=False),
                       lambda raw: raw['episodes'][0].update(metrics=[]),
                       lambda raw: raw.update(total_episodes=True)):
            raw = copy.deepcopy(self.raw)
            mutate(raw)
            with self.subTest(raw=raw), self.assertRaises(BenchmarkError) as failure:
                validate_native_result(self.config, case(), 'native-1', self.tool, raw, 'completed')
            self.assertEqual(failure.exception.code, ErrorCode.RESULT_INVALID)

    def test_stub_wrong_identity_and_inconsistent_counts_are_rejected(self):
        original = copy.deepcopy(self.raw)
        for change in (lambda r: r["reproducibility"].update(stub=True),
                       lambda r: r["episodes"][0].update(task_id=9),
                       lambda r: r.update(completed_episodes=2),
                       lambda r: r.update(success_rate=0),
                       lambda r: r["episodes"][0]["metrics"]["step_limits"].update(effective_max_steps=3),
                       lambda r: r["episodes"][0].update(termination="failed")):
            self.raw = copy.deepcopy(original);change(self.raw)
            with self.assertRaises(ValueError):self.validate()

    def test_result_path_cannot_escape_or_refer_to_another_invocation(self):
        for path in (self.root / "foreign.json", self.root / "results" / "other-invocation.json"):
            self.tool["outputs"]["result_file"] = str(path)
            with self.assertRaises(ValueError):
                validate_result(self.config, case(), "native-1", self.envelope, "succeeded")

    def test_raw_result_bytes_are_preserved_and_conflicting_archive_is_rejected(self):
        directory = self.root / "evidence"
        first = archive_result(directory, {}, self.config, case(), "native-1", self.envelope, "succeeded")
        self.assertEqual((directory / "native-result.json").read_bytes(), Path(first["result_path"]).read_bytes())
        archive_result(directory, {}, self.config, case(), "native-1", self.envelope, "succeeded")
        self.tool["outputs"]["failure_summary"] = "changed"
        with self.assertRaises(ValueError):
            archive_result(directory, {}, self.config, case(), "native-1", self.envelope, "succeeded")

    def test_stop_before_any_completed_episode_is_cancelled(self):
        self.raw.update(status="stopped", episodes=[], completed_episodes=0, valid_episodes=0,
                        successes=0, success_rate=0, num_steps=0)
        self.tool.update(status="stopped")
        self.tool["outputs"].update(status="stopped", completed_episodes=0, success_rate=0)
        Path(self.tool["outputs"]["result_file"]).write_text(json.dumps(self.raw))
        summary, _ = validate_result(self.config, case(), "native-1", self.envelope, "stopped")
        self.assertEqual((summary["outcome"], summary["native_score"]), ("cancelled", None))

    def test_video_reference_is_limited_to_its_session(self):
        video = self.root / "videos/native-1/frame.mp4"
        video.parent.mkdir(parents=True)
        video.write_bytes(b"synthetic fixture, not an actual video")
        self.raw["episodes"][0]["metrics"]["video"] = str(video)
        self.assertEqual(self.validate()["video"]["path"], str(video))
        self.raw["episodes"][0]["metrics"]["video"] = str(self.root / "videos/native-2/frame.mp4")
        with self.assertRaises(ValueError):self.validate()


class PlanAndRuntimeTests(unittest.TestCase):
    def test_plan_requires_explicit_budgets_and_single_worker(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "plan.json"
            config = {"layout_seed": 0}
            path.write_text(json.dumps(plan()))
            self.assertEqual(len(read_plan(path, config)["cases"]), 2)
            bad = plan();bad.pop("episode_timeout_s")
            path.write_text(json.dumps(bad))
            with self.assertRaises(ValueError):read_plan(path, config)
            bad = plan();bad["parallelism"] = 2
            path.write_text(json.dumps(bad))
            with self.assertRaises(ValueError):read_plan(path, config)

    def test_unmanaged_gateway_is_not_adopted(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = {"work_root": str(root), "gateway_port": 19031}
            bundle = root / "skills/robodojo-bench-pi05"
            bundle.mkdir(parents=True)
            (bundle / "deployment.json").write_text(json.dumps(config))
            from support import sha256
            (bundle / "PROVENANCE.json").write_text(json.dumps({"bundled_files": {
                "deployment.json": sha256(bundle / "deployment.json")}}))
            with self.assertRaisesRegex(ValueError, "unmanaged Gateway"):
                NativeRuntime(config).require_ready()


if __name__ == "__main__":
    unittest.main()
