"""Private Core control tools over a durable, single-worker RoboDojo adapter."""

import asyncio
import copy
import fcntl
import json
import os
from pathlib import Path

import httpx

from PhyAgentOS.forge.benchmark.client import SCOPE_KEYS
from PhyAgentOS.forge.benchmark.contracts import BenchmarkPlan
from PhyAgentOS.forge.interaction.contracts import digest

from results import archive_result
from support import ErrorCode, atomic_write, deployment_identity, immutable_file, json_bytes
from lifecycle import (EpisodeState, RootState, ROOT_KEYS, claim_episode, require_scope, request_episode_stop,
                       request_root_stop, settle_roots, transition_episode, validate_journal)


class RoboDojoAdapter:
    def __init__(self, native, plan, *, worker="robodojo-worker", allow_new_batch=False):
        self.native = native
        self.plan = BenchmarkPlan.model_validate(plan).model_dump()
        if self.plan["parallelism"] != 1 or self.plan["rounds"] != 1:
            raise ValueError("first implementation supports one worker and one round")
        for case in self.plan["cases"]:
            native.arguments_for(case)
        self.worker = worker
        self.directory = Path(native.config["work_root"]) / "control"
        self.directory.mkdir(parents=True, exist_ok=True)
        if self.directory.is_symlink() or self.directory.stat().st_uid != os.getuid():
            raise ValueError("controller directory must be owned by this user")
        self.path = self.directory / "state.json"
        self.last_journal = None
        lock = self.directory / "owner.lock"
        if lock.is_symlink() or self.path.is_symlink():
            raise ValueError("controller journal/lock cannot be symlinks")
        self.owner = lock.open("a+")
        try:
            fcntl.flock(self.owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.resumed = self.path.exists()
            self.state = json.loads(self.path.read_text()) if self.resumed else {
                "schema_version": 1, "config_digest": deployment_identity(native.config), "plan_digest": digest(self.plan),
                "roots": {}, "episodes": {}, "receipts": {}, "invocations": {}, "active": None,
            }
            if self.resumed:
                self.last_journal = self.path.read_text()
            if (self.state.get("schema_version") != 1 or self.state["config_digest"] != deployment_identity(native.config)
                    or self.state["plan_digest"] != digest(self.plan)):
                raise ValueError("controller journal belongs to another configuration/plan")
            if self.resumed and allow_new_batch:
                if (self.state['active'] is not None or not self.state['roots']
                        or any(not r['stopped'] for r in self.state['roots'].values())
                        or any(not e['snapshot']['stopped'] for e in self.state['episodes'].values())):
                    raise ValueError('Explicit new batch still requires all prior work authoritatively stopped')
                self.resumed = False  # All historical roots/receipts/evidence remain in this journal.
            validate_journal(self.state)
            self.persist()
        except BaseException:
            self.owner.close()
            raise
        self.lock = asyncio.Lock()
        self.specs = {f"benchmark.{operation}": {
            "tool_id": f"benchmark.{operation}", "endpoint_id": "benchmark", "operation": operation,
            "semantics": semantics, "available": True, "input_schema": {"type": "object"},
        } for operation, semantics in (("run", "session"), ("snapshot", "query"), ("submit", "action"))}

    def persist(self):
        if self.last_journal is not None and (not self.path.exists() or self.path.read_text() != self.last_journal):
            raise RuntimeError("controller journal disappeared or changed outside its owner")
        validate_journal(self.state)
        value = json_bytes(self.state)
        atomic_write(self.path, value)
        self.last_journal = value.decode()

    def close(self):
        # Keep the durable journal even when unresolved. Restart never resends starts.
        self.owner.close()

    def root(self, arguments):
        root = self.state["roots"][arguments["task_id"]]
        RootState.from_record(root)
        require_scope(arguments, root, ROOT_KEYS)
        return root

    def episode(self, arguments):
        self.root(arguments)
        episode = self.state["episodes"][arguments["subtask_id"]]
        require_scope(arguments, episode["scope"])
        return episode

    def save(self, episode, *, state, observation, outcome=None, score=None, stopped=False):
        snapshot = transition_episode(self.state, episode, state=state, observation=observation,
                                      outcome=outcome, score=score, stopped=stopped)
        self.persist()
        return snapshot

    def settle_roots(self):
        settle_roots(self.state)

    def receipt(self, arguments):
        invocation = "submission_" + digest(arguments)[:32]
        value = {**{k: arguments[k] for k in SCOPE_KEYS}, "request_id": arguments["request_id"],
                 "transport_invocation_id": invocation, "payload_digest": digest(arguments), "accepted": True}
        self.state["receipts"][arguments["request_id"]] = value
        self.state["invocations"][invocation] = {"kind": "receipt", "key": arguments["request_id"]}
        self.persist()
        return value

    async def begin(self, arguments):
        root = self.root(arguments)
        if self.resumed or RootState.from_record(root).stop_requested:
            raise ValueError("restart/closed root forbids new native dispatch; reconcile the original journal")
        if self.state["active"] is not None:
            raise ValueError("worker still owns an unacknowledged or unresolved episode")
        if arguments["worker_id"] != self.worker or arguments.get("lessons_digest") != digest([]):
            raise ValueError("unauthorized worker or online lessons")
        if arguments["case"] not in self.plan["cases"]:
            raise ValueError("case is outside the frozen plan")
        if any(e["scope"]["task_id"] == arguments["task_id"]
               and e["case"]["case_key"] == arguments["case"]["case_key"]
               for e in self.state["episodes"].values()):
            raise ValueError("this case was already dispatched; automatic retries are disabled")
        key = arguments["subtask_id"]
        if key in self.state["episodes"]:
            raise ValueError("episode identity already used; replay its original request")
        for field in SCOPE_KEYS:
            if not isinstance(arguments.get(field), str) or not arguments[field]:
                raise ValueError("missing episode identity: " + field)
        native_arguments = self.native.arguments_for(arguments["case"])
        episode = {"scope": {k: arguments[k] for k in SCOPE_KEYS}, "case": arguments["case"],
                   "native_arguments": native_arguments, "native_invocation": None, "history": [],
                   "directory": str(self.directory / "episodes" / digest(arguments["subtask_id"])[:32]),
                   "start_intent": True, "stop_requested": False}
        claim_episode(self.state, episode)
        self.save(episode, state="running", observation={"stage": "native_start_intent"})
        # Persist the local acceptance before the native POST. A lost native
        # acceptance leaves this worker occupied; the local receipt never means
        # that the physical episode succeeded or even started.
        self.receipt(arguments)
        try:
            invocation, acceptance = await self.native.start(native_arguments, arguments["request_id"])
            episode.update(native_invocation=invocation, acceptance=acceptance)
            self.save(episode, state="running", observation={"stage": "native_running", "native_invocation": invocation})
        except BaseException as error:
            self.save(episode, state="unknown", observation={"stage": "native_acceptance_unknown",
                      "error": type(error).__name__, "error_code": ErrorCode.ACCEPTANCE_UNKNOWN.value, "automatic_resend": False})
            if isinstance(error, (asyncio.CancelledError, KeyboardInterrupt, SystemExit)):
                raise

    async def poll(self, episode):
        if EpisodeState.from_record(episode).stopped or episode["native_invocation"] is None:
            return episode["snapshot"]
        try:
            value = await self.native.poll(episode["native_invocation"])
            episode["last_native_status"] = value["status"]
            if not value["terminal"] or value["result"] is None:
                return self.save(episode, state="running", observation={
                    "stage": "native_stop_pending" if episode["stop_requested"] else "native_running",
                    "native_invocation": episode["native_invocation"],
                })
            native_phase = value['phase']
            try:
                index = archive_result(episode["directory"], episode["scope"], self.native.config,
                                       episode["case"], episode["native_invocation"], value["result"], native_phase)
                observation = {"stage": "native_finished", "evidence": index}
                outcome, score = index["outcome"], index["native_score"]
            except (ValueError, KeyError, TypeError, OSError) as error:
                observation = {"stage": "native_result_invalid", "error": str(error),
                               "error_code": ErrorCode.RESULT_INVALID.value,
                               "native_invocation": episode["native_invocation"], "native_phase": native_phase}
                directory = Path(episode["directory"])
                directory.mkdir(parents=True, exist_ok=True)
                immutable_file(directory / "invalid-result.json", json_bytes(observation))
                outcome, score = "infra_error", None
            return self.save(episode, state="terminal", observation=observation,
                             outcome=outcome, score=score, stopped=True)
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
            return self.save(episode, state="unknown", observation={"stage": "native_query_unknown",
                             "error": type(error).__name__, "error_code": ErrorCode.QUERY_UNKNOWN.value,
                             "native_invocation": episode["native_invocation"]})

    async def stop(self, episode):
        if request_episode_stop(self.state, episode):
            self.persist()  # Persist intent before POST; do not blindly repeat an uncertain stop.
            if episode["native_invocation"] and not episode["snapshot"]["stopped"]:
                try:
                    episode["stop_acceptance"] = await self.native.stop(episode["native_invocation"])
                except (OSError, RuntimeError) as error:
                    episode["stop_error"] = type(error).__name__
                    episode["stop_error_code"] = ErrorCode.STOP_UNKNOWN.value
                self.persist()
        await self.poll(episode)
        self.settle_roots()
        self.persist()

    async def submit(self, arguments):
        request_id = arguments["request_id"]
        self.root(arguments)
        old = self.state["receipts"].get(request_id)
        if old:
            if old["payload_digest"] != digest(arguments):
                raise ValueError("request ID reused with different payload")
            self.episode(arguments)
            return old
        operation = arguments["operation"]
        if operation == "episode.start":
            await self.begin(arguments)
        elif operation == "episode.stop":
            episode = self.episode(arguments)
            self.receipt(arguments)
            await self.stop(episode)
        else:
            raise ValueError("pure Pi0.5 does not accept outer action decisions")
        return self.state["receipts"][request_id]

    async def handle(self, request):
        async with self.lock:
            try:
                return await self.route(request)
            except (ValueError, KeyError, TypeError) as error:
                return httpx.Response(409, json={"ok": False, "error": {
                    "code": ErrorCode.CONTROL_REJECTED.value, "message": str(error), "retryable": False}})

    async def route(self, request):
        path = request.url.path
        arguments = json.loads(request.content).get("arguments", {}) if request.content else {}
        def response(value, status=200):
            return httpx.Response(status, json={"ok": True, "data": copy.deepcopy(value)})
        if request.method == "GET" and path == "/tools":
            return response({"tools": list(self.specs.values())})
        if request.method == "GET" and path.startswith("/tools/"):
            tool = path.split("/")[2]
            spec = self.specs[tool]
            return response({"ready": True, "binding_error": None} if path.endswith("/context") else spec)
        if request.method == "POST" and path == "/tools/benchmark.run:invoke":
            if set(arguments) != {"task_id", "run_id", "binding_digest", "plan_digest"}:
                raise ValueError("invalid root start")
            if arguments["plan_digest"] != digest(self.plan):
                raise ValueError("root plan differs from the frozen manifest")
            key = arguments["task_id"]
            root = self.state["roots"].get(key)
            if root:
                if root["start_digest"] != digest(arguments):
                    raise ValueError("root start identity conflict")
            else:
                if self.resumed or any(not r["stopped"] for r in self.state["roots"].values()):
                    raise ValueError("original root requires reconciliation before another admission")
                invocation = "root_" + digest(arguments)[:32]
                root = {**arguments, "start_digest": digest(arguments), "session_invocation_id": invocation,
                        "stopped": False, "stop_requested": False}
                self.state["roots"][key] = root
                self.state["invocations"][invocation] = {"kind": "root", "key": key}
                self.persist()
            return response({"invocation_id": root["session_invocation_id"], "attempt_id": root["session_invocation_id"]}, 202)
        if request.method == "POST" and path == "/tools/benchmark.submit:invoke":
            receipt = await self.submit(arguments)
            return response({"invocation_id": receipt["transport_invocation_id"], "attempt_id": receipt["transport_invocation_id"]}, 202)
        if request.method == "POST" and path == "/tools/benchmark/snapshot:invoke":
            root = self.root(arguments)
            view = arguments["view"]
            if view == "root":
                self.settle_roots()
                output = root
            else:
                episode = self.episode(arguments)
                if view == "episode":
                    output = await self.poll(episode)
                elif view == "history":
                    output = {"complete": episode["snapshot"]["stopped"], "records": episode["history"]}
                elif view == "receipt":
                    receipt = self.state["receipts"].get(arguments["request_id"])
                    if receipt:
                        require_scope(arguments, receipt)
                    output = {"status": "found", "receipt": receipt} if receipt else {"status": "unknown"}
                else:
                    raise ValueError("unsupported snapshot view")
            return response({"response": {"result": {"status": "succeeded", "outputs": output}}})
        if path.startswith("/invocations/"):
            record = self.state["invocations"][path.split("/")[2]]
            if record["kind"] == "receipt":
                if request.method != "GET":
                    raise ValueError("local acceptance invocations cannot be controlled")
                value = self.state["receipts"][record["key"]]
                phase = "succeeded"
            else:
                value = self.state["roots"][record["key"]]
                if request.method == "POST" and path.endswith("/stop"):
                    request_root_stop(value)
                    self.persist()
                    active = self.state["active"]
                    if active and RootState.from_record(value).owns(self.state["episodes"][active]):
                        await self.stop(self.state["episodes"][active])
                    self.settle_roots()
                    self.persist()
                    return response({"status": "requested"}, 202)
                if request.method != "GET":
                    raise ValueError("unsupported root control operation")
                self.settle_roots()
                phase = "stopped" if value["stopped"] else "running"
            if path.endswith("/result"):
                if phase == "running":
                    return response({"status": "pending"}, 202)
                return response({"status": "available", "result": {"status": "succeeded", "outputs": value}})
            return response({"invocation_id": path.split("/")[2], "status": phase, "phase": phase})
        raise ValueError("unsupported private route")
