"""Pinned RoboDojo Query/Session client; one Core case is one native episode."""

import asyncio
from contextvars import Context
from urllib.parse import urlsplit

from PhyAgentOS.forge.interaction.client import query_outputs


class NativeClient:
    def __init__(self, client, config):
        self.client, self.config = client, config
        address = urlsplit(client.base_url)
        if (address.scheme != "http" or address.hostname != "127.0.0.1"
                or address.port != config["gateway_port"] or address.path
                or address.username or address.password or address.query or address.fragment):
            raise ValueError("native client must use the configured personal loopback Gateway")
        self.capabilities = None

    async def call(self, operation, *args, **kwargs):
        # The host-injected adapter has its own native authority. Do not inherit
        # the outer Core client's dispatch ContextVars into this second client.
        task = asyncio.create_task(operation(*args, **kwargs), context=Context())
        try:
            return await task
        except BaseException:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            raise

    async def describe(self):
        for tool, operation, semantics in (("benchmark.describe", "describe", "query"),
                                            ("benchmark.run", "run", "session")):
            spec = (await self.call(self.client.get_tool, tool))["data"]
            if (spec.get("tool_id") != tool or spec.get("endpoint_id") != "benchmark.controller"
                    or spec.get("operation") != operation or spec.get("semantics") != semantics):
                raise ValueError("unexpected native tool binding: " + tool)
        caps = query_outputs(await self.call(self.client.invoke_query_tool, "benchmark.describe", {}))
        expected = {"benchmark": "robodojo", "suite": self.config["suite"], "policy_id": "Pi_05"}
        if any(caps.get(key) != value for key, value in expected.items()):
            raise ValueError("native platform/suite/policy does not match deployment")
        repro = caps.get("reproducibility", {})
        expected = {"env_cfg_type": "arx_x5", "action_type": "joint", "policy_id": "Pi_05",
                    "step_limit_mode": self.config["step_limit_mode"]}
        if repro.get("stub") or any(repro.get(key) != value for key, value in expected.items()):
            raise ValueError("native capability configuration is incompatible")
        if caps.get("defaults", {}).get("seed") != self.config["layout_seed"]:
            raise ValueError("native layout directory differs from the configured seed")
        tasks = caps.get("tasks")
        if not isinstance(tasks, list) or len(tasks) != caps.get("num_tasks"):
            raise ValueError("native task catalogue is malformed")
        ids = [entry.get("task_id") for entry in tasks]
        if any(type(value) is not int or value < 0 for value in ids) or len(set(ids)) != len(ids):
            raise ValueError("native task IDs are invalid or duplicated")
        self.capabilities = caps
        return caps

    def arguments_for(self, case):
        if self.capabilities is None:
            raise RuntimeError("read and validate native capabilities before admitting a case")
        env = case["environment"]
        required = {"native_task_id", "task_name", "init_state_id", "layout_seed"}
        if set(env) != required:
            raise ValueError("case environment must explicitly identify task, initial state and layout seed")
        for key in ("native_task_id", "init_state_id", "layout_seed"):
            if type(env[key]) is not int or env[key] < 0:
                raise ValueError("invalid case field: " + key)
        if env["layout_seed"] != self.config["layout_seed"]:
            raise ValueError("case cannot switch the deployment's layout directory")
        entry = next((item for item in self.capabilities["tasks"]
                      if item["task_id"] == env["native_task_id"]), None)
        if entry is None or entry.get("name") != env["task_name"]:
            raise ValueError("case task ID/name does not match native describe")
        count = entry.get("num_init_states")
        if type(count) is not int or not 0 <= env["init_state_id"] < count:
            raise ValueError("case initial state is outside native capabilities")
        return {"task_ids": [env["native_task_id"]], "init_state_ids": [env["init_state_id"]],
                "num_runs": 1, "seed": env["layout_seed"], "max_steps": self.config["max_steps"]}

    async def start(self, arguments, caller_id):
        reply = await self.call(self.client.start_session, "benchmark.run", arguments, caller_id=caller_id)
        invocation = reply["data"].get("invocation_id")
        if (not isinstance(invocation, str) or not invocation
                or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for char in invocation)):
            raise ValueError("native acceptance has no safe invocation identity")
        return invocation, reply

    async def poll(self, invocation):
        status = await self.call(self.client.invocation_status, invocation)
        value = status["data"]
        if value.get("invocation_id") != invocation:
            raise ValueError("native status has a different invocation identity")
        phase = value.get('phase', value.get('status'))
        if not isinstance(phase, str) or not phase:
            raise ValueError('native status has no execution phase')
        if 'phase' in value and 'status' in value and value['phase'] != value['status']:
            raise ValueError('native execution phase/status disagree')
        if phase == "unknown":
            raise RuntimeError("native Gateway reports an unknown invocation")
        terminal = phase in {"completed", "succeeded", "failed", "stopped", "cancelled"}
        if not terminal:
            return {"terminal": False, "phase": phase, "status": status, "result": None}
        result = await self.call(self.client.invocation_result, invocation)
        return {"terminal": True, "phase": phase, "status": status,
                "result": result if result["data"].get("status") == "available" else None}

    async def stop(self, invocation):
        return await self.call(self.client.stop_session, invocation)
