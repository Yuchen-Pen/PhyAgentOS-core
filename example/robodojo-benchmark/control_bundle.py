"""Install the private controller through the real Core installer and bind it."""

import asyncio
import json
import time
from pathlib import Path
from uuid import uuid4

import yaml

from PhyAgentOS.agent.experience.activation import SkillActivationManager
from PhyAgentOS.agent.experience.store import ExperienceStore
from PhyAgentOS.config.schema import ForgeConfig
from PhyAgentOS.forge.benchmark.private import PrivateBenchmarkDeployment, PrivateBenchmarkTransport
from PhyAgentOS.forge.binding import ForgeSkillBindingResolver
from PhyAgentOS.forge.task import AgentTaskCoordinator
from PhyAgentOS.forge.tool_client import ForgeToolClient
from PhyAgentOS.skill_runtime.catalog import SkillCatalog
from PhyAgentOS.skill_runtime.installer import SkillInstaller
from PhyAgentOS.skill_runtime.integration import ActiveRuntimeRegistry, ActiveSkillRuntime, DynamicForgeToolClient
from PhyAgentOS.skill_runtime.state import RuntimeStateStore
from PhyAgentOS.verification.contracts import TaskVerificationContract
from scripts.package_skill import package

from results import immutable_file, json_bytes

NAME = "robodojo-benchmark-control"


def install_control(root):
    root = Path(root)
    source = root / "control-source" / NAME
    source.mkdir(parents=True)  # A prior run's source/evidence is never overwritten.
    (source / "SKILL.md").write_text(
        f"---\nname: {NAME}\ndescription: Private RoboDojo Pi0.5 evaluation management\n---\n"
        "Manage the frozen plan through Core. Pi0.5 acts autonomously; no outer model decisions or online lessons.\n")
    (source / "flow.yaml").write_text("nodes: []\n")
    (source / "skill.yaml").write_text(yaml.safe_dump({
        "manifest_version": 2, "name": NAME, "version": "0.1.0",
        "description": "Private single-worker RoboDojo Pi0.5 plan controller", "skill_document": "SKILL.md",
        "gateway_url": "http://benchmark.private",
        "required_tools": ["benchmark.run", "benchmark.snapshot", "benchmark.submit"],
        "profiles": {"local": {"dataflow": "flow.yaml"}},
    }))
    (source / "benchmark.yaml").write_text(yaml.safe_dump({
        "protocol": "benchmark_contract_v1", "session_tool": "benchmark.run",
        "snapshot_tool": "benchmark.snapshot", "submission_tool": "benchmark.submit",
        "proposal_schema_file": "proposal.json", "snapshot_views": ["root", "episode", "receipt", "history"],
        "access_policy": "private_control_plane",
    }))
    # Core requires an object schema. The always-failing `not` rejects proposals.
    (source / "proposal.json").write_text(json.dumps({"type": "object",
        "properties": {"mode": {"const": "autonomous"}}, "required": ["mode"],
        "additionalProperties": False, "not": {}}))
    installed = root / "control-skills"
    SkillInstaller(installed, state_store=RuntimeStateStore(root / "control-install-state")).install(
        package(source, root / "control-archives"))
    return installed


class CoreExperiment:
    """Use Core's normal freeze/supervisor/runner/evidence path without a model API."""
    def __init__(self, root, adapter):
        self.root, self.adapter = Path(root), adapter
        self.supervisor = None
        self.decision_calls = 0

    async def setup(self):
        installed = install_control(self.root)
        self.transport = PrivateBenchmarkTransport(self.adapter)
        self.client = ForgeToolClient("http://benchmark.private", transport=self.transport, timeout_s=30)
        identity = uuid4().hex
        runtime = ActiveSkillRuntime(NAME, "0.1.0", "local", "private_" + identity,
                                     "http://benchmark.private", "host_" + identity, self.client, set(), set(), set())
        self.registry = ActiveRuntimeRegistry(runtime)
        resolver = ForgeSkillBindingResolver(self.registry, catalog=SkillCatalog(installed))
        workspace = self.root / "core-workspace"
        self.activation = SkillActivationManager(workspace=workspace, store=ExperienceStore(workspace),
            runtime_availability_provider=lambda name: name == NAME,
            binding_resolver=resolver, installed_skills_dir=installed)
        self.coordinator = AgentTaskCoordinator(workspace=workspace, config=ForgeConfig(benchmark={"enabled": True}),
            client=DynamicForgeToolClient(self.registry), binding_resolver=resolver, activation_manager=self.activation,
            runtime_invocation_ids=runtime.invocation_ids, runtime_session_ids=runtime.session_ids,
            runtime_task_binding_ids=runtime.task_binding_ids)
        self.supervisor = self.coordinator.benchmark.enable_private_adapter(PrivateBenchmarkDeployment(
            transport=self.transport, workers=[self.adapter.worker], decide=self.no_outer_decisions,
            artifacts=None, control_timeout_s=30, poll_interval_s=0.2))

    async def no_outer_decisions(self, context):
        self.decision_calls += 1
        raise RuntimeError("pure Pi0.5 must never request an outer model decision")

    async def run(self, *, stop_event=None):
        session = "robodojo:" + uuid4().hex
        self.activation.begin_turn(session, "Run the frozen RoboDojo Pi0.5 evaluation plan")
        activated, _, _ = await self.activation.activate(session_key=session, name=NAME, role="primary")
        task = await self.coordinator.create_task(
            task_description="RoboDojo Pi0.5 native single-episode cases",
            activation_id=activated.activation_id, origin_session_key=session,
            benchmark_plan=self.adapter.plan,
            verification=TaskVerificationContract(mode="enforce", goal="Complete evaluation with native evidence",
                success_criteria=["evaluation_complete"], evidence_policy={
                    "profile": "benchmark_evaluation", "required_kinds": ["benchmark_report"]}))
        self.task_id = task.task_id
        await self.coordinator.start_session(task.task_id, "benchmark.run", {})
        async def request_stop():
            await stop_event.wait()
            immutable_file(self.root / "stop-request.json", json_bytes({
                "task_id": self.task_id, "reason": "operator_interrupt", "requested_at": time.time(),
                "semantics": "close admission and wait for the current native episode boundary"}))
            await self.coordinator.cancel_task(self.task_id, reason="operator_interrupt")
            immutable_file(self.root / "stop-acknowledgement.json", json_bytes(self.report()))

        stopper = asyncio.create_task(request_stop()) if stop_event is not None else None
        try:
            await self.supervisor.monitors[task.task_id]
            # A stop acknowledgement can precede the physical boundary. Core's
            # reconciliation watcher retains ownership until authoritative stop.
            deadline = time.monotonic() + self.adapter.plan['episode_timeout_s']
            while (self.coordinator.get_task(self.task_id).cancellation_requested
                   and not self.report()['run']['settled'] and time.monotonic() < deadline):
                if stopper is not None and stopper.done():
                    stopper.result()
                await asyncio.sleep(0.2)
            if stopper is not None and stopper.done():
                stopper.result()
        finally:
            if stopper is not None:
                stopper.cancel()
                await asyncio.gather(stopper, return_exceptions=True)
        report = self.report()
        immutable_file(self.root / "core-report.json", json_bytes(report))
        return report

    def report(self):
        store = self.coordinator.benchmark.store
        return {"run": store.get("run", self.task_id), "children": store.list("subtask", self.task_id),
                "task": self.coordinator.get_task(self.task_id).model_dump(mode="json"),
                "outer_model_calls": self.decision_calls}

    async def close(self):
        if self.supervisor is not None:
            await self.supervisor.shutdown()
        if hasattr(self, "client"):
            await self.client.close()
