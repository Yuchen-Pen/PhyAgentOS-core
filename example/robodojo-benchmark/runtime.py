"""Connect only to the native Skill managed under this deployment's own state."""

import json
from pathlib import Path

from PhyAgentOS.skill_runtime.catalog import SkillCatalog
from PhyAgentOS.skill_runtime.manager import RuntimeManager
from PhyAgentOS.skill_runtime.state import RuntimeStateStore

from preflight import sha256
from dora_scope import DoraScope

NAME = "robodojo-bench-pi05"


class NativeRuntime:
    def __init__(self, config):
        self.config = config
        self.work = Path(config["work_root"])
        self.scope = DoraScope(self.work)
        if self.scope.definition.exists():
            self.scope.activate()
        self.state = RuntimeStateStore(self.work / "state")
        self.manager = RuntimeManager(catalog=SkillCatalog(self.work / "skills"), state_store=self.state,
            runtime_root=self.work / "forge-runtime", logs_root=self.work / "runtime-logs")

    def verify_bundle(self):
        bundle = self.work / "skills" / NAME
        if json.loads((bundle / "deployment.json").read_text()) != self.config:
            raise ValueError("installed native Skill uses another deployment; rebuild before running")
        provenance = json.loads((bundle / "PROVENANCE.json").read_text())
        for relative, expected in provenance["bundled_files"].items():
            path = (bundle / relative).resolve()
            if not path.is_relative_to(bundle.resolve()) or sha256(path) != expected:
                raise ValueError("native bundle source/config changed: " + relative)
        return provenance

    def require_ready(self):
        self.verify_bundle()
        state = self.state.load(NAME)
        gateway = f"http://127.0.0.1:{self.config['gateway_port']}"
        if state is None or state.status != "running" or state.profile != "pi05" or state.gateway_url != gateway:
            raise ValueError("no matching managed native runtime; refuse to attach to an unmanaged Gateway")
        self.scope.owned()
        if not self.manager.status(NAME).ready:
            raise ValueError("managed native runtime is not ready")
        return state
