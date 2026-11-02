"""Connect only to the native Skill managed under this deployment's own state."""

from pathlib import Path

from PhyAgentOS.skill_runtime.catalog import SkillCatalog
from PhyAgentOS.skill_runtime.manager import RuntimeManager
from PhyAgentOS.skill_runtime.state import RuntimeStateStore

from support import verify_bundle
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
        return verify_bundle(bundle, self.config)

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
