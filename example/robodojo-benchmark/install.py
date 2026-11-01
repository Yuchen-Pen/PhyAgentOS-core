"""Package and install this native Skill into a personal runtime, without starting it."""

import argparse
import json
import sys
from pathlib import Path

from preflight import load_config, sha256

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def install(bundle):
    # Use the existing Core installers so their archive validation and receipts apply.
    from scripts.package_skill import package
    from PhyAgentOS.skill_runtime.installer import NodeInstaller, SkillInstaller, SkillEnvironmentBuilder
    from PhyAgentOS.skill_runtime.manifest import load_manifest
    from PhyAgentOS.skill_runtime.state import RuntimeStateStore

    bundle = Path(bundle).resolve()
    config = load_config(bundle / "deployment.json")
    work = Path(config["work_root"])
    if not bundle.is_relative_to(work):
        raise ValueError("bundle must be inside its personal work_root")
    manifest = load_manifest(bundle / "skill.yaml")
    if manifest.name != "robodojo-bench-pi05" or set(manifest.profiles) != {"pi05"}:
        raise ValueError("unexpected native Skill identity")
    state = RuntimeStateStore(work / "state")
    runtime_root = work / "forge-runtime"
    nodes = NodeInstaller(runtime_root, state_store=state)
    lock = manifest.artifacts.nodes["gateway"]
    nodes.install(Path(config["gateway_archive"]), lock)
    # Keep each build's archive instead of overwriting an earlier package.
    archives = work / "archives" / sha256(bundle / "PROVENANCE.json")[:16]
    archive = archives / f"{manifest.name}-{manifest.version}.tar.gz"
    if archive.exists():
        raise ValueError("this build was already packaged; inspect its existing installation")
    archive = package(bundle, archives)
    installed = SkillInstaller(work / "skills", state_store=state).install(
        archive, expected_sha256=sha256(archive),
    )
    executable_view = SkillEnvironmentBuilder(runtime_root, state_store=state).prepare(installed, "pi05")
    (work / "tmp/gateway").mkdir(parents=True, exist_ok=True)
    return {
        "skill": installed.name, "profile": "pi05", "archive": str(archive),
        "installed_bundle": str(installed.bundle_root), "executable_view": str(executable_view),
        "state_root": str(work / "state"), "runtime_root": str(runtime_root),
        "runtime_verified": False, "simulation_started": False, "process_started": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle")
    args = parser.parse_args()
    print(json.dumps(install(args.bundle), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
