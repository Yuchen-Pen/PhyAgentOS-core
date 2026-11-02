"""Build a Pi0.5 source-node Skill in a personal directory, without starting it."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

import yaml

from preflight import UPSTREAM_COMMIT, UPSTREAM_URL, load_config, sha256, source_identity
from source_patches import CATALOGUE, apply_native_patches
from support import deployment_identity

HERE = Path(__file__).resolve().parent


def gateway_lock(archive):
    archive = Path(archive)
    if archive.is_symlink() or not archive.is_file():
        raise ValueError("gateway archive must be a regular file")
    with tarfile.open(archive, "r:gz") as stream:
        files = [member for member in stream.getmembers() if not member.isdir()]
        if len(files) != 1 or files[0].name != "gateway" or not files[0].isfile():
            raise ValueError("gateway archive must contain only a root-level regular gateway executable")
        if not 0 < files[0].size <= 512 * 1024**2:
            raise ValueError("gateway executable size is outside the allowed range")
    digest = sha256(archive)
    return {
        "artifact_id": f"gateway-1.1.1-{digest[:12]}-linux-x86_64", "version": "1.1.1",
        "platform": "linux", "arch": "x86_64", "artifact_type": "executable_tar_gz",
        "entrypoint": "gateway", "sha256": digest,
    }


def build(config, output):
    source_identity(config)
    lock = gateway_lock(config["gateway_archive"])
    work = Path(config["work_root"])
    output = Path(output).resolve()
    if output.name != "robodojo-bench-pi05":
        raise ValueError("bundle directory must be named robodojo-bench-pi05 for Core manifest validation")
    if not output.is_relative_to(work) or output == work:
        raise ValueError("bundle output must be below the personal work_root")
    if output.exists():
        raise ValueError("bundle output already exists; use a new directory to retain earlier builds")
    work.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    repo = Path(config["source_repo"])
    upstream = repo / "forge_robodojo_eval/robodojo"
    with tempfile.TemporaryDirectory(prefix=".prepare-pi05-", dir=output.parent) as temporary:
        bundle = Path(temporary) / "bundle"
        tracked = subprocess.check_output(
            ["git", "-C", str(repo), "ls-files", "--", "forge_robodojo_eval/robodojo"], text=True,
        ).splitlines()
        for relative in tracked:
            source = repo / relative
            if source.is_symlink() or not source.is_file():
                raise ValueError("upstream source must contain only regular files: " + relative)
            target = bundle / "assets/robodojo" / source.relative_to(upstream)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        shutil.copyfile(repo / "LICENSE", bundle / "UPSTREAM_LICENSE.txt")
        original_hashes = {(repo / p).relative_to(upstream).as_posix(): sha256(repo / p) for p in tracked}
        patch_receipt = apply_native_patches(bundle / "assets/robodojo")
        shutil.copyfile(CATALOGUE, bundle / "NATIVE_PATCHES.json")
        shutil.copyfile(HERE / "step_limits.py", bundle / "assets/robodojo/_core/step_limits.py")
        shutil.copyfile(HERE / "render_state.py", bundle / "assets/robodojo/_core/render_state.py")
        for name in ("launch.py", "preflight.py", "support.py"):
            shutil.copyfile(HERE / name, bundle / name)
        (bundle / "deployment.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
        profile = bundle / "profiles/pi05"
        profile.mkdir(parents=True)
        base = repo / "skill-src/robodojo-bench/profiles/g05"
        flow = yaml.safe_load((base / "dataflow.yaml").read_text())
        for node in flow["nodes"]:
            node.pop("env", None)
            if node["id"] == "gateway":
                node["path"] = "${FORGE_RUNTIME_BIN}/gateway"
                node["env"] = {"TMPDIR": str(work / "tmp/gateway"),
                               "LD_LIBRARY_PATH": str(work / 'native-libs/root/usr/lib/x86_64-linux-gnu')}
            else:
                role = {"bench_tool": "tool", "bench_data": "simulation", "policy": "policy"}[node["id"]]
                # Dora 0.4.1 splits `args` at whitespace without shell quoting.
                # Keep the path (which may contain spaces) in a single executable
                # field and let this tiny shell entrypoint quote Python arguments.
                entry = bundle / ("launch-" + role)
                entry.write_text('#!/bin/sh\nexec ' + shlex.quote(config["python"]["launcher"]) +
                                 ' -s -B -u "$(dirname "$0")/launch.py" ' + shlex.quote(role) + '\n')
                entry.chmod(0o755)
                node["path"] = "${PAOS_SKILL_ROOT}/launch-" + role
                node.pop("args", None)
        gateway = yaml.safe_load((base / "gateway.yaml").read_text())
        gateway.update(host="127.0.0.1", port=config["gateway_port"])
        run = next(tool for tool in gateway["tools"]["specs"] if tool["tool_id"] == "benchmark.run")
        run["description"] = (
            "Run exactly one RoboDojo episode with Pi_05. Session stop lands at the episode boundary. "
            f"Step limit mode is {config['step_limit_mode']}; native uses the task horizon, "
            "cap uses min(requested max_steps, native task horizon). Actual limits are in episode metrics."
        )
        properties = run["input_schema"]["properties"]
        for name in ("task_ids", "init_state_ids"):
            properties[name] = {"type": "array", "minItems": 1, "maxItems": 1,
                                "items": {"type": "integer", "minimum": 0}}
        properties["num_runs"].update(maximum=1)
        # The adapter reuses an environment for a task; changing the layout directory
        # mid-service would otherwise reuse the previous seed manager silently.
        properties["seed"] = {"type": "integer", "enum": [config["layout_seed"]]}
        run["input_schema"]["required"] = ["task_ids", "init_state_ids", "num_runs"]
        bench = yaml.safe_load((base / "bench_config.yaml").read_text())
        bench.update(
            suite=config["suite"], device_id=0, policy_id="Pi_05", env_cfg_type="arx_x5",
            action_type="joint", serve=True, init_states_per_task=1, num_runs=1,
            max_total_episodes=1, max_steps=config["max_steps"], seed=config["layout_seed"],
            result_dir=str(work / "results"), record_dir=str(work / "videos"),
        )
        for name, value in (("dataflow.yaml", flow), ("gateway.yaml", gateway), ("bench_config.yaml", bench)):
            (profile / name).write_text(yaml.safe_dump(value, sort_keys=False, allow_unicode=True))
        (bundle / "SKILL.md").write_text(
            "# RoboDojo Pi0.5 native evaluation\n\n"
            "Read benchmark.describe, then start one task/init-state with num_runs=1 using benchmark.run. "
            "Poll the Session status/result; stop is effective at the episode boundary. "
            "Tool completion does not imply robot task success; retain and inspect result_file.\n"
            "This Skill is the native evaluator. Core plan management uses a separate private control Skill.\n"
        )
        manifest = {
            "manifest_version": 2, "name": "robodojo-bench-pi05", "version": "0.1.0",
            "description": "RoboDojo with pure Pi_05, single-episode native Session",
            "skill_document": "SKILL.md", "gateway_url": f"http://127.0.0.1:{config['gateway_port']}",
            "required_tools": ["benchmark.describe", "benchmark.run"],
            "profiles": {"pi05": {
                "dataflow": "profiles/pi05/dataflow.yaml", "startup_timeout_s": 900,
                "required_binaries": ["gateway"],
                "required_assets": ["profiles/pi05/gateway.yaml", "profiles/pi05/bench_config.yaml",
                                    "deployment.json", "PROVENANCE.json", "NATIVE_PATCHES.json",
                                    "launch.py", "preflight.py", "support.py"],
                "required_environment": [], "environment": {},
            }},
            "artifacts": {"resolver": "local", "nodes": {"gateway": lock}},
        }
        (bundle / "skill.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))
        provenance = {
            "upstream_url": UPSTREAM_URL, "upstream_commit": UPSTREAM_COMMIT,
            "upstream_files": original_hashes,
            "deployment_identity": deployment_identity(config),
            "patches": patch_receipt,
            "changes": ["Explicit native/cap horizon and metrics in robodojo_adapter.py",
                        "Enable Isaac Lab camera rendering for real RGB policy observations",
                        "Pause the persistent policy between Sessions; keep native two resets and RNG semantics",
                        "Separate native video directories by Session invocation in _core/session.py",
                        "Add _core/step_limits.py; Pi0.5 profile and isolated Python launch entrypoints"],
            "bundled_files": {p.relative_to(bundle).as_posix(): sha256(p)
                              for p in bundle.rglob("*") if p.is_file()},
        }
        (bundle / "PROVENANCE.json").write_text(json.dumps(provenance, sort_keys=True, indent=2) + "\n")
        os.rename(bundle, output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = build(load_config(args.config), args.output)
    print(json.dumps({"bundle": str(output), "packaged": False, "installed": False,
                      "runtime_verified": False, "simulation_started": False}))


if __name__ == "__main__":
    main()
