"""Read-only static resource checks; never import a simulator or policy."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

from support import sha256

UPSTREAM_COMMIT = "d827bc35b81f6625b10464660bb71ead3937f953"
UPSTREAM_URL = "https://github.com/liuy2839/PhyAgentOS-Repo"
MODEL_LOCK_SHA256 = "a7040c52335750a50131a50cff9169ce22dd18b9fe9d545d8bc470512e452cc1"
CONFIG_KEYS = {
    "schema_version", "work_root", "source_repo", "robodojo_root", "models_receipt",
    "gateway_archive", "python", "simulation_gpu", "policy_gpu", "gateway_port",
    "suite", "layout_seed", "max_steps", "step_limit_mode", "jax_memory_fraction",
}


def absolute(value, name):
    if not isinstance(value, str) or not Path(value).is_absolute() or "\x00" in value:
        raise ValueError(f"{name} must be an absolute path")
    return str(Path(value).resolve())


def load_config(path):
    config = json.loads(Path(path).read_text())
    if not isinstance(config, dict) or set(config) != CONFIG_KEYS:
        raise ValueError("deployment configuration has missing or unknown keys")
    if type(config["schema_version"]) is not int or config["schema_version"] != 1:
        raise ValueError("unsupported deployment configuration")
    for key in ("work_root", "source_repo", "robodojo_root", "models_receipt", "gateway_archive"):
        config[key] = absolute(config[key], key)
    if not isinstance(config["python"], dict) or set(config["python"]) != {"launcher", "simulation", "tool", "policy"}:
        raise ValueError("python must name launcher, simulation, tool and policy interpreters")
    # Preserve venv entrypoints: resolving a Python symlink changes sys.prefix.
    for role, value in config["python"].items():
        if not isinstance(value, str) or not Path(value).is_absolute() or "\x00" in value:
            raise ValueError(f"python.{role} must be an absolute path")
    for key in ("simulation_gpu", "policy_gpu"):
        value = config[key]
        if value is not None and (not isinstance(value, str) or not re.fullmatch(r"(?:[0-9]+|GPU-[A-Za-z0-9-]+|MIG-[A-Za-z0-9/-]+)", value)):
            raise ValueError(f"{key} must select exactly one GPU or remain null before allocation")
    for key, minimum, maximum in (("gateway_port", 1024, 65535), ("max_steps", 1, 100000)):
        if type(config[key]) is not int or not minimum <= config[key] <= maximum:
            raise ValueError(f"invalid {key}")
    if type(config["layout_seed"]) is not int:
        raise ValueError("layout_seed must be an integer")
    if not isinstance(config["suite"], str) or not re.fullmatch(r"[A-Za-z0-9_]+", config["suite"]) or config["suite"] == "all":
        raise ValueError("choose one explicit task suite before deployment")
    if config["step_limit_mode"] not in {"native", "cap"}:
        raise ValueError("step_limit_mode must be native or cap")
    fraction = config["jax_memory_fraction"]
    if type(fraction) not in (int, float) or not 0 < fraction <= 1:
        raise ValueError("jax_memory_fraction must be in (0, 1]")
    root = Path(config["work_root"])
    ancestor = next(p for p in (root, *root.parents) if p.exists())
    if ancestor.stat().st_uid != os.getuid():
        raise ValueError("work_root must be inside a personal directory owned by this user")
    return config


def source_identity(config):
    repo = Path(config["source_repo"])
    def git(*arguments):
        return subprocess.check_output(
            ["git", "-C", str(repo), *arguments], text=True, stderr=subprocess.DEVNULL,
        ).strip()
    if git("rev-parse", "HEAD") != UPSTREAM_COMMIT:
        raise ValueError("benchmark source is not the pinned upstream commit")
    if git("status", "--porcelain", "--untracked-files=all"):
        raise ValueError("benchmark dependency checkout must be clean, including untracked files")
    for name in ("LICENSE", "forge_robodojo_eval/robodojo/main.py",
                 "forge_robodojo_eval/robodojo/bench_tool/main.py",
                 "forge_robodojo_eval/robodojo/policy_node.py"):
        if not (repo / name).is_file():
            raise ValueError(f"incomplete benchmark source: {name}")


def model_resource(config):
    receipt = json.loads(Path(config["models_receipt"]).read_text())
    if receipt.get("schema_version") != 1 or receipt.get("status") != "ready":
        raise ValueError("Pi0.5 models receipt is not ready")
    checkpoint = Path(absolute(receipt["checkpoint_root"], "checkpoint_root"))
    if not (checkpoint / "params").is_dir():
        raise ValueError("checkpoint params are missing")
    norm = checkpoint / "assets/arx_x5_sim/norm_stats.json"
    if not norm.is_file():
        raise ValueError("checkpoint normalization statistics are missing")
    identity = receipt.get("checkpoint_lock_sha256")
    if identity != MODEL_LOCK_SHA256:
        raise ValueError("models receipt does not match the pinned Pi0.5 checkpoint lock")
    return receipt


def check(config):
    checks = []
    def record(name, function):
        try:
            detail = function()
            checks.append({"name": name, "ok": True, "detail": detail or "present; execution not verified"})
        except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
            checks.append({"name": name, "ok": False, "detail": str(error)})

    def file(path):
        with Path(path).open("rb") as stream:
            stream.read(1)
        return str(path)

    record("pinned_benchmark_source", lambda: source_identity(config))
    dojo = Path(config["robodojo_root"])
    for name in ("src/eval_client/main.py", "env/global_configs.py", "task/RoboDojo/task_registry.py",
                 "third_party/curobo/pyproject.toml", "third_party/curobo/curobo/motion_planner.py",
                 "Assets/Robots/x5/curobo.yml",
                 "XPolicyLab/policy/Pi_05/model.py", "XPolicyLab/policy/Pi_05/deploy.yml",
                 "XPolicyLab/policy/Pi_05/openpi/pyproject.toml"):
        record("source:" + name, lambda name=name: file(dojo / name))
    layout = dojo / "Assets/Eval_Layout/RoboDojo/arx_x5" / str(config["layout_seed"])
    def layouts():
        if not any(layout.iterdir()):
            raise ValueError("layout directory is empty")
        return str(layout)
    record("layout_directory", layouts)
    for role, python in config["python"].items():
        def interpreter(python=python):
            if not Path(python).is_file() or not os.access(python, os.X_OK):
                raise ValueError("Python executable unavailable: " + python)
            return python
        record("python:" + role, interpreter)
    record("models_receipt", lambda: model_resource(config) and "ready receipt and payload layout present; hashes not rechecked")
    def policy_receipt():
        value = json.loads((Path(config["work_root"]) / "policy-ready.json").read_text())
        if value.get("status") != "ready" or value.get("python") != config["python"]["policy"] or value.get("simulation_python") != config["python"]["simulation"]:
            raise ValueError("Pi0.5 environment CPU probe receipt is missing or does not match the interpreter")
        return value.get("ready_scope", "CPU preparation only; GPU not verified")
    record("policy_environment_receipt", policy_receipt)
    def simulation_receipt():
        value = json.loads((Path(config["work_root"]) / "simulation-ready.json").read_text())
        if value.get("status") != "ready" or value.get("python") != config["python"]["simulation"]:
            raise ValueError("simulation environment import/message probe receipt is missing or stale")
        return value["ready_scope"]
    record("simulation_environment_receipt", simulation_receipt)
    def assets_receipt():
        actual_root = (dojo / "Assets").resolve()
        value = json.loads((actual_root.parent / "assets-ready.json").read_text())
        if value.get("status") != "ready" or value.get("assets_root") != str(actual_root):
            raise ValueError("asset preparation has not completed")
        if value.get("revision") != "43dacb13d3051a9ccd421f4e7e08e4e3eb29dfab":
            raise ValueError("asset receipt does not match the pinned revision")
        return str(value["files"]) + " files passed LFS SHA256 verification; GPU scene loading not verified"
    record("assets_receipt", assets_receipt)
    record("gateway_archive", lambda: file(config["gateway_archive"]))
    for key in ("simulation_gpu", "policy_gpu"):
        def gpu(key=key):
            if config[key] is None:
                raise ValueError("GPU allocation not selected; required before launch")
            return config[key] + " (availability not probed)"
        record(key, gpu)
    source_revisions = {}
    for name, root in (("robodojo", dojo), ("xpolicylab", dojo / "XPolicyLab"),
                       ("curobo", dojo / "third_party/curobo")):
        try:
            source_revisions[name] = subprocess.check_output(
                ["git", "-C", str(root), "rev-parse", "HEAD"], text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.SubprocessError):
            source_revisions[name] = None
    return {
        "static_resources_present": all(item["ok"] for item in checks),
        "source_revisions": source_revisions,
        "runtime_verified": False, "simulation_started": False,
        "note": "File checks do not prove imports, URDF closure, CUDA, port availability or runnable deployment.",
        "checks": checks,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    args = parser.parse_args()
    result = check(load_config(args.config))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["static_resources_present"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
