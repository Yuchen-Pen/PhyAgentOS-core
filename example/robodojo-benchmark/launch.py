#!/usr/bin/env python3
"""Launch a bundled source node in its own interpreter; --check-launch is read-only."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from preflight import load_config, model_resource
from support import verify_bundle


def library_paths(python):
    prefix = Path(python).parent.parent
    paths = [str(prefix / "lib")]
    configuration = prefix / "pyvenv.cfg"
    if configuration.is_file():
        for line in configuration.read_text().splitlines():
            if line.startswith("home = "):
                base = Path(line.partition(" = ")[2]).parent / "lib"
                if str(base) not in paths:
                    paths.append(str(base))
    return os.pathsep.join(paths)


def graphics_environment(config, environment):
    """Check both APIs select the same UUID before initializing the simulator."""
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,pci.bus_id',
                                    '--format=csv,noheader,nounits'], text=True, timeout=10)
    devices = [list(map(str.strip, line.split(','))) for line in rows.splitlines()]
    device = next((d for d in devices if config['simulation_gpu'] in d[:2]), None)
    if device is None:
        raise ValueError('Simulation GPU UUID/index is unavailable')
    # Kit uses the physical NVIDIA index; CUDA_VISIBLE_DEVICES renumbers only CUDA.
    environment['ROBODOJO_RENDER_GPU_INDEX'] = device[0]
    domain, bus, slot = device[2].lower().split(':')
    environment.update(DRI_PRIME='pci-' + domain[-4:] + '_' + bus + '_' + slot.replace('.', '_'),
                       MESA_VK_DEVICE_SELECT_FORCE_DEFAULT_DEVICE='1')
    report = subprocess.run(['vulkaninfo', '--summary'], env=environment,
                            capture_output=True, text=True, timeout=20)
    found = re.findall(r'deviceUUID\s*=\s*([0-9a-f-]+)', report.stdout)
    expected = device[1].removeprefix('GPU-')
    if report.returncode or found != [expected]:
        raise ValueError('Vulkan must enumerate exactly the selected CUDA GPU; ' + report.stderr[-500:])
    work = Path(config['work_root'])
    (work / 'runtime-logs').mkdir(parents=True, exist_ok=True)
    (work / 'runtime-logs/vulkan-simulation-gpu.txt').write_text(report.stdout + report.stderr)


def launch_plan(bundle, role, inherited=None):
    bundle = Path(bundle).resolve()
    config = load_config(bundle / "deployment.json")
    verify_bundle(bundle, config)
    python = config["python"][role]
    if not Path(python).is_file() or not os.access(python, os.X_OK):
        raise ValueError("Python executable unavailable: " + python)
    source = bundle / "assets/robodojo"
    work = Path(config["work_root"])
    dojo = Path(config["robodojo_root"])
    if role in {"simulation", "policy"}:
        if not (dojo / "src/eval_client/main.py").is_file():
            raise ValueError("RoboDojo source is unavailable")
        gpu = config[role + "_gpu"]
        if gpu is None:
            raise ValueError("GPU allocation not selected for " + role)
    env = dict(os.environ if inherited is None else inherited)
    for key in list(env):
        if key in {"PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "CONDA_PREFIX",
                   "LD_LIBRARY_PATH", "LD_LIBRARY_PATH_ORIG", "CUDA_VISIBLE_DEVICES",
                   "CKPT_NAME", "PI05_CKPT_PATH"} or key.startswith("_PYI_"):
            env.pop(key, None)
    env.update({
        "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
        "TMPDIR": str(work / "tmp" / role),
        "TEMP": str(work / "tmp" / role), "TMP": str(work / "tmp" / role),
        "XDG_CACHE_HOME": str(work / "cache" / role),
        "HF_HOME": str(work / "cache/huggingface"),
        "OPENPI_DATA_HOME": str(work / "cache/openpi"),
    })
    if role == "tool":
        command = [python, "-s", "-B", "-u", "-m", "bench_tool.main"]
        env["PYTHONPATH"] = str(source)
    else:
        env.update({
            "ROBODOJO_ROOT": str(dojo), "XPOLICYLAB_ROOT": str(dojo / "XPolicyLab"),
            "CUDA_VISIBLE_DEVICES": gpu,
            "ENV_CFG_TYPE": "arx_x5", "ACTION_TYPE": "joint", "POLICY_NAME": "Pi_05",
            "TASK_NAME": config["suite"],
            "PYTHONPATH": os.pathsep.join((str(dojo), str(dojo / "XPolicyLab"), str(source))),
            "LD_LIBRARY_PATH": library_paths(python),
        })
        if role == "simulation":
            env.update({
                "BENCHMARK_CONFIG": str(bundle / "profiles/pi05/bench_config.yaml"),
                "ROBODOJO_STEP_LIMIT_MODE": config["step_limit_mode"],
                "OMNI_KIT_ACCEPT_EULA": "YES",
                "ROBODOJO_KIT_ROOT": str(work / "cache/simulation/kit"),
            })
            native_libs = work / 'native-libs/root/usr/lib/x86_64-linux-gnu'
            if native_libs.is_dir():
                env['LD_LIBRARY_PATH'] = str(native_libs) + os.pathsep + env['LD_LIBRARY_PATH']
            if Path('/etc/vulkan/icd.d/nvidia_icd.json').is_file():
                env['VK_ICD_FILENAMES'] = '/etc/vulkan/icd.d/nvidia_icd.json'
            command = [python, "-s", "-B", "-u", str(source / "main.py"),
                       '--portable-root', env['ROBODOJO_KIT_ROOT'],
                       '--/renderer/multiGpu/enabled=false',
                       '--/physics/updateToUsd=true',
                       '--enable', 'isaacsim.sensors.camera',
                       '--enable', 'isaacsim.replicator.behavior']
        else:
            models = model_resource(config)
            env.update({
                "CKPT_NAME": models["checkpoint_root"],
                "PI05_CKPT_PATH": models["checkpoint_root"],
                "XLA_PYTHON_CLIENT_MEM_FRACTION": str(config["jax_memory_fraction"]),
                "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            })
            command = [python, "-s", "-B", "-u", str(source / "policy_node.py")]
    # RoboDojo main and the native policy deliberately switch to their own source CWD.
    return {"role": role, "command": command, "cwd": str(work / "work" / role), "environment": env}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("role", choices=("simulation", "tool", "policy"))
    parser.add_argument("--check-launch", action="store_true")
    args = parser.parse_args()
    plan = launch_plan(Path(__file__).resolve().parent, args.role)
    if args.check_launch:
        # Do not print the inherited environment: it can contain account credentials.
        print(json.dumps({k: v for k, v in plan.items() if k != "environment"} | {
            "simulation_started": False, "process_started": False,
        }, ensure_ascii=False))
        return
    config = load_config(Path(__file__).resolve().parent / "deployment.json")
    work = Path(config["work_root"])
    for path in (Path(plan["cwd"]), work / "results", work / "videos",
                 Path(plan["environment"]["TMPDIR"]), Path(plan["environment"]["XDG_CACHE_HOME"]),
                 work / "cache/huggingface", work / "cache/openpi"):
        path.mkdir(parents=True, exist_ok=True)
    if args.role == 'simulation':
        Path(plan['environment']['ROBODOJO_KIT_ROOT']).mkdir(parents=True, exist_ok=True)
        graphics_environment(config, plan['environment'])
        plan['command'].append('--/renderer/activeGpu=' + plan['environment']['ROBODOJO_RENDER_GPU_INDEX'])
    os.chdir(plan["cwd"])
    os.execve(plan["command"][0], plan["command"], plan["environment"])


if __name__ == "__main__":
    main()
