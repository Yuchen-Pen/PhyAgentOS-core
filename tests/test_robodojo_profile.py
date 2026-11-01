"""Offline deployment behavior using synthetic sources, models and an inert gateway.

No simulator, policy inference, CUDA or external model API is executed.
"""

import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

EXAMPLE = Path(__file__).resolve().parents[1] / "example/robodojo-benchmark"
sys.path.insert(0, str(EXAMPLE))
import launch  # noqa: E402
import install as profile_install  # noqa: E402
import preflight  # noqa: E402
import prepare  # noqa: E402
from step_limits import resolve_step_limit  # noqa: E402


class StepLimitsTests(unittest.TestCase):
    def test_native_preserves_task_protocol_and_audits_request(self):
        audit = resolve_step_limit(300, 1050, "native")
        self.assertEqual(audit["effective_max_steps"], 1050)
        self.assertEqual(audit["requested_max_steps"], 300)

    def test_cap_honors_short_budget_and_never_extends_native_task(self):
        self.assertEqual(resolve_step_limit(100, 1050, "cap")["effective_max_steps"], 100)
        self.assertEqual(resolve_step_limit(2000, 1050, "cap")["effective_max_steps"], 1050)

    def test_missing_horizon_or_unknown_mode_is_rejected(self):
        for arguments in ((10, 0, "native"), (0, 100, "cap"), (10, 100, "guess"), (True, 100, "cap")):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                resolve_step_limit(*arguments)


class ProfileTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="robodojo profile fixture ")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.repo = self.root / "upstream"
        native = self.repo / "forge_robodojo_eval/robodojo"
        native.mkdir(parents=True)
        adapter = native / "adapters/robodojo_adapter.py"
        adapter.parent.mkdir()
        adapter.write_text('''import os
class Adapter:
    def reproducibility(self):
        return {
            "policy_id": self.cfg.policy_id,
        }
    def reset_episode(self, env):
        env.reset(seed=[layout_id])
        env_lim = int(getattr(env, "step_lim", 0) or 0)
        if env_lim and env_lim != self.cfg.max_steps:
            self._log(f"按环境的 step_lim 修正 max_steps：{self.cfg.max_steps} → {env_lim}")
            self.cfg.max_steps = env_lim
    def step(self):
        if False:
            self._last_obs = env.get_obs()
        return dict(
            metrics={
                "layout_id": self._current_layout_id,
            },
        )
    def _ensure_env(self, task_name):
        self._env = create_eval_env(self._build_env_cfg(task_name), self._app)
''')
        (native / "_core").mkdir()
        (native / "_core/__init__.py").write_text("")
        (native / "_core/session.py").write_text('''import time
class Session:
    def __init__(self, cfg_template, plan, invocation_id):
        self.cfg = replace(
            cfg_template,
            max_steps=int(plan["max_steps"]),
        )
''')
        (native / "bench_tool").mkdir()
        for path in (native / "main.py", native / "policy_node.py", native / "bench_tool/main.py"):
            path.write_text("raise RuntimeError('fixture must never be executed')\n")
        (native / 'main.py').write_text('def fixture():\n    app_args.headless = True\n')
        (native / '_core/runner.py').write_text('''def fixture(self):
    closing = (
            lambda: self.policy_command("stop"),
    )
    if True:
        self.last_metrics = outcome.metrics or {}
''')
        (native / 'policy_node.py').write_text('''def main():
    phase = "idle"
    for event in events:
        eid = event["id"]
        if eid == "policy_command":
            cmd = event["value"]
            if cmd.command == "start":
                phase = "running"
            elif cmd.command in ("stop", "cancel"):
                break
            elif cmd.command == "reset_scene":
                model.reset()
                model.reset()
            continue
        if eid == "proprio_state":
            observations.append(event["value"])
    return observations
''')
        (self.repo / "LICENSE").write_text("Synthetic fixture license\n")
        base = self.repo / "skill-src/robodojo-bench/profiles/g05"
        base.mkdir(parents=True)
        nodes = [{"id": name, "path": "/data0/stale", "env": {"CUDA_VISIBLE_DEVICES": "7"}}
                 for name in ("gateway", "bench_tool", "bench_data", "policy")]
        nodes[0]["args"] = "--config gateway.yaml"
        nodes[-1]["inputs"] = {"policy_command": {"source": "bench_data/policy_command", "queue_size": 32}}
        (base / "dataflow.yaml").write_text(yaml.safe_dump({"nodes": nodes}))
        run = {"tool_id": "benchmark.run", "semantics": "session", "input_schema": {
            "type": "object", "properties": {"task_ids": {}, "init_state_ids": {}, "num_runs": {}}}}
        (base / "gateway.yaml").write_text(yaml.safe_dump({"host": "127.0.0.1", "port": 1,
                                                            "tools": {"specs": [run]}}))
        (base / "bench_config.yaml").write_text("benchmark: robodojo\n")
        self.git("init", "-q")
        self.git("add", ".")
        self.git("-c", "user.name=fixture", "-c", "user.email=fixture@example.invalid", "commit", "-qm", "synthetic fixture")
        revision = self.git("rev-parse", "HEAD").strip()
        patch = mock.patch.object(preflight, "UPSTREAM_COMMIT", revision)
        patch.start()
        self.addCleanup(patch.stop)
        archive = self.root / "gateway.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            payload = b"#!/bin/sh\nexit 88\n"
            member = tarfile.TarInfo("gateway")
            member.size, member.mode = len(payload), 0o755
            tar.addfile(member, io.BytesIO(payload))
        dojo = self.root / "dojo"
        for relative in ("src/eval_client/main.py", "env/global_configs.py", "task/RoboDojo/task_registry.py",
                         "XPolicyLab/policy/Pi_05/model.py", "XPolicyLab/policy/Pi_05/deploy.yml",
                         "XPolicyLab/policy/Pi_05/openpi/pyproject.toml"):
            path = dojo / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# synthetic source, never executed\n")
        checkpoint = self.root / "models/59999"
        (checkpoint / "params").mkdir(parents=True)
        (checkpoint / "assets/arx_x5_sim").mkdir(parents=True)
        (checkpoint / "assets/arx_x5_sim/norm_stats.json").write_text("{}")
        receipt = self.root / "models.json"
        receipt.write_text(json.dumps({"schema_version": 1, "status": "ready",
            "checkpoint_root": str(checkpoint), "checkpoint_lock_sha256": preflight.MODEL_LOCK_SHA256}))
        self.config = json.loads((EXAMPLE / "config.example.json").read_text())
        self.config.update(work_root=str(self.root / "personal"), source_repo=str(self.repo),
                           robodojo_root=str(dojo), models_receipt=str(receipt), gateway_archive=str(archive),
                           simulation_gpu="2", policy_gpu="GPU-fixture", gateway_port=19123)
        self.config["python"] = dict.fromkeys(("launcher", "simulation", "tool", "policy"), sys.executable)

    def git(self, *args):
        return subprocess.check_output(["git", "-C", str(self.repo), *args], text=True, stderr=subprocess.DEVNULL)

    def build(self):
        output = Path(self.config["work_root"]) / "robodojo-bench-pi05"
        return prepare.build(self.config, output)

    def test_consecutive_sessions_cannot_overwrite_each_others_videos(self):
        from types import SimpleNamespace
        bundle = self.build()
        namespace = {"replace": lambda obj, **kw: SimpleNamespace(**(vars(obj) | kw))}
        exec((bundle / "assets/robodojo/_core/session.py").read_text(), namespace)
        template = SimpleNamespace(record_dir=str(self.root / "videos"), max_steps=700)
        first = namespace["Session"](template, {"max_steps": 700}, "native-1")
        second = namespace["Session"](template, {"max_steps": 700}, "native-2")
        self.assertNotEqual(first.cfg.record_dir, second.cfg.record_dir)
        self.assertEqual(Path(first.cfg.record_dir), self.root / "videos/native-1")
        self.assertEqual(template.record_dir, str(self.root / "videos"))

    def test_persistent_policy_ignores_frames_until_reset_and_between_sessions(self):
        from types import SimpleNamespace
        bundle = self.build()
        def command(name):
            return {'id': 'policy_command', 'value': SimpleNamespace(command=name)}
        def frame(number):
            return {'id': 'proprio_state', 'value': number}
        reset = mock.Mock()
        namespace = {'events': [frame(0), command('start'), frame(1), command('reset_scene'),
                     frame(2), command('pause'), frame(3), command('start'), frame(4),
                     command('reset_scene'), frame(5), command('stop')],
                     'model': SimpleNamespace(reset=reset), 'observations': [], 'log': lambda _: None}
        exec((bundle / 'assets/robodojo/policy_node.py').read_text(), namespace)
        self.assertEqual(namespace['main'](), [2, 5])
        self.assertEqual(reset.call_count, 4)

    def test_profile_has_one_episode_guard_and_explicit_paths(self):
        bundle = self.build()
        gateway = yaml.safe_load((bundle / "profiles/pi05/gateway.yaml").read_text())
        self.assertEqual(gateway["port"], 19123)
        properties = gateway["tools"]["specs"][0]["input_schema"]["properties"]
        self.assertEqual(properties["num_runs"]["maximum"], 1)
        self.assertEqual(properties["task_ids"]["maxItems"], 1)
        self.assertEqual(properties["seed"]["enum"], [self.config["layout_seed"]])
        bench = yaml.safe_load((bundle / "profiles/pi05/bench_config.yaml").read_text())
        self.assertTrue(bench["serve"])
        self.assertEqual(bench["max_total_episodes"], 1)
        self.assertEqual(bench["policy_id"], "Pi_05")
        flow = (bundle / "profiles/pi05/dataflow.yaml").read_text()
        self.assertNotIn("/data0", flow)
        self.assertNotIn("CUDA_VISIBLE_DEVICES", flow)
        self.assertIn("queue_size: 32", flow)
        self.assertEqual((bundle / "UPSTREAM_LICENSE.txt").read_text(), "Synthetic fixture license\n")

    def test_hostile_environment_cannot_select_another_gpu_or_policy(self):
        bundle = self.build()
        inherited = {"CUDA_VISIBLE_DEVICES": "7", "PYTHONPATH": "/stale", "PYTHONHOME": "/wrong",
                     "CKPT_NAME": "/wrong", "PI05_CKPT_PATH": "/wrong", "_PYI_ARCHIVE_FILE": "bad"}
        policy = launch.launch_plan(bundle, "policy", inherited)
        env = policy["environment"]
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "GPU-fixture")
        self.assertEqual(env["POLICY_NAME"], "Pi_05")
        self.assertNotIn("PYTHONHOME", env)
        self.assertNotIn("_PYI_ARCHIVE_FILE", env)
        self.assertNotIn("/stale", env["PYTHONPATH"])
        self.assertEqual(env["CKPT_NAME"], env["PI05_CKPT_PATH"])
        simulation = launch.launch_plan(bundle, "simulation", inherited)
        self.assertEqual(simulation["environment"]["CUDA_VISIBLE_DEVICES"], "2")
        self.assertEqual(yaml.safe_load((bundle / "profiles/pi05/bench_config.yaml").read_text())["device_id"], 0)

    def test_executable_entry_preserves_spaces_in_python_and_bundle_paths(self):
        capture = self.root / 'interpreter with spaces'
        capture.write_text('#!/usr/bin/python3\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n')
        capture.chmod(0o755)
        self.config['python']['launcher'] = str(capture)
        bundle = self.build()
        arguments = json.loads(subprocess.check_output([str(bundle / 'launch-simulation')], text=True))
        self.assertEqual(arguments, ['-s', '-B', '-u', str(bundle / 'launch.py'), 'simulation'])
        nodes = yaml.safe_load((bundle / 'profiles/pi05/dataflow.yaml').read_text())['nodes']
        self.assertTrue(all('args' not in n for n in nodes if n['id'] != 'gateway'))

    def test_graphics_check_rejects_more_than_the_assigned_gpu(self):
        self.config['simulation_gpu'] = '2'
        report = subprocess.CompletedProcess([], 0, 'deviceUUID = aaaa\ndeviceUUID = bbbb\n', '')
        with mock.patch.object(launch.subprocess, 'check_output', return_value='2, GPU-aaaa, 00000000:21:00.0\n'), \
                mock.patch.object(launch.subprocess, 'run', return_value=report):
            with self.assertRaisesRegex(ValueError, 'exactly the selected'):
                launch.graphics_environment(self.config, {})

    def test_graphics_check_matches_uuid_and_portable_cache_is_personal(self):
        self.config['simulation_gpu'] = '2'
        env = {}
        report = subprocess.CompletedProcess([], 0, 'deviceUUID = aaaa\n', '')
        with mock.patch.object(launch.subprocess, 'check_output', return_value='2, GPU-aaaa, 00000000:21:00.0\n'), \
                mock.patch.object(launch.subprocess, 'run', return_value=report):
            launch.graphics_environment(self.config, env)
        self.assertEqual(env['DRI_PRIME'], 'pci-0000_21_00_0')
        self.assertEqual(env['MESA_VK_DEVICE_SELECT_FORCE_DEFAULT_DEVICE'], '1')
        self.assertEqual(env['ROBODOJO_RENDER_GPU_INDEX'], '2')
        bundle = self.build()
        plan = launch.launch_plan(bundle, 'simulation', {})
        self.assertIn('--portable-root', plan['command'])
        self.assertIn('isaacsim.sensors.camera', plan['command'])
        self.assertIn('--/physics/updateToUsd=true', plan['command'])
        self.assertIn('isaacsim.replicator.behavior', plan['command'])
        self.assertIn('add_zero_delay_kit_args(app_args)',
                      (bundle / 'assets/robodojo/main.py').read_text())
        self.assertTrue(Path(plan['environment']['ROBODOJO_KIT_ROOT']).is_relative_to(self.config['work_root']))

    def test_check_launch_starts_no_process_and_creates_no_runtime_dirs(self):
        bundle = self.build()
        result = subprocess.check_output([sys.executable, "-B", str(bundle / "launch.py"),
                                          "policy", "--check-launch"], text=True)
        report = json.loads(result)
        self.assertFalse(report["process_started"])
        self.assertFalse(report["simulation_started"])
        self.assertFalse((Path(self.config["work_root"]) / "work").exists())
        self.assertNotIn("environment", report)

    def test_changed_source_is_rejected_before_launch(self):
        bundle = self.build()
        with (bundle / "assets/robodojo/main.py").open("a") as file:
            file.write("# changed\n")
        with self.assertRaisesRegex(ValueError, "changed"):
            launch.launch_plan(bundle, "simulation", {})

    def test_missing_gpu_refuses_launch(self):
        self.config["policy_gpu"] = None
        bundle = self.build()
        with self.assertRaisesRegex(ValueError, "GPU allocation"):
            launch.launch_plan(bundle, "policy", {})

    def test_rebuild_does_not_overwrite_previous_bundle(self):
        bundle = self.build()
        before = preflight.sha256(bundle / "PROVENANCE.json")
        with self.assertRaisesRegex(ValueError, "already exists"):
            prepare.build(self.config, bundle)
        self.assertEqual(before, preflight.sha256(bundle / "PROVENANCE.json"))

    def test_bundle_basename_must_match_core_skill_identity(self):
        with self.assertRaisesRegex(ValueError, "directory must be named"):
            prepare.build(self.config, Path(self.config["work_root"]) / "wrong-name")

    def test_preflight_reports_missing_resources_without_claiming_runtime(self):
        self.config["policy_gpu"] = None
        Path(self.config["models_receipt"]).unlink()
        report = preflight.check(self.config)
        failures = {item["name"] for item in report["checks"] if not item["ok"]}
        self.assertIn("models_receipt", failures)
        self.assertIn("policy_gpu", failures)
        self.assertIn("layout_directory", failures)
        self.assertIn("source:third_party/curobo/pyproject.toml", failures)
        self.assertIn("source:Assets/Robots/x5/curobo.yml", failures)
        self.assertFalse(report["runtime_verified"])
        self.assertFalse(report["simulation_started"])

    def test_wrong_model_identity_is_rejected(self):
        path = Path(self.config["models_receipt"])
        value = json.loads(path.read_text())
        value["checkpoint_lock_sha256"] = "f" * 64
        path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "pinned Pi0.5"):
            preflight.model_resource(self.config)

    def test_dependency_revision_and_untracked_source_are_checked(self):
        preflight.source_identity(self.config)
        (self.repo / "forge_robodojo_eval/robodojo/untracked.py").write_text("# not pinned")
        with self.assertRaisesRegex(ValueError, "clean"):
            preflight.source_identity(self.config)
        with mock.patch.object(preflight, "UPSTREAM_COMMIT", "0" * 40):
            with self.assertRaisesRegex(ValueError, "pinned upstream"):
                preflight.source_identity(self.config)

    def test_unknown_config_keys_and_multi_gpu_values_are_rejected(self):
        path = self.root / "config.json"
        for change in ({"policy_gpu": "0,1"}, {"policy_gpu": "-1"}, {"typo": 1}):
            path.write_text(json.dumps(self.config | change))
            with self.subTest(change=change), self.assertRaises(ValueError):
                preflight.load_config(path)

    def test_venv_python_symlink_is_preserved(self):
        python = self.root / "venv/bin/python"
        python.parent.mkdir(parents=True)
        python.symlink_to(sys.executable)
        self.config["python"]["policy"] = str(python)
        path = self.root / "config.json"
        path.write_text(json.dumps(self.config))
        self.assertEqual(preflight.load_config(path)["python"]["policy"], str(python))

    def test_simulation_overlay_keeps_base_library_directory(self):
        python = self.root / 'simulation-overlay/bin/python'
        python.parent.mkdir(parents=True)
        python.symlink_to(sys.executable)
        (python.parent.parent/'pyvenv.cfg').write_text('home = '+str(self.root/'simulation-base/bin')+'\n')
        self.config['python']['simulation'] = str(python)
        bundle = self.build()
        plan = launch.launch_plan(bundle,'simulation',{})
        self.assertIn(str(self.root/'simulation-base/lib'),plan['environment']['LD_LIBRARY_PATH'])

    def test_gateway_archive_rejects_symlink_and_path_escape(self):
        for name, kind in (("../gateway", tarfile.REGTYPE), ("gateway", tarfile.SYMTYPE)):
            path = self.root / "bad.tar.gz"
            with tarfile.open(path, "w:gz") as archive:
                member = tarfile.TarInfo(name)
                member.type = kind
                archive.addfile(member)
            with self.subTest(name=name, kind=kind), self.assertRaises(ValueError):
                prepare.gateway_lock(path)

    @unittest.skipUnless(sys.version_info >= (3, 11), "Core installer requires Python >=3.11")
    def test_core_installers_accept_generated_skill_and_verify_node_receipt(self):
        bundle = self.build()
        report = profile_install.install(bundle)
        self.assertFalse(report["simulation_started"])
        self.assertFalse(report["process_started"])
        from PhyAgentOS.skill_runtime.manifest import load_manifest
        from PhyAgentOS.skill_runtime.installer import NodeInstaller
        from PhyAgentOS.skill_runtime.state import RuntimeStateStore
        installed = load_manifest(Path(report["installed_bundle"]) / "skill.yaml")
        self.assertEqual(installed.tools_for_profile("pi05"), ("benchmark.describe", "benchmark.run"))
        nodes = NodeInstaller(Path(report["runtime_root"]), state_store=RuntimeStateStore(Path(report["state_root"])))
        binary = nodes.load(installed.artifacts.nodes["gateway"])
        self.assertTrue(binary.is_file())
        rendered = next(Path(report["executable_view"]).parent.glob("launch/profiles/pi05/dataflow.yaml"))
        self.assertNotIn("${PAOS_SKILL_ROOT}", rendered.read_text())
        self.assertNotIn("${FORGE_RUNTIME_BIN}", rendered.read_text())


if __name__ == "__main__":
    unittest.main()
