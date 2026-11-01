"""Prepare the locked Pi0.5 Python environment and verify it on CPU only."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import urllib.request
from pathlib import Path

from preflight import load_config, model_resource, sha256
from launch import library_paths


# The upstream lerobot group forces NumPy<2 but resolves rerun-sdk>=0.24,
# whose wheel metadata requires NumPy>=2. Use the older dataset API already
# supported by openpi.training.data_loader, preserving all model libraries.
COMPATIBILITY = {
    "lerobot": "0.3.2", "datasets": "3.6.0", "gymnasium": "0.29.1",
    "rerun-sdk": "0.22.1", "pytest": "8.3.5",
}


def command(arguments, *, cwd, env, log):
    with log.open("a") as output:
        subprocess.run([str(part) for part in arguments], cwd=cwd, env=env,
                       stdout=output, stderr=subprocess.STDOUT, check=True)


def tokenizer(work):
    target = work / "cache/openpi/big_vision/paligemma_tokenizer.model"
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.is_file():
        part = target.with_suffix(".partial")
        with urllib.request.urlopen("https://storage.googleapis.com/big_vision/paligemma_tokenizer.model", timeout=90) as source:
            with part.open("wb") as output:
                while data := source.read(1024**2):
                    output.write(data)
        part.replace(target)
    return {"path": str(target), "sha256": sha256(target), "bytes": target.stat().st_size}


PROBE = r'''
import importlib.metadata as metadata
import json, sys
from pathlib import Path
import numpy as np
import jax, dora, pyarrow, sentencepiece
from openpi.training import config as training_config
from openpi.shared import normalize
from XPolicyLab.policy.Pi_05.model import _resolve_pi05_model_root

config=json.loads(Path(sys.argv[1]).read_text())
models=json.loads(Path(config['models_receipt']).read_text())
checkpoint=Path(models['checkpoint_root'])
assert _resolve_pi05_model_root({'ckpt_name':str(checkpoint),'checkpoint_num':59999}).resolve()==checkpoint.resolve()
training=training_config.get_config('pi05_base_aloha_full_sim_arx-x5_seed_0')
assert training.model.pi05 is True
stats=normalize.load(checkpoint/'assets/arx_x5_sim')
assert len(stats['state'].mean)==14 and len(stats['actions'].mean)==14
sys.path.insert(0,str(Path(sys.argv[2])/'assets/robodojo'))
import policy_node
observations=policy_node._build_obs_spec('arx_x5')
actions=policy_node._build_state_spec('arx_x5','joint')
assert sum(size for _,size in observations)==28
assert sum(size for _,size in actions)==14
action=policy_node._flatten_action({name:np.zeros(size) for name,size in actions},actions)
assert action.shape==(14,) and np.isfinite(action).all()
from adapters.messages import JointCommand, PolicyCommand
names=[f'{name}.{i}' for name,size in actions for i in range(size)]
message=JointCommand(name=names,position=action.tolist(),mode='position')
roundtrip=JointCommand.from_arrow(message.to_arrow())
assert roundtrip.name==names and roundtrip.position==action.tolist()
reset=PolicyCommand.from_inputs(policy_id='Pi_05',command='reset_scene',inputs={})
assert PolicyCommand.from_arrow(reset.to_arrow()).command=='reset_scene'
fixture=Path(config['work_root'])/'verification/message-fixtures'
incoming=pyarrow.ipc.open_stream((fixture/'simulation-joint.arrow').read_bytes()).read_next_batch()
cross=JointCommand.from_arrow(incoming)
assert cross.position==[float(i) for i in range(14)] and cross.effort==[1.0]*14
with pyarrow.OSFile(str(fixture/'policy-joint.arrow'),'wb') as stream:
    with pyarrow.ipc.new_stream(stream,cross.to_arrow().schema) as writer:
        writer.write_batch(cross.to_arrow())
from adapters.messages import Image
image=Image.from_arrow(pyarrow.ipc.open_stream((fixture/'simulation-image.arrow').read_bytes()).read_next_batch())
assert image.to_numpy().shape==(2,3,3)
assert np.array_equal(image.to_numpy(),np.arange(18,dtype=np.uint8).reshape(2,3,3))
tokens=sentencepiece.SentencePieceProcessor(model_file=sys.argv[3])
assert tokens.encode('Put the bottles into the dustbin.')
report={'python':sys.executable,'python_base':sys.base_prefix,
        'jax_backend':jax.default_backend(),'observation_dimension':28,'action_dimension':14,
        'normalization_dimension':14,'message_roundtrip':True,'cross_environment_arrow':True,
        'tokenizer_vocabulary':tokens.get_piece_size(),
        'versions':{name:metadata.version(name) for name in ('openpi','xpolicylab','jax','flax','torch','numpy','pyarrow','dora-rs')},
        'model_loaded':False,'policy_inference_started':False,'simulation_started':False,
        'gpu_verified':False}
assert report['jax_backend']=='cpu'
print(json.dumps(report))
'''


SIMULATION_MESSAGE_PROBE = r'''
from pathlib import Path
import sys
import numpy as np
import pyarrow as pa
sys.path.insert(0,sys.argv[1])
from adapters.messages import JointCommand,Image
target=Path(sys.argv[2]);target.mkdir(parents=True,exist_ok=True)
if sys.argv[3]=='write':
    joint=JointCommand(name=[f'fixture.{i}' for i in range(14)],position=[float(i) for i in range(14)],effort=[1.0]*14)
    image=Image.from_numpy(np.arange(18,dtype=np.uint8).reshape(2,3,3))
    for name,record in [('simulation-joint.arrow',joint.to_arrow()),('simulation-image.arrow',image.to_arrow())]:
        with pa.OSFile(str(target/name),'wb') as stream:
            with pa.ipc.new_stream(stream,record.schema) as writer:writer.write_batch(record)
else:
    record=pa.ipc.open_stream((target/'policy-joint.arrow').read_bytes()).read_next_batch()
    decoded=JointCommand.from_arrow(record)
    assert decoded.position==[float(i) for i in range(14)] and decoded.effort==[1.0]*14
print('CROSS_ENVIRONMENT_ARROW_OK',pa.__version__)
'''


def prepare(config_path, uv, python_install_dir, *, skip_sync=False):
    config_path = Path(config_path).resolve()
    config = load_config(config_path)
    model_resource(config)
    work = Path(config["work_root"])
    root = Path(config["robodojo_root"])
    xpl = root / "XPolicyLab"
    openpi = xpl / "policy/Pi_05/openpi"
    python = openpi / ".venv/bin/python"
    if str(python) != config["python"]["policy"]:
        raise ValueError("configured policy interpreter must be the pinned openpi project's .venv")
    if openpi.stat().st_uid != os.getuid():
        raise ValueError("only prepare a policy checkout owned by this user")
    # A failed re-preparation must not leave an earlier ready receipt usable.
    (work / "policy-ready.json").write_text(json.dumps({
        "schema_version": 1, "status": "preparing", "python": str(python),
        "simulation_python": config["python"]["simulation"],
        "model_loaded": False, "simulation_started": False,
    }, indent=2) + "\n")
    lock = openpi / "uv.lock"
    lock_sha = sha256(lock)
    log = work / "verification/policy-overlay.log"
    env = dict(os.environ)
    for key in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "CONDA_PREFIX", "LD_LIBRARY_PATH"):
        env.pop(key, None)
    env.update({"UV_CACHE_DIR": str(work / "cache/uv"), "UV_PYTHON_INSTALL_DIR": str(Path(python_install_dir).resolve()),
                "UV_LINK_MODE": "copy", "TMPDIR": str(work / "tmp"), "PIP_CACHE_DIR": str(work / "cache/pip"),
                "CUDA_VISIBLE_DEVICES": "", "JAX_PLATFORMS": "cpu", "PYTHONDONTWRITEBYTECODE": "1",
                "OPENPI_DATA_HOME": str(work / "cache/openpi"), "HF_HOME": str(work / "cache/huggingface")})
    if not skip_sync:
        command([uv, "sync", "--frozen", "--no-default-groups", "--group", "lerobot", "--managed-python", "--python", "3.11"],
                cwd=openpi, env=env, log=log)
    # Keep the model stack unchanged; document the narrow dataset compatibility overlay.
    inventory = subprocess.check_output([str(python), "-I", "-B", "-c",
        "import importlib.metadata as m,json;print(json.dumps({d.metadata['Name']:d.version for d in m.distributions()}))"], env=env, text=True)
    versions = json.loads(inventory)
    constraints = work / "verification/policy-locked-constraints.txt"
    constraints.write_text("".join(f"{name}=={version}\n" for name, version in sorted(versions.items())
                                  if name.lower() not in {"openpi", "openpi-client", *COMPATIBILITY}))
    opencv = next(version for name, version in versions.items() if name.lower() == "opencv-python")
    command([uv, "pip", "install", "--python", python, "--constraint", constraints,
             "-e", xpl, "dora-rs==0.4.1", f"opencv-python-headless=={opencv}",
             *(f"{name}=={version}" for name, version in COMPATIBILITY.items())], cwd=openpi, env=env, log=log)
    command([uv, "pip", "check", "--python", python], cwd=openpi, env=env, log=log)
    cached_tokenizer = tokenizer(work)
    installed = work / "skills/robodojo-bench-pi05"
    fixture = work / "verification/message-fixtures"
    simulation_env = dict(env, LD_LIBRARY_PATH=library_paths(config["python"]["simulation"]))
    command([config["python"]["simulation"], "-I", "-B", "-c", SIMULATION_MESSAGE_PROBE,
             installed / "assets/robodojo", fixture, "write"], cwd=work, env=simulation_env, log=log)
    result = subprocess.run([str(python), "-I", "-B", "-c", PROBE, str(config_path), str(installed), cached_tokenizer["path"]],
                            cwd=openpi, env=env, text=True, capture_output=True)
    (work / "verification/policy-probe.stdout").write_text(result.stdout)
    (work / "verification/policy-probe.stderr").write_text(result.stderr)
    if result.returncode:
        raise RuntimeError("policy CPU probe failed; see verification/policy-probe.stderr")
    command([config["python"]["simulation"], "-I", "-B", "-c", SIMULATION_MESSAGE_PROBE,
             installed / "assets/robodojo", fixture, "verify"], cwd=work, env=simulation_env, log=log)
    simulation_metadata = subprocess.check_output([
        config["python"]["simulation"], "-I", "-B", "-c",
        "import numpy,pyarrow,json,sys;print(json.dumps({'python':sys.executable,'numpy':numpy.__version__,'numpy_path':numpy.__file__,'pyarrow':pyarrow.__version__,'pyarrow_path':pyarrow.__file__}))"
    ], env=simulation_env, text=True)
    simulation_report = json.loads(simulation_metadata)
    simulation_report.update(status="ready", ready_scope="CPU imports and cross-environment Arrow fixtures only", simulation_started=False)
    (work / "simulation-ready.json").write_text(json.dumps(simulation_report, indent=2) + "\n")
    if sha256(lock) != lock_sha:
        raise RuntimeError("openpi lock changed during preparation")
    report = json.loads(result.stdout.splitlines()[-1])
    report.update(schema_version=1, status="ready", simulation_python=config["python"]["simulation"], ready_scope="Python imports, config, normalization, messages and tokenizer; CPU only",
                  uv_lock_sha256=lock_sha, tokenizer=cached_tokenizer,
                  compatibility_overlay=COMPATIBILITY)
    freeze = subprocess.check_output([str(uv), "pip", "freeze", "--python", str(python)], env=env, text=True)
    (work / "verification/policy-environment.txt").write_text(freeze)
    (work / "policy-ready.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--uv", required=True)
    parser.add_argument("--python-install-dir", required=True)
    parser.add_argument("--skip-sync", action="store_true", help="Base uv sync already completed successfully")
    args = parser.parse_args()
    print(json.dumps(prepare(args.config, args.uv, args.python_install_dir, skip_sync=args.skip_sync), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
