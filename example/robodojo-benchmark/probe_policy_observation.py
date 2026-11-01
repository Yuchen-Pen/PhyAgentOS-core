"""Inspect one real Pi0.5 chunk from a saved observation; send no robot actions."""
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image


def main():
    bundle = Path(os.environ['ROBODOJO_PROBE_BUNDLE'])
    saved = Path(os.environ['ROBODOJO_PROBE_OBSERVATION'])
    output = Path(os.environ['ROBODOJO_PROBE_OUTPUT'])
    output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(bundle / 'assets/robodojo'))
    from policy_node import _build_model, _flatten_action, _build_state_spec
    report = json.loads((saved / 'observation.json').read_text())
    observation = {
        'instruction': report['instruction'],
        'state': {k: (v if k.endswith('ee_joint_state') else np.asarray(v, dtype=np.float32))
                  for k, v in report['state'].items()},
        'vision': {k: {'color': np.array(Image.open(saved / (k + '.png')).convert('RGB'))}
                   for k in report['cameras']},
    }
    started = time.monotonic()
    model = _build_model()
    model.reset()
    model.reset()
    model.update_obs(observation)
    before_infer = time.monotonic()
    actions = model.get_action()
    spec = _build_state_spec('arx_x5', 'joint')
    flat = np.stack([_flatten_action(action, spec) for action in actions])
    assert flat.ndim == 2 and flat.shape[1] == 14 and np.isfinite(flat).all()
    initial = np.concatenate([np.asarray(observation['state'][key]).reshape(-1) for key, _ in spec])
    arm_indices = list(range(6)) + list(range(7, 13))
    summary = {
        'scope': 'one actual local policy inference on archived RGB/state, no simulation or Dora',
        'native_sessions_dispatched': 0, 'robot_actions_sent': 0,
        'observation_source': str(saved), 'model_initialization_s': before_infer - started,
        'inference_s': time.monotonic() - before_infer, 'chunk_shape': list(flat.shape),
        'component_names': [f'{key}.{i}' for key, count in spec for i in range(count)],
        'initial_state': initial.tolist(), 'first_action': flat[0].tolist(),
        'min_per_component': flat.min(axis=0).tolist(), 'max_per_component': flat.max(axis=0).tolist(),
        'max_arm_target_delta_rad': float(np.abs(flat[:, arm_indices] - initial[arm_indices]).max()),
        'actions': flat.tolist(),
    }
    (output / 'policy-chunk.json').write_text(json.dumps(summary, indent=2) + '\n')
    print('POLICY_CHUNK_READY ' + json.dumps({k: v for k, v in summary.items() if k != 'actions'}), flush=True)


if __name__ == '__main__':
    main()
