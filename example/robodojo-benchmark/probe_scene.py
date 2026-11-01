"""Create the native scene and archive its first observation without policy actions.

Run in the simulation interpreter with the bundled launch environment. This is
an initialization check, not a scored episode or a native Session dispatch.
"""

import json
import os
from pathlib import Path
import sys
import time
import traceback


def main(after_reset=None):
    started = time.monotonic()
    bundle = Path(os.environ['ROBODOJO_PROBE_BUNDLE'])
    output = Path(os.environ['ROBODOJO_PROBE_OUTPUT'])
    output.mkdir(parents=True, exist_ok=False)
    deployment = json.loads((bundle / 'deployment.json').read_text())
    source = bundle / 'assets/robodojo'
    sys.path.insert(0, str(source))
    os.chdir(deployment['robodojo_root'])
    from config import load_config
    cfg = load_config(str(bundle / 'profiles/pi05/bench_config.yaml'))
    cfg.record_dir = ''
    cfg.result_dir = str(output / 'unscored-results')
    sys.argv = [__file__, '--portable-root', str(output / 'kit'),
                '--/renderer/multiGpu/enabled=false',
                '--enable', 'isaacsim.sensors.camera',
                '--enable', 'isaacsim.replicator.behavior']
    if os.environ.get('ROBODOJO_PROBE_SYNC_USD', '1') == '1':
        # Diagnostic switch only; compare to the unchanged native scene first.
        sys.argv.append('--/physics/updateToUsd=true')
    sys.argv.append('--/renderer/activeGpu=' + os.environ['ROBODOJO_RENDER_GPU_INDEX'])

    from isaaclab.app import AppLauncher
    import argparse
    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args([])
    args.headless = True
    args.enable_cameras = True
    args.device = 'cuda:0'
    from env.camera_manager.capture.render_sync import add_zero_delay_kit_args
    add_zero_delay_kit_args(args)
    print('SCENE_PROBE_APP_START', flush=True)
    app = AppLauncher(args).app
    print('SCENE_PROBE_APP_READY', flush=True)
    adapter = None
    try:
        from adapters.robodojo_adapter import RoboDojoAdapter
        import numpy as np
        from PIL import Image
        adapter = RoboDojoAdapter(cfg, app, log=lambda text: print(text, flush=True))
        print('SCENE_PROBE_RESET_BEGIN', flush=True)
        observation = adapter.reset_episode(0, 0, 0, 0)
        states = {key: np.asarray(value) for key, value in observation['state'].items()}
        assert sum(value.size for value in states.values()) == 28, 'Expected native 28D observation'
        assert all(np.isfinite(value).all() for value in states.values()), 'Non-finite state'
        cameras = {}
        for camera in ('cam_head', 'cam_left_wrist', 'cam_right_wrist'):
            frame = np.asarray(observation['vision'][camera]['color'])
            assert frame.ndim == 3 and frame.shape[-1] == 3 and frame.dtype == np.uint8, camera
            path = output / (camera + '.png')
            Image.fromarray(frame).save(path)
            cameras[camera] = {'shape': list(frame.shape), 'dtype': str(frame.dtype), 'path': str(path)}
        report = {'scope': 'native scene reset and first observation only; unscored',
                  'scene_initialized': True, 'native_sessions_dispatched': 0,
                  'policy_actions_sent': 0, 'model_loaded_by_probe': False,
                  'task': deployment['suite'], 'task_id': 0, 'init_state_id': 0,
                  'layout_seed': deployment['layout_seed'], 'layout_id': adapter._current_layout_id,
                  'render_state_sync': getattr(adapter, '_render_state_sync', None),
                  'state': {key: value.tolist() for key, value in states.items()},
                  'instruction': observation.get('instruction'), 'cameras': cameras,
                  'elapsed_s': time.monotonic() - started}
        (output / 'observation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        print('FIRST_OBSERVATION_READY ' + json.dumps(report, ensure_ascii=False), flush=True)
        if after_reset is not None:
            after_reset(adapter, app, output, observation)
    except BaseException as error:
        # Kit fast shutdown may exit(0), masking a pending Python exception.
        # Archive the failure before app.close(), and require the observation
        # artifact in the parent launcher instead of trusting the exit code.
        failure = {'scene_initialized': False, 'policy_actions_sent': 0,
                   'error_type': type(error).__name__, 'message': str(error),
                   'traceback': traceback.format_exc()}
        (output / 'failure.json').write_text(json.dumps(failure, indent=2) + '\n')
        traceback.print_exc()
        sys.stderr.flush()
        raise
    finally:
        if adapter is not None:
            adapter.close()
        app.close()


if __name__ == '__main__':
    main()
