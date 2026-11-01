"""Trace three archived Pi0.5 commands through native control; unscored diagnosis."""
import json
import os
from pathlib import Path

from probe_scene import main


def inspect_control(adapter, app, output, observation):
    import numpy as np
    import omni.timeline
    import carb.settings
    import omni.usd
    from pxr import UsdGeom
    from PIL import Image

    commands = json.loads(Path(os.environ['ROBODOJO_PROBE_ACTIONS']).read_text())['actions'][:3]
    env = adapter._require_env()
    manager = env.robot_manager
    timeline = omni.timeline.get_timeline_interface()
    sim = env.sim.unwrapped

    def robot_snapshot():
        result = []
        for index, (robot, arm) in enumerate(zip(manager.robot_list, manager.robot_key)):
            prim = omni.usd.get_context().get_stage().GetPrimAtPath(f'/World/envs/env_0/robot{index}/link6')
            result.append({'arm': robot.arm_name, 'joint_names': arm.joint_names,
                           'joint_positions': arm.data.joint_pos.cpu().tolist(),
                           'joint_targets': arm.data.joint_pos_target.cpu().tolist(),
                           'physx_joint_positions': arm.root_physx_view.get_dof_positions().cpu().tolist(),
                           'usd_link6_world_transform': (list(map(list, UsdGeom.XformCache().GetLocalToWorldTransform(prim)))
                                                         if prim.IsValid() else None)})
        return {'timeline_playing': timeline.is_playing(), 'timeline_time': timeline.get_current_time(),
                'physics_time': sim.sim.current_time, 'physics_step_counter': sim._sim_step_counter,
                'simulation_device': str(sim.device), 'fabric_enabled': sim.sim.is_fabric_enabled(),
                'physics_update_to_usd': carb.settings.get_settings().get('/physics/updateToUsd'),
                'robots': result}

    trace = []
    original = manager.control_robot

    def traced_control(*args, **kwargs):
        original(*args, **kwargs)
        trace.append(robot_snapshot())

    manager.control_robot = traced_control
    report = {'scope': 'three native control actions from the saved first model chunk; unscored',
              'native_sessions_dispatched': 0, 'commands_source': os.environ['ROBODOJO_PROBE_ACTIONS'],
              'before': robot_snapshot(), 'actions': [], 'control_targets': trace}
    (output / 'control-probe.json').write_text(json.dumps(report, indent=2))
    for command in commands:
        outcome = adapter.step(np.asarray(command, dtype=np.float32), chunk_end=False)
        report['actions'].append({'command': command, 'after': robot_snapshot(),
                                  'observed_state': {k: np.asarray(v).tolist()
                                                     for k, v in outcome.obs['state'].items()},
                                  'done': outcome.done})
        (output / 'control-probe.json').write_text(json.dumps(report, indent=2))
        if outcome.done:
            break
    report['actions_executed'] = len(report['actions'])
    for camera in ('cam_head', 'cam_left_wrist', 'cam_right_wrist'):
        Image.fromarray(np.asarray(outcome.obs['vision'][camera]['color'])).save(output / (camera + '_after.png'))
    (output / 'control-probe.json').write_text(json.dumps(report, indent=2))
    print('CONTROL_PROBE_READY actions=' + str(report['actions_executed']), flush=True)


if __name__ == '__main__':
    main(after_reset=inspect_control)
