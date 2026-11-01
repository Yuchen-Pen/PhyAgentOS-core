"""Check physics-to-render synchronization and retain the native terminal frame."""


def verify_render_sync(simulation, settings, expected_render_gpu=None):
    fabric = bool(simulation.is_fabric_enabled())
    usd = settings.get('/physics/updateToUsd') is True
    if not fabric and not usd:
        raise RuntimeError('Camera state cannot follow physics: Fabric and physics/updateToUsd are both disabled')
    render_gpu = settings.get('/renderer/activeGpu')
    if expected_render_gpu is not None and str(render_gpu) != str(expected_render_gpu):
        raise RuntimeError('Kit renderer GPU differs from the selected physical NVIDIA GPU')
    return {'fabric_enabled': fabric, 'physics_update_to_usd': usd, 'render_gpu_index': render_gpu}


class TerminalObservation:
    """Tap the observation already acquired by native is_episode_end; never render again."""
    def __init__(self, environment):
        self.observation = None
        original = environment.get_obs_batch

        def capture(env_idx_list=None, last_frame=False):
            result = original(env_idx_list=env_idx_list, last_frame=last_frame)
            if last_frame:
                self.observation = next((item for item in result if item.get('env_idx') == 0), None)
            return result

        environment.get_obs_batch = capture


def record_final_frame(recorder, adapter, outcome, step_index, max_steps):
    if outcome.done or outcome.success or step_index >= max_steps:
        recorder.add(adapter.to_images(outcome.obs))
