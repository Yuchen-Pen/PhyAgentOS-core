"""Explicit RoboDojo task horizon versus an engineering step cap."""


def resolve_step_limit(requested: int, native_horizon: int, mode: str) -> dict:
    if type(requested) is not int or requested < 1:
        raise ValueError("requested max_steps must be a positive integer")
    if type(native_horizon) is not int or native_horizon < 1:
        raise ValueError("RoboDojo must expose a positive task step_lim")
    if mode not in {"native", "cap"}:
        raise ValueError("step_limit_mode must be native or cap")
    return {
        "mode": mode,
        "requested_max_steps": requested,
        "native_horizon": native_horizon,
        "effective_max_steps": native_horizon if mode == "native" else min(requested, native_horizon),
    }
