"""Validate and archive one native result. Never equate tool success with robot success."""

import hashlib
import json
import math
import os
from pathlib import Path

from step_limits import resolve_step_limit
from support import immutable_file, json_bytes, require


def finite(value):
    return type(value) in (float, int) and math.isfinite(value)


def validate_tool_result(tool, invocation):
    require(isinstance(tool, dict), "ToolResult must be an object")
    require(tool.get("status") in {"succeeded", "failed", "stopped"}, "invalid ToolResult status")
    outputs = tool.get("outputs", {})
    require(isinstance(outputs, dict), "ToolResult outputs must be an object")
    require(outputs.get("run_id") == invocation, "native result invocation does not match acceptance")
    return outputs


def validate_result(config, case, invocation, envelope, native_phase):
    require(isinstance(envelope, dict) and isinstance(envelope.get("data"), dict), "invalid result envelope")
    tool = envelope["data"].get("result")
    outputs = validate_tool_result(tool, invocation)
    path = Path(outputs.get("result_file") or "")
    allowed = Path(config["work_root"]) / "results"
    require(allowed.resolve() == allowed and not allowed.is_symlink(), "results directory is redirected")
    require(path.is_absolute() and path.resolve().parent == allowed and not path.is_symlink(),
            "native result path is outside the personal results directory")
    require(path.name == f"{config['suite']}_{invocation}.json", "native result filename has a different invocation")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as source:
        content = source.read(4 * 1024 * 1024 + 1)
    require(len(content) <= 4 * 1024 * 1024, "native result exceeds the single-episode size limit")
    def bad_constant(value):
        raise ValueError("non-finite JSON number: " + value)
    raw = json.loads(content, parse_constant=bad_constant)
    summary = validate_native_result(config, case, invocation, tool, raw, native_phase)
    video = None
    episodes = raw["episodes"]
    if episodes and episodes[0].get("metrics", {}).get("video"):
        video_path = Path(episodes[0]["metrics"]["video"])
        video_root = Path(config["work_root"]) / "videos" / invocation
        require(video_root.resolve() == video_root and video_path.is_absolute()
                and video_path.resolve().parent == video_root and not video_path.is_symlink(),
                "video path is outside this native Session's directory")
        checksum = hashlib.sha256()
        with video_path.open("rb") as source:
            while data := source.read(1024 * 1024):
                checksum.update(data)
        video = {"path": str(video_path), "sha256": checksum.hexdigest(), "bytes": video_path.stat().st_size}
    return {**summary, "native_result": raw, "video": video,
            "result_path": str(path), "result_sha256": hashlib.sha256(content).hexdigest()}, content


def validate_native_result(config, case, invocation, tool, raw, native_phase):
    """Pure structure/identity/count/budget validation, shared by every archive."""
    outputs = validate_tool_result(tool, invocation)
    require(isinstance(raw, dict), "native result must be an object")
    require(raw.get("schema_version") == "benchmark_execution_result_v1", "unsupported native result schema")
    require(raw.get("execution_mode") == "target_native", "unexpected execution mode")
    require(raw.get("benchmark") == "robodojo" and raw.get("suite") == config["suite"], "platform/suite mismatch")
    require(type(raw.get("seed")) is int and raw["seed"] == case["environment"]["layout_seed"],
            "layout directory seed mismatch")
    repro = raw.get("reproducibility", {})
    require(isinstance(repro, dict), "reproducibility must be an object")
    require(not repro.get("stub"), "stub result is not an actual evaluation")
    expected = {"policy_id": "Pi_05", "env_cfg_type": "arx_x5", "action_type": "joint",
                "step_limit_mode": config["step_limit_mode"]}
    require(all(repro.get(k) == v for k, v in expected.items()), "policy/robot/action/budget configuration mismatch")
    episodes = raw.get("episodes")
    require(isinstance(episodes, list) and len(episodes) <= 1, "expected at most one native episode")
    require(all(isinstance(e, dict) for e in episodes), "episode records must be objects")
    completed = len(episodes)
    expected_counts = {"total_episodes": 1, "completed_episodes": completed,
                       "successes": sum(e.get("success") is True for e in episodes),
                       "valid_episodes": sum(e.get("error_code") is None for e in episodes)}
    for key, expected_count in expected_counts.items():
        require(type(raw.get(key)) is int and raw[key] == expected_count, "inconsistent native count: " + key)
    for key in ("total_episodes", "completed_episodes"):
        require(type(outputs.get(key)) is int and outputs[key] == raw[key], "ToolResult/native count mismatch")
    score = raw["successes"]
    require(finite(raw.get("success_rate")) and raw["success_rate"] == score
            and finite(outputs.get("success_rate")) and outputs["success_rate"] == score,
            "inconsistent success rate")
    status = raw.get("status")
    require(status in {"succeeded", "failed", "stopped"}, "native result is not terminal")
    require((status == "succeeded" and native_phase in {"completed", "succeeded"})
            or (status == "failed" and native_phase == "failed")
            or (status == "stopped" and native_phase in {"stopped", "cancelled"}), "native status/result disagreement")
    expected_tool = "partial" if status == "stopped" and completed else status
    require(outputs.get("status") == expected_tool and tool["status"] == status, "ToolResult batch status disagreement")
    outcome, native_score = "infra_error", None
    if episodes:
        episode = episodes[0]
        metrics = episode.get("metrics", {})
        require(isinstance(metrics, dict), "episode metrics must be an object")
        env = case["environment"]
        require(type(episode.get("task_id")) is int and episode["task_id"] == env["native_task_id"]
                and type(episode.get("init_state_id")) is int and episode["init_state_id"] == env["init_state_id"]
                and type(episode.get("run_index")) is int and episode["run_index"] == 0, "episode identity mismatch")
        require(type(episode.get("success")) is bool, "episode success must be boolean")
        require(type(episode.get("num_steps")) is int and episode["num_steps"] >= 0
                and type(raw.get("num_steps")) is int and raw["num_steps"] == episode["num_steps"], "invalid step count")
        if episode.get("error_code") is None and not tool.get("error") and status != "failed":
            limits = metrics.get("step_limits", {})
            require(isinstance(limits, dict), "step limits must be an object")
            require(limits == resolve_step_limit(config["max_steps"], limits.get("native_horizon"),
                                                config["step_limit_mode"]), "step budget audit mismatch")
            horizon = limits["effective_max_steps"]
            require(type(raw.get("max_steps")) is int and raw["max_steps"] == horizon
                    and episode["num_steps"] <= horizon, "effective horizon mismatch")
            require(type(metrics.get("layout_id")) is int and metrics["layout_id"] >= 0, "missing actual layout ID")
            termination = episode.get("termination")
            require(termination in {"success", "failed", "partial", "timed_out"}, "unsupported termination")
            require(episode["success"] == (termination == "success"), "success/termination disagreement")
            require(termination != "timed_out" or episode["num_steps"] == horizon, "timeout before recorded horizon")
            outcome, native_score = ("success" if episode["success"] else "failure"), score
    elif status == "stopped" and not tool.get("error"):
        outcome = "cancelled"
    return {"outcome": outcome, "native_score": native_score}


def archive_result(directory, scope, config, case, invocation, envelope, native_phase):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    immutable_file(directory / "tool-result.json", json_bytes(envelope))
    summary, content = validate_result(config, case, invocation, envelope, native_phase)
    immutable_file(directory / "native-result.json", content)
    index = {**summary, "core_scope": scope, "case": case, "native_invocation": invocation}
    immutable_file(directory / "result-index.json", json_bytes(index))
    return index
