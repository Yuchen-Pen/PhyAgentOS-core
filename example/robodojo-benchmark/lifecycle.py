"""Typed lifecycle rules over the version-1 journal's stable wire format.

As in Forge's invocation store, outcome and concurrency ownership are separate.
Here unknown is unresolved: it may recover, but can never release the worker.
"""

import copy
from dataclasses import dataclass
from enum import Enum

from PhyAgentOS.forge.benchmark.client import SCOPE_KEYS
from PhyAgentOS.forge.interaction.contracts import digest

from support import ErrorCode, require

ROOT_KEYS = ("task_id", "run_id", "binding_digest")


def control_require(condition, message):
    require(condition, message, ErrorCode.CONTROL_REJECTED)


def require_scope(arguments, scope, keys=SCOPE_KEYS):
    control_require(all(arguments.get(k) == scope[k] for k in keys), "cross-scope control request")


def flag(record, key):
    value = record.get(key, False)
    control_require(type(value) is bool, "invalid lifecycle flag: " + key)
    return value


class RootPhase(str, Enum):
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"


class EpisodePhase(str, Enum):
    RUNNING = "running"
    UNKNOWN = "unknown"
    TERMINAL = "terminal"


@dataclass(frozen=True)
class RootState:
    task_id: str
    run_id: str
    binding_digest: str
    stop_requested: bool = False
    stopped: bool = False

    @classmethod
    def from_record(cls, record):
        value = cls(*(record[k] for k in ROOT_KEYS), flag(record, "stop_requested"), flag(record, "stopped"))
        control_require(all(isinstance(record[k], str) and record[k] for k in ROOT_KEYS), "invalid root identity")
        control_require(not value.stopped or value.stop_requested, "root stopped without a stop request")
        return value

    @property
    def phase(self):
        return RootPhase.STOPPED if self.stopped else RootPhase.STOPPING if self.stop_requested else RootPhase.RUNNING

    def owns(self, episode):
        return all(episode["scope"].get(k) == getattr(self, k) for k in ROOT_KEYS)


@dataclass(frozen=True)
class EpisodeState:
    phase: EpisodePhase
    stopped: bool
    stop_requested: bool

    @classmethod
    def from_record(cls, record):
        snapshot = record["snapshot"]
        value = cls(EpisodePhase(snapshot["state"]), flag(snapshot, "stopped"), flag(record, "stop_requested"))
        control_require(value.stopped == (value.phase is EpisodePhase.TERMINAL), "terminal/stopped disagreement")
        return value

    @property
    def releasable(self):
        # A completed episode is acknowledged by Core's stop before reuse.
        return self.stopped and self.stop_requested


def settle_roots(journal):
    active = journal["active"]
    if active is not None and EpisodeState.from_record(journal["episodes"][active]).releasable:
        journal["active"] = None
    for record in journal["roots"].values():
        root = RootState.from_record(record)
        episodes = [e for e in journal["episodes"].values() if root.owns(e)]
        if root.stop_requested and all(EpisodeState.from_record(e).stopped for e in episodes):
            record["stopped"] = True


def validate_journal(journal):
    """Reject ambiguous ownership on recovery; never repair it by dispatching."""
    roots, episodes = journal["roots"], journal["episodes"]
    active = journal["active"]
    control_require(active is None or active in episodes, "worker references a missing episode")
    for key, record in roots.items():
        control_require(RootState.from_record(record).task_id == key, "root journal key mismatch")
    for key, record in episodes.items():
        scope = record["scope"]
        control_require(all(isinstance(scope.get(k), str) and scope[k] for k in SCOPE_KEYS), "invalid episode identity")
        control_require(scope["subtask_id"] == key, "episode journal key mismatch")
        root = RootState.from_record(roots[scope["task_id"]])
        control_require(root.owns(record), "episode belongs to another root")
        require_scope(record["snapshot"], scope)
        state = EpisodeState.from_record(record)
        control_require(state.releasable or active == key, "episode lost its worker ownership")
        control_require(not root.stopped or state.stopped, "closed root has unresolved work")
        control_require(active != key or not state.releasable, "acknowledged episode still owns the worker")


def claim_episode(journal, episode):
    key = episode["scope"]["subtask_id"]
    root = RootState.from_record(journal["roots"][episode["scope"]["task_id"]])
    control_require(not root.stop_requested and root.owns(episode), "episode requires an open matching root")
    control_require(journal["active"] is None, "worker still owns an unresolved episode")
    control_require(key not in journal["episodes"], "episode identity already used")
    journal["episodes"][key] = episode
    journal["active"] = key


def request_episode_stop(journal, episode):
    if EpisodeState.from_record(episode).stop_requested:
        return False
    episode["stop_requested"] = True
    settle_roots(journal)
    return True


def request_root_stop(root):
    RootState.from_record(root)
    root["stop_requested"] = True


def transition_episode(journal, episode, *, state, observation, outcome=None, score=None, stopped=False):
    phase = EpisodePhase(state)
    control_require(type(stopped) is bool and stopped == (phase is EpisodePhase.TERMINAL),
                    "only a verified terminal observation can stop an episode")
    control_require((outcome in {"success", "failure", "infra_error", "cancelled"}) if stopped
                    else outcome is None and score is None, "outcome is only valid at a terminal state")
    previous = episode.get("snapshot")
    snapshot = {**episode["scope"], "observation_version": previous["observation_version"] if previous else 0,
                "observation": observation, "state": phase.value, "outcome": outcome,
                "native_score": score, "stopped": stopped}
    old = {k: v for k, v in previous.items() if k != "history_digest"} if previous else None
    if previous and EpisodeState.from_record(episode).stopped:
        control_require(snapshot == old, "terminal episode cannot be rewritten or reopened")
    if snapshot != old:
        if previous:
            snapshot["observation_version"] += 1
        episode["history"].append(copy.deepcopy(snapshot))
    snapshot["history_digest"] = digest(episode["history"])
    episode["snapshot"] = snapshot
    settle_roots(journal)
    return snapshot
