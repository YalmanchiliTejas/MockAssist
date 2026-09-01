import json

import pytest

from interviewEnv.candidate.candidate_actions import (
    CandidateAction,
    CandidateActionType,
)
from interviewEnv.interviewer.interviewer_actions import (
    InterviewerAction,
    InterviewerActionType,
)
from interviewEnv.server.interviewEnv_environment import InterviewEnvironment
from interviewEnv.server.code_execution import CodeExecutionResult
from trainer.TRL_env import TRL_Env


PROBLEM = {"id": "1", "title": "Example", "description": "Solve it."}


class _Candidate:
    def __init__(self, responses=()):
        self.responses = iter(responses)
        self.calls = 0

    def respond(self, message):
        self.calls += 1
        return next(self.responses)


class _Executor:
    def __init__(self, results):
        self.results = iter(results)
        self.calls = []

    def execute(self, *, code, language, problem):
        self.calls.append(
            {"code": code, "language": language, "problem_id": problem["id"]}
        )
        return next(self.results)


def _action(kind, message=""):
    return InterviewerAction(action_type=kind, message=message, hint_level=0)


def test_premature_end_is_penalized_and_skips_candidate_generation():
    environment = InterviewEnvironment(problems=[PROBLEM])
    environment.reset(seed=0, profile="strong")
    candidate = _Candidate()
    environment.candidate = candidate

    observation = environment.step(_action(InterviewerActionType.END))

    assert observation.done is True
    assert observation.reward == pytest.approx(-0.95)
    assert environment.state.end_reason == "interviewer_end"
    assert candidate.calls == 0


def test_code_and_complexity_enable_successful_terminal_bonus():
    executor = _Executor([CodeExecutionResult(status="passed", exit_code=0)])
    environment = InterviewEnvironment(problems=[PROBLEM], code_executor=executor)
    environment.reset(seed=0, profile="strong")
    candidate = _Candidate(
        [
            CandidateAction(
                action_type=CandidateActionType.WRITE_CODE,
                spoken_response="Here is the implementation.",
                code_patch="def solve(): pass",
                complexity_claim="O(n) time, O(1) space",
                confidence=1.0,
            )
        ]
    )
    environment.candidate = candidate

    code_observation = environment.step(_action(InterviewerActionType.REQUEST_CODE))
    end_observation = environment.step(_action(InterviewerActionType.END))

    assert code_observation.reward == pytest.approx(0.05)
    assert code_observation.code_execution["status"] == "passed"
    assert code_observation.candidate_code == "def solve(): pass"
    assert executor.calls == [
        {"code": "def solve(): pass", "language": "python", "problem_id": "1"}
    ]
    assert environment.state.solution_reached is True
    assert end_observation.reward == pytest.approx(2.05)


def test_failed_code_is_reported_and_relayed_retry_earns_minor_reward():
    executor = _Executor(
        [
            CodeExecutionResult(
                status="failed",
                stderr="NameError: name 'n' is not defined",
                exit_code=1,
            ),
            CodeExecutionResult(status="passed", exit_code=0),
        ]
    )
    environment = InterviewEnvironment(problems=[PROBLEM], code_executor=executor)
    environment.reset(seed=0, profile="strong")
    environment.candidate = _Candidate(
        [
            CandidateAction(
                action_type=CandidateActionType.WRITE_CODE,
                spoken_response="Here is my code.",
                code_patch="print(n)",
                complexity_claim="O(1) time, O(1) space",
            ),
            CandidateAction(
                action_type=CandidateActionType.MODIFY_CODE,
                spoken_response="I fixed the variable name.",
                code_patch="n = 1\nprint(n)",
            ),
        ]
    )

    failed = environment.step(_action(InterviewerActionType.REQUEST_CODE))
    recovered = environment.step(
        _action(
            InterviewerActionType.REQUEST_CODE,
            "The sandbox failed with a NameError. Please fix it and try again.",
        )
    )

    assert failed.reward == pytest.approx(0.05)
    assert "NameError" in failed.candidate_message
    assert environment.state.solution_reached is True
    assert recovered.reward == pytest.approx(0.08)


def test_requesting_code_without_relaying_failure_gets_no_recovery_reward():
    executor = _Executor(
        [
            CodeExecutionResult(status="failed", stderr="SyntaxError", exit_code=1),
            CodeExecutionResult(status="passed", exit_code=0),
        ]
    )
    environment = InterviewEnvironment(problems=[PROBLEM], code_executor=executor)
    environment.reset(seed=0, profile="strong")
    environment.candidate = _Candidate(
        [
            CandidateAction(
                action_type=CandidateActionType.WRITE_CODE,
                code_patch="def bad(:",
            ),
            CandidateAction(
                action_type=CandidateActionType.MODIFY_CODE,
                code_patch="def good(): pass",
            ),
        ]
    )

    environment.step(_action(InterviewerActionType.REQUEST_CODE))
    retry = environment.step(
        _action(InterviewerActionType.REQUEST_CODE, "Please provide the code.")
    )

    assert retry.reward == pytest.approx(0.05)


def test_timeout_without_solution_is_penalized():
    environment = InterviewEnvironment(problems=[PROBLEM])
    environment.reset(seed=0, profile="strong")
    environment.candidate = _Candidate(
        [
            CandidateAction(
                action_type=CandidateActionType.THINK_ALOUD,
                spoken_response="Still thinking.",
            )
            for _ in range(6)
        ]
    )

    observations = [
        environment.step(_action(InterviewerActionType.ASK)) for _ in range(6)
    ]

    assert observations[-1].done is True
    assert observations[-1].reward == pytest.approx(-0.45)
    assert environment.state.end_reason == "time_limit"


def test_incomplete_training_episode_gets_penalty_once(tmp_path):
    data_path = tmp_path / "problems.jsonl"
    data_path.write_text(json.dumps(PROBLEM) + "\n", encoding="utf-8")
    environment = TRL_Env(data_path, rollout_dir=tmp_path / "rollouts")
    environment.reset(profile="strong", seed=0)

    first = environment._get_reward()
    second = environment._get_reward()

    assert first == pytest.approx(-1.0)
    assert second == pytest.approx(-1.0)
    rollout = json.loads((tmp_path / "rollouts/rollouts-rank-0.jsonl").read_text())
    assert rollout["end_reason"] == "incomplete"
    assert rollout["total_reward"] == pytest.approx(-1.0)
