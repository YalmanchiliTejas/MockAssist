import json
import os
from pathlib import Path

from openenv.core.env_server.interfaces import Environment

from interviewEnv.candidate.candidate_actions import (
    CandidateAction,
    CandidateActionType,
)
from interviewEnv.candidate.simulator import PROFILES, CandidateSimulator
from interviewEnv.interviewer.interview_state import CurrentPhase, InterviewState
from interviewEnv.interviewer.interviewer_actions import (
    InterviewerAction,
    InterviewerActionType,
)
from interviewEnv.interviewer.interviewer_observation import InterviewerObservation
from interviewEnv.server.code_execution import (
    CodeExecutionResult,
    CodeExecutor,
    build_code_executor,
)

DEFAULT_DATA = Path(
    os.environ.get(
        "MOCKASSIST_DATA",
        Path(__file__).resolve().parents[3] / "data" / "leetcode-training.jsonl",
    )
)
MINUTES_PER_TURN = 5
INTERVIEW_TIME_LIMIT_MINUTES = 30
# The simulator advances a fixed amount of interview time per interviewer turn.
MAX_TURNS = INTERVIEW_TIME_LIMIT_MINUTES // MINUTES_PER_TURN
TOTAL_MINUTES = INTERVIEW_TIME_LIMIT_MINUTES
STUCK_THRESHOLD_MINUTES = 5
MAX_CLARIFICATION_AWARDS = 3
CODE_FAILURE_RECOVERY_REWARD = 0.03

INTERVIEWER_VISIBLE_FIELDS = {
    "id",
    "title",
    "description",
    "optimal_time",
    "optimal_space",
    "concepts",
    "hints",
}


def filter_interview_problem(problem: dict) -> dict:
    return {k: v for k, v in problem.items() if k in INTERVIEWER_VISIBLE_FIELDS}


def load_problems(path: Path = DEFAULT_DATA) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class InterviewEnvironment(
    Environment[InterviewerAction, InterviewerObservation, InterviewState]
):
    def __init__(
        self,
        data_path: Path = DEFAULT_DATA,
        *,
        problems: list[dict] | None = None,
        code_executor: CodeExecutor | None = None,
    ):
        super().__init__()
        self.problems = problems if problems is not None else load_problems(data_path)
        self.profile_ids = sorted(PROFILES)
        self.problem: dict = {}
        self.candidate: CandidateSimulator | None = None
        self.code_executor = code_executor or build_code_executor()
        self._state = InterviewState()

    @property
    def state(self) -> InterviewState:
        return self._state

    def reset(self, seed=None, episode_id=None, **kwargs) -> InterviewerObservation:
        seed = 0 if seed is None else seed
        self.problem = kwargs.get("problem") or self.problems[seed % len(self.problems)]
        profile = (
            kwargs.get("profile")
            or self.profile_ids[(seed // len(self.problems)) % len(self.profile_ids)]
        )
        self.candidate = CandidateSimulator(self.problem, profile)
        self._state = InterviewState(
            episode_id=episode_id,
            problem_id=str(self.problem.get("id", "")),
            scenario_id=f"{self.problem.get('id', '')}:{profile}:{seed}",
            current_phase=CurrentPhase.START,
        )
        return self._observation(candidate_message="", reward=None)

    def step(
        self, action: InterviewerAction, timeout_s=None, **kwargs
    ) -> InterviewerObservation:
        if self.candidate is None:
            raise RuntimeError("reset() must be called before step()")

        state = self._state
        # Snapshot what the interviewer was reacting to. The candidate's reply has
        # not happened yet, so scoring against post-reply state would be backwards.
        stuck_before = state.stuck_minutes()
        clarification_pending = (
            state.last_candidate_action == CandidateActionType.ASK_CLARIFICATION.value
        )
        recovered_code_failure_number = _recovered_code_failure_number(action, state)
        state.turn += 1
        state.step_count += 1
        state.elapsed_minutes += MINUTES_PER_TURN
        if state.current_phase is CurrentPhase.START:
            state.current_phase = CurrentPhase.INTERVIEW
        if action.action_type is InterviewerActionType.HINT:
            state.hints_used += 1
        if action.action_type is InterviewerActionType.END:
            # The interviewer has already ended the interview. Avoid paying for
            # a candidate generation that cannot affect the outcome.
            response = CandidateAction(
                action_type=CandidateActionType.END,
                spoken_response="Thank you for your time.",
                confidence=1.0,
            )
        else:
            response = self.candidate.respond(_prompt_for(action))

        execution: CodeExecutionResult | None = None
        if response.code_patch:
            state.code_written = True
            state.progress_level = max(state.progress_level, 1)
            execution = self.code_executor.execute(
                code=response.code_patch,
                language="python",
                problem=self.problem,
            )
            state.last_code_execution = execution.to_dict()
            state.code_validated = execution.passed
            if execution.candidate_failure:
                state.pending_code_failure = True
                state.code_failure_number += 1
            elif execution.passed:
                state.pending_code_failure = False
        if response.complexity_claim:
            state.complexity_stated = True
            state.current_time_complexity = response.complexity_claim
        state.solution_reached = state.code_validated and state.complexity_stated
        if state.solution_reached:
            state.progress_level = max(state.progress_level, 2)

        if state.elapsed_minutes >= INTERVIEW_TIME_LIMIT_MINUTES:
            end_reason = "time_limit"
        elif action.action_type is InterviewerActionType.END:
            end_reason = "interviewer_end"
        elif response.action_type is CandidateActionType.END:
            end_reason = "candidate_end"
        else:
            end_reason = ""
        ended = bool(end_reason)
        if ended:
            state.current_phase = CurrentPhase.END
            state.end_reason = end_reason

        reward = self._reward(
            action,
            response,
            stuck_before=stuck_before,
            clarification_pending=clarification_pending,
            recovered_code_failure_number=recovered_code_failure_number,
            done=ended,
            end_reason=end_reason,
        )

        # Stuck tracking runs after scoring so this turn is judged on prior state.
        if response.is_stuck:
            if state.stuck_since_minute is None:
                state.stuck_since_minute = state.elapsed_minutes
        else:
            state.stuck_since_minute = None
        state.last_candidate_action = response.action_type.value

        candidate_message = response.spoken_response
        if execution is not None:
            candidate_message = _append_execution_feedback(
                candidate_message, execution
            )
        return self._observation(
            candidate_message=candidate_message,
            candidate_code=response.code_patch,
            code_execution=execution.to_dict() if execution else None,
            reward=reward,
            done=ended,
        )

    def _reward(
        self,
        action,
        response,
        *,
        stuck_before: int,
        clarification_pending: bool,
        recovered_code_failure_number: int | None,
        done: bool,
        end_reason: str,
    ) -> float:
        """Deterministic half of the reward table.

        The five rows needing semantic judgment -- restating clearly, answering a
        clarification *correctly*, hint too strong, leaking the solution, and
        misleading guidance -- are not scored here. They need an LLM judge.
        """

        state = self._state
        reward = +0.05  # reward for a valid interview turn
        was_stuck = stuck_before >= STUCK_THRESHOLD_MINUTES
        gave_hint = action.action_type is InterviewerActionType.HINT

        if was_stuck and gave_hint:
            reward += 0.4
        elif was_stuck and not gave_hint:
            reward -= 0.4
        elif gave_hint:
            reward -= 0.15  # unnecessary hint: candidate was not stuck

        # ponytail: pairing is detectable, correctness is not. This awards any
        # answer to a pending clarification; gate it on the judge when that lands.
        if (
            clarification_pending
            and action.action_type is InterviewerActionType.ASK
            and state.clarifications_answered < MAX_CLARIFICATION_AWARDS
        ):
            state.clarifications_answered += 1
            reward += 0.15

        # Code submission, execution success, testing, and complexity claims have
        # no direct reward. The one code-execution shaping reward recognizes the
        # interviewer recovering constructively from a concrete failed run.
        if recovered_code_failure_number is not None:
            reward += CODE_FAILURE_RECOVERY_REWARD
            state.recovered_code_failure_number = recovered_code_failure_number

        if done:
            if state.solution_reached:
                reward += 2.0
            elif end_reason == "time_limit":
                reward -= 0.5
            else:
                reward -= 1.0
        return reward

    def _observation(
        self,
        *,
        candidate_message: str,
        reward: float | None,
        candidate_code: str | None = None,
        code_execution: dict | None = None,
        done: bool = False,
    ) -> InterviewerObservation:
        return InterviewerObservation(
            done=done,
            reward=reward,
            problem=filter_interview_problem(self.problem),
            current_phase=self._state.current_phase.value,
            candidate_message=candidate_message,
            candidate_code=candidate_code,
            code_execution=code_execution,
            turn=self._state.turn,
        )


def _recovered_code_failure_number(
    action: InterviewerAction, state: InterviewState
) -> int | None:
    """Detect a one-time relay-and-retry response to the latest code failure."""
    if (
        not state.pending_code_failure
        or state.recovered_code_failure_number >= state.code_failure_number
        or action.action_type is not InterviewerActionType.REQUEST_CODE
    ):
        return None
    message = action.message.lower()
    relays_failure = any(
        term in message
        for term in (
            "failed",
            "failure",
            "error",
            "exception",
            "timed out",
            "timeout",
            "test did not pass",
            "tests did not pass",
        )
    )
    probes_retry = any(
        term in message
        for term in ("try again", "fix", "correct", "revise", "update", "retry")
    )
    return state.code_failure_number if relays_failure and probes_retry else None


def _append_execution_feedback(
    spoken_response: str, execution: CodeExecutionResult
) -> str:
    if execution.passed:
        feedback = "Code execution passed in the sandbox."
    elif execution.candidate_failure:
        diagnostic = execution.stderr or execution.detail or execution.stdout
        diagnostic = diagnostic.strip()[:1500]
        label = "timed out" if execution.timed_out else "failed"
        exit_text = (
            f" with exit code {execution.exit_code}"
            if execution.exit_code is not None
            else ""
        )
        feedback = f"Code execution {label}{exit_text}."
        if diagnostic:
            feedback += f" Diagnostic: {diagnostic}"
    else:
        feedback = (
            "Code execution could not be evaluated because the sandbox was "
            f"unavailable. {execution.detail}"
        ).strip()
    return f"{spoken_response}\n\n[Sandbox result: {feedback}]".strip()


def _prompt_for(action: InterviewerAction) -> str:
    """What the candidate actually hears. Action type is a stage direction, not speech."""
    if action.message:
        return action.message
    return {
        InterviewerActionType.ASK: "Walk me through your approach.",
        InterviewerActionType.HINT: "Here is a hint: think about the data structure.",
        InterviewerActionType.CHALLENGE: "Are you sure? What about the edge cases?",
        InterviewerActionType.REQUEST_CODE: "Go ahead and write the code.",
        InterviewerActionType.REQUEST_TEST: "Trace your code on an example.",
        InterviewerActionType.REQUEST_COMPLEXITY: "What is the time and space complexity?",
        InterviewerActionType.TRANSITION: "Let's move to the next part of the interview.",
        InterviewerActionType.END: "That is all the time we have. Thank you.",
    }[action.action_type]
