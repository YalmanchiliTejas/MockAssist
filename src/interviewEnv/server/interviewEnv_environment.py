import json
import os
from pathlib import Path

from openenv.core.env_server.interfaces import Environment

from interviewEnv.candidate.candidate_actions import CandidateActionType
from interviewEnv.candidate.simulator import PROFILES, CandidateSimulator
from interviewEnv.interviewer.interview_state import CurrentPhase, InterviewState
from interviewEnv.interviewer.interviewer_actions import InterviewerAction, InterviewerActionType
from interviewEnv.interviewer.interviewer_observation import InterviewerObservation

DEFAULT_DATA = Path(
    os.environ.get(
        "MOCKASSIST_DATA",
        Path(__file__).resolve().parents[3] / "data" / "leetcode-training.jsonl",
    )
)
MAX_TURNS = 30
MINUTES_PER_TURN = 5
TOTAL_MINUTES = MAX_TURNS * MINUTES_PER_TURN
STUCK_THRESHOLD_MINUTES = 5
MAX_CLARIFICATION_AWARDS = 3

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
    def __init__(self, data_path: Path = DEFAULT_DATA):
        super().__init__()
        self.problems = load_problems(data_path)
        self.profile_ids = sorted(PROFILES)
        self.problem: dict = {}
        self.candidate: CandidateSimulator | None = None
        self._state = InterviewState()

    @property
    def state(self) -> InterviewState:
        return self._state

    def reset(self, seed=None, episode_id=None, **kwargs) -> InterviewerObservation:
        seed = 0 if seed is None else seed
        self.problem = kwargs.get("problem") or self.problems[seed % len(self.problems)]
        profile = kwargs.get("profile") or self.profile_ids[
            (seed // len(self.problems)) % len(self.profile_ids)
        ]
        self.candidate = CandidateSimulator(self.problem, profile)
        self._state = InterviewState(
            episode_id=episode_id,
            problem_id=str(self.problem.get("id", "")),
            scenario_id=f"{self.problem.get('id', '')}:{profile}:{seed}",
            current_phase=CurrentPhase.START,
        )
        return self._observation(candidate_message="", reward=None)

    def step(self, action: InterviewerAction, timeout_s=None, **kwargs) -> InterviewerObservation:
        if self.candidate is None:
            raise RuntimeError("reset() must be called before step()")

        state = self._state
        # Snapshot what the interviewer was reacting to. The candidate's reply has
        # not happened yet, so scoring against post-reply state would be backwards.
        stuck_before = state.stuck_minutes()
        clarification_pending = (
            state.last_candidate_action == CandidateActionType.ASK_CLARIFICATION.value
        )

        state.turn += 1
        state.step_count += 1
        state.elapsed_minutes += MINUTES_PER_TURN
        if state.current_phase is CurrentPhase.START:
            state.current_phase = CurrentPhase.INTERVIEW
        if action.action_type is InterviewerActionType.HINT:
            state.hints_used += 1

        response = self.candidate.respond(_prompt_for(action))

        if response.code_patch:
            state.progress_level = max(state.progress_level, 1)
        if response.action_type is CandidateActionType.STATE_COMPLEXITY and response.complexity_claim:
            state.current_time_complexity = response.complexity_claim
            state.progress_level = max(state.progress_level, 2)
        if state.progress_level >= 2:
            state.solution_reached = True

        ended = (
            action.action_type is InterviewerActionType.END
            or response.action_type is CandidateActionType.END
            or state.turn >= MAX_TURNS
        )
        if ended:
            state.current_phase = CurrentPhase.END

        reward = self._reward(
            action,
            response,
            stuck_before=stuck_before,
            clarification_pending=clarification_pending,
            done=ended,
        )

        # Stuck tracking runs after scoring so this turn is judged on prior state.
        if response.is_stuck:
            if state.stuck_since_minute is None:
                state.stuck_since_minute = state.elapsed_minutes
        else:
            state.stuck_since_minute = None
        state.last_candidate_action = response.action_type.value

        return self._observation(
            candidate_message=response.spoken_response,
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
        done: bool,
    ) -> float:
        """Deterministic half of the reward table.

        The five rows needing semantic judgment -- restating clearly, answering a
        clarification *correctly*, hint too strong, leaking the solution, and
        misleading guidance -- are not scored here. They need an LLM judge.
        """
        state = self._state
        reward = -0.05  # time cost
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

        time_remains = TOTAL_MINUTES - state.elapsed_minutes >= MINUTES_PER_TURN
        if (
            state.solution_reached
            and time_remains
            and action.action_type
            in (InterviewerActionType.CHALLENGE, InterviewerActionType.REQUEST_TEST)
        ):
            reward += 0.25

        if done:
            reward += 2.0
        return reward

    def _observation(
        self, *, candidate_message: str, reward: float | None, done: bool = False
    ) -> InterviewerObservation:
        return InterviewerObservation(
            done=done,
            reward=reward,
            problem=filter_interview_problem(self.problem),
            current_phase=self._state.current_phase.value,
            candidate_message=candidate_message,
            turn=self._state.turn,
        )


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
        InterviewerActionType.END: "That is all the time we have. Thank you.",
    }[action.action_type]
