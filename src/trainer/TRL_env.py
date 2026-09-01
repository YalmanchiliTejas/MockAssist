from pathlib import Path
import os

from interviewEnv import InterviewerAction
from interviewEnv.interviewer.interview_state import CurrentPhase
from interviewEnv.interviewer.interviewer_prompt import build_interviewer_prompt
from interviewEnv.server.interviewEnv_environment import InterviewEnvironment
from interviewEnv.interviewer.interviewer_actions import InterviewerActionType
import json


class TRL_Env:
    def __init__(self, data_path, rollout_dir=None):
        # Run the environment in the same Modal process as the trainer.  There is
        # no HTTP server or localhost dependency in the training path.
        self.env = InterviewEnvironment(data_path=data_path)
        self.reward = 0.0
        self._reward_finalized = False
        self.rollout_dir = Path(rollout_dir) if rollout_dir else None
        self._episode = None
        self._rollout_file = None
        if self.rollout_dir:
            self.rollout_dir.mkdir(parents=True, exist_ok=True)
            rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))
            self._rollout_file = self.rollout_dir / f"rollouts-rank-{rank}.jsonl"

    def reset(self, **kwargs):
        self._write_episode()
        self.reward = 0.0
        self._reward_finalized = False
        observation = self.env.reset(
            seed=kwargs.get("seed"),
            profile=kwargs.get("profile", "strong"),
        )

        prompt = build_interviewer_prompt(observation.problem, kwargs["profile"])
        self._episode = {
            "problem": observation.problem,
            "profile": kwargs.get("profile", "strong"),
            "seed": kwargs.get("seed"),
            "prompt": prompt,
            "turns": [],
        }
        return prompt

    def interviewer_turn(
        self, action_type: str, message: str = "", hint_level: int = 0
    ) -> str:
        """Send one interviewer message and return the candidate's response.

        Args:
            action_type: One of ASK, HINT, CHALLENGE, REQUEST_CODE,
                REQUEST_TEST, REQUEST_COMPLEXITY, TRANSITION, or END.
            message: The exact natural-language message for the candidate.
            hint_level: Hint strength from 0 through 3.

        Returns:
            The candidate response and current interview state as JSON.
        """
        # Models sometimes emit enum values with different casing. Normalize
        # them before Pydantic validation so valid calls are not rejected.
        action_kind = InterviewerActionType(str(action_type).upper())
        action = InterviewerAction(
            action_type=action_kind,
            message=message,
            hint_level=hint_level,
        )

        observation = self.env.step(action)
        self.reward += float(observation.reward or 0.0)

        result = {
            "candidate_message": observation.candidate_message,
            "candidate_code": observation.candidate_code,
            "code_execution": observation.code_execution,
            "turn": observation.turn,
            "current_phase": observation.current_phase,
            "done": observation.done,
        }
        if self._episode is not None:
            self._episode["turns"].append(
                {
                    "action_type": action.action_type.value,
                    "message": action.message,
                    "hint_level": action.hint_level,
                    "candidate_message": observation.candidate_message,
                    "candidate_code": observation.candidate_code,
                    "code_execution": observation.code_execution,
                    "turn": observation.turn,
                    "current_phase": observation.current_phase,
                    "reward": float(observation.reward or 0.0),
                    "done": observation.done,
                }
            )
        return json.dumps(result)

    def _get_reward(self) -> float:
        """Return the accumulated reward for the completed interview."""
        if not self._reward_finalized:
            if self.env.state.current_phase is not CurrentPhase.END:
                self.reward -= 1.0
            self._reward_finalized = True
        self._write_episode()
        return self.reward

    def _write_episode(self):
        if not self._episode or not self._rollout_file:
            return
        self._episode["total_reward"] = self.reward
        self._episode["num_turns"] = len(self._episode["turns"])
        self._episode["solution_reached"] = self.env.state.solution_reached
        self._episode["end_reason"] = self.env.state.end_reason or "incomplete"
        with self._rollout_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(self._episode, ensure_ascii=False) + "\n")
            handle.flush()
        self._episode = None
