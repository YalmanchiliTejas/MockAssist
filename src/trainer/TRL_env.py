from pathlib import Path

from interviewEnv import InterviewerAction
from interviewEnv.server.interviewEnv_environment import InterviewEnvironment
from interviewEnv.interviewer.interviewer_actions import InterviewerActionType
import json

class TRL_Env:
    def __init__(self, data_path):
        # Run the environment in the same Modal process as the trainer.  There is
        # no HTTP server or localhost dependency in the training path.
        self.env = InterviewEnvironment(data_path=data_path)
        self.reward = 0.0


    def reset(self, **kwargs):
        self.reward = 0.0
        observation = self.env.reset(
            seed=kwargs.get("seed"),
            profile=kwargs.get("profile", "strong"),
        )

        return f"""You are the interviewer in a technical coding interview.

                Interview problem:
                {json.dumps(observation.problem, indent=2)}

                Candidate profile:
                {kwargs['profile']}

                Interview state:
                - Turn: 0
                - Phase: START
                - The candidate has not responded yet.

                Your task is to conduct the interview intelligently. Decide what to say
                based on the candidate's response after each turn.

                Use the interviewer_turn tool for every interviewer message.

                The tool's action_type is evaluation metadata. The message field must contain
                the exact natural-language words you want to say to the candidate. Do not use
                canned action-to-message mappings.

                Guidelines:
                - Begin by asking the candidate to explain their approach.
                - Give hints only when appropriate.
                - Do not reveal the complete solution prematurely.
                - Ask for code, testing, or complexity when useful.
                - Keep messages concise and conversational.
                - End the interview when it is sufficiently evaluated or time expires.
                """.strip()

    def interviewer_turn(
        self, action_type: str, message: str = "", hint_level: int = 0
    ) -> str:
        """Send one interviewer message and return the candidate's response.

        Args:
            action_type: One of ASK, HINT, CHALLENGE, REQUEST_CODE,
                REQUEST_TEST, REQUEST_COMPLEXITY, or END.
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

        return json.dumps({
            "candidate_message": observation.candidate_message,
            "turn": observation.turn,
            "current_phase": observation.current_phase,
            "done": observation.done
        })


    def _get_reward(self) -> float:
        """Return the accumulated reward for the completed interview."""
        return self.reward
