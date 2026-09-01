from pydantic import Field

from openenv.core.env_server import Action
from enum import Enum
from typing import Any, Mapping

class InterviewerActionType(str, Enum):
    ASK = "ASK"
    HINT = "HINT"
    CHALLENGE = "CHALLENGE"
    REQUEST_CODE = "REQUEST_CODE"
    REQUEST_TEST = "REQUEST_TEST"
    REQUEST_COMPLEXITY = "REQUEST_COMPLEXITY"
    TRANSITION = "TRANSITION"
    END = "END"


class InterviewerAction(Action):
    action_type: InterviewerActionType = InterviewerActionType.ASK
    message: str=""
    target_concept: str | None = None
    hint_level: int= Field(default=0, ge=0, le=3)
    bug_being_challenged: str | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "InterviewerAction":
        action_type = InterviewerActionType(data.get("action_type", "ASK"))
        message = data.get("message", "")
        target_concept = data.get("target_concept")
        hint_level = data.get("hint_level", 0)
        bug_being_challenged = data.get("bug_being_challenged")
        return cls(
            action_type=action_type,
            message=message,
            target_concept=target_concept,
            hint_level=hint_level,
            bug_being_challenged=bug_being_challenged,
        )

