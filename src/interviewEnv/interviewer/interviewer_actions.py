from enum import Enum

from openenv.core.env_server import Action
from pydantic import Field


class InterviewerActionType(str, Enum):
    ASK = "ASK"
    HINT = "HINT"
    CHALLENGE = "CHALLENGE"
    REQUEST_CODE = "REQUEST_CODE"
    REQUEST_TEST = "REQUEST_TEST"
    REQUEST_COMPLEXITY = "REQUEST_COMPLEXITY"
    END = "END"


class InterviewerAction(Action):
    action_type: InterviewerActionType
    message: str = ""
    hint_level: int = Field(0, ge=0, le=3)
