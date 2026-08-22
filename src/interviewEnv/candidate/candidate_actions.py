from enum import Enum

from openenv.core.env_server import Action
from pydantic import Field


class CandidateActionType(str, Enum):
    ASK_CLARIFICATION = "ASK_CLARIFICATION"
    THINK_ALOUD = "THINK_ALOUD"
    PROPOSE_APPROACH = "PROPOSE_APPROACH"
    WRITE_CODE = "WRITE_CODE"
    MODIFY_CODE = "MODIFY_CODE"
    TEST_CODE = "TEST_CODE"
    STATE_COMPLEXITY = "STATE_COMPLEXITY"
    ASK_FOR_HINT = "ASK_FOR_HINT"
    GO_SILENT = "GO_SILENT"
    END = "END"


class CandidateAction(Action):
    action_type: CandidateActionType
    spoken_response: str = ""
    code_patch: str | None = None
    complexity_claim: str | None = None
    confidence: float = Field(0.5, ge=0.0, le=1.0)
    # Self-reported: the candidate knows it is stuck, and asking is cheaper and
    # more accurate than inferring it from action types.
    is_stuck: bool = False
