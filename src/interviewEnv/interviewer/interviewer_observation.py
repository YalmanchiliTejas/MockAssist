from openenv.core.env_server import Observation
from typing import Any
from pydantic import Field

class InterviewerObservation(Observation):

    problem: dict[str, Any] = Field(default_factory=dict)
    current_phase: str = ""
    candidate_message: str = ""
    candidate_code: str | None = None
    code_execution: dict[str, Any] | None = None
    turn: int = 0

    
   
