from openenv.core.env_server import Observation
from typing import Any
from pydantic import Field

class InterviewerObservation(Observation):

    problem: dict[str, Any] = Field(default_factory=dict)
    current_phase: str = ""
    candidate_message: str = ""
    turn: int = 0

    
   
