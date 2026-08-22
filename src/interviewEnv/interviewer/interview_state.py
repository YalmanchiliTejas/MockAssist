from enum import Enum

from openenv.core.env_server import State


class CurrentPhase(str, Enum):
    START = "START"
    INTERVIEW = "INTERVIEW"
    END = "END"


class InterviewState(State):

    problem_id: str = ""
    scenario_id: str = ""
    progress_level: int = 0
    current_phase: CurrentPhase = CurrentPhase.START
    hints_used: int = 0
    solution_reached: bool = False
    current_time_complexity: str = ""
    current_space_complexity: str = ""
    turn: int = 0

    # --- reward bookkeeping ---
    elapsed_minutes: int = 0
    stuck_since_minute: int | None = None
    clarifications_answered: int = 0
    last_candidate_action: str = ""

    def stuck_minutes(self) -> int:
        """How long the candidate has been stuck. 0 when not stuck."""
        if self.stuck_since_minute is None:
            return 0
        return self.elapsed_minutes - self.stuck_since_minute
