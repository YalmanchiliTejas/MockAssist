# SPDX-License-Identifier: BSD-3-Clause

"""Interviewenv Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import InterviewenvAction, InterviewenvObservation


class InterviewenvEnv(
    EnvClient[InterviewenvAction, InterviewenvObservation, State]
):
    """
    Client for the Interviewenv Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with InterviewenvEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.echoed_message)
        ...
        ...     result = client.step(InterviewenvAction(message="Hello!"))
        ...     print(result.observation.echoed_message)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = InterviewenvEnv.from_docker_image("interviewEnv-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(InterviewenvAction(message="Test"))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: InterviewenvAction) -> Dict:
        """
        Convert InterviewenvAction to JSON payload for step message.

        Args:
            action: InterviewenvAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "action_type": action.action_type.value,
            "message": action.message,
            "hint_level": action.hint_level,
        }

    def _parse_result(self, payload: Dict) -> StepResult[InterviewenvObservation]:
        """
        Parse server response into StepResult[InterviewenvObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with InterviewenvObservation
        """
        obs_data = payload.get("observation", {})
        observation = InterviewenvObservation(
            problem=obs_data.get("problem", {}),
            current_phase=obs_data.get("current_phase", ""),
            candidate_message=obs_data.get("candidate_message", ""),
            turn=obs_data.get("turn", 0),
            done=payload.get("done", False),
            reward=payload.get("reward"),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
            metadata=payload.get("metadata"),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
