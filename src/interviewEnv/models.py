# SPDX-License-Identifier: BSD-3-Clause

"""Public client-side model names for the interview environment."""

from .interviewer.interviewer_actions import InterviewerAction
from .interviewer.interviewer_observation import InterviewerObservation

# Keep the names used by the OpenEnv client template while exposing the
# interview-specific schemas used by the server.
InterviewenvAction = InterviewerAction
InterviewenvObservation = InterviewerObservation

__all__ = ["InterviewenvAction", "InterviewenvObservation"]
