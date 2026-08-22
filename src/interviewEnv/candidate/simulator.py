"""LLM-backed candidate. The interviewer is the agent; this is the environment."""

from __future__ import annotations

import os
from dataclasses import dataclass

from openai import OpenAI

from .candidate_actions import CandidateAction, CandidateActionType

DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")


@dataclass(frozen=True)
class CandidateProfile:
    profile_id: str
    description: str
    knows_optimal: bool


PROFILES = {
    "strong": CandidateProfile(
        "strong",
        "You are strong. You reason out loud clearly, ask one good clarifying "
        "question, and reach an efficient solution without much prompting.",
        knows_optimal=True,
    ),
    "average": CandidateProfile(
        "average",
        "You are average. You start with a brute-force idea and stay on it until "
        "the interviewer pushes you on efficiency. You make one off-by-one or "
        "edge-case mistake when you write code.",
        knows_optimal=False,
    ),
    "nervous": CandidateProfile(
        "nervous",
        "You are nervous and quiet. You give short answers, second-guess yourself, "
        "and need an explicit prompt before you write any code.",
        knows_optimal=False,
    ),
}

SYSTEM_PROMPT = """You are role-playing a CANDIDATE in a technical coding interview.

Persona: {persona}

Problem you were given:
{problem}

Reply with ONLY a JSON object with these keys:
  action_type       one of: {action_types}
  spoken_response   what you say out loud (one short paragraph)
  code_patch        your full current code, or null if you have not written any
  complexity_claim  e.g. "O(n) time, O(n) space", or null if you have not stated it
  confidence        0.0 to 1.0
  is_stuck          true if you are genuinely stuck and cannot make progress
                    without help; false if you are still making headway

Rules:
- Stay in character. Never break role or mention that you are an AI.
- Do not solve the whole problem in your first message. Interviews are gradual.
- Only reveal the optimal approach if your persona would realistically get there,
  and only after the conversation has built up to it.
- Use WRITE_CODE only when asked to code, and set code_patch when you do.
- Use STATE_COMPLEXITY only when asked about complexity, and set complexity_claim.
- Never mention hints you were not given.
"""


class CandidateSimulator:
    """Holds one candidate's conversation. Build a fresh one per episode."""

    def __init__(
        self,
        problem: dict,
        profile: str | CandidateProfile = "average",
        *,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.8,
        client: OpenAI | None = None,
    ) -> None:
        self.profile = PROFILES[profile] if isinstance(profile, str) else profile
        self.model = model
        self.temperature = temperature
        # Delay credential validation until the first candidate response. This
        # lets environment construction and dataset checks run without a key.
        self.client = client
        self.messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT.format(
                    persona=self.profile.description,
                    problem=_render_problem(problem, self.profile.knows_optimal),
                    action_types=", ".join(a.value for a in CandidateActionType),
                ),
            }
        ]

    def respond(self, interviewer_message: str) -> CandidateAction:
        """One interviewer turn in, one structured candidate action out."""
        self.messages.append({"role": "user", "content": interviewer_message})
        if self.client is None:
            self.client = OpenAI()
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            temperature=self.temperature,
            response_format={"type": "json_object"},
        )
        raw = (completion.choices[0].message.content or "").strip()
        self.messages.append({"role": "assistant", "content": raw})
        return _parse(raw)


def _parse(raw: str) -> CandidateAction:
    """Model output is untrusted; a malformed reply degrades to thinking aloud."""
    try:
        return CandidateAction.model_validate_json(raw)
    except Exception:
        return CandidateAction(
            action_type=CandidateActionType.THINK_ALOUD,
            spoken_response=raw,
            confidence=0.5,
        )


def _render_problem(problem: dict, knows_optimal: bool) -> str:
    """Flatten a corpus record into prompt text, gated on what this persona knows."""
    lines = [
        f"Title: {problem.get('title', '')}",
        problem.get("description") or problem.get("statement", ""),
    ]
    if constraints := problem.get("constraints"):
        lines.append("Constraints: " + "; ".join(map(str, constraints)))
    # ponytail: a weak persona is told the optimal solution exists but not what it
    # is, so it cannot leak it. Swap for a per-concept mask if that gets too blunt.
    if knows_optimal and (solutions := problem.get("solutions")):
        best = solutions[0]
        lines.append(
            f"You know an approach like this works: {best.get('title', '')} — "
            f"{best.get('time_complexity', '')} time, "
            f"{best.get('space_complexity', '')} space."
        )
    return "\n".join(part for part in lines if part)
