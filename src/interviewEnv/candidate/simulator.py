"""LLM-backed candidate. The interviewer is the agent; this is the environment."""

from __future__ import annotations

import os
from dataclasses import dataclass

from .candidate_actions import CandidateAction, CandidateActionType

DEFAULT_MODEL = os.environ.get("MOCKASSIST_CANDIDATE_MODEL", "Qwen/Qwen3.5-4B")

_LOCAL_PROCESSOR = None
_LOCAL_MODEL = None


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
- Write Python code in code_patch so it can be executed by the interview sandbox.
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
    ) -> None:
        self.profile = PROFILES[profile] if isinstance(profile, str) else profile
        self.model = model
        self.temperature = temperature
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

        print("This is in the respond fucntion ")
        print("This is before the client message: ", interviewer_message)
        try:
            raw = _local_completion(
                model_name=self.model,
                messages=self.messages,
                temperature=self.temperature,
            )
            print("Qwen completion returned", flush=True)
        except Exception as exc:
            print(
                f"Qwen completion failed: {type(exc).__name__}: {exc}",
                flush=True,
            )
            raise
        self.messages.append({"role": "assistant", "content": raw})
        return _parse(raw)


def _local_completion(
    *, model_name: str, messages: list[dict[str, str]], temperature: float
) -> str:
    """Generate one candidate response with a local Hugging Face model."""
    global _LOCAL_MODEL, _LOCAL_PROCESSOR
    import torch

    if _LOCAL_MODEL is None or _LOCAL_PROCESSOR is None:
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        requested_device = os.environ.get("MOCKASSIST_CANDIDATE_DEVICE", "auto")
        use_cuda = torch.cuda.is_available() and requested_device != "cpu"
        dtype = torch.bfloat16 if use_cuda else torch.float32
        if use_cuda:
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            device_map = {"": f"cuda:{local_rank}"}
        else:
            device_map = {"": "cpu"}
        _LOCAL_PROCESSOR = AutoProcessor.from_pretrained(model_name)
        _LOCAL_MODEL = AutoModelForMultimodalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device_map,
        )
        _LOCAL_MODEL.eval()
        print(f"Loaded local candidate model: {model_name}", flush=True)

    # Qwen3.5's processor accepts text-only messages when represented as a
    # single text content block. Keep the conversation format unchanged for
    # the environment while adapting it at the model boundary.
    model_messages = [
        {
            "role": message["role"],
            "content": [{"type": "text", "text": message["content"]}],
        }
        for message in messages
    ]
    inputs = _LOCAL_PROCESSOR.apply_chat_template(
        model_messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False,
    ).to(_LOCAL_MODEL.device)

    with torch.inference_mode():
        outputs = _LOCAL_MODEL.generate(
            **inputs,
            max_new_tokens=int(
                os.environ.get("MOCKASSIST_CANDIDATE_MAX_NEW_TOKENS", "512")
            ),
            do_sample=True,
            temperature=temperature,
            top_p=0.8,
        )
    prompt_length = inputs["input_ids"].shape[-1]
    return _LOCAL_PROCESSOR.decode(
        outputs[0][prompt_length:], skip_special_tokens=True
    ).strip()


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
