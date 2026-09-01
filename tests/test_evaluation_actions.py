import json

from trainer.evaluation import (
    MALFORMED_ACTION_PENALTY,
    _episode,
    _episode_key,
    _load_completed_episodes,
    _processor_messages,
    parse_action,
)


def test_resume_loads_jsonl_and_builds_stable_scenario_key(tmp_path):
    rollout = tmp_path / "evaluation-rollouts.jsonl"
    episode = {
        "policy": "trained",
        "problem_id": "4",
        "profile": "strong",
        "seed": 0,
    }
    rollout.write_text(json.dumps(episode) + "\n", encoding="utf-8")

    loaded = _load_completed_episodes(rollout)

    assert loaded == [episode]
    assert _episode_key(loaded[0]) == ("trained", "4", "strong", 0)


def test_processor_messages_converts_strings_to_qwen_content_blocks():
    messages = [
        {"role": "user", "content": "Run the interview."},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "1"}]},
        {"role": "tool", "content": '{"done": false}'},
    ]

    normalized = _processor_messages(messages)

    assert normalized[0]["content"] == [
        {"type": "text", "text": "Run the interview."}
    ]
    assert normalized[1]["content"] == []
    assert normalized[1]["tool_calls"] == [{"id": "1"}]
    assert normalized[2]["content"] == [
        {"type": "text", "text": '{"done": false}'}
    ]
    assert messages[0]["content"] == "Run the interview."


def test_parse_action_accepts_qwen_xml_tool_call():
    raw = """<think>ignored</think>
<tool_call>
<function=interviewer_turn>
<parameter=action_type>
HINT
</parameter>
<parameter=message>
Think about a hash map.
</parameter>
<parameter=hint_level>
2
</parameter>
</function>
</tool_call>"""

    action, malformed = parse_action(raw)

    assert malformed is False
    assert action == {
        "action_type": "HINT",
        "message": "Think about a hash map.",
        "hint_level": 2,
    }


def test_parse_action_accepts_json_after_reasoning():
    raw = 'Reasoning first. {"action_type":"ASK","message":"Explain.","hint_level":0}'

    action, malformed = parse_action(raw)

    assert malformed is False
    assert action["action_type"] == "ASK"


def test_parse_action_does_not_turn_malformed_output_into_end():
    action, malformed = parse_action("truncated reasoning without a tool call")

    assert action is None
    assert malformed is True


class _AlwaysMalformedPolicy:
    def act(self, **kwargs):
        return "not a complete tool call"


class _RepairablePolicy:
    def __init__(self):
        self.calls = 0

    def act(self, **kwargs):
        self.calls += 1
        if self.calls == 1:
            return "truncated"
        return """<tool_call>
<function=interviewer_turn>
<parameter=action_type>END</parameter>
<parameter=message>Thank you for your time.</parameter>
</function>
</tool_call>"""


def test_episode_penalizes_malformed_output_without_terminal_bonus():
    problem = {"id": "1", "title": "Example", "description": "Solve it."}

    episode = _episode(_AlwaysMalformedPolicy(), problem, "strong", 0)

    assert episode["total_reward"] == MALFORMED_ACTION_PENALTY
    assert episode["end_reason"] == "malformed_action"
    assert episode["malformed_actions"] == 1
    assert episode["format_retries"] == 1
    assert episode["elapsed_minutes"] == 0
    assert episode["turns"][0]["action"]["action_type"] == "INVALID"


def test_episode_repairs_format_once_before_scoring_action():
    problem = {"id": "1", "title": "Example", "description": "Solve it."}

    episode = _episode(_RepairablePolicy(), problem, "strong", 0)

    assert episode["total_reward"] == -0.95
    assert episode["end_reason"] == "interviewer_end"
    assert episode["malformed_actions"] == 0
    assert episode["format_retries"] == 1
    assert len(episode["turns"][0]["raw_model_outputs"]) == 2
