import json

from interviewEnv.server.code_execution import (
    GauntletDropletCodeExecutor,
    LocalPythonCodeExecutor,
    build_code_executor,
)


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return json.dumps(self.payload).encode()


JUDGED_PROBLEM = {
    "id": "7",
    "title": "Reverse Integer",
    "description": """<pre><strong>Input:</strong> x = 123
<strong>Output:</strong> 321</pre>
<pre><strong>Input:</strong> x = -123
<strong>Output:</strong> -321</pre>""",
    "metadata": {
        "reference_code_python": """class Solution:
    def reverse(self, x: int) -> int:
        sign = -1 if x < 0 else 1
        value = int(str(abs(x))[::-1])
        return sign * value"""
    },
}


DESIGN_PROBLEM = {
    "id": "352",
    "title": "Data Stream as Disjoint Intervals",
    "description": """<pre>
<strong>Input</strong>
[&quot;SummaryRanges&quot;, &quot;addNum&quot;, &quot;getIntervals&quot;, &quot;addNum&quot;, &quot;getIntervals&quot;]
[[], [1], [], [3], []]
<strong>Output</strong>
[null, null, [[1, 1]], null, [[1, 1], [3, 3]]]
<strong>Explanation</strong> example
</pre>""",
    "metadata": {},
}


def test_gauntlet_executor_posts_code_and_normalizes_result(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["authorization"] = request.headers["Authorization"]
        captured["payload"] = json.loads(request.data)
        captured["timeout"] = timeout
        return _Response(
            {
                "status": "failed",
                "stdout": "",
                "stderr": "SyntaxError: invalid syntax",
                "exit_code": 1,
            }
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    executor = GauntletDropletCodeExecutor(
        "https://runner.example/sandbox/execute",
        token="secret",
        timeout_seconds=7,
    )

    result = executor.execute(
        code="def bad(:",
        language="python",
        problem={"id": "1", "title": "Example", "solutions": ["private"]},
    )

    assert result.candidate_failure is True
    assert result.stderr == "SyntaxError: invalid syntax"
    assert captured["url"] == "https://runner.example/sandbox/execute"
    assert captured["authorization"] == "Bearer secret"
    assert captured["payload"]["code"] == "def bad(:"
    assert captured["payload"]["problem"] == {"id": "1", "title": "Example"}
    assert captured["timeout"] == 12


def test_missing_executor_configuration_is_infrastructure_error(monkeypatch):
    monkeypatch.delenv("MOCKASSIST_CODE_EXECUTOR_URL", raising=False)
    monkeypatch.delenv("SANDBOX_RUNNER_URL", raising=False)

    result = GauntletDropletCodeExecutor().execute(
        code="print('hello')", language="python", problem={"id": "1"}
    )

    assert result.status == "infrastructure_error"
    assert result.candidate_failure is False


def test_local_executor_runs_python_without_remote_configuration(monkeypatch):
    monkeypatch.delenv("MOCKASSIST_CODE_EXECUTOR", raising=False)

    executor = build_code_executor()
    result = executor.execute(
        code="""class Solution:
    def reverse(self, x: int) -> int:
        sign = -1 if x < 0 else 1
        return sign * int(str(abs(x))[::-1])""",
        language="python",
        problem=JUDGED_PROBLEM,
    )

    assert isinstance(executor, LocalPythonCodeExecutor)
    assert result.passed is True


def test_local_executor_rejects_algorithmically_wrong_code():
    result = LocalPythonCodeExecutor().execute(
        code="""class Solution:
    def reverse(self, x: int) -> int:
        return x""",
        language="python",
        problem=JUDGED_PROBLEM,
    )

    assert result.candidate_failure is True
    assert result.exit_code == 0
    assert "tests failed" in result.detail


def test_local_executor_detects_syntax_errors():
    result = LocalPythonCodeExecutor().execute(
        code="class Solution def broken(:",
        language="python",
        problem=JUDGED_PROBLEM,
    )

    assert result.candidate_failure is True
    assert "SyntaxError" in result.stderr


def test_local_executor_detects_top_level_exceptions():
    result = LocalPythonCodeExecutor().execute(
        code="raise RuntimeError('top-level failure')",
        language="python",
        problem=JUDGED_PROBLEM,
    )

    assert result.candidate_failure is True
    assert "RuntimeError: top-level failure" in result.stderr


def test_local_executor_detects_timeouts():
    result = LocalPythonCodeExecutor(timeout_seconds=0.2).execute(
        code="while True: pass",
        language="python",
        problem=JUDGED_PROBLEM,
    )

    assert result.candidate_failure is True
    assert result.timed_out is True


def test_local_executor_detects_process_crashes():
    result = LocalPythonCodeExecutor().execute(
        code="import os; os._exit(7)",
        language="python",
        problem=JUDGED_PROBLEM,
    )

    assert result.candidate_failure is True
    assert result.exit_code == 7


def test_local_executor_does_not_validate_without_oracle_tests():
    result = LocalPythonCodeExecutor().execute(
        code="print('looks fine')", language="python", problem={"id": "1"}
    )

    assert result.status == "infrastructure_error"
    assert result.passed is False


def test_local_executor_runs_stateful_design_operation_sequences():
    result = LocalPythonCodeExecutor().execute(
        code="""class SummaryRanges:
    def __init__(self): self.values = set()
    def addNum(self, value): self.values.add(value)
    def getIntervals(self): return [[value, value] for value in sorted(self.values)]""",
        language="python",
        problem=DESIGN_PROBLEM,
    )

    assert result.passed is True


def test_local_executor_rejects_wrong_stateful_design_code():
    result = LocalPythonCodeExecutor().execute(
        code="""class SummaryRanges:
    def addNum(self, value): pass
    def getIntervals(self): return []""",
        language="python",
        problem=DESIGN_PROBLEM,
    )

    assert result.candidate_failure is True
    assert "tests failed" in result.detail
