"""Resource-limited local and optional remote execution of candidate code."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from interviewEnv.server.python_judge import build_test_program, parse_test_report


@dataclass(frozen=True)
class CodeExecutionResult:
    """Normalized result returned by the remote code sandbox."""

    status: str
    stdout: str = ""
    stderr: str = ""
    exit_code: int | None = None
    timed_out: bool = False
    detail: str = ""

    @property
    def passed(self) -> bool:
        return self.status == "passed"

    @property
    def candidate_failure(self) -> bool:
        return self.status == "failed"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CodeExecutor(Protocol):
    def execute(
        self, *, code: str, language: str, problem: dict[str, Any]
    ) -> CodeExecutionResult: ...


class LocalPythonCodeExecutor:
    """Execute Python in a short-lived, resource-limited child process.

    This is intentionally a POC executor, not a hardened multi-tenant sandbox.
    It removes inherited environment variables, uses Python isolated mode, limits
    time/memory/processes/output, and runs from a temporary directory. Production
    use should select the remote container executor instead.
    """

    def __init__(
        self,
        *,
        timeout_seconds: float = 5.0,
        memory_bytes: int = 512 * 1024 * 1024,
        output_bytes: int = 1024 * 1024,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.memory_bytes = memory_bytes
        self.output_bytes = output_bytes

    def execute(
        self, *, code: str, language: str, problem: dict[str, Any]
    ) -> CodeExecutionResult:
        if language.lower() not in {"py", "python", "python3"}:
            return CodeExecutionResult(
                status="infrastructure_error",
                detail=f"Local POC executor supports Python only, not {language!r}.",
            )

        program = build_test_program(code, problem)
        if program is None:
            return CodeExecutionResult(
                status="infrastructure_error",
                detail=(
                    "No executable oracle tests could be built from this problem's "
                    "reference solution and examples."
                ),
            )

        with tempfile.TemporaryDirectory(prefix="mockassist-code-") as directory:
            workdir = Path(directory)
            script = workdir / "candidate.py"
            script.write_text(program, encoding="utf-8")
            stdout_path = workdir / "stdout.txt"
            stderr_path = workdir / "stderr.txt"
            with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
                process = subprocess.Popen(
                    [sys.executable, "-I", "-S", str(script)],
                    cwd=workdir,
                    env={
                        "PATH": os.defpath,
                        "PYTHONIOENCODING": "utf-8",
                    },
                    stdin=subprocess.DEVNULL,
                    stdout=stdout,
                    stderr=stderr,
                    preexec_fn=self._limit_child,
                )
                timed_out = False
                try:
                    exit_code = process.wait(timeout=self.timeout_seconds)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    exit_code = process.wait()

            stdout_text = stdout_path.read_text(
                encoding="utf-8", errors="replace"
            )[: self.output_bytes]
            stderr_text = stderr_path.read_text(
                encoding="utf-8", errors="replace"
            )[: self.output_bytes]
            report = parse_test_report(stdout_text) if exit_code == 0 else None
            visible_stdout = "\n".join(
                line
                for line in stdout_text.splitlines()
                if not line.startswith("__MOCKASSIST_TEST_RESULT__=")
            )
            if timed_out or exit_code != 0:
                status = "failed"
                detail = (
                    f"Execution exceeded {self.timeout_seconds:g} seconds."
                    if timed_out
                    else "Candidate or test harness crashed."
                )
            elif report is None or not report.get("executed"):
                status = "infrastructure_error"
                detail = "The oracle test harness could not execute any valid tests."
            elif report.get("failures"):
                failures = report["failures"]
                status = "failed"
                detail = (
                    f"{len(failures)}/{report['executed']} executed tests failed. "
                    f"First failure: {json.dumps(failures[0], default=str)[:1200]}"
                )
            else:
                status = "passed"
                detail = (
                    f"Passed {report['executed']} oracle tests "
                    f"({report.get('skipped', 0)} invalid generated variants skipped)."
                )
            return CodeExecutionResult(
                status=status,
                stdout=visible_stdout,
                stderr=stderr_text,
                exit_code=exit_code,
                timed_out=timed_out,
                detail=detail,
            )

    def _limit_child(self) -> None:
        os.setsid()
        try:
            import resource

            cpu_seconds = max(1, int(self.timeout_seconds) + 1)
            limits = (
                (resource.RLIMIT_CPU, cpu_seconds),
                (resource.RLIMIT_AS, self.memory_bytes),
                (resource.RLIMIT_FSIZE, self.output_bytes),
                (resource.RLIMIT_NOFILE, 64),
                (resource.RLIMIT_NPROC, 32),
                (resource.RLIMIT_CORE, 0),
            )
            for kind, value in limits:
                resource.setrlimit(kind, (value, value))
        except (ImportError, OSError, ValueError):
            # Wall-clock timeout and isolated environment still apply on systems
            # that do not expose every POSIX resource limit.
            pass


def build_code_executor() -> CodeExecutor:
    """Select local POC execution by default; remote remains opt-in."""
    mode = os.environ.get("MOCKASSIST_CODE_EXECUTOR", "local").lower()
    if mode == "local":
        return LocalPythonCodeExecutor()
    if mode == "remote":
        return GauntletDropletCodeExecutor()
    raise ValueError("MOCKASSIST_CODE_EXECUTOR must be 'local' or 'remote'")


class GauntletDropletCodeExecutor:
    """HTTP client for the code endpoint hosted by Gauntlet's Docker droplet.

    ``MOCKASSIST_CODE_EXECUTOR_URL`` should be the complete endpoint URL. The
    expected endpoint is ``/sandbox/execute`` on the same runner used by
    Gauntlet. Alternatively, setting ``SANDBOX_RUNNER_URL`` appends that path.
    Authentication is optional and is read from
    ``MOCKASSIST_CODE_EXECUTOR_TOKEN`` or ``SANDBOX_RUNNER_TOKEN``.
    """

    def __init__(
        self,
        endpoint: str | None = None,
        *,
        token: str | None = None,
        timeout_seconds: float = 15.0,
    ) -> None:
        runner_url = os.environ.get("SANDBOX_RUNNER_URL", "").rstrip("/")
        self.endpoint = (
            endpoint
            or os.environ.get("MOCKASSIST_CODE_EXECUTOR_URL")
            or (f"{runner_url}/sandbox/execute" if runner_url else "")
        )
        self.token = (
            token
            or os.environ.get("MOCKASSIST_CODE_EXECUTOR_TOKEN")
            or os.environ.get("SANDBOX_RUNNER_TOKEN")
        )
        self.timeout_seconds = timeout_seconds

    def execute(
        self, *, code: str, language: str, problem: dict[str, Any]
    ) -> CodeExecutionResult:
        if not self.endpoint:
            return CodeExecutionResult(
                status="infrastructure_error",
                detail=(
                    "Code executor is not configured. Set "
                    "MOCKASSIST_CODE_EXECUTOR_URL or SANDBOX_RUNNER_URL."
                ),
            )

        payload = {
            "code": code,
            "language": language,
            "timeout_seconds": self.timeout_seconds,
            "problem": {
                key: problem.get(key)
                for key in ("id", "slug", "title", "description", "constraints")
                if problem.get(key) is not None
            },
        }
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.timeout_seconds + 5
            ) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            return CodeExecutionResult(
                status="infrastructure_error",
                detail=f"Executor returned HTTP {exc.code}: {detail[:1000]}",
            )
        except (OSError, TimeoutError, json.JSONDecodeError) as exc:
            return CodeExecutionResult(
                status="infrastructure_error",
                detail=f"Executor request failed: {type(exc).__name__}: {exc}",
            )

        if not isinstance(body, dict):
            return CodeExecutionResult(
                status="infrastructure_error",
                detail="Executor returned JSON that was not an object.",
            )

        status = str(body.get("status", "")).lower()
        passed_value = body.get("passed", body.get("success"))
        exit_code = body.get("exit_code")
        timed_out = bool(body.get("timed_out", status == "timeout"))
        if status not in {"passed", "failed", "infrastructure_error"}:
            if passed_value is True:
                status = "passed"
            elif passed_value is False or timed_out or (
                isinstance(exit_code, int) and exit_code != 0
            ):
                status = "failed"
            else:
                status = "infrastructure_error"
        return CodeExecutionResult(
            status=status,
            stdout=str(body.get("stdout", "")),
            stderr=str(body.get("stderr", "")),
            exit_code=exit_code if isinstance(exit_code, int) else None,
            timed_out=timed_out,
            detail=str(body.get("detail", body.get("error", ""))),
        )
