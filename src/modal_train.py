import os
from pathlib import Path

import modal


app = modal.App("mockassist-training")
DATA_PATH = Path("/root/data/leetcode-training.jsonl")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = modal.Volume.from_name("mockassist-checkpoints", create_if_missing=True)
# Use a fresh volume because the previous cache was missing one model shard.
HF_CACHE = modal.Volume.from_name("mockassist-hf-cache-v2", create_if_missing=True)

# Smoke evaluation budget. Rates are Modal's public per-second prices as of
# 2026-09-01. Explicit CPU/memory requests make the one-hour upper bound
# auditable instead of depending on platform defaults.
EVALUATION_GPU = "L4"
EVALUATION_TIMEOUT_SECONDS = 60 * 60
EVALUATION_CPU_CORES = 2.0
EVALUATION_MEMORY_MIB = 32 * 1024
EVALUATION_GPU_USD_PER_SECOND = 0.000222
EVALUATION_CPU_USD_PER_CORE_SECOND = 0.0000131
EVALUATION_MEMORY_USD_PER_GIB_SECOND = 0.00000222
EVALUATION_MAX_COMPUTE_USD = EVALUATION_TIMEOUT_SECONDS * (
    EVALUATION_GPU_USD_PER_SECOND
    + EVALUATION_CPU_CORES * EVALUATION_CPU_USD_PER_CORE_SECOND
    + (EVALUATION_MEMORY_MIB / 1024) * EVALUATION_MEMORY_USD_PER_GIB_SECOND
)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "torch",
        "transformers==5.2.0",
        "pyarrow>=21,<22",
        "datasets>=4.7.0,<5",
        "trl==1.8.0",
        "peft==0.20.0",
        "openenv>=0.4.1",
        "accelerate>=1.4.0,<2",
        "openai>=1.0.0",
        "Pillow",
        "torchvision",
        "num2words==0.5.14",
        "jmespath",
        "tensorboard",
    )
    .run_commands("python -m pip check")
    .run_commands(
        "python -c \"from trl import GRPOConfig, GRPOTrainer; import openai, jmespath; print('TRL dependency check passed')\""
    )
    .env(
        {
            "HF_HOME": "/cache",
            "TRL_EXPERIMENTAL_SILENCE": "1",
            "PYTHONPATH": "/root/app/src",
            "MOCKASSIST_CANDIDATE_MODEL": "Qwen/Qwen3.5-4B",
        }
    )
    .workdir("/root/app")
    .add_local_dir(
        PROJECT_ROOT,
        remote_path="/root/app",
        ignore=[
            ".git",
            "**/.git",
            "data",
            "data/**",
            "src/output",
            "src/output/**",
            "**/__pycache__",
        ],
    )
    .add_local_dir(PROJECT_ROOT / "data", remote_path="/root/data")
)


@app.function(image=image, timeout=300)
def dependency_check():
    from importlib.metadata import version

    from interviewEnv.server.interviewEnv_environment import InterviewEnvironment
    from trainer.TRL_env import TRL_Env
    from trainer.training import make_dataset

    environment = TRL_Env(DATA_PATH)
    initial_prompt = environment.reset(profile="strong", seed=0)
    print(
        {
            "trl": version("trl"),
            "transformers": version("transformers"),
            "datasets": version("datasets"),
            "pyarrow": version("pyarrow"),
            "openenv": version("openenv"),
            "environment": InterviewEnvironment.__name__,
            "trainer_environment": TRL_Env.__name__,
            "dataset_rows": len(make_dataset(DATA_PATH)),
            "initial_prompt_chars": len(initial_prompt),
        }
    )


@app.function(
    image=image,
    gpu="L40S",
    timeout=600,
    volumes={"/cache": HF_CACHE},
)
def smoke_test():
    from interviewEnv.interviewer.interviewer_actions import (
        InterviewerAction,
        InterviewerActionType,
    )
    from interviewEnv.server.interviewEnv_environment import InterviewEnvironment

    environment = InterviewEnvironment(DATA_PATH)
    initial = environment.reset(seed=0, profile="strong")
    result = environment.step(
        InterviewerAction(
            action_type=InterviewerActionType.ASK,
            message="Explain your approach.",
            hint_level=0,
        )
    )
    print(
        {
            "problem": initial.problem["title"],
            "candidate": result.candidate_message,
            "reward": result.reward,
        }
    )


@app.function(
    image=image,
    gpu="A100-80GB:2",
    timeout=86400,
    volumes={"/checkpoints": CHECKPOINTS, "/cache": HF_CACHE},
)
def train(
    model: str = "Qwen/Qwen3.5-9B",
    run_name: str = "run-001",
    steps: int = 2,
    max_completion_length: int = 1024,
    examples: int | None = None,
    heldout_fraction: float = 0.1,
    split_salt: str = "mockassist-v1",
    code_executor_url: str | None = None,
    code_executor_token: str | None = None,
    candidate_max_new_tokens: int = 512,
):
    from pathlib import Path
    import subprocess
    from huggingface_hub import snapshot_download

    if code_executor_url:
        os.environ["MOCKASSIST_CODE_EXECUTOR"] = "remote"
        os.environ["MOCKASSIST_CODE_EXECUTOR_URL"] = code_executor_url
    if code_executor_token:
        os.environ["MOCKASSIST_CODE_EXECUTOR_TOKEN"] = code_executor_token
    os.environ["MOCKASSIST_CANDIDATE_MAX_NEW_TOKENS"] = str(
        candidate_max_new_tokens
    )

    def prepare_model(model_id: str) -> str:
        model_path = Path(model_id)
        if model_path.exists():
            return str(model_path)
        print(f"Downloading model once before torchrun: {model_id}", flush=True)
        return snapshot_download(repo_id=model_id)

    # Download from one process before starting distributed workers. If rank 0
    # and rank 1 download the same sharded checkpoint concurrently, one worker
    # can observe the index before the final shard has been materialized.
    training_model_path = prepare_model(model)
    os.environ["MOCKASSIST_BASE_MODEL_ID"] = model
    candidate_model_path = prepare_model(
        os.environ.get("MOCKASSIST_CANDIDATE_MODEL", "Qwen/Qwen3.5-4B")
    )
    os.environ["MOCKASSIST_CANDIDATE_MODEL"] = candidate_model_path

    command = [
        "torchrun",
        "--standalone",
        "--nproc_per_node=2",
        "--log_dir=/tmp/torchrun-logs",
        "--redirects=3",
        "--tee=3",
        "-m",
        "trainer.training",
        "--data",
        str(DATA_PATH),
        "--output",
        f"/checkpoints/{run_name}",
        "--model",
        training_model_path,
        "--steps",
        str(steps),
        "--max-completion-length",
        str(max_completion_length),
        "--heldout-fraction",
        str(heldout_fraction),
        "--split-salt",
        split_salt,
    ]
    if examples is not None:
        command.extend(["--examples", str(examples)])
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError:
        print("torchrun failed; worker logs:", flush=True)
        log_root = Path("/tmp/torchrun-logs")
        if log_root.exists():
            for log_file in sorted(log_root.rglob("*.log")):
                print(f"\n===== {log_file} =====", flush=True)
                lines = log_file.read_text(errors="replace").splitlines()
                print("\n".join(lines[-200:]), flush=True)
        raise
    CHECKPOINTS.commit()
    HF_CACHE.commit()


@app.function(
    image=image,
    gpu=EVALUATION_GPU,
    cpu=EVALUATION_CPU_CORES,
    memory=EVALUATION_MEMORY_MIB,
    timeout=EVALUATION_TIMEOUT_SECONDS,
    volumes={"/checkpoints": CHECKPOINTS, "/cache": HF_CACHE},
)
def evaluate(
    run_name: str = "run-001",
    baseline_checkpoint: str | None = None,
    heldout_fraction: float = 0.1,
    split_salt: str = "mockassist-v1",
    profiles: str = "strong",
    seeds: str = "0",
    max_problems: int | None = 1,
    max_new_tokens: int = 256,
    candidate_model: str = "Qwen/Qwen3.5-0.8B",
    candidate_max_new_tokens: int = 256,
    evaluation_name: str = "evaluation-smoke",
    resume: bool = True,
    bootstrap_samples: int = 200,
    max_budget_usd: float = 5.0,
    code_executor_url: str | None = None,
    code_executor_token: str | None = None,
):
    """Evaluate a saved adapter on the training-excluded problem partition.

    ``profiles`` and ``seeds`` are comma-separated because Modal's CLI cannot
    parse variadic tuple annotations.
    """
    import subprocess

    if not 0 < max_budget_usd <= 30:
        raise ValueError("max_budget_usd must be greater than zero and at most $30")
    if EVALUATION_MAX_COMPUTE_USD > max_budget_usd:
        raise ValueError(
            f"Configured resources can cost up to ${EVALUATION_MAX_COMPUTE_USD:.2f}, "
            f"above the ${max_budget_usd:.2f} evaluation budget."
        )
    print(
        f"Evaluation preflight: gpu={EVALUATION_GPU}, timeout="
        f"{EVALUATION_TIMEOUT_SECONDS}s, maximum estimated compute="
        f"${EVALUATION_MAX_COMPUTE_USD:.2f} (budget=${max_budget_usd:.2f})",
        flush=True,
    )

    if code_executor_url:
        os.environ["MOCKASSIST_CODE_EXECUTOR"] = "remote"
        os.environ["MOCKASSIST_CODE_EXECUTOR_URL"] = code_executor_url
    if code_executor_token:
        os.environ["MOCKASSIST_CODE_EXECUTOR_TOKEN"] = code_executor_token
    os.environ["MOCKASSIST_CANDIDATE_MAX_NEW_TOKENS"] = str(
        candidate_max_new_tokens
    )
    os.environ["MOCKASSIST_CANDIDATE_MODEL"] = candidate_model
    os.environ["MOCKASSIST_CANDIDATE_DEVICE"] = "cpu"

    profile_values = [
        profile.strip() for profile in profiles.split(",") if profile.strip()
    ]
    seed_values = [seed.strip() for seed in seeds.split(",") if seed.strip()]
    if not profile_values:
        raise ValueError("profiles must contain at least one comma-separated profile")
    if not seed_values:
        raise ValueError("seeds must contain at least one comma-separated integer")
    try:
        seed_values = [str(int(seed)) for seed in seed_values]
    except ValueError as exc:
        raise ValueError("seeds must be comma-separated integers") from exc

    if not evaluation_name or Path(evaluation_name).name != evaluation_name:
        raise ValueError("evaluation_name must be a single directory name")
    output = f"/checkpoints/{run_name}/{evaluation_name}"
    command = [
        "python",
        "-m",
        "trainer.evaluation",
        "--checkpoint",
        f"/checkpoints/{run_name}",
        "--data",
        str(DATA_PATH),
        "--output",
        output,
        "--heldout-fraction",
        str(heldout_fraction),
        "--split-salt",
        split_salt,
        "--profiles",
        *profile_values,
        "--seeds",
        *seed_values,
        "--max-new-tokens",
        str(max_new_tokens),
        "--bootstrap-samples",
        str(bootstrap_samples),
    ]
    if max_problems is not None:
        if max_problems <= 0:
            raise ValueError("max_problems must be greater than zero")
        command.extend(["--max-problems", str(max_problems)])
    if baseline_checkpoint:
        command.extend(["--baseline-checkpoint", baseline_checkpoint])
    if resume:
        command.append("--resume")
    subprocess.run(command, check=True)
    CHECKPOINTS.commit()
    return output
