from pathlib import Path

import modal


app = modal.App("mockassist-training")
DATA_PATH = Path("/root/data/leetcode-training.jsonl")
CHECKPOINTS = modal.Volume.from_name("mockassist-checkpoints", create_if_missing=True)
# Use a fresh volume because the previous cache was missing one model shard.
HF_CACHE = modal.Volume.from_name("mockassist-hf-cache-v2", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "torch",
        "transformers>=5.2.0,<6",
        "pyarrow>=21,<22",
        "datasets>=4.7.0,<5",
        "trl==1.8.0",
        "peft>=0.17.0,<1",
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
    .env({"HF_HOME": "/cache", "TRL_EXPERIMENTAL_SILENCE": "1"})
    .workdir("/root/app")
    .add_local_dir(".", remote_path="/root/app")
    .add_local_dir("../data", remote_path="/root/data")
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
    gpu="A100",
    timeout=600,
    volumes={"/cache": HF_CACHE},
    secrets=[modal.Secret.from_name("openai-secret")],
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
    print({"problem": initial.problem["title"], "candidate": result.candidate_message, "reward": result.reward})


@app.function(
    image=image,
    gpu="A100-80GB:2",
    timeout=86400,
    volumes={"/checkpoints": CHECKPOINTS, "/cache": HF_CACHE},
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("openai-secret"),
    ],
)
def train(
    model: str = "Qwen/Qwen3.5-9B",
    steps: int = 2,
    max_completion_length: int = 1024,
):
    from pathlib import Path
    import subprocess

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
        "/checkpoints/run-001",
        "--model",
        model,
        "--steps",
        str(steps),
        "--max-completion-length",
        str(max_completion_length),
    ]
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError:
        print("torchrun failed; worker logs:")
        log_root = Path("/tmp/torchrun-logs")
        if log_root.exists():
            for log_file in sorted(log_root.rglob("*.log")):
                print(f"\n===== {log_file} =====")
                lines = log_file.read_text(errors="replace").splitlines()
                print("\n".join(lines[-200:]))
        raise
    CHECKPOINTS.commit()
    HF_CACHE.commit()
