import json
import os
from pathlib import Path

from datasets import Dataset
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer
from transformers import TrainerCallback
from torch.utils.tensorboard import SummaryWriter

from .TRL_env import TRL_Env
from .evaluation import load_problem_records, split_problem_records


MODEL_NAME = os.environ.get("MOCKASSIST_MODEL", "Qwen/Qwen3.5-9B")


def make_dataset(
    data_path: Path,
    limit: int | None = None,
    heldout_fraction: float = 0.0,
    split_salt: str = "mockassist-v1",
) -> Dataset:
    problems = load_problem_records(data_path)
    if heldout_fraction:
        problems = split_problem_records(
            problems,
            heldout_fraction=heldout_fraction,
            split_salt=split_salt,
            split="train",
        )
    rows = []
    for seed, _problem in enumerate(problems):
        for profile in ("strong", "average", "nervous"):
            rows.append(
                {
                    # TRL expects a list of chat messages, not a nested list.
                    "prompt": [{"role": "user", "content": "Run the interview."}],
                    "profile": profile,
                    "seed": seed,
                }
            )
    if limit is not None:
        rows = rows[:limit]
    return Dataset.from_list(rows)


def reward_func(environments, **kwargs):
    rewards = []
    for environment in environments:
        rewards.append(environment._get_reward())
    return rewards


class ExplicitTensorBoardCallback(TrainerCallback):
    """Write scalar logs explicitly to the output volume."""

    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir=str(log_dir))

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            for name, value in logs.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(name, value, state.global_step)
            self.writer.flush()

    def on_train_end(self, args, state, control, **kwargs):
        self.writer.flush()
        self.writer.close()


class MockAssistGRPOTrainer(GRPOTrainer):
    """GRPO trainer with a workaround for Qwen 3.5's stale MRoPE cache.

    Qwen's multimodal wrapper stores ``rope_deltas`` during generation using
    the generation batch size. TRL subsequently scores those generations in
    smaller chunks. Reusing the larger cached tensor can produce an empty
    position-id batch (for example, ``1 // 32 == 0``) and crash rotary
    attention. Text-only scoring does not need that generation cache, so clear
    it before each log-probability pass.
    """

    @staticmethod
    def _clear_qwen_rope_deltas(model):
        for module in model.modules():
            if module.__class__.__name__ == "Qwen3_5Model":
                module.rope_deltas = None

    def _get_per_token_logps_and_entropies(self, model, *args, **kwargs):
        self._clear_qwen_rope_deltas(model)
        return super()._get_per_token_logps_and_entropies(
            model, *args, **kwargs
        )


def trainer_main(
    data_path,
    output_dir,
    model_name=MODEL_NAME,
    max_steps=2,
    max_completion_length=1024,
    examples=None,
    heldout_fraction=0.0,
    split_salt="mockassist-v1",
):
    data_path = Path(data_path)
    dataset = make_dataset(
        data_path,
        limit=examples,
        heldout_fraction=heldout_fraction,
        split_salt=split_salt,
    )
    output_dir = Path(output_dir)
    tensorboard_dir = output_dir / "tensorboard"
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    rollout_dir = output_dir / "rollouts"
    config = GRPOConfig(
        output_dir=str(output_dir),
        logging_steps=1,
        report_to="tensorboard",
        learning_rate=5e-7,
        gradient_accumulation_steps=1,
        num_generations=8,
        max_steps=max_steps,
        per_device_train_batch_size=1,
        bf16=True,
        gradient_checkpointing=True,
        beta=0.0,
        epsilon=0.2,
        num_iterations=1,
        temperature=0.9,
        top_p=0.95,
        max_completion_length=max_completion_length,
        use_vllm=False,
        chat_template_kwargs={"enable_thinking": False},
        scale_rewards="batch",
        generation_batch_size=32,
    )
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )
    trainer = MockAssistGRPOTrainer(
        model=model_name,
        args=config,
        train_dataset=dataset,
        reward_funcs=reward_func,
        environment_factory=lambda: TRL_Env(data_path, rollout_dir=rollout_dir),
        peft_config=peft_config,
        callbacks=[ExplicitTensorBoardCallback(tensorboard_dir)],
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    adapter_config_path = output_dir / "adapter_config.json"
    if trainer.is_world_process_zero() and adapter_config_path.exists():
        adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
        adapter_config["base_model_name_or_path"] = os.environ.get(
            "MOCKASSIST_BASE_MODEL_ID", model_name
        )
        adapter_config_path.write_text(
            json.dumps(adapter_config, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("./output"))
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--max-completion-length", type=int, default=1024)
    parser.add_argument("--examples", type=int, default=None)
    parser.add_argument(
        "--heldout-fraction",
        type=float,
        default=0.0,
        help="Problem-level test-set fraction to exclude from training.",
    )
    parser.add_argument("--split-salt", default="mockassist-v1")
    args = parser.parse_args()
    trainer_main(
        args.data,
        args.output,
        args.model,
        args.steps,
        args.max_completion_length,
        args.examples,
        args.heldout_fraction,
        args.split_salt,
    )
