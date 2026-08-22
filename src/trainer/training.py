import json
import os
from pathlib import Path

from datasets import Dataset
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

from .TRL_env import TRL_Env


MODEL_NAME = os.environ.get("MOCKASSIST_MODEL", "Qwen/Qwen3.5-9B")


def make_dataset(data_path: Path) -> Dataset:
    problems = [
        json.loads(line)
        for line in data_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
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
    return Dataset.from_list(rows)


def reward_func(environments, **kwargs):
    return [environment.reward for environment in environments]


def trainer_main(
    data_path,
    output_dir,
    model_name=MODEL_NAME,
    max_steps=2,
    max_completion_length=1024,
):
    data_path = Path(data_path)
    dataset = make_dataset(data_path)
    config = GRPOConfig(
        output_dir=str(output_dir),
        logging_dir=str(Path(output_dir) / "tensorboard"),
        logging_steps=1,
        report_to="tensorboard",
        learning_rate=1e-6,
        gradient_accumulation_steps=1,
        num_generations=2,
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
    )
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )
    trainer = GRPOTrainer(
        model=model_name,
        args=config,
        train_dataset=dataset,
        reward_funcs=reward_func,
        environment_factory=lambda: TRL_Env(data_path),
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(str(output_dir))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("./output"))
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--max-completion-length", type=int, default=1024)
    args = parser.parse_args()
    trainer_main(
        args.data,
        args.output,
        args.model,
        args.steps,
        args.max_completion_length,
    )
