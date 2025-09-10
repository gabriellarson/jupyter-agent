import os
from dataclasses import dataclass, field

from datasets import load_dataset
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
from trl import ModelConfig, SFTTrainer, TrlParser, SFTConfig
from accelerate.state import PartialState

# get slurm job id
slurm_job_id = os.environ.get("SLURM_JOB_ID")
if slurm_job_id is None:
    raise Exception("No SLURM_JOB_ID found")


@dataclass
class CustomScriptArguments:
    take_first_n_samples: int | None = field(
        default=None,
        metadata={"help": ("Number of samples to take from the dataset.")},
    )


def load_local_dataset(mode):
    """
    Load a local dataset from fsx into a huggingface dataset for TRL
    """
    print("Loading Jupyter Agent Dataset")

    split = "thinking" if mode else "non_thinking"

    print(f"Loading dataset for {split} mode")

    dataset = load_dataset("jupyter-agent/jupyter-agent-dataset", split=split)

    print(f"Dataset loaded for {split} mode")

    cols = [
        "id",
        "edu_score",
        "files_used",
        "packages_used",
        "question",
        "answer",
        "kaggle_dataset_name",
        "executor_type",
        "original_notebook",
    ]

    dataset = dataset.remove_columns(cols)
    dataset = dataset.shuffle(seed=42)

    return dataset


def train(script_args, training_args, model_args):
    thinking_mode = False  # use False for Qwen-3-4B-Instruct-2507

    state = PartialState()  # this is done to avoid multi-GPU preprocessing cache error
    if state.is_main_process:
        dataset = load_local_dataset(thinking_mode)
        if (
            script_args.take_first_n_samples is not None
            and script_args.take_first_n_samples > 0
            and script_args.take_first_n_samples < len(dataset)
        ):
            print(f"Taking first {script_args.take_first_n_samples} samples")
            dataset = dataset.select(range(script_args.take_first_n_samples))
        print(f"length of dataset: {len(dataset)}")
        eval_size = int(0.02 * len(dataset))
        if eval_size < 1:
            eval_size = 1
        train_size = len(dataset) - eval_size
        train_dataset = dataset.select(range(train_size))
        eval_dataset = dataset.select(range(train_size, len(dataset)))

    state.wait_for_everyone()
    print("Dataset loaded")

    # this will be used from newly created dataset cache folder
    dataset = load_local_dataset(thinking_mode)

    if (
        script_args.take_first_n_samples is not None
        and script_args.take_first_n_samples > 0
        and script_args.take_first_n_samples < len(dataset)
    ):
        dataset = dataset.select(range(script_args.take_first_n_samples))

    # Split 5% of the dataset for evaluation
    # This can be reduced to 1% in case of OOM errors
    eval_size = int(0.02 * len(dataset))
    if eval_size < 1:
        eval_size = 1
    train_size = len(dataset) - eval_size
    train_dataset = dataset.select(range(train_size))
    eval_dataset = dataset.select(range(train_size, len(dataset)))

    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

    with (
        open("templates/qwen3_chat_non_thinking_template.jinja") as f
    ):  # original chat template of Qwen3-4B-Instruct-2507 with added generation tags for assistant_only_loss=True
        template = f.read()

    if thinking_mode:
        with (
            open("templates/qwen3_chat_template.jinja") as f
        ):  # original chat template of Qwen3-4B/Thinking-2507 with added generation tags for assistant_only_loss=True
            template = f.read()

    tokenizer.chat_template = template

    print(f"Train samples: {len(train_dataset)}, Eval samples: {len(eval_dataset)}")

    trainer = SFTTrainer(
        model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        processing_class=tokenizer,
    )

    state = PartialState()  # this is done to avoid multi-GPU preprocessing cache error

    state.wait_for_everyone()

    trainer.train()

    trainer.save_model()


def main():
    parser = TrlParser((CustomScriptArguments, SFTConfig, ModelConfig))  # type: ignore
    script_args, training_args, model_args = parser.parse_args_and_config()
    train(script_args, training_args, model_args)


if __name__ == "__main__":
    main()
