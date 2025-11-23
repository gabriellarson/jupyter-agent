import os
from dataclasses import dataclass, field

from datasets import load_dataset
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
from trl import ModelConfig, SFTTrainer, TrlParser, SFTConfig
from accelerate.state import PartialState
import torch

dataset = load_dataset("jupyter-agent/jupyter-agent-dataset", split="non_thinking")

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

eval_size = int(0.02 * len(dataset))
if eval_size < 1:
    eval_size = 1
train_size = len(dataset) - eval_size
train_dataset = dataset.select(range(train_size))
eval_dataset = dataset.select(range(train_size, len(dataset)))


model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-Coder-30B-A3B-Instruct",attn_implementation="flash_attention_2",torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Coder-30B-A3B-Instruct")
with (
        open("templates/qwen3_chat_non_thinking_template.jinja") as f
    ):  # original chat template of Qwen3-4B-Instruct-2507 with added generation tags for assistant_only_loss=True
        template = f.read()
tokenizer.chat_template = template

training_args = SFTConfig(
    eval_strategy="steps",
    eval_steps=500,
    eval_on_start=True,
    gradient_accumulation_steps=1,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant":False},
    logging_steps=1,
    logging_strategy="steps",
    lr_scheduler_type="cosine_with_min_lr",
    learning_rate=5e-06,
    neftune_noise_alpha=7,
    lr_scheduler_kwargs={"min_lr_rate":0.1},
    packing=False,
    max_length=32768,
    num_train_epochs=5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    report_to="tensorboard",
    save_strategy="epoch",
    save_total_limit=10,
    seed=42,
    use_liger_kernel=True,
    warmup_ratio=0.03,
    assistant_only_loss=True,
)

trainer = SFTTrainer(
        model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        processing_class=tokenizer,
    )

trainer.train()

trainer.save_model()