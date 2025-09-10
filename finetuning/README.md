# Finetuning

This contains the code to fine-tune the Jupyter Agent model using TRL and DeepSpeed ZeRO-2 for multi-node training.

## Installation

```bash
uv sync
```

## Multi-Node Training

The training setup uses **SLURM** for multi-node distributed training with DeepSpeed ZeRO-2 optimization:

```bash
sbatch slurm/train.slurm
```

### Configuration

- **Model**: Qwen3-4B-Instruct (full-parameter fine-tuning)
- **Nodes**: Configurable via SLURM (default: 1 node, 8 GPUs)
- **Memory optimization**: DeepSpeed ZeRO-2 (`recipes/accelerate_configs/zero2.yaml`)
- **Dataset**: [jupyter-agent-dataset](https://huggingface.co/datasets/data-agents/jupyter-agent-dataset)

### Key Training Parameters

- `assistant_only_loss: true` - Only compute loss on assistant tokens
- `neftune_noise_alpha: 7` - NEFTune noise for full-parameter training
- `learning_rate: 5e-06` - Lower LR with cosine scheduling
- `max_length: 32768` - Support long context notebooks
- `num_train_epochs: 1` - Single epoch training

## Chat Template Adaptation

Modified Qwen3 chat templates to support `assistant_only_loss=True`:

- **Standard**: `templates/qwen3_chat_template.jinja` (thinking models)
- **Non-thinking**: `templates/qwen3_chat_non_thinking_template.jinja`

The templates wrap assistant responses in `{% generation %}` tags, enabling TRL to compute loss only on model-generated content during training.

## Training Results

Our fine-tuned model achieves significant improvements on DABStep benchmark:

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/jupyter-agent-2/training_dabstep_easy.png" alt="DABstep Easy Score" width="500"/>

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/jupyter-agent-2/training_dabstep_hard.png" alt="DABstep Hard Score" width="500"/>

### Epoch Ablation Study

| Model | No. of epochs | DABstep (Easy) |
|----------|---------|---------|
| Qwen-3-4B-Instruct-2507 (Base) | 0 | 38.67% |
| Qwen-3-4B-Instruct-2507 (Our Scaffolding) | 0 | 52.78% |
| Qwen-3-4B-Instruct-2507 | 2 | 63.89% |
| Qwen-3-4B-Instruct-2507 | 3 | 73.61% |
| Qwen-3-4B-Instruct-2507 | 5 | **75%** |
| Qwen-3-4B-Instruct-2507 | 7 | 70.83% |

The results show that 5 epochs provides optimal performance, with diminishing returns beyond that point. The combination of improved scaffolding and fine-tuning delivers up to **36%** improvement over the base model on easy tasks.