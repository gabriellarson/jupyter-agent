# Jupyter Agent 🤓

![Thumbnail](https://cdn-uploads.huggingface.co/production/uploads/650ed7adf141bc34f91a12ae/ZyF9foqe5SLECwkq0dOpT.png)

Jupyter Agent is an **open-source data science agent** that lives inside your Jupyter notebook.
It can:

* Read notebook + dataset context
* Execute Python code (`pandas`, `numpy`, `matplotlib`, …)
* Produce step-by-step reasoning traces with intermediate computations

👉 Think of it as *Cursor*, but built natively for data analysis workflows.

📖 Learn more in our [blog post](https://huggingface.co/blog/jupyter-agent-2) or try the [live demo](https://huggingface.co/spaces/lvwerra/jupyter-agent-2).


## 🚀 What’s Included

We release:

* **Dataset:** [Jupyter Agent Dataset](https://huggingface.co/datasets/data-agents/jupyter-agent-dataset) (51k synthetic notebooks, \~0.2B tokens)
* **Models:**

  * [Jupyter-Agent-Qwen3-4B-Instruct](https://huggingface.co/data-agents/jupyter-agent-qwen3-4b-instruct)
  * [Jupyter-Agent-Qwen3-4B-Thinking](https://huggingface.co/data-agents/jupyter-agent-qwen3-4b-thinking)
* **Pipeline:** Code to generate training data from Kaggle notebooks + fine-tuning scripts 

## 🎯 Why This Matters

* Jupyter notebooks are the **de facto environment for scientists and analysts**.
* We built a dataset + training pipeline that helps small models become **strong data agents**.
* On the [DABStep benchmark](https://huggingface.co/spaces/adyen/DABstep), our tuned 4B model reaches **SOTA performance for its size** on realistic data science tasks.

## 🏗️ Pipeline Overview

Our pipeline processes the [Meta Kaggle Notebooks dataset](https://www.kaggle.com/datasets/kaggle/meta-kaggle-code) (2TB) into training-ready data:

1. Deduplicate notebooks (\~90% duplicates)
2. Fetch linked datasets for executability
3. Score notebooks for educational quality
4. Filter irrelevant content
5. Generate dataset-grounded QA pairs
6. Produce reasoning + execution traces
7. Curate final dataset (\~2B tokens)

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/jupyter-agent-2/jupyter-agent-dataset-pipeline.png" alt="Pipeline" width="600"/>  

## 🔧 Quick Start

Clone the repo:

```bash
git clone https://github.com/huggingface/jupyter-agent.git
cd jupyter-agent
```

### Run the Code

* To **generate the dataset**, check the [`data/`](https://github.com/huggingface/jupyter-agent/tree/main/data) folder.
* To **fine-tune the model**, check the [`finetuning/`](https://github.com/huggingface/jupyter-agent/tree/main/finetuning) folder.

### Load the Dataset

```python
from datasets import load_dataset
ds = load_dataset("data-agents/jupyter-agent-dataset", split="non-thinking")
```

### Run a Fine-Tuned Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = "data-agents/jupyter-agent-qwen3-4b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model)
model = AutoModelForCausalLM.from_pretrained(model, torch_dtype="auto", device_map="auto")
```

## 📊 Results

* Base Qwen3-4B-Instruct (easy split): **38.7%**
* With scaffolding: **52.8%**
* After fine-tuning on our dataset: **75%**

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/jupyter-agent-2/training_dabstep_easy.png" alt="DABstep Easy Score" width="500"/> 

Our fine-tuned model is the **current SOTA small-model agent** on DABStep.

## 📚 Resources

* [Blog post](https://huggingface.co/blog/jupyter-agent-2) – full story + insights
* [Dataset on Hub](https://huggingface.co/datasets/data-agents/jupyter-agent-dataset)
* [Models on Hub](https://huggingface.co/collections/data-agents/jupyter-agent-66f43f63b3d87c9ac69039eb)
* [DABStep Benchmark](https://huggingface.co/spaces/adyen/DABstep)

## 📜 Citation

```bibtex
@misc{jupyteragentdataset,
  title={Jupyter Agent Dataset},
  author={Colle, Baptiste and Yukhymenko, Hanna and von Werra, Leandro},
  year={2025}
}
```