How to run:

1) Run `merge_meta_sources.py` to retireve merged metadata DataFrame, saved into .csv file. It contains Kernel/notebook IDs and their respective linked dataset Id.
2) Run the pipeline steps in `pipelines/kaggle_data_mapping.py` to retrieve mapping of Kaggle source datasets used in original notebooks and whether they are locally available or not.

**Important!** Change `KAGGLEHUB_CACHE` env variable to some local folder convenient for you, or just use `fsx/data-agents`, for example:
```bash
os.environ["KAGGLEHUB_CACHE"] = "/fsx/jupyter-agent/kaggle-datasets"
```
Setting Kaggle API credentials (Kaggle account required) from [Settings tab](https://www.kaggle.com/settings) is optional, but apparently it can have higher rate limits:
```bash
os.environ["KAGGLE_USERNAME"] = "<your_username>"
os.environ["KAGGLE_KEY"] = "<your_api_key>"
```
Rate limits are not specified anywhere, but they might be around 1k/day.