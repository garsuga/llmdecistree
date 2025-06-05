# LLM-Powered Decision Trees For Classification

Creating and traversing decision trees using large language models for classification.

## Documentation And Results

To view documentation and experimentation results, follow the instructions in [the website subdirectory](./website/README.md).

## Directory

`model/*`: Common code used to run, display, and test the algorithm. Specific workflows are available in the notebooks listed below.\
`tree_creation.ipynb`: Notebook used to create a tree for evaluation purposes. Amazon product categories are used for this notebook. They were retrieved from [here](https://www.kaggle.com/datasets/asaniczka/amazon-products-dataset-2023-1-4m-products).\
`classification.ipynb`: Notebook used to perform classifications. Amazon products were used as classification examples. They were retrieved from [here](https://www.kaggle.com/datasets/asaniczka/amazon-products-dataset-2023-1-4m-products).\
`tree_improvement.ipynb`: Notebook used to perform post processing on existing trees. This should be used before classification.\
`tagging_util.py`: Quick utility file to perform interactive checking on classification results.\
`stats_util.py`: Quick utility file to calculate statistics from classification results.\
`*.pkl`: Saved trees to be reused during testing and evaluation.\
`website/*`: Website for documentation and results discussion. See above.

## Environment Setup

A python virtual environment was used during the creation and execution of the code in this repository. Use the following commands to recreate this environment.

```
python -m venv .venv
./.venv/Scripts/activate
python -m pip install -r requirements.txt
```
