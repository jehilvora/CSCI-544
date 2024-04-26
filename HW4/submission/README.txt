Prerequisites:
- "conlleval.py" and "glove.6B.100d.txt" are in the current directory.
- Checkpoint files "task1.pt" and "task2.pt" are present in the same directory. These can be copied from the submission or by running the training script.
- The training script is an ipynb that will install requirements itself. The inference scripts require accelerate and datasets to be installed before running.
- Execution was performed solely in a Google Colab environment with .py files exported from individual notebooks. Please use similar environments if possible.

To reproduce the results for the given assignment, please follow the steps below:
1. Running the entire notebook HW4.ipynb will train the models for task1 and task2. They will produce two checkpoint files task1.pt and task2.pt for each of the tasks. These checkpoint files are attached with the submission as well
2. You can now run task1-test-infer.py with the command `python task1-test-infer.py` to run the inference on the test set using the task1 model.
2. You can now run task2-test-infer.py with the command `python task2-test-infer.py` to run the inference on the test set using the GloVe model.