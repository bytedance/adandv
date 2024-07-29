# Introduction
This repository contains the implementation of **AdaNDV**. We provide details in our paper, including but not limited to the train/validation/test dataset splits, preprocessed data, and model training. You can obtain the results presented in our paper by following the instructions below.

# Instruction
## Reproduce the results in the paper
We are unable to provide the implementation of statistical estimators (base estimators) and the raw data in this repository due to license issues. You can reproduce the results in the paper by following the instructions below.

1. Establish the experimental environment.
```bash
pip3 install -r requirement.txt
```
2. Download the TabLib dataset from [Huggingface](https://huggingface.co/datasets/approximatelabs/tablib-v1-sample), and put the parquet files in a folder.
```
tablib-sample
|-0ad462e9.parquet
|-......
```

3. The train/validation/test datasets are split by:
```python
import os
file_list = os.listdir('tablib-sample') # all parquet files
# We ignore three files ('2d7d54b8', '8e1450ee', 'dc0e820c') for the memory issue
file_list = sorted(file_list) # fix the orders
train_size = int(len(file_list) * 0.6)
test_size = int(len(file_list) * 0.2)
val_size = len(file_list) - train_size - test_size
train_files = file_list[:train_size]
test_files = file_list[train_size:train_size+test_size] 
val_files = file_list[-val_size:]
```
4. Implement the base traditional estimators. Refer to [pydistinct](https://github.com/chanedwin/pydistinct) and [the paper](https://vldb.org/conf/1995/P311.PDF) for details, we can not provide the code of them for license issues.


5. Sampling and preprocess data. You will get the three pickle files in the `data/` folder.

- Get the table content by following the instructions provided by [TabLib]((https://huggingface.co/datasets/approximatelabs/tablib-v1-sample))
- For each column, record the number of rows (N), ground truth NDV (D), uniformly at random sample 1% data to build frequency profile (f).
- Each pickle file in the `data/` folder has 4 lists, the basic item of each list is shown as follows.

    - data_profile: f[1:H-3] || logn || logd || logN
    - rank_label: y^over || y^under
    - esimate_ndv: estimation results (in log) of the following base estimators: `['EB', 'GEE', 'Chao', 'Shlosser', 'ChaoLee', 'Goodman', 'Jackknife', 'Sichel', 'Method of Movement', 'Bootstrap', 'Horvitz Thompson', 'Method of Movement v2', 'Method of Movement v3', 'Smoothed Jackknife']`
    - D_list: ground truth NDV 

6. For more details, refer to `process_data.py` and implement it according to the comments. Then execute the script.

```bash
python3 process_data.py
```
7. Train the model and observe the results.

```bash
python3 train_adandv.py
```

8. Show the q-error distributions of base estimators and the learned estimator (the primary results of Table 3 in the paper).
```bash
python3 show_base.py
```

# LICENSE
The code in this repository is licensed under [MIT LICENSE](LICENSE)

