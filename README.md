# ANRL
ANRL: Attributed Network Representation Learning via Deep Neural Networks (IJCAI-18)

This is a Tensorflow implementation of the ANRL algorithm, which learns a low-dimensional representations for each node in a network. Specifically, ANRL consists of two modules, i.e., neighbor enhancement autoencoder and attribute-aware skip-gram model, to jointly capture the node attribute proximity and network topology proximity.

## Requirements
* python2.7 or python3.6
* tensorflow
* networkx
* numpy
* scipy
* scikit-learn

All required packages are defined in requirements.txt. To install all requirement, just use the following commands:
```
pip install -r requirements.txt
```

## Basic Usage

### Input Data 
For node classification, each dataset contains 3 files: edgelist, features and labels.
```
1. citeseer.edgelist: each line contains two connected nodes.
node_1 node_2
node_2 node_3
...

2. citeseer.feature: this file has n+1 lines.
The first line has the following format:
node_number feature_dimension
The next n lines are as follows: (each node per line ordered by node id)
(for node_1) feature_1 feature_2 ... feature_n
(for node_2) feature_1 feature_2 ... feature_n
...

3. citeseer.label: each line represents a node and its class label.
node_1 label_1
node_2 label_2
...
```
For link prediction, each dataset contains 3 files: training edgelist, features and test edgelist.
```
1. xxx_train.edgelist: each line contains two connected nodes.
node_1 node_2
node_2 node_3
...

2. xxx.feature: this file has n+1 lines.
The first line has the following format:
node_number feature_dimension
The next n lines are as follows: (each node per line ordered by node id)
(for node_1) feature_1 feature_2 ... feature_n
(for node_2) feature_1 feature_2 ... feature_n
...

3. xxx_test.edgelist: each line contains two connected nodes.
node_1 node_2 1 (positive sample)
node_2 node_3 0 (negative sample)
...
```

### Output Data
The output file has n+1 lines as the input feature files. The first line has the following format:
```
node_number embedding_dimension
```
The next n lines are as follows:
node_id dim_1, dim_2, ... dim_d

### Run
To run ANRL, just execute the following command for node classification task:
```
python main.py
```


Note:
As for simulating random walks, we directly use the code provided in [node2vec](https://github.com/aditya-grover/node2vec), which levearges alias sampling to faciliate the procedure.

## Citing
If you find ANRL useful for your research, please consider citing the following paper:
```
@inproceedings{ijcai2018-438,
  title     = {ANRL: Attributed Network Representation Learning via Deep Neural Networks},
  author    = {Zhen Zhang and Hongxia Yang and Jiajun Bu and Sheng Zhou and Pinggang Yu and Jianwei Zhang and Martin Ester and Can Wang},
  booktitle = {Proceedings of the Twenty-Seventh International Joint Conference on
               Artificial Intelligence, {IJCAI-18}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},             
  pages     = {3155--3161},
  year      = {2018},
  month     = {7},
  doi       = {10.24963/ijcai.2018/438},
  url       = {https://doi.org/10.24963/ijcai.2018/438},
}
``` 
